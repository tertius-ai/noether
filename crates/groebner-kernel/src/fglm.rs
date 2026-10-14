// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Coefficient-generic FGLM order conversion for zero-dimensional ideals.

use core::cmp::Ordering;
use std::collections::BinaryHeap;

use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use poly_core::{
    ArenaPolynomial, Monomial, MonomialArena, MonomialId, MonomialOrder, SparsePolynomial,
};

use crate::engine::{normal_form, validate_inputs, verify_compact};
use crate::{Field, GroebnerError, GroebnerOptions};

/// Work counters from a verified FGLM order conversion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FglmMetrics {
    /// Dimension of the source quotient algebra as a vector space.
    pub quotient_dimension: usize,
    /// Sparse columns constructed across the variable multiplication operators.
    pub multiplication_operator_columns: usize,
    /// Nonzero entries in those sparse columns.
    pub multiplication_operator_nonzeros: usize,
    /// Target-order monomials removed from the candidate queue.
    pub candidates_visited: usize,
    /// Linear-dependence relations that became target-order basis members.
    pub dependency_relations: usize,
    /// Largest number of nonzeros retained in the incremental echelon basis.
    pub peak_echelon_nonzeros: usize,
}

/// Exactly verified result of one FGLM order conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FglmOutput<C> {
    /// Reduced monic basis in the requested target order.
    pub basis: Vec<SparsePolynomial<C>>,
    /// Operation counters.
    pub metrics: FglmMetrics,
}

pub(crate) type SparseVector<C> = Vec<(usize, C)>;
pub(crate) type MultiplicationOperators<C> = Vec<Vec<SparseVector<C>>>;

/// Sparse standard-basis description of a finite-dimensional quotient
/// algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteQuotientAlgebraOutput<C> {
    pub(crate) arity: usize,
    pub(crate) dimension: usize,
    pub(crate) standard: Vec<Monomial>,
    pub(crate) operators: MultiplicationOperators<C>,
    pub(crate) operator_nonzeros: usize,
    pub(crate) is_unit: bool,
}

impl<C> FiniteQuotientAlgebraOutput<C> {
    /// Number of source variables and multiplication operators.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Vector-space dimension of the quotient algebra.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Canonically ordered standard monomials.
    pub fn standard_monomials(&self) -> &[Monomial] {
        &self.standard
    }

    /// Sparse columns for multiplication by each source variable.
    ///
    /// The outer index is the variable, the next index is the input basis
    /// column, and each `(row, coefficient)` pair is a nonzero output entry.
    pub fn multiplication_operators(&self) -> &[Vec<Vec<(usize, C)>>] {
        &self.operators
    }

    /// Total number of nonzero entries in all multiplication operators.
    pub fn multiplication_operator_nonzeros(&self) -> usize {
        self.operator_nonzeros
    }

    /// Whether the source ideal is the unit ideal and the quotient is zero.
    pub fn is_zero_algebra(&self) -> bool {
        self.is_unit
    }
}

/// Construct an exactly verified sparse finite-dimensional quotient algebra.
///
/// The source must be a reduced Gröbner basis of a zero-dimensional ideal.
/// Unlike a rational univariate representation, this description does not
/// require the quotient algebra to admit a primitive element.
pub fn finite_quotient_algebra<F: Field>(
    field: &F,
    source_basis: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<FiniteQuotientAlgebraOutput<F::Element>, GroebnerError> {
    let (arity, source_order) = validate_inputs(source_basis)?;
    let input_terms = source_basis
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let mut source_arena =
        MonomialArena::with_capacity(arity, source_order, input_terms.saturating_mul(4).max(64));
    let compact_source = source_basis
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut source_arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_compact(
        field,
        &compact_source,
        options.max_reduction_steps,
        &mut source_arena,
    )?;
    let is_unit = compact_source.iter().any(|polynomial| {
        polynomial
            .leading_term()
            .is_some_and(|(monomial, _)| source_arena.total_degree(monomial) == 0)
    });
    if is_unit {
        return Ok(FiniteQuotientAlgebraOutput {
            arity,
            dimension: 0,
            standard: Vec::new(),
            operators: (0..arity).map(|_| Vec::new()).collect(),
            operator_nonzeros: 0,
            is_unit: true,
        });
    }

    require_pure_power_bounds(&compact_source, &source_arena)?;
    let standard = standard_monomials(
        &compact_source,
        &source_arena,
        options.max_quotient_dimension,
        options.max_fglm_candidates,
    )?;
    let dimension = standard.len();
    let (_, standard_indices) = intern_standard_monomials(&standard, &mut source_arena)?;
    let (operators, operator_nonzeros) = multiplication_operators(
        field,
        &compact_source,
        &standard,
        &standard_indices,
        options.max_reduction_steps,
        &mut source_arena,
    )?;
    Ok(FiniteQuotientAlgebraOutput {
        arity,
        dimension,
        standard,
        operators,
        operator_nonzeros,
        is_unit: false,
    })
}

/// Convert a reduced Gröbner basis of a zero-dimensional ideal to `target_order`.
///
/// The algorithm is coefficient-generic over [`Field`]. It constructs sparse
/// quotient multiplication operators, performs classical incremental FGLM,
/// and then independently verifies the target basis and both ideal
/// containments. The source basis must itself be reduced and exactly verified.
pub fn fglm_basis<F: Field>(
    field: &F,
    source_basis: &[SparsePolynomial<F::Element>],
    target_order: MonomialOrder,
    options: &GroebnerOptions,
) -> Result<FglmOutput<F::Element>, GroebnerError> {
    let quotient = finite_quotient_algebra(field, source_basis, options)?;
    let arity = quotient.arity;
    if quotient.is_unit {
        let basis = vec![SparsePolynomial::new(
            arity,
            target_order,
            [(Monomial::one(arity), field.one())],
            field,
        )];
        verify_equivalent(field, source_basis, &basis, options.max_reduction_steps)?;
        return Ok(FglmOutput {
            basis,
            metrics: FglmMetrics::default(),
        });
    }

    let quotient_dimension = quotient.dimension;
    let one = Monomial::one(arity);
    let one_index = quotient
        .standard
        .iter()
        .position(|monomial| monomial == &one)
        .ok_or(GroebnerError::VerificationFailed)?;
    let mut one_vector = zero_vector(field, quotient_dimension);
    one_vector[one_index] = field.one();

    let mut metrics = FglmMetrics {
        quotient_dimension,
        multiplication_operator_columns: arity.saturating_mul(quotient_dimension),
        multiplication_operator_nonzeros: quotient.operator_nonzeros,
        ..FglmMetrics::default()
    };
    let mut queue = BinaryHeap::new();
    let mut scheduled = HashSet::new();
    schedule_candidate(
        &mut queue,
        &mut scheduled,
        QueuedMonomial {
            monomial: one,
            order: target_order,
            parent: None,
        },
        options.max_fglm_candidates,
    )?;

    let variables = (0..arity)
        .map(|variable| {
            let mut exponents = vec![0; arity];
            exponents[variable] = 1;
            Monomial::new(exponents.into_boxed_slice())
        })
        .collect::<Vec<_>>();
    let mut accepted_monomials = Vec::with_capacity(quotient_dimension);
    let mut accepted_vectors: Vec<Vec<F::Element>> = Vec::with_capacity(quotient_dimension);
    let mut echelon = LinearBasis::new(quotient_dimension);
    let mut leading_relations = Vec::<Monomial>::new();
    let mut relations = Vec::new();

    while let Some(candidate) = queue.pop() {
        metrics.candidates_visited = metrics.candidates_visited.saturating_add(1);
        if leading_relations
            .iter()
            .any(|leading| leading.divides(&candidate.monomial))
        {
            continue;
        }
        let vector = match candidate.parent {
            None => one_vector.clone(),
            Some((parent, variable)) => apply_operator(
                field,
                &quotient.operators[variable],
                &accepted_vectors[parent],
                quotient_dimension,
            ),
        };
        match echelon.classify(field, &vector, accepted_monomials.len())? {
            LinearDependence::Dependent(coefficients) => {
                let relation = SparsePolynomial::new(
                    arity,
                    target_order,
                    core::iter::once((candidate.monomial.clone(), field.one())).chain(
                        accepted_monomials
                            .iter()
                            .cloned()
                            .zip(coefficients)
                            .filter_map(|(monomial, coefficient)| {
                                let coefficient = field.neg(&coefficient);
                                (!field.is_zero(&coefficient)).then_some((monomial, coefficient))
                            }),
                    ),
                    field,
                );
                let leading = relation
                    .leading_monomial()
                    .ok_or(GroebnerError::VerificationFailed)?;
                if leading != &candidate.monomial {
                    return Err(GroebnerError::VerificationFailed);
                }
                leading_relations.push(candidate.monomial);
                relations.push(relation);
                metrics.dependency_relations = metrics.dependency_relations.saturating_add(1);
            }
            LinearDependence::Independent => {
                if accepted_monomials.len() == quotient_dimension {
                    return Err(GroebnerError::VerificationFailed);
                }
                let accepted = accepted_monomials.len();
                for (variable, generator) in variables.iter().enumerate() {
                    let monomial = candidate
                        .monomial
                        .checked_product(generator)
                        .map_err(|_| GroebnerError::ExponentOverflow)?;
                    schedule_candidate(
                        &mut queue,
                        &mut scheduled,
                        QueuedMonomial {
                            monomial,
                            order: target_order,
                            parent: Some((accepted, variable)),
                        },
                        options.max_fglm_candidates,
                    )?;
                }
                accepted_monomials.push(candidate.monomial);
                accepted_vectors.push(vector);
                metrics.peak_echelon_nonzeros = metrics
                    .peak_echelon_nonzeros
                    .max(echelon.nonzero_count(field));
            }
        }
    }
    if accepted_monomials.len() != quotient_dimension {
        return Err(GroebnerError::VerificationFailed);
    }

    relations.sort_by(|left, right| {
        right
            .leading_monomial()
            .expect("FGLM relations are nonzero")
            .compare(
                left.leading_monomial().expect("FGLM relations are nonzero"),
                target_order,
            )
    });
    verify_equivalent(field, source_basis, &relations, options.max_reduction_steps)?;
    Ok(FglmOutput {
        basis: relations,
        metrics,
    })
}

fn require_pure_power_bounds<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    arena: &MonomialArena,
) -> Result<(), GroebnerError> {
    for variable in 0..arena.arity() {
        let has_bound = basis.iter().any(|polynomial| {
            let Some((leading, _)) = polynomial.leading_term() else {
                return false;
            };
            arena.exponent(leading, variable) != 0
                && (0..arena.arity())
                    .filter(|candidate| *candidate != variable)
                    .all(|candidate| arena.exponent(leading, candidate) == 0)
        });
        if !has_bound {
            return Err(GroebnerError::NotZeroDimensional);
        }
    }
    Ok(())
}

fn standard_monomials<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    arena: &MonomialArena,
    max_dimension: usize,
    max_candidates: usize,
) -> Result<Vec<Monomial>, GroebnerError> {
    let leading = basis
        .iter()
        .filter_map(|polynomial| {
            polynomial
                .leading_term()
                .map(|(monomial, _)| arena.get(monomial))
        })
        .collect::<Vec<_>>();
    let variables = (0..arena.arity())
        .map(|variable| {
            let mut exponents = vec![0; arena.arity()];
            exponents[variable] = 1;
            Monomial::new(exponents.into_boxed_slice())
        })
        .collect::<Vec<_>>();
    let mut queue = BinaryHeap::new();
    let mut scheduled = HashSet::new();
    schedule_candidate(
        &mut queue,
        &mut scheduled,
        QueuedMonomial {
            monomial: Monomial::one(arena.arity()),
            order: arena.order(),
            parent: None,
        },
        max_candidates,
    )?;
    let mut standard = Vec::new();
    while let Some(candidate) = queue.pop() {
        if leading
            .iter()
            .any(|divisor| divisor.divides(&candidate.monomial))
        {
            continue;
        }
        if standard.len() == max_dimension {
            return Err(resource("FGLM quotient dimension", max_dimension));
        }
        for generator in &variables {
            schedule_candidate(
                &mut queue,
                &mut scheduled,
                QueuedMonomial {
                    monomial: candidate
                        .monomial
                        .checked_product(generator)
                        .map_err(|_| GroebnerError::ExponentOverflow)?,
                    order: arena.order(),
                    parent: None,
                },
                max_candidates,
            )?;
        }
        standard.push(candidate.monomial);
    }
    Ok(standard)
}

fn intern_standard_monomials(
    standard: &[Monomial],
    arena: &mut MonomialArena,
) -> Result<(Vec<MonomialId>, HashMap<MonomialId, usize>), GroebnerError> {
    let mut ids = Vec::with_capacity(standard.len());
    let mut indices = HashMap::with_capacity(standard.len());
    for (index, monomial) in standard.iter().enumerate() {
        let id = arena
            .intern(monomial.clone())
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        ids.push(id);
        indices.insert(id, index);
    }
    Ok((ids, indices))
}

fn multiplication_operators<F: Field>(
    field: &F,
    source_basis: &[ArenaPolynomial<F::Element>],
    standard: &[Monomial],
    standard_indices: &HashMap<MonomialId, usize>,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(MultiplicationOperators<F::Element>, usize), GroebnerError> {
    let variables = (0..arena.arity())
        .map(|variable| {
            let mut exponents = vec![0; arena.arity()];
            exponents[variable] = 1;
            Monomial::new(exponents.into_boxed_slice())
        })
        .collect::<Vec<_>>();
    let mut operators = Vec::with_capacity(arena.arity());
    let mut nonzeros = 0_usize;
    for variable in variables {
        let mut columns = Vec::with_capacity(standard.len());
        for monomial in standard {
            let product = monomial
                .checked_product(&variable)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            let product = arena
                .intern(product)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            let value = ArenaPolynomial::from_sorted_terms(arena, [(product, field.one())], field);
            let remainder = normal_form(field, value, source_basis, max_steps, arena)?;
            let column = remainder
                .iter()
                .map(|(term, coefficient)| {
                    standard_indices
                        .get(&term)
                        .copied()
                        .map(|index| (index, coefficient.clone()))
                        .ok_or(GroebnerError::VerificationFailed)
                })
                .collect::<Result<Vec<_>, _>>()?;
            nonzeros = nonzeros.saturating_add(column.len());
            columns.push(column);
        }
        operators.push(columns);
    }
    Ok((operators, nonzeros))
}

pub(crate) fn apply_operator<F: Field>(
    field: &F,
    columns: &[SparseVector<F::Element>],
    vector: &[F::Element],
    dimension: usize,
) -> Vec<F::Element> {
    let mut output = zero_vector(field, dimension);
    for (column, coefficient) in columns.iter().zip(vector) {
        if field.is_zero(coefficient) {
            continue;
        }
        for (row, value) in column {
            let product = field.mul(coefficient, value);
            output[*row] = field.add(&output[*row], &product);
        }
    }
    output
}

pub(crate) fn zero_vector<F: Field>(field: &F, dimension: usize) -> Vec<F::Element> {
    (0..dimension).map(|_| field.zero()).collect()
}

struct PivotRow<C> {
    pivot: usize,
    vector: Vec<C>,
    coordinates: Vec<C>,
}

pub(crate) struct LinearBasis<C> {
    dimension: usize,
    rows: Vec<PivotRow<C>>,
}

pub(crate) enum LinearDependence<C> {
    Dependent(Vec<C>),
    Independent,
}

impl<C: Clone + Eq> LinearBasis<C> {
    pub(crate) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            rows: Vec::with_capacity(dimension),
        }
    }

    pub(crate) fn classify<F: Field<Element = C>>(
        &mut self,
        field: &F,
        value: &[C],
        accepted: usize,
    ) -> Result<LinearDependence<C>, GroebnerError> {
        if value.len() != self.dimension || accepted != self.rows.len() {
            return Err(GroebnerError::VerificationFailed);
        }
        let mut working = value.to_vec();
        let mut coordinates = zero_vector(field, accepted);
        for row in &self.rows {
            let scalar = working[row.pivot].clone();
            if field.is_zero(&scalar) {
                continue;
            }
            subtract_multiple(field, &mut working, &row.vector, &scalar);
            for (target, source) in coordinates.iter_mut().zip(&row.coordinates) {
                let product = field.mul(&scalar, source);
                *target = field.add(target, &product);
            }
        }
        let Some(pivot) = working.iter().position(|value| !field.is_zero(value)) else {
            return Ok(LinearDependence::Dependent(coordinates));
        };
        let inverse = field.inverse(&working[pivot]);
        for value in &mut working {
            *value = field.mul(value, &inverse);
        }
        let mut row_coordinates = coordinates
            .iter()
            .map(|coefficient| field.neg(&field.mul(coefficient, &inverse)))
            .collect::<Vec<_>>();
        row_coordinates.push(inverse);
        let position = self
            .rows
            .binary_search_by_key(&pivot, |row| row.pivot)
            .unwrap_or_else(|position| position);
        self.rows.insert(
            position,
            PivotRow {
                pivot,
                vector: working,
                coordinates: row_coordinates,
            },
        );
        Ok(LinearDependence::Independent)
    }

    fn nonzero_count<F: Field<Element = C>>(&self, field: &F) -> usize {
        self.rows
            .iter()
            .map(|row| {
                row.vector
                    .iter()
                    .filter(|value| !field.is_zero(value))
                    .count()
                    + row
                        .coordinates
                        .iter()
                        .filter(|value| !field.is_zero(value))
                        .count()
            })
            .sum()
    }
}

fn subtract_multiple<F: Field>(
    field: &F,
    target: &mut [F::Element],
    source: &[F::Element],
    scalar: &F::Element,
) {
    for (target, source) in target.iter_mut().zip(source) {
        let product = field.mul(scalar, source);
        *target = field.add(target, &field.neg(&product));
    }
}

#[derive(Clone)]
struct QueuedMonomial {
    monomial: Monomial,
    order: MonomialOrder,
    parent: Option<(usize, usize)>,
}

impl PartialEq for QueuedMonomial {
    fn eq(&self, other: &Self) -> bool {
        self.monomial == other.monomial
    }
}

impl Eq for QueuedMonomial {}

impl PartialOrd for QueuedMonomial {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedMonomial {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert_eq!(self.order, other.order);
        other.monomial.compare(&self.monomial, self.order)
    }
}

fn schedule_candidate(
    queue: &mut BinaryHeap<QueuedMonomial>,
    scheduled: &mut HashSet<Monomial>,
    candidate: QueuedMonomial,
    limit: usize,
) -> Result<(), GroebnerError> {
    if !scheduled.insert(candidate.monomial.clone()) {
        return Ok(());
    }
    if scheduled.len() > limit {
        return Err(resource("FGLM candidates", limit));
    }
    queue.push(candidate);
    Ok(())
}

fn verify_equivalent<F: Field>(
    field: &F,
    source: &[SparsePolynomial<F::Element>],
    target: &[SparsePolynomial<F::Element>],
    max_steps: usize,
) -> Result<(), GroebnerError> {
    let first = source.first().ok_or(GroebnerError::ContextMismatch)?;
    let arity = first.arity();
    let source_order = first.order();
    let target_order = target.first().map_or(source_order, SparsePolynomial::order);

    let mut source_arena = MonomialArena::new(arity, source_order);
    let source_compact = source
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut source_arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    for polynomial in target {
        let reordered = reorder(polynomial, source_order, field);
        let compact = ArenaPolynomial::from_sparse(&mut source_arena, &reordered, field)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        if !normal_form(
            field,
            compact,
            &source_compact,
            max_steps,
            &mut source_arena,
        )?
        .is_zero()
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }

    let mut target_arena = MonomialArena::new(arity, target_order);
    let target_compact = target
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut target_arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_compact(field, &target_compact, max_steps, &mut target_arena)?;
    for polynomial in source {
        let reordered = reorder(polynomial, target_order, field);
        let compact = ArenaPolynomial::from_sparse(&mut target_arena, &reordered, field)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        if !normal_form(
            field,
            compact,
            &target_compact,
            max_steps,
            &mut target_arena,
        )?
        .is_zero()
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn reorder<C: Clone + Eq, R: poly_core::CoefficientRing<Element = C>>(
    polynomial: &SparsePolynomial<C>,
    order: MonomialOrder,
    ring: &R,
) -> SparsePolynomial<C> {
    SparsePolynomial::new(
        polynomial.arity(),
        order,
        polynomial.terms().iter().cloned(),
        ring,
    )
}

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{groebner_basis_prime, PrimeField32};

    fn polynomial(
        field: &PrimeField32,
        order: MonomialOrder,
        terms: &[(i64, &[u32])],
    ) -> SparsePolynomial<u32> {
        SparsePolynomial::new(
            terms.first().map_or(0, |(_, exponents)| exponents.len()),
            order,
            terms.iter().map(|(coefficient, exponents)| {
                (
                    Monomial::new(exponents.to_vec().into_boxed_slice()),
                    field.element(*coefficient),
                )
            }),
            field,
        )
    }

    fn reorder_prime(
        polynomial: &SparsePolynomial<u32>,
        order: MonomialOrder,
        field: &PrimeField32,
    ) -> SparsePolynomial<u32> {
        reorder(polynomial, order, field)
    }

    #[test]
    fn fglm_matches_a_direct_lexicographic_basis() {
        let field = PrimeField32::new(101).unwrap();
        let source_order = MonomialOrder::GradedReverseLexicographic;
        let target_order = MonomialOrder::Lexicographic;
        let generators = vec![
            polynomial(&field, source_order, &[(1, &[2, 0]), (-1, &[0, 1])]),
            polynomial(&field, source_order, &[(1, &[0, 2]), (-1, &[1, 0])]),
        ];
        let source = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        let converted =
            fglm_basis(&field, &source, target_order, &GroebnerOptions::default()).unwrap();
        let target_generators = generators
            .iter()
            .map(|value| reorder_prime(value, target_order, &field))
            .collect::<Vec<_>>();
        let direct =
            groebner_basis_prime(&field, &target_generators, &GroebnerOptions::default()).unwrap();

        assert_eq!(converted.basis, direct.basis);
        assert_eq!(converted.metrics.quotient_dimension, 4);
        assert_eq!(converted.metrics.multiplication_operator_columns, 8);
    }

    #[test]
    fn positive_dimensional_input_is_rejected_explicitly() {
        let field = PrimeField32::new(7).unwrap();
        let order = MonomialOrder::GradedReverseLexicographic;
        let generators = vec![polynomial(&field, order, &[(1, &[1, 1])])];
        let source = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        assert_eq!(
            fglm_basis(
                &field,
                &source,
                MonomialOrder::Lexicographic,
                &GroebnerOptions::default(),
            ),
            Err(GroebnerError::NotZeroDimensional)
        );
    }

    #[test]
    fn quotient_dimension_limit_is_enforced_before_linear_algebra() {
        let field = PrimeField32::new(7).unwrap();
        let order = MonomialOrder::GradedReverseLexicographic;
        let generators = vec![
            polynomial(&field, order, &[(1, &[3, 0])]),
            polynomial(&field, order, &[(1, &[0, 3])]),
        ];
        let source = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        let options = GroebnerOptions {
            max_quotient_dimension: 8,
            ..GroebnerOptions::default()
        };
        assert_eq!(
            fglm_basis(&field, &source, MonomialOrder::Lexicographic, &options),
            Err(GroebnerError::ResourceLimit {
                resource: "FGLM quotient dimension",
                limit: 8,
            })
        );
    }
}
