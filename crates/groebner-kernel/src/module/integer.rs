// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strong module Gröbner arithmetic over the integers.

use std::collections::VecDeque;

use flint_core::Integer;
use module_core::ModuleOrder;
use poly_core::{ArenaPolynomial, Monomial, MonomialArena, MonomialId};

use super::{
    compare_terms, resource, ArenaModuleElement, SparseModuleElement, SparseModuleOrder,
    SparseModuleTerm,
};
use crate::engine::GroebnerError;
use crate::integer::{
    extended_gcd, IntegerCoefficientRing, StrongGroebnerMetrics, StrongGroebnerOptions,
};

type IntegerPolynomial = ArenaPolynomial<Integer>;
type IntegerRepresentation = Vec<IntegerPolynomial>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrackedIntegerModuleElement {
    value: ArenaModuleElement<Integer>,
    representation: IntegerRepresentation,
}

/// Verified normalized strong module basis over `Z`, with exact coordinates
/// in the supplied generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongModuleGroebnerOutput {
    /// Positive-leading-coefficient strong basis in deterministic order.
    pub basis: Vec<SparseModuleElement<Integer>>,
    /// Basis-major polynomial change matrix into the input generators.
    pub transformations: Vec<Vec<poly_core::SparsePolynomial<Integer>>>,
    /// S/G-pair and insertion counters.
    pub metrics: StrongGroebnerMetrics,
}

/// Euclidean strong module reduction with exact quotient multipliers.
///
/// The result satisfies
/// `value = sum(quotients[i] * basis[i]) + remainder` over `Z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongModuleNormalFormOutput {
    /// Canonical Euclidean remainder for the declared reducer order.
    pub remainder: SparseModuleElement<Integer>,
    /// Polynomial multiplier for each reducer, in input order.
    pub quotients: Vec<poly_core::SparsePolynomial<Integer>>,
}

/// Verified generators of all relations among an integral module-generator
/// family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongModuleKernelOutput {
    /// Strong-basis generators in the original source-generator frame.
    pub generators: Vec<SparseModuleElement<Integer>>,
    /// Counters for the graph-module elimination computation.
    pub metrics: StrongGroebnerMetrics,
}

/// Compute a normalized strong Gröbner basis of a submodule of a polynomial
/// free module over `Z`.
///
/// S-polynomials control monomial overlap while G-polynomials close leading
/// coefficient ideals. Every returned basis value is reconstructed from the
/// exact transformation row, and a separate S/G-pair sweep verifies
/// completeness.
pub fn strong_module_groebner_basis(
    generators: &[SparseModuleElement<Integer>],
    order: &SparseModuleOrder,
    options: &StrongGroebnerOptions,
) -> Result<StrongModuleGroebnerOutput, GroebnerError> {
    let (mut arena, packed_order, inputs) = pack_inputs(generators, order)?;
    let one = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let ring = IntegerCoefficientRing;
    let mut basis = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        if input.is_zero() {
            continue;
        }
        let mut representation = (0..inputs.len())
            .map(|_| IntegerPolynomial::zero(&arena))
            .collect::<Vec<_>>();
        representation[index] =
            IntegerPolynomial::new(&arena, [(one, Integer::one())], &IntegerCoefficientRing);
        let entry = normalize_tracked_sign(
            TrackedIntegerModuleElement {
                value: input.clone(),
                representation,
            },
            &ring,
            &arena,
        );
        if !basis
            .iter()
            .any(|known: &TrackedIntegerModuleElement| known.value == entry.value)
        {
            basis.push(entry);
        }
    }
    if basis.len() > options.max_basis_size {
        return Err(resource("strong module basis size", options.max_basis_size));
    }

    let mut pending = VecDeque::new();
    for right in 1..basis.len() {
        install_pairs(&basis, right, &mut pending, options.max_pending_pairs)?;
    }
    let mut metrics = StrongGroebnerMetrics {
        peak_pending_pairs: pending.len(),
        ..StrongGroebnerMetrics::default()
    };

    while let Some((left, right)) = pending.pop_front() {
        if metrics.pairs_processed == options.max_processed_pairs {
            return Err(resource(
                "strong module processed pairs",
                options.max_processed_pairs,
            ));
        }
        metrics.pairs_processed += 1;
        let s = integer_module_s_element(&basis[left], &basis[right], &packed_order, &mut arena)?;
        metrics.s_polynomials += 1;
        let g = integer_module_g_element(&basis[left], &basis[right], &packed_order, &mut arena)?;
        metrics.g_polynomials += 1;
        for candidate in [s, g] {
            let remainder = tracked_normal_form(
                candidate,
                &basis,
                &packed_order,
                options.max_reduction_steps,
                &mut arena,
            )?;
            if remainder.value.is_zero() {
                continue;
            }
            let remainder = normalize_tracked_sign(remainder, &ring, &arena);
            if basis.iter().any(|known| known.value == remainder.value) {
                continue;
            }
            if basis.len() == options.max_basis_size {
                return Err(resource("strong module basis size", options.max_basis_size));
            }
            let new_index = basis.len();
            basis.push(remainder);
            metrics.basis_insertions += 1;
            install_pairs(&basis, new_index, &mut pending, options.max_pending_pairs)?;
            metrics.peak_pending_pairs = metrics.peak_pending_pairs.max(pending.len());
        }
    }

    autoreduce_tracked(
        &mut basis,
        &packed_order,
        options.max_reduction_steps,
        &mut arena,
    )?;
    verify_strong_module_basis(
        &inputs,
        &basis,
        &packed_order,
        options.max_reduction_steps,
        &mut arena,
    )?;

    Ok(StrongModuleGroebnerOutput {
        basis: basis
            .iter()
            .map(|entry| entry.value.to_sparse(&ring, &arena))
            .collect(),
        transformations: basis
            .iter()
            .map(|entry| {
                entry
                    .representation
                    .iter()
                    .map(|polynomial| polynomial.to_sparse(&arena, &ring))
                    .collect()
            })
            .collect(),
        metrics,
    })
}

/// Compute Euclidean strong normal form over `Z`.
pub fn strong_module_normal_form(
    value: &SparseModuleElement<Integer>,
    basis: &[SparseModuleElement<Integer>],
    order: &SparseModuleOrder,
    max_steps: usize,
) -> Result<SparseModuleElement<Integer>, GroebnerError> {
    Ok(strong_module_normal_form_with_quotients(value, basis, order, max_steps)?.remainder)
}

/// Compute Euclidean strong normal form and exact reducer multipliers over
/// `Z`.
pub fn strong_module_normal_form_with_quotients(
    value: &SparseModuleElement<Integer>,
    basis: &[SparseModuleElement<Integer>],
    order: &SparseModuleOrder,
    max_steps: usize,
) -> Result<StrongModuleNormalFormOutput, GroebnerError> {
    if order.base_order() != value.polynomial_order()
        || basis.iter().any(|entry| {
            entry.arity() != value.arity()
                || entry.rank() != value.rank()
                || entry.polynomial_order() != value.polynomial_order()
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = value
        .terms()
        .len()
        .saturating_add(basis.iter().map(|entry| entry.terms().len()).sum());
    let mut arena = MonomialArena::with_capacity(
        value.arity(),
        value.polynomial_order(),
        input_terms.max(256),
    );
    let packed_order = order.pack(&mut arena, value.rank())?;
    let ring = IntegerCoefficientRing;
    let packed_value = ArenaModuleElement::from_sparse(&ring, value, &packed_order, &mut arena)?;
    let packed_basis = basis
        .iter()
        .map(|entry| ArenaModuleElement::from_sparse(&ring, entry, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    require_positive_leaders(&packed_basis)?;
    let (remainder, quotients) = normal_form_with_quotients_arena(
        packed_value,
        &packed_basis,
        &packed_order,
        max_steps,
        &mut arena,
    )?;
    Ok(StrongModuleNormalFormOutput {
        remainder: remainder.to_sparse(&ring, &arena),
        quotients: quotients
            .iter()
            .map(|quotient| quotient.to_sparse(&arena, &ring))
            .collect(),
    })
}

/// Compute all relations among integral module generators through exact
/// graph-module elimination.
///
/// Position-over-term places every target component before every source
/// component. The pure-source part of the resulting strong basis is therefore
/// the graph module's intersection with the source summand, hence the full
/// kernel. The extracted relations are strong-autoreduced again in their own
/// source frame and checked by direct substitution.
pub fn strong_module_relations(
    generators: &[SparseModuleElement<Integer>],
    options: &StrongGroebnerOptions,
) -> Result<StrongModuleKernelOutput, GroebnerError> {
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    if generators.iter().any(|entry| {
        entry.arity() != first.arity()
            || entry.rank() != first.rank()
            || entry.polynomial_order() != first.polynomial_order()
    }) {
        return Err(GroebnerError::ContextMismatch);
    }
    let ring = IntegerCoefficientRing;
    let target_rank = first.rank();
    let source_rank = generators.len();
    let one = Monomial::new(vec![0; first.arity()].into_boxed_slice());
    let graph = generators
        .iter()
        .enumerate()
        .map(|(source_component, generator)| {
            SparseModuleElement::new(
                first.arity(),
                target_rank.saturating_add(source_rank),
                first.polynomial_order(),
                generator.terms().iter().cloned().chain([(
                    SparseModuleTerm::new(
                        one.clone(),
                        target_rank.saturating_add(source_component),
                    ),
                    Integer::one(),
                )]),
                &ring,
            )
        })
        .collect::<Vec<_>>();
    let elimination_order = SparseModuleOrder::PositionOverTerm(first.polynomial_order());
    let graph_basis = strong_module_groebner_basis(&graph, &elimination_order, options)?;
    let mut relations = graph_basis
        .basis
        .iter()
        .filter(|entry| {
            entry
                .terms()
                .iter()
                .all(|(term, _)| term.component() >= target_rank)
        })
        .map(|entry| {
            SparseModuleElement::new(
                first.arity(),
                source_rank,
                first.polynomial_order(),
                entry.terms().iter().map(|(term, coefficient)| {
                    (
                        SparseModuleTerm::new(
                            term.monomial().clone(),
                            term.component() - target_rank,
                        ),
                        coefficient.clone(),
                    )
                }),
                &ring,
            )
        })
        .collect::<Vec<_>>();
    if !relations.is_empty() {
        relations = strong_module_groebner_basis(&relations, &elimination_order, options)?.basis;
    }
    verify_relations(generators, &relations)?;
    Ok(StrongModuleKernelOutput {
        generators: relations,
        metrics: graph_basis.metrics,
    })
}

fn pack_inputs(
    generators: &[SparseModuleElement<Integer>],
    order: &SparseModuleOrder,
) -> Result<(MonomialArena, ModuleOrder, Vec<ArenaModuleElement<Integer>>), GroebnerError> {
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    if order.base_order() != first.polynomial_order()
        || generators.iter().any(|entry| {
            entry.arity() != first.arity()
                || entry.rank() != first.rank()
                || entry.polynomial_order() != first.polynomial_order()
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = generators
        .iter()
        .map(|entry| entry.terms().len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        first.arity(),
        first.polynomial_order(),
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let packed_order = order.pack(&mut arena, first.rank())?;
    let inputs = generators
        .iter()
        .map(|entry| {
            ArenaModuleElement::from_sparse(
                &IntegerCoefficientRing,
                entry,
                &packed_order,
                &mut arena,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((arena, packed_order, inputs))
}

fn install_pairs(
    basis: &[TrackedIntegerModuleElement],
    new_index: usize,
    pending: &mut VecDeque<(usize, usize)>,
    max_pending_pairs: usize,
) -> Result<(), GroebnerError> {
    let new_component = basis[new_index]
        .value
        .leading_term()
        .expect("installed strong basis member is nonzero")
        .0
        .component();
    for (old, entry) in basis.iter().enumerate().take(new_index) {
        if entry
            .value
            .leading_term()
            .expect("installed strong basis member is nonzero")
            .0
            .component()
            != new_component
        {
            continue;
        }
        if pending.len() == max_pending_pairs {
            return Err(resource("strong module pending pairs", max_pending_pairs));
        }
        pending.push_back((old, new_index));
    }
    Ok(())
}

fn integer_module_s_element(
    left: &TrackedIntegerModuleElement,
    right: &TrackedIntegerModuleElement,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<TrackedIntegerModuleElement, GroebnerError> {
    let (left_term, left_coefficient) = left.value.leading_term().expect("nonzero basis member");
    let (right_term, right_coefficient) = right.value.leading_term().expect("nonzero basis member");
    let lcm = arena
        .lcm(left_term.monomial(), right_term.monomial())
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let coefficient_lcm = left_coefficient.abs().lcm(&right_coefficient.abs());
    tracked_pair_combination(
        left,
        arena
            .quotient(lcm, left_term.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?,
        &(&coefficient_lcm / left_coefficient),
        right,
        arena
            .quotient(lcm, right_term.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?,
        &(&coefficient_lcm / right_coefficient),
        false,
        order,
        arena,
    )
}

fn integer_module_g_element(
    left: &TrackedIntegerModuleElement,
    right: &TrackedIntegerModuleElement,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<TrackedIntegerModuleElement, GroebnerError> {
    let (left_term, left_coefficient) = left.value.leading_term().expect("nonzero basis member");
    let (right_term, right_coefficient) = right.value.leading_term().expect("nonzero basis member");
    let lcm = arena
        .lcm(left_term.monomial(), right_term.monomial())
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let (_, left_bezout, right_bezout) = extended_gcd(left_coefficient, right_coefficient);
    tracked_pair_combination(
        left,
        arena
            .quotient(lcm, left_term.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?,
        &left_bezout,
        right,
        arena
            .quotient(lcm, right_term.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?,
        &right_bezout,
        true,
        order,
        arena,
    )
}

#[allow(clippy::too_many_arguments)]
fn tracked_pair_combination(
    left: &TrackedIntegerModuleElement,
    left_monomial: MonomialId,
    left_scalar: &Integer,
    right: &TrackedIntegerModuleElement,
    right_monomial: MonomialId,
    right_scalar: &Integer,
    add_right: bool,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<TrackedIntegerModuleElement, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let left_value = left
        .value
        .term_mul(&ring, left_monomial, left_scalar, order, arena)?;
    let right_value = right
        .value
        .term_mul(&ring, right_monomial, right_scalar, order, arena)?;
    let value = if add_right {
        add_module_values(&left_value, &right_value, order, arena)?
    } else {
        left_value.sub(&ring, &right_value, order, arena)?
    };
    if left.representation.len() != right.representation.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut representation = Vec::with_capacity(left.representation.len());
    for (left_polynomial, right_polynomial) in left.representation.iter().zip(&right.representation)
    {
        let left_polynomial = left_polynomial
            .term_mul(left_monomial, left_scalar, arena, &ring)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        let right_polynomial = right_polynomial
            .term_mul(right_monomial, right_scalar, arena, &ring)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        representation.push(if add_right {
            left_polynomial.add(&right_polynomial, arena, &ring)
        } else {
            left_polynomial.sub(&right_polynomial, arena, &ring)
        });
    }
    Ok(TrackedIntegerModuleElement {
        value,
        representation,
    })
}

fn tracked_normal_form(
    mut current: TrackedIntegerModuleElement,
    basis: &[TrackedIntegerModuleElement],
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<TrackedIntegerModuleElement, GroebnerError> {
    let ring = IntegerCoefficientRing;
    require_positive_leaders(
        &basis
            .iter()
            .map(|entry| entry.value.clone())
            .collect::<Vec<_>>(),
    )?;
    let mut remainder = Vec::new();
    let mut steps = 0;
    while let Some((leading, coefficient)) = current.value.leading_term() {
        if steps == max_steps {
            return Err(resource("strong module normal-form steps", max_steps));
        }
        let reduction = basis.iter().enumerate().find_map(|(index, reducer)| {
            let (reducer_term, reducer_coefficient) = reducer.value.leading_term()?;
            if reducer_term.component() != leading.component() {
                return None;
            }
            let multiplier = arena
                .quotient(leading.monomial(), reducer_term.monomial())
                .ok()??;
            let quotient = coefficient.div_floor(&reducer_coefficient.abs());
            (!quotient.is_zero()).then_some((index, multiplier, quotient))
        });
        if let Some((index, multiplier, quotient)) = reduction {
            let multiple = basis[index]
                .value
                .term_mul(&ring, multiplier, &quotient, order, arena)?;
            current.value = current.value.sub(&ring, &multiple, order, arena)?;
            for (coordinate, reducer_coordinate) in current
                .representation
                .iter_mut()
                .zip(&basis[index].representation)
            {
                let multiple = reducer_coordinate
                    .term_mul(multiplier, &quotient, arena, &ring)
                    .map_err(|_| GroebnerError::ExponentOverflow)?;
                *coordinate = coordinate.sub(&multiple, arena, &ring);
            }
        } else {
            remainder.push((leading, coefficient.clone()));
            current.value = ArenaModuleElement::new(
                &ring,
                current.value.rank,
                current
                    .value
                    .iter()
                    .skip(1)
                    .map(|(term, coefficient)| (term, coefficient.clone())),
                order,
                arena,
            )?;
        }
        steps += 1;
    }
    current.value = ArenaModuleElement::new(&ring, current.value.rank, remainder, order, arena)?;
    Ok(current)
}

fn normal_form_with_quotients_arena(
    mut current: ArenaModuleElement<Integer>,
    basis: &[ArenaModuleElement<Integer>],
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(ArenaModuleElement<Integer>, Vec<IntegerPolynomial>), GroebnerError> {
    let ring = IntegerCoefficientRing;
    let rank = current.rank;
    let mut remainder = Vec::new();
    let mut quotients = basis
        .iter()
        .map(|_| IntegerPolynomial::zero(arena))
        .collect::<Vec<_>>();
    let mut steps = 0;
    while let Some((leading, coefficient)) = current.leading_term() {
        if steps == max_steps {
            return Err(resource("strong module normal-form steps", max_steps));
        }
        let reduction = basis.iter().enumerate().find_map(|(index, reducer)| {
            let (reducer_term, reducer_coefficient) = reducer.leading_term()?;
            if reducer_term.component() != leading.component() {
                return None;
            }
            let multiplier = arena
                .quotient(leading.monomial(), reducer_term.monomial())
                .ok()??;
            let quotient = coefficient.div_floor(&reducer_coefficient.abs());
            (!quotient.is_zero()).then_some((index, multiplier, quotient))
        });
        if let Some((index, multiplier, quotient)) = reduction {
            let multiple = basis[index].term_mul(&ring, multiplier, &quotient, order, arena)?;
            current = current.sub(&ring, &multiple, order, arena)?;
            let quotient_term = IntegerPolynomial::new(&*arena, [(multiplier, quotient)], &ring);
            quotients[index] = quotients[index].add(&quotient_term, arena, &ring);
        } else {
            remainder.push((leading, coefficient.clone()));
            current = ArenaModuleElement::new(
                &ring,
                rank,
                current
                    .iter()
                    .skip(1)
                    .map(|(term, coefficient)| (term, coefficient.clone())),
                order,
                arena,
            )?;
        }
        steps += 1;
    }
    Ok((
        ArenaModuleElement::new(&ring, rank, remainder, order, arena)?,
        quotients,
    ))
}

fn autoreduce_tracked(
    basis: &mut Vec<TrackedIntegerModuleElement>,
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    let ring = IntegerCoefficientRing;
    let mut changed = true;
    let mut passes = 0;
    while changed && passes <= basis.len().saturating_add(1) {
        changed = false;
        let mut index = 0;
        while index < basis.len() {
            let reducers = basis
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, entry)| entry.clone())
                .collect::<Vec<_>>();
            let reduced = normalize_tracked_sign(
                tracked_normal_form(basis[index].clone(), &reducers, order, max_steps, arena)?,
                &ring,
                arena,
            );
            if reduced.value.is_zero() {
                basis.remove(index);
                changed = true;
            } else {
                if reduced != basis[index] {
                    basis[index] = reduced;
                    changed = true;
                }
                index += 1;
            }
        }
        deduplicate_tracked(basis);
        passes += 1;
    }
    if changed {
        return Err(GroebnerError::VerificationFailed);
    }
    sort_tracked(basis, order, arena)
}

fn verify_strong_module_basis(
    inputs: &[ArenaModuleElement<Integer>],
    basis: &[TrackedIntegerModuleElement],
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    let ring = IntegerCoefficientRing;
    let values = basis
        .iter()
        .map(|entry| entry.value.clone())
        .collect::<Vec<_>>();
    require_positive_leaders(&values)?;
    for entry in basis {
        if entry.representation.len() != inputs.len() {
            return Err(GroebnerError::VerificationFailed);
        }
        let mut represented = ArenaModuleElement::zero(entry.value.rank);
        for (input, multiplier) in inputs.iter().zip(&entry.representation) {
            let product = input.polynomial_mul(&ring, multiplier, order, arena)?;
            represented = add_module_values(&represented, &product, order, arena)?;
        }
        if represented != entry.value {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    for input in inputs {
        let (remainder, _) =
            normal_form_with_quotients_arena(input.clone(), &values, order, max_steps, arena)?;
        if !remainder.is_zero() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    for right in 1..basis.len() {
        let right_component = basis[right]
            .value
            .leading_term()
            .expect("nonzero")
            .0
            .component();
        for left in 0..right {
            if basis[left]
                .value
                .leading_term()
                .expect("nonzero")
                .0
                .component()
                != right_component
            {
                continue;
            }
            for pair in [
                integer_module_s_element(&basis[left], &basis[right], order, arena)?,
                integer_module_g_element(&basis[left], &basis[right], order, arena)?,
            ] {
                let (remainder, _) =
                    normal_form_with_quotients_arena(pair.value, &values, order, max_steps, arena)?;
                if !remainder.is_zero() {
                    return Err(GroebnerError::VerificationFailed);
                }
            }
        }
    }
    for index in 0..values.len() {
        let reducers = values
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let (remainder, _) = normal_form_with_quotients_arena(
            values[index].clone(),
            &reducers,
            order,
            max_steps,
            arena,
        )?;
        if remainder != values[index] {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn require_positive_leaders(basis: &[ArenaModuleElement<Integer>]) -> Result<(), GroebnerError> {
    if basis.iter().any(|entry| {
        entry
            .leading_term()
            .is_none_or(|(_, coefficient)| !coefficient.is_positive())
    }) {
        Err(GroebnerError::VerificationFailed)
    } else {
        Ok(())
    }
}

fn normalize_tracked_sign(
    mut value: TrackedIntegerModuleElement,
    ring: &IntegerCoefficientRing,
    arena: &MonomialArena,
) -> TrackedIntegerModuleElement {
    if value
        .value
        .leading_term()
        .is_some_and(|(_, coefficient)| coefficient.is_negative())
    {
        let negative_one = Integer::from(-1);
        value.value = value.value.scalar_mul(ring, &negative_one);
        for coordinate in &mut value.representation {
            *coordinate = coordinate.scalar_mul(&negative_one, arena, ring);
        }
    }
    value
}

fn add_module_values(
    left: &ArenaModuleElement<Integer>,
    right: &ArenaModuleElement<Integer>,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<ArenaModuleElement<Integer>, GroebnerError> {
    left.sub(
        &IntegerCoefficientRing,
        &right.scalar_mul(&IntegerCoefficientRing, &Integer::from(-1)),
        order,
        arena,
    )
}

fn deduplicate_tracked(basis: &mut Vec<TrackedIntegerModuleElement>) {
    let mut unique = Vec::with_capacity(basis.len());
    for entry in basis.drain(..) {
        if !unique
            .iter()
            .any(|known: &TrackedIntegerModuleElement| known.value == entry.value)
        {
            unique.push(entry);
        }
    }
    *basis = unique;
}

fn sort_tracked(
    basis: &mut [TrackedIntegerModuleElement],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for index in 1..basis.len() {
        let mut cursor = index;
        while cursor != 0 {
            let (left_term, left_coefficient) = basis[cursor]
                .value
                .leading_term()
                .expect("nonzero strong basis member");
            let (right_term, right_coefficient) = basis[cursor - 1]
                .value
                .leading_term()
                .expect("nonzero strong basis member");
            let comparison = compare_terms(order, left_term, right_term, arena)?
                .then_with(|| right_coefficient.cmp(left_coefficient));
            if comparison != core::cmp::Ordering::Greater {
                break;
            }
            basis.swap(cursor, cursor - 1);
            cursor -= 1;
        }
    }
    Ok(())
}

fn verify_relations(
    generators: &[SparseModuleElement<Integer>],
    relations: &[SparseModuleElement<Integer>],
) -> Result<(), GroebnerError> {
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let ring = IntegerCoefficientRing;
    let input_terms = generators
        .iter()
        .map(|entry| entry.terms().len())
        .sum::<usize>()
        .saturating_add(
            relations
                .iter()
                .map(|entry| entry.terms().len())
                .sum::<usize>(),
        );
    let mut arena = MonomialArena::with_capacity(
        first.arity(),
        first.polynomial_order(),
        input_terms.max(256),
    );
    let order = SparseModuleOrder::TermOverPosition(first.polynomial_order());
    let packed_order = order.pack(&mut arena, first.rank())?;
    let inputs = generators
        .iter()
        .map(|entry| ArenaModuleElement::from_sparse(&ring, entry, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    for relation in relations {
        if relation.rank() != generators.len()
            || relation.arity() != first.arity()
            || relation.polynomial_order() != first.polynomial_order()
        {
            return Err(GroebnerError::ContextMismatch);
        }
        let coordinates = relation
            .coordinates(&ring)
            .iter()
            .map(|polynomial| {
                IntegerPolynomial::from_sparse(&mut arena, polynomial, &ring)
                    .map_err(|_| GroebnerError::ExponentOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut image = ArenaModuleElement::zero(first.rank());
        for (input, coefficient) in inputs.iter().zip(&coordinates) {
            let product = input.polynomial_mul(&ring, coefficient, &packed_order, &mut arena)?;
            image = add_module_values(&image, &product, &packed_order, &mut arena)?;
        }
        if !image.is_zero() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use poly_core::{MonomialOrder, SparsePolynomial};

    use super::*;

    fn monomial(exponents: &[u32]) -> Monomial {
        Monomial::new(exponents.to_vec().into_boxed_slice())
    }

    fn module_element(rank: usize, terms: &[(i64, &[u32], usize)]) -> SparseModuleElement<Integer> {
        SparseModuleElement::new(
            1,
            rank,
            MonomialOrder::Lexicographic,
            terms.iter().map(|(coefficient, exponent, component)| {
                (
                    SparseModuleTerm::new(monomial(exponent), *component),
                    Integer::from(*coefficient),
                )
            }),
            &IntegerCoefficientRing,
        )
    }

    fn reconstruct(
        basis: &SparseModuleElement<Integer>,
        inputs: &[SparseModuleElement<Integer>],
        coordinates: &[poly_core::SparsePolynomial<Integer>],
    ) -> SparseModuleElement<Integer> {
        let ring = IntegerCoefficientRing;
        let mut output =
            vec![
                SparsePolynomial::new(basis.arity(), basis.polynomial_order(), [], &ring,);
                basis.rank()
            ];
        for (input, multiplier) in inputs.iter().zip(coordinates) {
            for (coordinate, value) in output.iter_mut().zip(input.coordinates(&ring)) {
                *coordinate = coordinate.add(&value.multiply(multiplier, &ring, None), &ring);
            }
        }
        SparseModuleElement::from_coordinates(output, &ring)
    }

    #[test]
    fn strong_module_basis_closes_leading_coefficient_ideals() {
        let inputs = [
            module_element(2, &[(2, &[1], 0)]),
            module_element(2, &[(3, &[1], 0)]),
            module_element(2, &[(5, &[1], 1)]),
        ];
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::Lexicographic);
        let output =
            strong_module_groebner_basis(&inputs, &order, &StrongGroebnerOptions::default())
                .unwrap();
        assert!(output.basis.iter().any(|entry| {
            entry.terms() == [(SparseModuleTerm::new(monomial(&[1]), 0), Integer::one())]
        }));
        assert!(output.basis.iter().any(|entry| {
            entry.terms() == [(SparseModuleTerm::new(monomial(&[1]), 1), Integer::from(5))]
        }));
        assert!(output.metrics.g_polynomials > 0);
        for (basis, coordinates) in output.basis.iter().zip(&output.transformations) {
            assert_eq!(reconstruct(basis, &inputs, coordinates), *basis);
        }
    }

    #[test]
    fn strong_module_normal_form_returns_exact_quotients() {
        let basis = [
            module_element(2, &[(2, &[1], 0), (1, &[0], 1)]),
            module_element(2, &[(3, &[0], 0)]),
        ];
        let value = module_element(2, &[(7, &[2], 0), (-5, &[1], 0), (11, &[0], 1)]);
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::Lexicographic);
        let output =
            strong_module_normal_form_with_quotients(&value, &basis, &order, 1_000_000).unwrap();
        let reconstructed = reconstruct(&value, &basis, &output.quotients);
        let ring = IntegerCoefficientRing;
        let sum = SparseModuleElement::new(
            value.arity(),
            value.rank(),
            value.polynomial_order(),
            reconstructed
                .terms()
                .iter()
                .cloned()
                .chain(output.remainder.terms().iter().cloned()),
            &ring,
        );
        assert_eq!(sum, value);
        assert_eq!(
            output.remainder,
            strong_module_normal_form(&value, &basis, &order, 1_000_000).unwrap()
        );
    }

    #[test]
    fn graph_elimination_recovers_all_integral_relations() {
        let inputs = [
            module_element(1, &[(2, &[1], 0)]),
            module_element(1, &[(3, &[1], 0)]),
        ];
        let output = strong_module_relations(&inputs, &StrongGroebnerOptions::default()).unwrap();
        assert_eq!(output.generators.len(), 1);
        let relation = &output.generators[0];
        assert_eq!(relation.rank(), 2);
        assert!(reconstruct(
            &inputs[0],
            &inputs,
            &relation.coordinates(&IntegerCoefficientRing)
        )
        .is_zero());
        assert_eq!(
            relation.terms(),
            [
                (SparseModuleTerm::new(monomial(&[0]), 0), Integer::from(3),),
                (SparseModuleTerm::new(monomial(&[0]), 1), Integer::from(-2),),
            ]
        );
    }
}
