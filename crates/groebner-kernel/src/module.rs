// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Degree-batched F4 for submodules of polynomial free modules.

use core::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use flint_core::{Integer, Rational};
use module_core::{ModuleOrder, ModuleTerm, PositionModuleOrder};
use poly_core::{
    ArenaPolynomial, CoefficientRing, Monomial, MonomialArena, MonomialId, MonomialOrder,
    PolynomialWorkspace, SparsePolynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::engine::{GroebnerError, GroebnerEvidence, GroebnerMetrics, GroebnerOptions};
use crate::field::{BinaryExtensionField16, BinaryField, Field, PrimeField32};
use crate::rational::RationalCoefficientField;
use crate::reducer::{
    BinaryExtensionReducer, BinaryReducer, F4Reduction, F4RowWork, GenericReducer, PrimeReducer,
    RationalReducer, RowReducer,
};

mod integer;

pub use integer::{
    strong_module_groebner_basis, strong_module_normal_form,
    strong_module_normal_form_with_quotients, strong_module_relations, StrongModuleGroebnerOutput,
    StrongModuleKernelOutput, StrongModuleNormalFormOutput,
};

#[cfg(feature = "parallel")]
const MIN_PARALLEL_SCHREYER_VERIFICATION_PAIRS: usize = 512;
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SCHREYER_BASIS_ENTRIES: usize = 128;
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SCHREYER_CELL_PAIRS: usize = 256;

/// A portable polynomial monomial times one free-module basis vector.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SparseModuleTerm {
    monomial: Monomial,
    component: usize,
}

impl SparseModuleTerm {
    /// Construct `monomial * e_component`.
    pub fn new(monomial: Monomial, component: usize) -> Self {
        Self {
            monomial,
            component,
        }
    }

    /// Polynomial monomial.
    pub fn monomial(&self) -> &Monomial {
        &self.monomial
    }

    /// Zero-based free-module basis component.
    pub const fn component(&self) -> usize {
        self.component
    }

    /// Consume the term into its monomial and basis component.
    pub fn into_parts(self) -> (Monomial, usize) {
        (self.monomial, self.component)
    }
}

/// Portable exact module order accepted by the fixed kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseModuleOrder {
    /// Compare polynomial monomials before positions.
    TermOverPosition(MonomialOrder),
    /// Compare positions before polynomial monomials.
    PositionOverTerm(MonomialOrder),
    /// Compare through an exact ordered target leading frame.
    Schreyer {
        /// Exact module order in the target frame.
        target: Box<Self>,
        /// Ordered target leaders of the source generators.
        leaders: Vec<SparseModuleTerm>,
    },
}

impl SparseModuleOrder {
    /// Underlying polynomial monomial order.
    pub const fn base_order(&self) -> MonomialOrder {
        match self {
            Self::TermOverPosition(order) | Self::PositionOverTerm(order) => *order,
            Self::Schreyer { target, .. } => target.base_order(),
        }
    }

    /// Construct an induced Schreyer order.
    pub fn schreyer(target: Self, leaders: Vec<SparseModuleTerm>) -> Self {
        Self::Schreyer {
            target: Box::new(target),
            leaders,
        }
    }

    fn pack(
        &self,
        arena: &mut MonomialArena,
        source_rank: usize,
    ) -> Result<ModuleOrder, GroebnerError> {
        if self.base_order() != arena.order() {
            return Err(GroebnerError::ContextMismatch);
        }
        match self {
            Self::TermOverPosition(_) => Ok(PositionModuleOrder::TermOverPosition.into()),
            Self::PositionOverTerm(_) => Ok(PositionModuleOrder::PositionOverTerm.into()),
            Self::Schreyer { target, leaders } => {
                if leaders.len() != source_rank
                    || leaders
                        .iter()
                        .any(|leader| leader.monomial.arity() != arena.arity())
                {
                    return Err(GroebnerError::ContextMismatch);
                }
                let packed = leaders
                    .iter()
                    .map(|leader| {
                        arena
                            .intern(leader.monomial.clone())
                            .map(|monomial| ModuleTerm::new(monomial, leader.component))
                            .map_err(|_| GroebnerError::ExponentOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let target_rank = match target.as_ref() {
                    Self::Schreyer { leaders, .. } => leaders.len(),
                    Self::TermOverPosition(_) | Self::PositionOverTerm(_) => packed
                        .iter()
                        .map(|leader| leader.component().saturating_add(1))
                        .max()
                        .unwrap_or(0),
                };
                Ok(ModuleOrder::schreyer(
                    target.pack(arena, target_rank)?,
                    packed,
                ))
            }
        }
    }
}

/// Canonical portable sparse element of a polynomial free module.
///
/// Terms are stored component-major, with each polynomial coordinate in the
/// declared polynomial order. The computation order is supplied separately
/// to the Gröbner operation because it is not part of the free-module value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseModuleElement<C> {
    arity: usize,
    rank: usize,
    polynomial_order: MonomialOrder,
    terms: Vec<(SparseModuleTerm, C)>,
}

impl<C: Clone + Eq> SparseModuleElement<C> {
    /// Canonicalize arbitrary module terms, summing duplicates and dropping
    /// zero coefficients.
    pub fn new<R>(
        arity: usize,
        rank: usize,
        polynomial_order: MonomialOrder,
        terms: impl IntoIterator<Item = (SparseModuleTerm, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut canonical = HashMap::<SparseModuleTerm, C>::new();
        for (term, coefficient) in terms {
            assert_eq!(
                term.monomial.arity(),
                arity,
                "module monomial arity mismatch"
            );
            assert!(term.component < rank, "module component outside rank");
            if ring.is_zero(&coefficient) {
                continue;
            }
            let combined = canonical
                .remove(&term)
                .map_or(coefficient.clone(), |previous| {
                    ring.add(&previous, &coefficient)
                });
            if !ring.is_zero(&combined) {
                canonical.insert(term, combined);
            }
        }
        let mut terms = canonical.into_iter().collect::<Vec<_>>();
        terms.sort_by(|(left, _), (right, _)| {
            left.component
                .cmp(&right.component)
                .then_with(|| right.monomial.compare(&left.monomial, polynomial_order))
        });
        Self {
            arity,
            rank,
            polynomial_order,
            terms,
        }
    }

    /// Construct from polynomial coordinates in basis order.
    pub fn from_coordinates<R>(coordinates: Vec<SparsePolynomial<C>>, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let rank = coordinates.len();
        let (arity, order) = coordinates
            .first()
            .map_or((0, MonomialOrder::default()), |value| {
                (value.arity(), value.order())
            });
        assert!(coordinates
            .iter()
            .all(|value| value.arity() == arity && value.order() == order));
        Self::new(
            arity,
            rank,
            order,
            coordinates
                .into_iter()
                .enumerate()
                .flat_map(|(component, value)| {
                    value
                        .terms()
                        .iter()
                        .cloned()
                        .map(move |(monomial, coefficient)| {
                            (SparseModuleTerm::new(monomial, component), coefficient)
                        })
                        .collect::<Vec<_>>()
                }),
            ring,
        )
    }

    /// Construct from canonical component-major terms, checking the invariant
    /// once and retaining coefficient ownership.
    pub fn from_sorted_terms<R>(
        arity: usize,
        rank: usize,
        polynomial_order: MonomialOrder,
        terms: Vec<(SparseModuleTerm, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        for (index, (term, coefficient)) in terms.iter().enumerate() {
            assert_eq!(
                term.monomial.arity(),
                arity,
                "module monomial arity mismatch"
            );
            assert!(term.component < rank, "module component outside rank");
            assert!(!ring.is_zero(coefficient), "canonical coefficient is zero");
            if let Some((previous, _)) = index.checked_sub(1).and_then(|i| terms.get(i)) {
                assert!(
                    previous.component < term.component
                        || (previous.component == term.component
                            && previous.monomial.compare(&term.monomial, polynomial_order)
                                == Ordering::Greater),
                    "module terms are not strictly component-major"
                );
            }
        }
        Self {
            arity,
            rank,
            polynomial_order,
            terms,
        }
    }

    /// Number of polynomial variables.
    pub const fn arity(&self) -> usize {
        self.arity
    }

    /// Rank of the ambient free module.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Underlying polynomial monomial order.
    pub const fn polynomial_order(&self) -> MonomialOrder {
        self.polynomial_order
    }

    /// Canonical nonzero terms in component-major order.
    pub fn terms(&self) -> &[(SparseModuleTerm, C)] {
        &self.terms
    }

    /// Consume the element into canonical component-major terms.
    pub fn into_terms(self) -> Vec<(SparseModuleTerm, C)> {
        self.terms
    }

    /// Whether every coordinate is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Materialize polynomial coordinates in basis order.
    pub fn coordinates<R>(&self, ring: &R) -> Vec<SparsePolynomial<C>>
    where
        R: CoefficientRing<Element = C>,
    {
        let mut coordinates = vec![Vec::new(); self.rank];
        for (term, coefficient) in &self.terms {
            coordinates[term.component].push((term.monomial.clone(), coefficient.clone()));
        }
        coordinates
            .into_iter()
            .map(|terms| {
                SparsePolynomial::from_sorted_terms(self.arity, self.polynomial_order, terms, ring)
            })
            .collect()
    }
}

/// Verified reduced module basis with exact input-generator coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleGroebnerOutput<C> {
    /// Reduced monic basis, sorted by descending leading module term.
    pub basis: Vec<SparseModuleElement<C>>,
    /// Basis-major sparse change vectors into the input generators.
    pub transformations: Vec<SparseModuleElement<C>>,
    /// F4 phase and representation counters.
    pub metrics: GroebnerMetrics,
    /// Exact evidence established by the computation and final sweep.
    pub evidence: GroebnerEvidence,
}

/// Verified reduced module Gröbner basis without generator coordinates.
///
/// This result is intended for mathematical playbooks that consume only the
/// canonical basis, such as Hilbert and Betti computations. Exact F4 row
/// operations keep every constructed basis element in the input span; a final
/// independent sweep checks reducedness, input-generator containment, and
/// every critical pair. Unlike [`ModuleGroebnerOutput`], this result does not
/// retain replayable generator coordinates or construct an augmented change
/// matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleBasisOutput<C> {
    /// Reduced monic basis, sorted by descending leading module term.
    pub basis: Vec<SparseModuleElement<C>>,
    /// F4 phase counters for the unaugmented computation.
    pub metrics: GroebnerMetrics,
    /// Exact evidence established by the construction and final sweep.
    pub evidence: GroebnerEvidence,
}

/// Verified reduced Gröbner basis for the first syzygy module of an ordered
/// module Gröbner basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleSyzygyOutput<C> {
    /// Reduced syzygy basis in the ordered basis-generator free module.
    pub generators: Vec<SparseModuleElement<C>>,
    /// Degree-cell F4 counters.
    pub metrics: GroebnerMetrics,
    /// Exact evidence established by a complete Schreyer-pair sweep, or by
    /// exact Hilbert completion followed by an independent basis sweep.
    pub evidence: GroebnerEvidence,
}

/// Exact normal form and ordered basis quotients for one module value.
///
/// The identity is `input = sum(quotients[i] * basis[i]) + remainder` in the
/// supplied module order. A zero remainder therefore certifies membership and
/// the quotients provide a checked lift through the ordered basis map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleNormalFormOutput<C> {
    /// Canonical remainder after division by the ordered module basis.
    pub remainder: SparseModuleElement<C>,
    /// Polynomial coefficient of each ordered basis generator.
    pub quotients: Vec<SparsePolynomial<C>>,
}

/// Exact relations among an original module-generator family.
///
/// Construction starts from a verified module Gröbner basis and its checked
/// coordinates in the original generators. The retained evidence is the
/// evidence for the canonical basis's Schreyer kernel; exact change-map and
/// normal-form identities are checked again before the relations are
/// returned in the original source frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleKernelOutput<C> {
    /// Generators of the kernel in the original generator coordinates.
    pub generators: Vec<SparseModuleElement<C>>,
    /// Evidence used for the canonical basis's complete Schreyer kernel.
    pub basis_syzygy_evidence: GroebnerEvidence,
}

/// Exact standard-graded quotient Hilbert numerator for a module fast path.
///
/// `generator_degrees` records the internal degree of every ambient free
/// generator. The target is trusted only as a completion theorem's input;
/// callers must derive it from exact graded data. Every returned relation and
/// the resulting reduced module basis are checked independently of the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardModuleHilbertTarget {
    generator_degrees: Vec<usize>,
    quotient_numerator: Vec<Integer>,
}

impl StandardModuleHilbertTarget {
    /// Construct a standard-graded module quotient target.
    pub fn new(generator_degrees: Vec<usize>, quotient_numerator: Vec<Integer>) -> Self {
        Self {
            generator_degrees,
            quotient_numerator,
        }
    }

    /// Derive the exact next quotient target from a standard-graded
    /// submodule's leading terms.
    ///
    /// If `K` is the represented submodule of `F`, the returned numerator is
    /// that of `K`, viewed as the target quotient of the free module on the
    /// current basis generators in the next Schreyer step.
    pub fn from_submodule_leading_terms(
        arity: usize,
        ambient_generator_degrees: &[usize],
        basis_generator_degrees: Vec<usize>,
        leading_terms: &[(Monomial, usize)],
    ) -> Result<Self, GroebnerError> {
        Ok(Self::new(
            basis_generator_degrees,
            crate::hilbert::standard_module_submodule_hilbert_numerator(
                arity,
                ambient_generator_degrees,
                leading_terms,
            )?,
        ))
    }

    /// Internal degrees of the ambient free generators.
    pub fn generator_degrees(&self) -> &[usize] {
        &self.generator_degrees
    }

    /// Dense exact numerator of the target quotient Hilbert series.
    pub fn quotient_numerator(&self) -> &[Integer] {
        &self.quotient_numerator
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArenaModuleElement<C> {
    rank: usize,
    support: Vec<ModuleTerm>,
    coefficients: Vec<C>,
}

impl<C: Clone + Eq> ArenaModuleElement<C> {
    fn zero(rank: usize) -> Self {
        Self {
            rank,
            support: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    fn from_sparse<R: CoefficientRing<Element = C>>(
        ring: &R,
        value: &SparseModuleElement<C>,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let mut terms = Vec::with_capacity(value.terms.len());
        for (term, coefficient) in &value.terms {
            terms.push((
                ModuleTerm::new(
                    arena
                        .intern(term.monomial.clone())
                        .map_err(|_| GroebnerError::ExponentOverflow)?,
                    term.component,
                ),
                coefficient.clone(),
            ));
        }
        Self::new(ring, value.rank, terms, order, arena)
    }

    fn to_sparse<R: CoefficientRing<Element = C>>(
        &self,
        ring: &R,
        arena: &MonomialArena,
    ) -> SparseModuleElement<C> {
        SparseModuleElement::new(
            arena.arity(),
            self.rank,
            arena.order(),
            self.iter().map(|(term, coefficient)| {
                (
                    SparseModuleTerm::new(arena.get(term.monomial()), term.component()),
                    coefficient.clone(),
                )
            }),
            ring,
        )
    }

    fn new<R: CoefficientRing<Element = C>>(
        ring: &R,
        rank: usize,
        terms: impl IntoIterator<Item = (ModuleTerm, C)>,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let mut canonical = HashMap::<ModuleTerm, C>::new();
        for (term, coefficient) in terms {
            if term.component() >= rank || !arena.contains(term.monomial()) {
                return Err(GroebnerError::ContextMismatch);
            }
            if ring.is_zero(&coefficient) {
                continue;
            }
            let combined = canonical
                .remove(&term)
                .map_or(coefficient.clone(), |previous| {
                    ring.add(&previous, &coefficient)
                });
            if !ring.is_zero(&combined) {
                canonical.insert(term, combined);
            }
        }
        let mut terms = canonical.into_iter().collect::<Vec<_>>();
        sort_module_terms(&mut terms, order, arena)?;
        let (support, coefficients) = terms.into_iter().unzip();
        Ok(Self {
            rank,
            support,
            coefficients,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (ModuleTerm, &C)> {
        self.support.iter().copied().zip(&self.coefficients)
    }

    fn leading_term(&self) -> Option<(ModuleTerm, &C)> {
        self.support.first().copied().zip(self.coefficients.first())
    }

    fn is_zero(&self) -> bool {
        self.support.is_empty()
    }

    fn len(&self) -> usize {
        self.support.len()
    }

    fn monomial_mul(
        &self,
        multiplier: MonomialId,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let terms = self
            .iter()
            .map(|(term, coefficient)| {
                arena
                    .product(term.monomial(), multiplier)
                    .map(|monomial| {
                        (
                            ModuleTerm::new(monomial, term.component()),
                            coefficient.clone(),
                        )
                    })
                    .map_err(|_| GroebnerError::ExponentOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rank: self.rank,
            support: terms.iter().map(|(term, _)| *term).collect(),
            coefficients: terms
                .into_iter()
                .map(|(_, coefficient)| coefficient)
                .collect(),
        })
        .and_then(|value| value.resort(order, arena))
    }

    fn term_mul<R: CoefficientRing<Element = C>>(
        &self,
        ring: &R,
        multiplier: MonomialId,
        scalar: &C,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let mut value = self.monomial_mul(multiplier, order, arena)?;
        for coefficient in &mut value.coefficients {
            *coefficient = ring.mul(coefficient, scalar);
        }
        Ok(value)
    }

    fn scalar_mul<R: CoefficientRing<Element = C>>(&self, ring: &R, scalar: &C) -> Self {
        Self {
            rank: self.rank,
            support: self.support.clone(),
            coefficients: self
                .coefficients
                .iter()
                .map(|coefficient| ring.mul(coefficient, scalar))
                .collect(),
        }
    }

    fn sub<R: CoefficientRing<Element = C>>(
        &self,
        ring: &R,
        other: &Self,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        if self.rank != other.rank {
            return Err(GroebnerError::ContextMismatch);
        }
        Self::new(
            ring,
            self.rank,
            self.iter()
                .map(|(term, coefficient)| (term, coefficient.clone()))
                .chain(
                    other
                        .iter()
                        .map(|(term, coefficient)| (term, ring.neg(coefficient))),
                ),
            order,
            arena,
        )
    }

    fn polynomial_mul<R: CoefficientRing<Element = C>>(
        &self,
        ring: &R,
        polynomial: &ArenaPolynomial<C>,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let mut terms = Vec::with_capacity(self.len().saturating_mul(polynomial.len()));
        for (module_term, module_coefficient) in self.iter() {
            for (monomial, coefficient) in polynomial.iter() {
                terms.push((
                    ModuleTerm::new(
                        arena
                            .product(module_term.monomial(), monomial)
                            .map_err(|_| GroebnerError::ExponentOverflow)?,
                        module_term.component(),
                    ),
                    ring.mul(module_coefficient, coefficient),
                ));
            }
        }
        Self::new(ring, self.rank, terms, order, arena)
    }

    fn sum_polynomial_products<R: CoefficientRing<Element = C>>(
        ring: &R,
        values: &[Self],
        multipliers: &[ArenaPolynomial<C>],
        rank: usize,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        if values.len() != multipliers.len() || values.iter().any(|value| value.rank != rank) {
            return Err(GroebnerError::ContextMismatch);
        }
        let term_count = values
            .iter()
            .zip(multipliers)
            .map(|(value, multiplier)| value.len().saturating_mul(multiplier.len()))
            .fold(0_usize, usize::saturating_add);
        let mut terms = Vec::with_capacity(term_count);
        for (value, multiplier) in values.iter().zip(multipliers) {
            for (module_term, module_coefficient) in value.iter() {
                for (monomial, coefficient) in multiplier.iter() {
                    terms.push((
                        ModuleTerm::new(
                            arena
                                .product(module_term.monomial(), monomial)
                                .map_err(|_| GroebnerError::ExponentOverflow)?,
                            module_term.component(),
                        ),
                        ring.mul(module_coefficient, coefficient),
                    ));
                }
            }
        }
        Self::new(ring, rank, terms, order, arena)
    }

    fn resort(
        mut self,
        order: &ModuleOrder,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        let mut terms = self
            .support
            .drain(..)
            .zip(self.coefficients.drain(..))
            .collect::<Vec<_>>();
        sort_module_terms(&mut terms, order, arena)?;
        (self.support, self.coefficients) = terms.into_iter().unzip();
        Ok(self)
    }
}

fn sparse_representation<C, R>(
    representation: Vec<ArenaPolynomial<C>>,
    arena: &MonomialArena,
    ring: &R,
) -> SparseModuleElement<C>
where
    C: Clone + Eq,
    R: CoefficientRing<Element = C>,
{
    let rank = representation.len();
    let term_count = representation.iter().map(ArenaPolynomial::len).sum();
    let mut terms = Vec::with_capacity(term_count);
    for (component, polynomial) in representation.into_iter().enumerate() {
        terms.extend(polynomial.into_terms().map(|(monomial, coefficient)| {
            (
                SparseModuleTerm::new(arena.get(monomial), component),
                coefficient,
            )
        }));
    }
    SparseModuleElement::from_sorted_terms(arena.arity(), rank, arena.order(), terms, ring)
}

fn sort_module_terms<C>(
    terms: &mut [(ModuleTerm, C)],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    // Stable insertion sorting keeps error propagation explicit and module
    // rows are sparse enough that the avoided exponent materialization is the
    // more important regime. Larger F4 matrices sort columns only once.
    for index in 1..terms.len() {
        let mut cursor = index;
        while cursor != 0
            && compare_terms(order, terms[cursor].0, terms[cursor - 1].0, arena)?
                == Ordering::Greater
        {
            terms.swap(cursor, cursor - 1);
            cursor -= 1;
        }
    }
    Ok(())
}

fn compare_terms(
    order: &ModuleOrder,
    left: ModuleTerm,
    right: ModuleTerm,
    arena: &mut MonomialArena,
) -> Result<Ordering, GroebnerError> {
    order
        .compare(left, right, arena)
        .map_err(|_| GroebnerError::ExponentOverflow)
}

type Representation<C> = Vec<ArenaPolynomial<C>>;
type TrackedParts<C> = (ArenaModuleElement<C>, Representation<C>);
type TrackedPrecondition<C> = (Vec<Tracked<C>>, Option<Vec<Representation<C>>>);

#[derive(Clone)]
struct Tracked<C> {
    value: ArenaModuleElement<C>,
    representation: Representation<C>,
}

trait ModuleValue<C> {
    fn module_value(&self) -> &ArenaModuleElement<C>;
}

impl<C> ModuleValue<C> for ArenaModuleElement<C> {
    fn module_value(&self) -> &ArenaModuleElement<C> {
        self
    }
}

impl<C> ModuleValue<C> for Tracked<C> {
    fn module_value(&self) -> &ArenaModuleElement<C> {
        &self.value
    }
}

#[derive(Clone, Copy)]
struct Pair {
    left: usize,
    right: usize,
    lcm: MonomialId,
    component: usize,
    degree: u64,
}

#[derive(Clone, Copy)]
struct RowPlan {
    basis_index: usize,
    multiplier: MonomialId,
}

const MIN_TRACKED_GENERATOR_ROW_REDUCTION: usize = 8;

/// Compute a complete reduced module Gröbner basis without a change matrix.
///
/// This is the coefficient-generic, word-exponent F4 lane for consumers that
/// need only the canonical basis. Completeness is still checked by reducing
/// every input generator and every critical pair through the final basis.
pub fn module_groebner_basis<F: Field>(
    field: &F,
    generators: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleBasisOutput<F::Element>, GroebnerError> {
    module_groebner_basis_without_transform_with_reducer(
        field,
        generators,
        order,
        options,
        &GenericReducer,
    )
}

/// Compute a complete reduced module basis over the rationals without a
/// change matrix.
///
/// This has the same exact contract as [`module_groebner_basis`] and uses
/// FLINT's fraction-free dense reduction inside rational F4 cells.
pub fn module_groebner_basis_rational(
    generators: &[SparseModuleElement<Rational>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleBasisOutput<Rational>, GroebnerError> {
    module_groebner_basis_without_transform_with_reducer(
        &RationalCoefficientField,
        generators,
        order,
        options,
        &RationalReducer,
    )
}

/// Compute a complete reduced module basis over a word-sized prime field
/// without a change matrix.
///
/// The result and verification contract match [`module_groebner_basis`],
/// while row reduction privately selects the specialized prime backend.
pub fn module_groebner_basis_prime(
    field: &PrimeField32,
    generators: &[SparseModuleElement<u32>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleBasisOutput<u32>, GroebnerError> {
    module_groebner_basis_without_transform_with_reducer(
        field,
        generators,
        order,
        options,
        &PrimeReducer,
    )
}

/// Compute a complete reduced module basis over `F_2` without a change
/// matrix, using bit-packed sparse or dense rows.
pub fn module_groebner_basis_f2(
    generators: &[SparseModuleElement<u8>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleBasisOutput<u8>, GroebnerError> {
    module_groebner_basis_without_transform_with_reducer(
        &BinaryField,
        generators,
        order,
        options,
        &BinaryReducer,
    )
}

/// Compute a complete reduced module basis over packed `F_{2^e}` without a
/// change matrix.
pub fn module_groebner_basis_binary_extension(
    field: &BinaryExtensionField16,
    generators: &[SparseModuleElement<u16>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleBasisOutput<u16>, GroebnerError> {
    module_groebner_basis_without_transform_with_reducer(
        field,
        generators,
        order,
        options,
        &BinaryExtensionReducer,
    )
}

/// Compute a complete reduced module Gröbner basis with exact change matrix.
///
/// This is the word-exponent, coefficient-generic F4 lane. Unsupported
/// exponents or exhausted explicit limits return an error; callers may retain
/// a separate arbitrary-exponent correctness floor.
pub fn module_groebner_basis_with_transform<F: Field>(
    field: &F,
    generators: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleGroebnerOutput<F::Element>, GroebnerError> {
    module_groebner_basis_with_reducer(field, generators, order, options, &GenericReducer)
}

/// Compute a complete reduced rational module basis with exact change matrix.
///
/// The result and verification contract match
/// [`module_groebner_basis_with_transform`]. Rational F4 cells use FLINT's
/// fraction-free dense matrix reduction, while all module and witness
/// construction remains native kernel logic.
pub fn module_groebner_basis_with_transform_rational(
    generators: &[SparseModuleElement<Rational>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleGroebnerOutput<Rational>, GroebnerError> {
    module_groebner_basis_with_reducer(
        &RationalCoefficientField,
        generators,
        order,
        options,
        &RationalReducer,
    )
}

/// Compute a complete reduced module basis over a word-sized prime field.
///
/// The mathematical result is identical to
/// [`module_groebner_basis_with_transform`], while F4 row reduction privately
/// selects the sparse, dense, or `A|B / C|D` word-prime backend.
pub fn module_groebner_basis_with_transform_prime(
    field: &PrimeField32,
    generators: &[SparseModuleElement<u32>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleGroebnerOutput<u32>, GroebnerError> {
    module_groebner_basis_with_reducer(field, generators, order, options, &PrimeReducer)
}

/// Compute a complete reduced module basis over `F_2` with an exact change
/// matrix, using bit-packed sparse or dense rows.
pub fn module_groebner_basis_with_transform_f2(
    generators: &[SparseModuleElement<u8>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleGroebnerOutput<u8>, GroebnerError> {
    module_groebner_basis_with_reducer(&BinaryField, generators, order, options, &BinaryReducer)
}

/// Compute a complete reduced module basis over packed `F_{2^e}` with an
/// exact change matrix.
pub fn module_groebner_basis_with_transform_binary_extension(
    field: &BinaryExtensionField16,
    generators: &[SparseModuleElement<u16>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleGroebnerOutput<u16>, GroebnerError> {
    module_groebner_basis_with_reducer(field, generators, order, options, &BinaryExtensionReducer)
}

/// Compute Schreyer generators in degree-batched augmented F4 cells.
///
/// `basis` must be an ordered reduced module Gröbner basis in `order`. The
/// returned relations use that exact input order as their free-module basis,
/// and every relation is independently checked against the supplied basis.
pub fn module_schreyer_syzygies<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        None,
        SchreyerPairPolicy::CHECKED,
        &GenericReducer,
    )
}

/// Compute degree-batched Schreyer generators with an exact Hilbert target.
///
/// A target match may skip higher-degree Schreyer cells. The target must be
/// derived from exact graded data; returned generators remain checked
/// relations and form an independently swept reduced Gröbner basis.
pub fn module_schreyer_syzygies_with_hilbert_target<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    target: &StandardModuleHilbertTarget,
) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        Some(target),
        SchreyerPairPolicy::CHECKED,
        &GenericReducer,
    )
}

/// Compute degree-batched Schreyer generators over a word-sized prime field.
pub fn module_schreyer_syzygies_prime(
    field: &PrimeField32,
    basis: &[SparseModuleElement<u32>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleSyzygyOutput<u32>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        None,
        SchreyerPairPolicy::CHECKED,
        &PrimeReducer,
    )
}

/// Compute prime-field Schreyer generators with an exact Hilbert target.
pub fn module_schreyer_syzygies_prime_with_hilbert_target(
    field: &PrimeField32,
    basis: &[SparseModuleElement<u32>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    target: &StandardModuleHilbertTarget,
) -> Result<ModuleSyzygyOutput<u32>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        Some(target),
        SchreyerPairPolicy::CHECKED,
        &PrimeReducer,
    )
}

/// Compute degree-batched Schreyer generators over `F_2` with bit-packed row
/// reduction.
pub fn module_schreyer_syzygies_f2(
    basis: &[SparseModuleElement<u8>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleSyzygyOutput<u8>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        &BinaryField,
        basis,
        order,
        options,
        None,
        SchreyerPairPolicy::CHECKED,
        &BinaryReducer,
    )
}

/// Compute `F_2` Schreyer generators with an exact Hilbert target and
/// bit-packed row reduction.
pub fn module_schreyer_syzygies_f2_with_hilbert_target(
    basis: &[SparseModuleElement<u8>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    target: &StandardModuleHilbertTarget,
) -> Result<ModuleSyzygyOutput<u8>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        &BinaryField,
        basis,
        order,
        options,
        Some(target),
        SchreyerPairPolicy::CHECKED,
        &BinaryReducer,
    )
}

/// Compute degree-batched Schreyer generators over packed `F_{2^e}`.
pub fn module_schreyer_syzygies_binary_extension(
    field: &BinaryExtensionField16,
    basis: &[SparseModuleElement<u16>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleSyzygyOutput<u16>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        None,
        SchreyerPairPolicy::CHECKED,
        &BinaryExtensionReducer,
    )
}

/// Compute packed `F_{2^e}` Schreyer generators with an exact Hilbert
/// target.
pub fn module_schreyer_syzygies_binary_extension_with_hilbert_target(
    field: &BinaryExtensionField16,
    basis: &[SparseModuleElement<u16>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    target: &StandardModuleHilbertTarget,
) -> Result<ModuleSyzygyOutput<u16>, GroebnerError> {
    module_schreyer_syzygies_with_reducer(
        field,
        basis,
        order,
        options,
        Some(target),
        SchreyerPairPolicy::CHECKED,
        &BinaryExtensionReducer,
    )
}

/// Extract the ordered leading terms of a canonical sparse module basis.
///
/// This is an operation-private bridge for callers that retain a packed
/// Schreyer frame across several certified kernel operations.
pub fn module_leading_terms<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
) -> Result<Vec<SparseModuleTerm>, GroebnerError> {
    let Some(first) = basis.first() else {
        return Ok(Vec::new());
    };
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || basis.iter().any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let term_count = basis.iter().map(|value| value.terms.len()).sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        term_count
            .max(term_count.saturating_mul(2).min(65_536))
            .max(64),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    basis
        .iter()
        .map(|value| {
            let value = ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena)?;
            let (term, coefficient) = value
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?;
            if coefficient != &field.one() {
                return Err(GroebnerError::VerificationFailed);
            }
            Ok(SparseModuleTerm::new(
                arena.get(term.monomial()),
                term.component(),
            ))
        })
        .collect()
}

/// Decide several module-membership queries against one ordered Gröbner basis.
///
/// The basis and values share one packed monomial arena, so callers avoid
/// rebuilding exponent storage for every query. `basis` must be a monic
/// Gröbner basis in `order`; each result is true exactly when the corresponding
/// value has zero normal form.
pub fn module_membership<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    values: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<Vec<bool>, GroebnerError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let first = basis.first().unwrap_or(&values[0]);
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || basis.iter().chain(values).any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = basis
        .iter()
        .chain(values)
        .map(|value| value.terms.len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        input_terms
            .max(input_terms.saturating_mul(2).min(65_536))
            .max(64),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    let packed_basis = basis
        .iter()
        .map(|value| ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    if packed_basis.iter().any(ArenaModuleElement::is_zero)
        || packed_basis.iter().any(|value| {
            value
                .leading_term()
                .is_none_or(|(_, coefficient)| coefficient != &field.one())
        })
    {
        return Err(GroebnerError::VerificationFailed);
    }
    values
        .iter()
        .map(|value| {
            ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena).and_then(
                |value| {
                    reduces_to_zero(
                        field,
                        value,
                        &packed_basis,
                        &packed_order,
                        options.max_reduction_steps,
                        &mut arena,
                    )
                },
            )
        })
        .collect()
}

/// Reduce several module values while retaining exact basis quotients.
///
/// The basis and all queries share one monomial arena. `basis` must be a
/// monic Gröbner basis in `order`; each output satisfies the reconstruction
/// identity documented by [`ModuleNormalFormOutput`]. This is the tracked
/// companion to [`module_membership`], whose boolean-only lane avoids the
/// quotient work.
pub fn module_normal_forms<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    values: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<Vec<ModuleNormalFormOutput<F::Element>>, GroebnerError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let first = basis.first().unwrap_or(&values[0]);
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || basis.iter().chain(values).any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = basis
        .iter()
        .chain(values)
        .map(|value| value.terms.len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        input_terms
            .max(input_terms.saturating_mul(2).min(65_536))
            .max(64),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    let packed_basis = basis
        .iter()
        .map(|value| ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    if packed_basis.iter().any(ArenaModuleElement::is_zero)
        || packed_basis.iter().any(|value| {
            value
                .leading_term()
                .is_none_or(|(_, coefficient)| coefficient != &field.one())
        })
    {
        return Err(GroebnerError::VerificationFailed);
    }

    let mut outputs = Vec::with_capacity(values.len());
    for value in values {
        let mut value = ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena)?;
        let mut remainder = ArenaModuleElement::zero(rank);
        let mut quotients = vec![ArenaPolynomial::zero(&arena); packed_basis.len()];
        let mut steps = 0;
        while let Some((leading, coefficient)) = value.leading_term() {
            if steps == options.max_reduction_steps {
                return Err(resource(
                    "module tracked normal-form steps",
                    options.max_reduction_steps,
                ));
            }
            if let Some((index, reducer)) =
                packed_basis.iter().enumerate().find(|(_, candidate)| {
                    candidate
                        .leading_term()
                        .is_some_and(|(divisor, _)| module_term_divides(divisor, leading, &arena))
                })
            {
                let (divisor, _) = reducer
                    .leading_term()
                    .expect("a selected module reducer is nonzero");
                let multiplier = arena
                    .quotient(leading.monomial(), divisor.monomial())
                    .map_err(|_| GroebnerError::ExponentOverflow)?
                    .ok_or(GroebnerError::VerificationFailed)?;
                let scalar = coefficient.clone();
                value = value.sub(
                    field,
                    &reducer.term_mul(field, multiplier, &scalar, &packed_order, &mut arena)?,
                    &packed_order,
                    &mut arena,
                )?;
                let quotient_term = ArenaPolynomial::new(&arena, [(multiplier, scalar)], field);
                quotients[index] = quotients[index].add(&quotient_term, &arena, field);
            } else {
                let term = ArenaModuleElement::new(
                    field,
                    rank,
                    [(leading, coefficient.clone())],
                    &packed_order,
                    &mut arena,
                )?;
                remainder = remainder.sub(
                    field,
                    &term.scalar_mul(field, &field.neg(&field.one())),
                    &packed_order,
                    &mut arena,
                )?;
                value = value.sub(field, &term, &packed_order, &mut arena)?;
            }
            steps += 1;
        }
        outputs.push(ModuleNormalFormOutput {
            remainder: remainder.to_sparse(field, &arena),
            quotients: quotients
                .into_iter()
                .map(|quotient| quotient.to_sparse(&arena, field))
                .collect(),
        });
    }
    Ok(outputs)
}

/// Compute original-generator relations from checked module Gröbner data.
///
/// `basis_transformations[i]` must express `basis[i]` through `values`. The
/// function verifies all those identities, computes the complete Schreyer
/// kernel of `basis`, reduces every original value with exact quotients, and
/// transports both families of relations into the original source frame.
pub fn module_relations_from_groebner_data<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    basis_transformations: &[SparseModuleElement<F::Element>],
    values: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleKernelOutput<F::Element>, GroebnerError> {
    let syzygies = if basis.is_empty() {
        ModuleSyzygyOutput {
            generators: Vec::new(),
            metrics: GroebnerMetrics::default(),
            evidence: GroebnerEvidence::MonomialSchreyerTheorem,
        }
    } else {
        module_schreyer_syzygies(field, basis, order, options)?
    };
    assemble_original_module_relations(
        field,
        basis,
        basis_transformations,
        values,
        order,
        options,
        syzygies,
    )
}

/// Prime-field specialization of [`module_relations_from_groebner_data`].
pub fn module_relations_from_groebner_data_prime(
    field: &PrimeField32,
    basis: &[SparseModuleElement<u32>],
    basis_transformations: &[SparseModuleElement<u32>],
    values: &[SparseModuleElement<u32>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleKernelOutput<u32>, GroebnerError> {
    let syzygies = if basis.is_empty() {
        ModuleSyzygyOutput {
            generators: Vec::new(),
            metrics: GroebnerMetrics::default(),
            evidence: GroebnerEvidence::MonomialSchreyerTheorem,
        }
    } else {
        module_schreyer_syzygies_prime(field, basis, order, options)?
    };
    assemble_original_module_relations(
        field,
        basis,
        basis_transformations,
        values,
        order,
        options,
        syzygies,
    )
}

/// `F_2` specialization of [`module_relations_from_groebner_data`].
pub fn module_relations_from_groebner_data_f2(
    basis: &[SparseModuleElement<u8>],
    basis_transformations: &[SparseModuleElement<u8>],
    values: &[SparseModuleElement<u8>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleKernelOutput<u8>, GroebnerError> {
    let syzygies = if basis.is_empty() {
        ModuleSyzygyOutput {
            generators: Vec::new(),
            metrics: GroebnerMetrics::default(),
            evidence: GroebnerEvidence::MonomialSchreyerTheorem,
        }
    } else {
        module_schreyer_syzygies_f2(basis, order, options)?
    };
    assemble_original_module_relations(
        &BinaryField,
        basis,
        basis_transformations,
        values,
        order,
        options,
        syzygies,
    )
}

/// Packed `F_{2^e}` specialization of
/// [`module_relations_from_groebner_data`].
pub fn module_relations_from_groebner_data_binary_extension(
    field: &BinaryExtensionField16,
    basis: &[SparseModuleElement<u16>],
    basis_transformations: &[SparseModuleElement<u16>],
    values: &[SparseModuleElement<u16>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<ModuleKernelOutput<u16>, GroebnerError> {
    let syzygies = if basis.is_empty() {
        ModuleSyzygyOutput {
            generators: Vec::new(),
            metrics: GroebnerMetrics::default(),
            evidence: GroebnerEvidence::MonomialSchreyerTheorem,
        }
    } else {
        module_schreyer_syzygies_binary_extension(field, basis, order, options)?
    };
    assemble_original_module_relations(
        field,
        basis,
        basis_transformations,
        values,
        order,
        options,
        syzygies,
    )
}

fn assemble_original_module_relations<F: Field>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    basis_transformations: &[SparseModuleElement<F::Element>],
    values: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    basis_syzygies: ModuleSyzygyOutput<F::Element>,
) -> Result<ModuleKernelOutput<F::Element>, GroebnerError> {
    if basis.len() != basis_transformations.len() {
        return Err(GroebnerError::ContextMismatch);
    }
    let Some(reference) = basis
        .first()
        .or_else(|| values.first())
        .or_else(|| basis_transformations.first())
    else {
        return Ok(ModuleKernelOutput {
            generators: Vec::new(),
            basis_syzygy_evidence: basis_syzygies.evidence,
        });
    };
    let arity = reference.arity;
    let polynomial_order = reference.polynomial_order;
    let ambient_rank = basis
        .first()
        .or_else(|| values.first())
        .map_or(0, |value| value.rank);
    if order.base_order() != polynomial_order
        || basis.iter().chain(values).any(|value| {
            value.arity != arity
                || value.rank != ambient_rank
                || value.polynomial_order != polynomial_order
        })
        || basis_transformations.iter().any(|value| {
            value.arity != arity
                || value.rank != values.len()
                || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }

    for (expected, transformation) in basis.iter().zip(basis_transformations) {
        let reconstructed = apply_sparse_module_map(
            field,
            values,
            transformation,
            ambient_rank,
            arity,
            polynomial_order,
        )?;
        if &reconstructed != expected {
            return Err(GroebnerError::VerificationFailed);
        }
    }

    let reductions = module_normal_forms(field, basis, values, order, options)?;
    if reductions
        .iter()
        .any(|reduction| !reduction.remainder.is_zero())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut generators = basis_syzygies
        .generators
        .iter()
        .map(|relation| {
            apply_sparse_module_map(
                field,
                basis_transformations,
                relation,
                values.len(),
                arity,
                polynomial_order,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    generators.retain(|relation| !relation.is_zero());

    let one = Monomial::one(arity);
    for (index, reduction) in reductions.into_iter().enumerate() {
        let quotient = if reduction.quotients.is_empty() {
            SparseModuleElement::new(arity, 0, polynomial_order, core::iter::empty(), field)
        } else {
            SparseModuleElement::from_coordinates(reduction.quotients, field)
        };
        let reconstructed = apply_sparse_module_map(
            field,
            basis_transformations,
            &quotient,
            values.len(),
            arity,
            polynomial_order,
        )?;
        let relation = SparseModuleElement::new(
            arity,
            values.len(),
            polynomial_order,
            core::iter::once((SparseModuleTerm::new(one.clone(), index), field.one())).chain(
                reconstructed
                    .terms
                    .iter()
                    .map(|(term, coefficient)| (term.clone(), field.neg(coefficient))),
            ),
            field,
        );
        if !relation.is_zero() {
            generators.push(relation);
        }
    }
    Ok(ModuleKernelOutput {
        generators,
        basis_syzygy_evidence: basis_syzygies.evidence,
    })
}

fn apply_sparse_module_map<F: Field>(
    field: &F,
    images: &[SparseModuleElement<F::Element>],
    coefficients: &SparseModuleElement<F::Element>,
    target_rank: usize,
    arity: usize,
    polynomial_order: MonomialOrder,
) -> Result<SparseModuleElement<F::Element>, GroebnerError> {
    if coefficients.arity != arity
        || coefficients.rank != images.len()
        || coefficients.polynomial_order != polynomial_order
        || images.iter().any(|image| {
            image.arity != arity
                || image.rank != target_rank
                || image.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let mut terms = Vec::new();
    for (coefficient_term, coefficient) in &coefficients.terms {
        for (image_term, image_coefficient) in &images[coefficient_term.component].terms {
            let monomial = coefficient_term
                .monomial
                .checked_product(&image_term.monomial)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            terms.push((
                SparseModuleTerm::new(monomial, image_term.component),
                field.mul(coefficient, image_coefficient),
            ));
        }
    }
    Ok(SparseModuleElement::new(
        arity,
        target_rank,
        polynomial_order,
        terms,
        field,
    ))
}

#[derive(Clone, Copy)]
struct SchreyerPairPolicy {
    colon_criterion: bool,
    direct_monomial: bool,
}

impl SchreyerPairPolicy {
    const CHECKED: Self = Self {
        colon_criterion: true,
        direct_monomial: true,
    };

    #[cfg(test)]
    const UNPRUNED: Self = Self {
        colon_criterion: false,
        direct_monomial: true,
    };

    #[cfg(all(test, feature = "parallel"))]
    const FORCE_F4: Self = Self {
        colon_criterion: true,
        direct_monomial: false,
    };
}

fn module_schreyer_syzygies_with_reducer<F, R>(
    field: &F,
    basis: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    hilbert_target: Option<&StandardModuleHilbertTarget>,
    policy: SchreyerPairPolicy,
    reducer: &R,
) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F> + Sync,
{
    let Some(first) = basis.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || basis.iter().any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = basis.iter().map(|value| value.terms.len()).sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    let packed_basis = basis
        .iter()
        .map(|value| ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    if packed_basis.iter().any(ArenaModuleElement::is_zero)
        || packed_basis.iter().any(|value| {
            value
                .leading_term()
                .is_none_or(|(_, coefficient)| coefficient != &field.one())
        })
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let leaders = packed_basis
        .iter()
        .map(|value| value.leading_term().expect("validated nonzero basis").0)
        .collect::<Vec<_>>();
    let source_order = ModuleOrder::schreyer(packed_order.clone(), leaders.clone());
    let schedule = schreyer_pair_cells(&leaders, options, policy.colon_criterion, &mut arena)?;
    let cells = schedule.cells.into_iter().collect::<Vec<_>>();
    let scheduled_pair_count = cells.iter().map(|(_, pairs)| pairs.len()).sum::<usize>();
    let mut metrics = GroebnerMetrics {
        pairs_generated: schedule.pairs_generated,
        schreyer_colon_pairs: schedule.pairs_pruned,
        ..GroebnerMetrics::default()
    };
    let mut relations = Vec::<ArenaModuleElement<F::Element>>::new();
    let used_monomial_lane =
        policy.direct_monomial && packed_basis.iter().all(|value| value.support.len() == 1);
    if used_monomial_lane {
        for (degree, pairs) in &cells {
            if metrics.rounds == options.max_rounds {
                return Err(resource(
                    "Schreyer monomial degree cells",
                    options.max_rounds,
                ));
            }
            observe_schreyer_round(&mut metrics, *degree, pairs.len());
            for pair in pairs {
                let left_leading = leaders[pair.left];
                let right_leading = leaders[pair.right];
                let left_multiplier = arena
                    .quotient(pair.lcm, left_leading.monomial())
                    .map_err(|_| GroebnerError::ExponentOverflow)?
                    .ok_or(GroebnerError::VerificationFailed)?;
                let right_multiplier = arena
                    .quotient(pair.lcm, right_leading.monomial())
                    .map_err(|_| GroebnerError::ExponentOverflow)?
                    .ok_or(GroebnerError::VerificationFailed)?;
                let relation = ArenaModuleElement::new(
                    field,
                    packed_basis.len(),
                    [
                        (ModuleTerm::new(left_multiplier, pair.left), field.one()),
                        (
                            ModuleTerm::new(right_multiplier, pair.right),
                            field.neg(&field.one()),
                        ),
                    ],
                    &source_order,
                    &mut arena,
                )?;
                if !relations.contains(&relation) {
                    relations.push(relation);
                }
            }
        }
    }

    let cell_count = cells.len();
    #[cfg(feature = "parallel")]
    let mut used_parallel_cells = false;
    #[cfg(not(feature = "parallel"))]
    let used_parallel_cells = false;
    #[cfg(feature = "parallel")]
    if !used_monomial_lane
        && hilbert_target.is_none()
        && cells.len() > 1
        && scheduled_pair_count >= MIN_PARALLEL_SCHREYER_CELL_PAIRS
        && rayon::current_num_threads() > 1
        && cells.len() <= options.max_rounds
        && schreyer_cell_workers(options, &arena, cells.len()) > 1
    {
        match reduce_schreyer_cells_parallel(
            field,
            &packed_basis,
            &leaders,
            &cells,
            rank,
            &packed_order,
            &source_order,
            options,
            reducer,
            &arena,
        ) {
            Ok(parallel) => {
                for output in parallel.outputs {
                    observe_schreyer_round(&mut metrics, output.degree, output.pair_count);
                    merge_schreyer_cell_metrics(&mut metrics, &output.metrics);
                    for relation in output.relations {
                        let relation = ArenaModuleElement::from_sparse(
                            field,
                            &relation,
                            &source_order,
                            &mut arena,
                        )?;
                        if !relations.contains(&relation) {
                            relations.push(relation);
                        }
                    }
                }
                metrics.peak_workspace_bytes = metrics
                    .peak_workspace_bytes
                    .max(parallel.peak_workspace_bytes);
                metrics.schreyer_parallel_cells = cell_count;
                used_parallel_cells = true;
            }
            Err(GroebnerError::ResourceLimit {
                resource: "module F4 workspace bytes",
                ..
            }) => {}
            Err(error) => return Err(error),
        }
    }

    if !used_monomial_lane && !used_parallel_cells {
        for (degree, pairs) in cells {
            if metrics.rounds == options.max_rounds {
                return Err(resource("Schreyer F4 degree cells", options.max_rounds));
            }
            observe_schreyer_round(&mut metrics, degree, pairs.len());

            let cell = reduce_schreyer_cell(
                field,
                &packed_basis,
                &leaders,
                &pairs,
                rank,
                &packed_order,
                &source_order,
                options,
                reducer,
                &mut arena,
            )?;
            merge_schreyer_cell_metrics(&mut metrics, &cell.metrics);
            for relation in cell.relations {
                if !relations.contains(&relation) {
                    relations.push(relation);
                }
            }
            if metrics.rounds < cell_count
                && hilbert_target.is_some_and(|target| {
                    module_hilbert_target_reached(&relations, target, arity, &arena, &mut metrics)
                })
            {
                metrics.hilbert_pruned_pairs = metrics
                    .hilbert_pruned_pairs
                    .saturating_add(scheduled_pair_count.saturating_sub(metrics.pairs_selected));
                break;
            }
        }
    }
    let schreyer_generators = relations;
    let mut tracked = schreyer_generators
        .iter()
        .cloned()
        .map(|value| Tracked {
            value,
            representation: zero_representation(0, &arena),
        })
        .collect::<Vec<_>>();
    let relations = if used_monomial_lane {
        sort_tracked(&mut tracked, &source_order, &mut arena)?;
        let direct = tracked
            .iter()
            .map(|entry| entry.value.clone())
            .collect::<Vec<_>>();
        if direct
            .iter()
            .enumerate()
            .all(|(index, entry)| schreyer_entry_is_reduced(index, entry, &direct, &arena))
        {
            direct
        } else {
            interreduce_tracked(
                field,
                tracked,
                &source_order,
                options.max_reduction_steps,
                &mut arena,
            )?
            .into_iter()
            .map(|entry| entry.value)
            .collect()
        }
    } else {
        interreduce_tracked(
            field,
            tracked,
            &source_order,
            options.max_reduction_steps,
            &mut arena,
        )?
        .into_iter()
        .map(|entry| entry.value)
        .collect()
    };
    let construction_monomials = retained_module_monomials(&packed_basis, &relations);
    let evidence = if used_monomial_lane {
        GroebnerEvidence::MonomialSchreyerTheorem
    } else {
        metrics.schreyer_parallel_verification_pairs = verify_reduced_schreyer_basis(
            field,
            &schreyer_generators,
            &relations,
            &source_order,
            options,
            &mut arena,
        )?;
        for relation in &relations {
            verify_relation(field, relation, &packed_basis, &packed_order, &mut arena)?;
        }
        GroebnerEvidence::IndependentBuchbergerSweep
    };
    metrics.monomials_interned = construction_monomials;
    Ok(ModuleSyzygyOutput {
        generators: relations
            .iter()
            .map(|relation| relation.to_sparse(field, &arena))
            .collect(),
        metrics,
        evidence,
    })
}

struct SchreyerCellReduction<C> {
    relations: Vec<ArenaModuleElement<C>>,
    metrics: GroebnerMetrics,
}

#[cfg(feature = "parallel")]
struct PortableSchreyerCell<C> {
    degree: u64,
    pair_count: usize,
    relations: Vec<SparseModuleElement<C>>,
    metrics: GroebnerMetrics,
}

#[cfg(feature = "parallel")]
struct ParallelSchreyerCells<C> {
    outputs: Vec<PortableSchreyerCell<C>>,
    peak_workspace_bytes: usize,
}

fn observe_schreyer_round(metrics: &mut GroebnerMetrics, degree: u64, pair_count: usize) {
    metrics.rounds = metrics.rounds.saturating_add(1);
    metrics.minimum_round_degree = Some(
        metrics
            .minimum_round_degree
            .map_or(degree, |current| current.min(degree)),
    );
    metrics.maximum_round_degree = Some(
        metrics
            .maximum_round_degree
            .map_or(degree, |current| current.max(degree)),
    );
    metrics.round_degree_sum = metrics.round_degree_sum.saturating_add(u128::from(degree));
    metrics.pairs_selected = metrics.pairs_selected.saturating_add(pair_count);
}

fn retained_module_monomials<C: Clone + Eq>(
    basis: &[ArenaModuleElement<C>],
    relations: &[ArenaModuleElement<C>],
) -> usize {
    basis
        .iter()
        .chain(relations)
        .flat_map(|value| value.support.iter().map(|term| term.monomial()))
        .collect::<HashSet<_>>()
        .len()
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn reduce_schreyer_cells_parallel<F, R>(
    field: &F,
    basis: &[ArenaModuleElement<F::Element>],
    leaders: &[ModuleTerm],
    cells: &[(u64, Vec<SchreyerPairWork>)],
    target_rank: usize,
    target_order: &ModuleOrder,
    source_order: &ModuleOrder,
    options: &GroebnerOptions,
    reducer: &R,
    arena: &MonomialArena,
) -> Result<ParallelSchreyerCells<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F> + Sync,
{
    let workers = schreyer_cell_workers(options, arena, cells.len());
    let retained_arena_bytes = arena.allocated_bytes();
    let worker_workspace = options
        .max_workspace_bytes
        .saturating_sub(retained_arena_bytes)
        / workers;
    let assignments = schreyer_cell_assignments(cells, workers);
    let worker_outputs = assignments
        .into_par_iter()
        .map(|assignment| {
            let mut worker_arena = arena.fork();
            let mut worker_options = options.clone();
            worker_options.max_workspace_bytes = worker_workspace;
            let outputs = assignment
                .iter()
                .map(|&cell_index| {
                    let (degree, pairs) = &cells[cell_index];
                    let cell = reduce_schreyer_cell(
                        field,
                        basis,
                        leaders,
                        pairs,
                        target_rank,
                        target_order,
                        source_order,
                        &worker_options,
                        reducer,
                        &mut worker_arena,
                    )?;
                    Ok((
                        cell_index,
                        PortableSchreyerCell {
                            degree: *degree,
                            pair_count: pairs.len(),
                            relations: cell
                                .relations
                                .iter()
                                .map(|relation| relation.to_sparse(field, &worker_arena))
                                .collect(),
                            metrics: cell.metrics,
                        },
                    ))
                })
                .collect::<Result<Vec<_>, GroebnerError>>()?;
            let peak_workspace_bytes = outputs
                .iter()
                .map(|(_, output)| output.metrics.peak_workspace_bytes)
                .max()
                .unwrap_or(0);
            Ok((outputs, peak_workspace_bytes))
        })
        .collect::<Result<Vec<_>, GroebnerError>>()?;
    let peak_workspace_bytes = worker_outputs
        .iter()
        .fold(retained_arena_bytes, |total, (_, worker_peak)| {
            total.saturating_add(*worker_peak)
        });
    let mut outputs = worker_outputs
        .into_iter()
        .flat_map(|(outputs, _)| outputs)
        .collect::<Vec<_>>();
    outputs.sort_unstable_by_key(|(cell_index, _)| *cell_index);
    Ok(ParallelSchreyerCells {
        outputs: outputs.into_iter().map(|(_, output)| output).collect(),
        peak_workspace_bytes,
    })
}

#[cfg(feature = "parallel")]
fn schreyer_cell_assignments<C>(cells: &[(u64, Vec<C>)], workers: usize) -> Vec<Vec<usize>> {
    let mut by_work = cells
        .iter()
        .enumerate()
        .map(|(index, (_, pairs))| (index, pairs.len()))
        .collect::<Vec<_>>();
    by_work.sort_unstable_by(|(left_index, left_work), (right_index, right_work)| {
        right_work
            .cmp(left_work)
            .then_with(|| left_index.cmp(right_index))
    });
    let mut loads = vec![0_usize; workers];
    let mut assignments = vec![Vec::new(); workers];
    for (cell_index, work) in by_work {
        let worker = loads
            .iter()
            .enumerate()
            .min_by_key(|(worker, load)| (**load, *worker))
            .expect("a parallel Schreyer schedule has a worker")
            .0;
        loads[worker] = loads[worker].saturating_add(work);
        assignments[worker].push(cell_index);
    }
    for assignment in &mut assignments {
        assignment.sort_unstable();
    }
    assignments
}

#[cfg(feature = "parallel")]
fn schreyer_cell_workers(options: &GroebnerOptions, arena: &MonomialArena, cells: usize) -> usize {
    let worker_bytes = arena.allocated_bytes().max(1);
    let memory_workers = (options.max_workspace_bytes / worker_bytes).max(1);
    rayon::current_num_threads().min(memory_workers).min(cells)
}

#[allow(clippy::too_many_arguments)]
fn reduce_schreyer_cell<F, R>(
    field: &F,
    basis: &[ArenaModuleElement<F::Element>],
    leaders: &[ModuleTerm],
    pairs: &[SchreyerPairWork],
    target_rank: usize,
    target_order: &ModuleOrder,
    source_order: &ModuleOrder,
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
) -> Result<SchreyerCellReduction<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let mut targets = Vec::with_capacity(pairs.len());
    let mut target_representations = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let left_leading = leaders[pair.left];
        let right_leading = leaders[pair.right];
        let left_multiplier = arena
            .quotient(pair.lcm, left_leading.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?;
        let right_multiplier = arena
            .quotient(pair.lcm, right_leading.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?;
        targets.push(s_element(
            field,
            &basis[pair.left],
            &basis[pair.right],
            pair.lcm,
            target_order,
            arena,
        )?);
        let mut representation = zero_representation(basis.len(), arena);
        representation[pair.left] =
            ArenaPolynomial::new(arena, [(left_multiplier, field.one())], field);
        representation[pair.right] =
            ArenaPolynomial::new(arena, [(right_multiplier, field.neg(&field.one()))], field);
        target_representations.push(representation);
    }
    let target_count = targets.len();
    let (rows, reducer_plans) = symbolic_preprocess(basis, targets, target_order, options, arena)?;
    let mut representations = target_representations;
    representations.extend(identity_representations_from_plans(
        field,
        basis.len(),
        &reducer_plans,
        arena,
    ));
    if rows.len() != representations.len() || rows.len() > options.max_matrix_rows {
        return Err(resource("Schreyer F4 matrix rows", options.max_matrix_rows));
    }
    let columns = matrix_columns(&rows, target_order, arena)?;
    let representation_columns = representation_columns(&representations, arena);
    let total_columns = columns.len().saturating_add(representation_columns.len());
    if total_columns > options.max_matrix_columns {
        return Err(resource(
            "Schreyer F4 augmented matrix columns",
            options.max_matrix_columns,
        ));
    }
    let matrix = assemble_augmented_work(
        &rows,
        &representations,
        &columns,
        &representation_columns,
        target_count,
    );
    let mut metrics = GroebnerMetrics::default();
    observe_matrix_metrics(
        field,
        reducer,
        &matrix,
        total_columns,
        arena,
        options,
        &mut metrics,
    )?;
    let F4Reduction {
        rows: reduced,
        reduced_tails,
    } = reducer.reduce_f4(field, total_columns, matrix, &mut metrics);
    let mut nonzero_value_rows = 0_usize;
    let mut relations = Vec::new();
    for row in reduced {
        let (value, representation) = split_augmented_row(
            field,
            target_rank,
            basis.len(),
            row,
            &columns,
            &representation_columns,
            target_order,
            arena,
        )?;
        if !value.is_zero() {
            nonzero_value_rows += 1;
            continue;
        }
        let relation = representation_as_module(field, representation, source_order, arena)?;
        if relation.is_zero() {
            continue;
        }
        let relation = monic_module_element(field, relation);
        if !relations.contains(&relation) {
            relations.push(relation);
        }
    }
    if (reduced_tails && nonzero_value_rows != 0)
        || (!reduced_tails && nonzero_value_rows != reducer_plans.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    Ok(SchreyerCellReduction { relations, metrics })
}

fn merge_schreyer_cell_metrics(total: &mut GroebnerMetrics, cell: &GroebnerMetrics) {
    total.matrix_rows = total.matrix_rows.saturating_add(cell.matrix_rows);
    total.matrix_columns = total.matrix_columns.saturating_add(cell.matrix_columns);
    total.matrix_nonzeros = total.matrix_nonzeros.saturating_add(cell.matrix_nonzeros);
    total.peak_matrix_rows = total.peak_matrix_rows.max(cell.peak_matrix_rows);
    total.peak_matrix_columns = total.peak_matrix_columns.max(cell.peak_matrix_columns);
    total.peak_matrix_nonzeros = total.peak_matrix_nonzeros.max(cell.peak_matrix_nonzeros);
    total.peak_monomial_bytes = total.peak_monomial_bytes.max(cell.peak_monomial_bytes);
    total.peak_sparse_row_bytes = total.peak_sparse_row_bytes.max(cell.peak_sparse_row_bytes);
    total.peak_dense_row_bytes = total.peak_dense_row_bytes.max(cell.peak_dense_row_bytes);
    total.peak_coefficient_bytes = total
        .peak_coefficient_bytes
        .max(cell.peak_coefficient_bytes);
    total.peak_workspace_bytes = total.peak_workspace_bytes.max(cell.peak_workspace_bytes);
    total.generic_row_reductions = total
        .generic_row_reductions
        .saturating_add(cell.generic_row_reductions);
    total.rational_dense_row_reductions = total
        .rational_dense_row_reductions
        .saturating_add(cell.rational_dense_row_reductions);
    total.binary_sparse_row_reductions = total
        .binary_sparse_row_reductions
        .saturating_add(cell.binary_sparse_row_reductions);
    total.binary_dense_row_reductions = total
        .binary_dense_row_reductions
        .saturating_add(cell.binary_dense_row_reductions);
    total.binary_extension_dense_row_reductions = total
        .binary_extension_dense_row_reductions
        .saturating_add(cell.binary_extension_dense_row_reductions);
    total.prime_sparse_row_reductions = total
        .prime_sparse_row_reductions
        .saturating_add(cell.prime_sparse_row_reductions);
    total.prime_abcd_row_reductions = total
        .prime_abcd_row_reductions
        .saturating_add(cell.prime_abcd_row_reductions);
    total.prime_dense_row_reductions = total
        .prime_dense_row_reductions
        .saturating_add(cell.prime_dense_row_reductions);
    total.abcd_left_columns = total
        .abcd_left_columns
        .saturating_add(cell.abcd_left_columns);
    total.abcd_right_columns = total
        .abcd_right_columns
        .saturating_add(cell.abcd_right_columns);
    total.abcd_dense_cells = total.abcd_dense_cells.saturating_add(cell.abcd_dense_cells);
    total.abcd_full_dense_cells = total
        .abcd_full_dense_cells
        .saturating_add(cell.abcd_full_dense_cells);
    total.abcd_parallel_projection_rounds = total
        .abcd_parallel_projection_rounds
        .saturating_add(cell.abcd_parallel_projection_rounds);
    total.abcd_parallel_projection_rows = total
        .abcd_parallel_projection_rows
        .saturating_add(cell.abcd_parallel_projection_rows);
    if cell.last_row_backend.is_some() {
        total.last_row_backend = cell.last_row_backend;
    }
}

fn module_hilbert_target_reached<C: Clone + Eq>(
    relations: &[ArenaModuleElement<C>],
    target: &StandardModuleHilbertTarget,
    arity: usize,
    arena: &MonomialArena,
    metrics: &mut GroebnerMetrics,
) -> bool {
    metrics.hilbert_target_checks = metrics.hilbert_target_checks.saturating_add(1);
    let leading_terms = relations
        .iter()
        .filter_map(|relation| {
            relation
                .leading_term()
                .map(|(term, _)| (arena.get(term.monomial()).clone(), term.component()))
        })
        .collect::<Vec<_>>();
    crate::hilbert::standard_module_quotient_hilbert_numerator(
        arity,
        target.generator_degrees.len(),
        &leading_terms,
        &target.generator_degrees,
    )
    .is_ok_and(|numerator| {
        equal_trimmed_integer_polynomials(&numerator, &target.quotient_numerator)
    })
}

fn equal_trimmed_integer_polynomials(left: &[Integer], right: &[Integer]) -> bool {
    fn trimmed_length(values: &[Integer]) -> usize {
        values
            .iter()
            .rposition(|value| !value.is_zero())
            .map_or(1, |index| index + 1)
    }
    let left_length = trimmed_length(left);
    let right_length = trimmed_length(right);
    left_length == right_length
        && (0..left_length).all(|index| {
            left.get(index).cloned().unwrap_or_else(Integer::zero)
                == right.get(index).cloned().unwrap_or_else(Integer::zero)
        })
}

fn module_groebner_basis_without_transform_with_reducer<F, R>(
    field: &F,
    generators: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    reducer: &R,
) -> Result<ModuleBasisOutput<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || generators.iter().any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = generators
        .iter()
        .map(|value| value.terms.len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    let inputs = generators
        .iter()
        .map(|value| ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    let mut basis = normalized_module_generators(field, &inputs, &packed_order, &mut arena)?;
    if basis.len() > options.max_basis_size {
        return Err(resource("module basis size", options.max_basis_size));
    }
    if inputs.iter().all(|value| value.len() <= 1) {
        basis = module_divisibility_antichain(basis, &arena)?;
        let metrics = monomial_module_metrics(&basis, &arena);
        return Ok(ModuleBasisOutput {
            basis: basis
                .iter()
                .map(|entry| entry.to_sparse(field, &arena))
                .collect(),
            metrics,
            evidence: GroebnerEvidence::MonomialDivisibilityTheorem,
        });
    }

    let mut metrics = GroebnerMetrics::default();
    let mut pending = Vec::<Pair>::new();
    for right in 1..basis.len() {
        install_pairs(
            &basis,
            right,
            options,
            &mut pending,
            &mut metrics,
            &mut arena,
        )?;
    }

    while !pending.is_empty() {
        if metrics.rounds == options.max_rounds {
            return Err(resource("module F4 rounds", options.max_rounds));
        }
        let degree = pending
            .iter()
            .map(|pair| pair.degree)
            .min()
            .expect("a nonempty pair set has a minimum degree");
        let mut selected = Vec::new();
        let mut retained = Vec::with_capacity(pending.len());
        for pair in pending.drain(..) {
            let room =
                options.max_pairs_per_round == 0 || selected.len() < options.max_pairs_per_round;
            if pair.degree == degree && room {
                selected.push(pair);
            } else {
                retained.push(pair);
            }
        }
        pending = retained;
        metrics.rounds += 1;
        metrics.minimum_round_degree = Some(
            metrics
                .minimum_round_degree
                .map_or(degree, |current| current.min(degree)),
        );
        metrics.maximum_round_degree = Some(
            metrics
                .maximum_round_degree
                .map_or(degree, |current| current.max(degree)),
        );
        metrics.round_degree_sum = metrics.round_degree_sum.saturating_add(u128::from(degree));
        metrics.pairs_selected = metrics.pairs_selected.saturating_add(selected.len());

        let (targets, target_plans) = selected_rows(&basis, selected, &packed_order, &mut arena)?;
        if targets.is_empty() {
            continue;
        }
        let (rows, _) = symbolic_preprocess(&basis, targets, &packed_order, options, &mut arena)?;
        if rows.len() > options.max_matrix_rows {
            return Err(resource("module F4 matrix rows", options.max_matrix_rows));
        }
        let columns = matrix_columns(&rows, &packed_order, &mut arena)?;
        if columns.len() > options.max_matrix_columns {
            return Err(resource(
                "module F4 matrix columns",
                options.max_matrix_columns,
            ));
        }
        let matrix = assemble_module_work(&rows, &columns, target_plans.len());
        observe_matrix_metrics(
            field,
            reducer,
            &matrix,
            columns.len(),
            &arena,
            options,
            &mut metrics,
        )?;
        let F4Reduction {
            rows: reduced,
            reduced_tails,
        } = reducer.reduce_f4(field, columns.len(), matrix, &mut metrics);

        let mut new_entries = Vec::<ArenaModuleElement<F::Element>>::new();
        for row in reduced {
            let value = split_module_row(field, rank, row, &columns, &packed_order, &mut arena)?;
            if value.is_zero()
                || value.leading_term().is_some_and(|(leading, _)| {
                    basis.iter().any(|known| {
                        known
                            .leading_term()
                            .is_some_and(|(known, _)| module_term_divides(known, leading, &arena))
                    })
                })
            {
                continue;
            }
            let value = if reduced_tails {
                make_monic_value(field, value)
            } else {
                normal_form_value(
                    field,
                    value,
                    &basis,
                    &new_entries,
                    None,
                    &packed_order,
                    options.max_reduction_steps,
                    &mut arena,
                )?
                .map(|value| make_monic_value(field, value))
                .unwrap_or_else(|| ArenaModuleElement::zero(rank))
            };
            if value.is_zero()
                || basis
                    .iter()
                    .chain(&new_entries)
                    .any(|known| known == &value)
            {
                continue;
            }
            new_entries.push(value);
        }
        for entry in new_entries {
            if basis.len() == options.max_basis_size {
                return Err(resource("module basis size", options.max_basis_size));
            }
            let index = basis.len();
            basis.push(entry);
            install_pairs(
                &basis,
                index,
                options,
                &mut pending,
                &mut metrics,
                &mut arena,
            )?;
        }
    }

    basis = interreduce_values(
        field,
        basis,
        &packed_order,
        options.max_reduction_steps,
        &mut arena,
    )?;
    verify_module_basis(field, &inputs, &basis, &packed_order, options, &mut arena)?;
    metrics.monomials_interned = arena.len();
    Ok(ModuleBasisOutput {
        basis: basis
            .iter()
            .map(|entry| entry.to_sparse(field, &arena))
            .collect(),
        metrics,
        evidence: GroebnerEvidence::IndependentBuchbergerSweep,
    })
}

fn module_groebner_basis_with_reducer<F, R>(
    field: &F,
    generators: &[SparseModuleElement<F::Element>],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
    reducer: &R,
) -> Result<ModuleGroebnerOutput<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    let arity = first.arity;
    let rank = first.rank;
    let polynomial_order = first.polynomial_order;
    if order.base_order() != polynomial_order
        || generators.iter().any(|value| {
            value.arity != arity || value.rank != rank || value.polynomial_order != polynomial_order
        })
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = generators
        .iter()
        .map(|value| value.terms.len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        polynomial_order,
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let packed_order = order.pack(&mut arena, rank)?;
    let inputs = generators
        .iter()
        .map(|value| ArenaModuleElement::from_sparse(field, value, &packed_order, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    let one = arena
        .intern_exponents(&vec![0; arity])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut basis = normalized_tracked_generators(field, &inputs, one, &packed_order, &mut arena)?;
    if basis.len() > options.max_basis_size {
        return Err(resource("module basis size", options.max_basis_size));
    }
    if inputs.iter().all(|value| value.len() <= 1) {
        let leaders = basis
            .iter()
            .map(|entry| {
                entry
                    .value
                    .leading_term()
                    .map(|(term, _)| term)
                    .ok_or(GroebnerError::VerificationFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let keep = leaders
            .iter()
            .enumerate()
            .map(|(index, leading)| {
                !leaders.iter().enumerate().any(|(other_index, other)| {
                    index != other_index
                        && module_term_divides(*other, *leading, &arena)
                        && (other != leading || other_index < index)
                })
            })
            .collect::<Vec<_>>();
        basis = basis
            .into_iter()
            .zip(keep)
            .filter_map(|(entry, keep)| keep.then_some(entry))
            .collect();
        let mut metrics = GroebnerMetrics {
            monomials_interned: arena.len(),
            ..GroebnerMetrics::default()
        };
        for left in 0..basis.len() {
            for right in (left + 1)..basis.len() {
                if basis[left]
                    .value
                    .leading_term()
                    .zip(basis[right].value.leading_term())
                    .is_some_and(|((left, _), (right, _))| left.component() == right.component())
                {
                    metrics.pairs_generated = metrics.pairs_generated.saturating_add(1);
                }
            }
        }
        return Ok(ModuleGroebnerOutput {
            basis: basis
                .iter()
                .map(|entry| entry.value.to_sparse(field, &arena))
                .collect(),
            transformations: basis
                .into_iter()
                .map(|entry| sparse_representation(entry.representation, &arena, field))
                .collect(),
            metrics,
            evidence: GroebnerEvidence::MonomialDivisibilityTheorem,
        });
    }
    let mut metrics = GroebnerMetrics::default();
    let (preconditioned_basis, source_transformation) = row_reduce_tracked_generators(
        field,
        basis,
        rank,
        inputs.len(),
        one,
        &packed_order,
        options,
        reducer,
        &mut arena,
        &mut metrics,
    )?;
    basis = preconditioned_basis;
    let representation_arity = basis
        .first()
        .map_or(inputs.len(), |entry| entry.representation.len());
    let mut pending = Vec::<Pair>::new();
    for right in 1..basis.len() {
        install_pairs(
            &basis,
            right,
            options,
            &mut pending,
            &mut metrics,
            &mut arena,
        )?;
    }

    while !pending.is_empty() {
        if metrics.rounds == options.max_rounds {
            return Err(resource("module F4 rounds", options.max_rounds));
        }
        let degree = pending
            .iter()
            .map(|pair| pair.degree)
            .min()
            .expect("a nonempty pair set has a minimum degree");
        let mut selected = Vec::new();
        let mut retained = Vec::with_capacity(pending.len());
        for pair in pending.drain(..) {
            let room =
                options.max_pairs_per_round == 0 || selected.len() < options.max_pairs_per_round;
            if pair.degree == degree && room {
                selected.push(pair);
            } else {
                retained.push(pair);
            }
        }
        pending = retained;
        metrics.rounds += 1;
        metrics.minimum_round_degree = Some(
            metrics
                .minimum_round_degree
                .map_or(degree, |current| current.min(degree)),
        );
        metrics.maximum_round_degree = Some(
            metrics
                .maximum_round_degree
                .map_or(degree, |current| current.max(degree)),
        );
        metrics.round_degree_sum = metrics.round_degree_sum.saturating_add(u128::from(degree));
        metrics.pairs_selected = metrics.pairs_selected.saturating_add(selected.len());

        let (targets, target_plans) = selected_rows(&basis, selected, &packed_order, &mut arena)?;
        if targets.is_empty() {
            continue;
        }
        let (rows, reducer_plans) =
            symbolic_preprocess(&basis, targets, &packed_order, options, &mut arena)?;
        let target_count = target_plans.len();
        let mut representations = representations_from_plans(&basis, &target_plans, &mut arena)?;
        representations.extend(representations_from_plans(
            &basis,
            &reducer_plans,
            &mut arena,
        )?);
        if rows.len() != representations.len() || rows.len() > options.max_matrix_rows {
            return Err(resource("module F4 matrix rows", options.max_matrix_rows));
        }
        let columns = matrix_columns(&rows, &packed_order, &mut arena)?;
        let representation_columns = representation_columns(&representations, &arena);
        let total_columns = columns.len().saturating_add(representation_columns.len());
        if total_columns > options.max_matrix_columns {
            return Err(resource(
                "module F4 augmented matrix columns",
                options.max_matrix_columns,
            ));
        }
        let matrix = assemble_augmented_work(
            &rows,
            &representations,
            &columns,
            &representation_columns,
            target_count,
        );
        observe_matrix_metrics(
            field,
            reducer,
            &matrix,
            total_columns,
            &arena,
            options,
            &mut metrics,
        )?;
        let F4Reduction {
            rows: reduced,
            reduced_tails,
        } = reducer.reduce_f4(field, total_columns, matrix, &mut metrics);

        let mut new_entries = Vec::<Tracked<F::Element>>::new();
        for row in reduced {
            let (value, representation) = split_augmented_row(
                field,
                rank,
                representation_arity,
                row,
                &columns,
                &representation_columns,
                &packed_order,
                &mut arena,
            )?;
            if value.is_zero()
                || value.leading_term().is_some_and(|(leading, _)| {
                    basis.iter().any(|known| {
                        known
                            .value
                            .leading_term()
                            .is_some_and(|(known, _)| module_term_divides(known, leading, &arena))
                    })
                })
            {
                continue;
            }
            let tracked = if reduced_tails {
                make_monic(
                    field,
                    Tracked {
                        value,
                        representation,
                    },
                    &arena,
                )
            } else {
                normal_form_tracked(
                    field,
                    Tracked {
                        value,
                        representation,
                    },
                    &basis
                        .iter()
                        .chain(&new_entries)
                        .cloned()
                        .collect::<Vec<_>>(),
                    None,
                    &packed_order,
                    options.max_reduction_steps,
                    &mut arena,
                )?
                .map(|entry| make_monic(field, entry, &arena))
                .unwrap_or_else(|| Tracked {
                    value: ArenaModuleElement::zero(rank),
                    representation: zero_representation(representation_arity, &arena),
                })
            };
            if tracked.value.is_zero()
                || basis
                    .iter()
                    .chain(&new_entries)
                    .any(|known| known.value == tracked.value)
            {
                continue;
            }
            new_entries.push(tracked);
        }
        for entry in new_entries {
            if basis.len() == options.max_basis_size {
                return Err(resource("module basis size", options.max_basis_size));
            }
            let index = basis.len();
            basis.push(entry);
            install_pairs(
                &basis,
                index,
                options,
                &mut pending,
                &mut metrics,
                &mut arena,
            )?;
        }
    }

    basis = interreduce_tracked(
        field,
        basis,
        &packed_order,
        options.max_reduction_steps,
        &mut arena,
    )?;
    if let Some(source_transformation) = source_transformation {
        for entry in &mut basis {
            entry.representation = compose_representations(
                field,
                &entry.representation,
                &source_transformation,
                &mut arena,
            )?;
        }
    }
    verify_output(field, &inputs, &basis, &packed_order, options, &mut arena)?;
    metrics.monomials_interned = arena.len();
    Ok(ModuleGroebnerOutput {
        basis: basis
            .iter()
            .map(|entry| entry.value.to_sparse(field, &arena))
            .collect(),
        transformations: basis
            .into_iter()
            .map(|entry| sparse_representation(entry.representation, &arena, field))
            .collect(),
        metrics,
        evidence: GroebnerEvidence::IndependentBuchbergerSweep,
    })
}

fn normalized_module_generators<F: Field>(
    field: &F,
    generators: &[ArenaModuleElement<F::Element>],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<Vec<ArenaModuleElement<F::Element>>, GroebnerError> {
    let mut entries = Vec::new();
    for generator in generators {
        let Some((_, coefficient)) = generator.leading_term() else {
            continue;
        };
        let value = generator.scalar_mul(field, &field.inverse(coefficient));
        if !entries.contains(&value) {
            entries.push(value);
        }
    }
    sort_module_values(&mut entries, order, arena)?;
    Ok(entries)
}

fn module_divisibility_antichain<C: Clone + Eq>(
    basis: Vec<ArenaModuleElement<C>>,
    arena: &MonomialArena,
) -> Result<Vec<ArenaModuleElement<C>>, GroebnerError> {
    let leaders = basis
        .iter()
        .map(|entry| {
            entry
                .leading_term()
                .map(|(term, _)| term)
                .ok_or(GroebnerError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let keep = leaders
        .iter()
        .enumerate()
        .map(|(index, leading)| {
            !leaders.iter().enumerate().any(|(other_index, other)| {
                index != other_index
                    && module_term_divides(*other, *leading, arena)
                    && (other != leading || other_index < index)
            })
        })
        .collect::<Vec<_>>();
    Ok(basis
        .into_iter()
        .zip(keep)
        .filter_map(|(entry, keep)| keep.then_some(entry))
        .collect())
}

fn monomial_module_metrics<C: Clone + Eq>(
    basis: &[ArenaModuleElement<C>],
    arena: &MonomialArena,
) -> GroebnerMetrics {
    let mut metrics = GroebnerMetrics {
        monomials_interned: arena.len(),
        ..GroebnerMetrics::default()
    };
    for left in 0..basis.len() {
        for right in (left + 1)..basis.len() {
            if basis[left]
                .leading_term()
                .zip(basis[right].leading_term())
                .is_some_and(|((left, _), (right, _))| left.component() == right.component())
            {
                metrics.pairs_generated = metrics.pairs_generated.saturating_add(1);
            }
        }
    }
    metrics
}

fn sort_module_values<C: Clone + Eq>(
    entries: &mut [ArenaModuleElement<C>],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for index in 1..entries.len() {
        let mut cursor = index;
        while cursor != 0 {
            let left = entries[cursor]
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            let right = entries[cursor - 1]
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            if compare_terms(order, left, right, arena)? != Ordering::Greater {
                break;
            }
            entries.swap(cursor, cursor - 1);
            cursor -= 1;
        }
    }
    Ok(())
}

fn normalized_tracked_generators<F: Field>(
    field: &F,
    generators: &[ArenaModuleElement<F::Element>],
    one: MonomialId,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<Vec<Tracked<F::Element>>, GroebnerError> {
    let mut entries = Vec::<Tracked<F::Element>>::new();
    for (index, generator) in generators.iter().enumerate() {
        let Some((_, coefficient)) = generator.leading_term() else {
            continue;
        };
        let inverse = field.inverse(coefficient);
        let value = generator.scalar_mul(field, &inverse);
        if entries.iter().any(|known| known.value == value) {
            continue;
        }
        let mut representation = zero_representation(generators.len(), arena);
        representation[index] = ArenaPolynomial::new(arena, [(one, inverse)], field);
        entries.push(Tracked {
            value,
            representation,
        });
    }
    sort_tracked(&mut entries, order, arena)?;
    Ok(entries)
}

#[allow(clippy::too_many_arguments)]
fn row_reduce_tracked_generators<F, R>(
    field: &F,
    basis: Vec<Tracked<F::Element>>,
    rank: usize,
    generator_count: usize,
    one: MonomialId,
    order: &ModuleOrder,
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
    metrics: &mut GroebnerMetrics,
) -> Result<TrackedPrecondition<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    if basis.len() < MIN_TRACKED_GENERATOR_ROW_REDUCTION {
        return Ok((basis, None));
    }
    let (values, representations): (Vec<_>, Vec<_>) = basis
        .into_iter()
        .map(|entry| (entry.value, entry.representation))
        .unzip();
    let value_columns = matrix_columns(&values, order, arena)?;
    let representation_columns = representation_columns(&representations, arena);
    let total_columns = value_columns
        .len()
        .saturating_add(representation_columns.len());
    if values.len() > options.max_matrix_rows {
        return Err(resource(
            "module generator row-reduction matrix rows",
            options.max_matrix_rows,
        ));
    }
    if total_columns > options.max_matrix_columns {
        return Err(resource(
            "module generator row-reduction matrix columns",
            options.max_matrix_columns,
        ));
    }
    let row_count = values.len();
    let matrix = assemble_augmented_work(
        &values,
        &representations,
        &value_columns,
        &representation_columns,
        row_count,
    );
    observe_matrix_metrics(
        field,
        reducer,
        &matrix,
        total_columns,
        arena,
        options,
        metrics,
    )?;
    let rows = reducer.reduce(field, total_columns, matrix.into_rows(), metrics);
    let mut reduced = Vec::with_capacity(row_count);
    for row in rows {
        let (value, representation) = split_augmented_row(
            field,
            rank,
            generator_count,
            row,
            &value_columns,
            &representation_columns,
            order,
            arena,
        )?;
        if !value.is_zero() {
            reduced.push(make_monic(
                field,
                Tracked {
                    value,
                    representation,
                },
                arena,
            ));
        }
    }
    sort_tracked(&mut reduced, order, arena)?;
    let reduced_count = reduced.len();
    let mut source_transformation = Vec::with_capacity(reduced_count);
    for (index, entry) in reduced.iter_mut().enumerate() {
        let mut identity = zero_representation(reduced_count, arena);
        identity[index] = ArenaPolynomial::new(arena, [(one, field.one())], field);
        source_transformation.push(std::mem::replace(&mut entry.representation, identity));
    }
    Ok((reduced, Some(source_transformation)))
}

fn compose_representations<F: Field>(
    field: &F,
    outer: &Representation<F::Element>,
    inner: &[Representation<F::Element>],
    arena: &mut MonomialArena,
) -> Result<Representation<F::Element>, GroebnerError> {
    if outer.len() != inner.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let Some(output_arity) = inner.first().map(Vec::len) else {
        return Ok(Vec::new());
    };
    if inner.iter().any(|row| row.len() != output_arity) {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut workspace = PolynomialWorkspace::new();
    let mut output = Vec::with_capacity(output_arity);
    for column in 0..output_arity {
        let mut polynomial = ArenaPolynomial::zero(arena);
        ArenaPolynomial::sum_products_into(
            outer
                .iter()
                .zip(inner)
                .map(|(left, row)| (left, &row[column])),
            arena,
            field,
            &mut workspace,
            &mut polynomial,
        )
        .map_err(|_| GroebnerError::ExponentOverflow)?;
        output.push(polynomial);
    }
    Ok(output)
}

fn zero_representation<C: Clone + Eq>(count: usize, arena: &MonomialArena) -> Representation<C> {
    (0..count).map(|_| ArenaPolynomial::zero(arena)).collect()
}

fn sort_tracked<C: Clone + Eq>(
    entries: &mut [Tracked<C>],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for index in 1..entries.len() {
        let mut cursor = index;
        while cursor != 0 {
            let left = entries[cursor]
                .value
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            let right = entries[cursor - 1]
                .value
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            if compare_terms(order, left, right, arena)? != Ordering::Greater {
                break;
            }
            entries.swap(cursor, cursor - 1);
            cursor -= 1;
        }
    }
    Ok(())
}

fn install_pairs<C: Clone + Eq, B: ModuleValue<C>>(
    basis: &[B],
    right: usize,
    options: &GroebnerOptions,
    pending: &mut Vec<Pair>,
    metrics: &mut GroebnerMetrics,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    let right_leading = basis[right]
        .module_value()
        .leading_term()
        .ok_or(GroebnerError::VerificationFailed)?
        .0;
    for (left, entry) in basis.iter().enumerate().take(right) {
        let left_leading = entry
            .module_value()
            .leading_term()
            .ok_or(GroebnerError::VerificationFailed)?
            .0;
        if left_leading.component() != right_leading.component() {
            continue;
        }
        if pending.len() == options.max_pending_pairs {
            return Err(resource(
                "pending module critical pairs",
                options.max_pending_pairs,
            ));
        }
        let lcm = arena
            .lcm(left_leading.monomial(), right_leading.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        pending.push(Pair {
            left,
            right,
            lcm,
            component: left_leading.component(),
            degree: arena.total_degree(lcm),
        });
        metrics.pairs_generated = metrics.pairs_generated.saturating_add(1);
    }
    Ok(())
}

fn selected_rows<C: Clone + Eq, B: ModuleValue<C>>(
    basis: &[B],
    mut pairs: Vec<Pair>,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(Vec<ArenaModuleElement<C>>, Vec<RowPlan>), GroebnerError> {
    pairs.sort_unstable_by(|left, right| {
        left.component
            .cmp(&right.component)
            .then_with(|| arena.compare(left.lcm, right.lcm))
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    let mut rows = Vec::new();
    let mut plans = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let lcm = pairs[start].lcm;
        let component = pairs[start].component;
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].lcm == lcm && pairs[end].component == component {
            end += 1;
        }
        let mut generators = pairs[start..end]
            .iter()
            .flat_map(|pair| [pair.left, pair.right])
            .collect::<Vec<_>>();
        generators.sort_unstable();
        generators.dedup();
        for generator in generators.into_iter().skip(1) {
            let leading = basis[generator]
                .module_value()
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            let multiplier = arena
                .quotient(lcm, leading.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .ok_or(GroebnerError::VerificationFailed)?;
            rows.push(
                basis[generator]
                    .module_value()
                    .monomial_mul(multiplier, order, arena)?,
            );
            plans.push(RowPlan {
                basis_index: generator,
                multiplier,
            });
        }
        start = end;
    }
    Ok((rows, plans))
}

fn symbolic_preprocess<C: Clone + Eq, B: ModuleValue<C>>(
    basis: &[B],
    targets: Vec<ArenaModuleElement<C>>,
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(Vec<ArenaModuleElement<C>>, Vec<RowPlan>), GroebnerError> {
    let mut visited = HashSet::<ModuleTerm>::new();
    let mut queue = VecDeque::<ModuleTerm>::new();
    for term in targets
        .iter()
        .flat_map(|value| value.support.iter().copied())
    {
        if visited.insert(term) {
            queue.push_back(term);
        }
    }
    let mut rows = targets;
    let mut plans = Vec::new();
    while let Some(term) = queue.pop_front() {
        let Some((index, leading)) = basis.iter().enumerate().find_map(|(index, entry)| {
            entry
                .module_value()
                .leading_term()
                .map(|(leading, _)| leading)
                .filter(|leading| module_term_divides(*leading, term, arena))
                .map(|leading| (index, leading))
        }) else {
            continue;
        };
        let multiplier = arena
            .quotient(term.monomial(), leading.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?;
        let row = basis[index]
            .module_value()
            .monomial_mul(multiplier, order, arena)?;
        for term in row.support.iter().copied() {
            if visited.insert(term) {
                queue.push_back(term);
            }
        }
        if rows.len() == options.max_matrix_rows {
            return Err(resource("module F4 matrix rows", options.max_matrix_rows));
        }
        rows.push(row);
        plans.push(RowPlan {
            basis_index: index,
            multiplier,
        });
    }
    Ok((rows, plans))
}

fn identity_representations_from_plans<F: Field>(
    field: &F,
    basis_len: usize,
    plans: &[RowPlan],
    arena: &MonomialArena,
) -> Vec<Representation<F::Element>> {
    plans
        .iter()
        .map(|plan| {
            let mut representation = zero_representation(basis_len, arena);
            representation[plan.basis_index] =
                ArenaPolynomial::new(arena, [(plan.multiplier, field.one())], field);
            representation
        })
        .collect()
}

fn representations_from_plans<C: Clone + Eq>(
    basis: &[Tracked<C>],
    plans: &[RowPlan],
    arena: &mut MonomialArena,
) -> Result<Vec<Representation<C>>, GroebnerError> {
    plans
        .iter()
        .map(|plan| {
            basis[plan.basis_index]
                .representation
                .iter()
                .map(|value| {
                    value
                        .monomial_mul(plan.multiplier, arena)
                        .map_err(|_| GroebnerError::ExponentOverflow)
                })
                .collect()
        })
        .collect()
}

fn matrix_columns<C: Clone + Eq>(
    rows: &[ArenaModuleElement<C>],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<Vec<ModuleTerm>, GroebnerError> {
    let mut columns = rows
        .iter()
        .flat_map(|row| row.support.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let dummy = columns.drain(..).map(|term| (term, ())).collect::<Vec<_>>();
    let mut dummy = dummy;
    sort_module_terms(&mut dummy, order, arena)?;
    Ok(dummy.into_iter().map(|(term, ())| term).collect())
}

fn representation_columns<C: Clone + Eq>(
    rows: &[Representation<C>],
    arena: &MonomialArena,
) -> Vec<ModuleTerm> {
    let mut columns = rows
        .iter()
        .flat_map(|row| {
            row.iter().enumerate().flat_map(|(component, polynomial)| {
                polynomial
                    .support()
                    .iter()
                    .copied()
                    .map(move |monomial| ModuleTerm::new(monomial, component))
            })
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    columns.sort_unstable_by(|left, right| {
        left.component()
            .cmp(&right.component())
            .then_with(|| arena.compare(right.monomial(), left.monomial()))
    });
    columns
}

fn assemble_module_work<C: Clone + Eq>(
    values: &[ArenaModuleElement<C>],
    value_columns: &[ModuleTerm],
    target_count: usize,
) -> F4RowWork<C> {
    let value_indices = value_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(index, term)| (term, index as u32))
        .collect::<HashMap<_, _>>();
    let mut work = F4RowWork::with_expected_rows(target_count, values.len() - target_count);
    for (row_index, value) in values.iter().enumerate() {
        let row = value
            .iter()
            .map(|(term, coefficient)| (value_indices[&term], coefficient.clone()))
            .collect::<Vec<_>>();
        debug_assert!(row.windows(2).all(|pair| pair[0].0 < pair[1].0));
        if row_index < target_count {
            work.push_target(row);
        } else {
            work.push_reducer(row);
        }
    }
    work
}

fn assemble_augmented_work<C: Clone + Eq>(
    values: &[ArenaModuleElement<C>],
    representations: &[Representation<C>],
    value_columns: &[ModuleTerm],
    representation_columns: &[ModuleTerm],
    target_count: usize,
) -> F4RowWork<C> {
    let value_indices = value_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(index, term)| (term, index as u32))
        .collect::<HashMap<_, _>>();
    let representation_indices = representation_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(index, term)| {
            (
                term,
                u32::try_from(value_columns.len() + index)
                    .expect("an augmented module F4 column fits u32"),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut work = F4RowWork::with_expected_rows(target_count, values.len() - target_count);
    for (row_index, (value, representation)) in values.iter().zip(representations).enumerate() {
        let mut row = value
            .iter()
            .map(|(term, coefficient)| (value_indices[&term], coefficient.clone()))
            .collect::<Vec<_>>();
        for (component, polynomial) in representation.iter().enumerate() {
            for (monomial, coefficient) in polynomial.iter() {
                row.push((
                    representation_indices[&ModuleTerm::new(monomial, component)],
                    coefficient.clone(),
                ));
            }
        }
        row.sort_unstable_by_key(|entry| entry.0);
        debug_assert!(row.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(row
            .iter()
            .all(|entry| (entry.0 as usize) < value_columns.len() + representation_columns.len()));
        if row_index < target_count {
            work.push_target(row);
        } else {
            work.push_reducer(row);
        }
    }
    work
}

fn split_module_row<F: Field>(
    field: &F,
    rank: usize,
    row: Vec<(u32, F::Element)>,
    value_columns: &[ModuleTerm],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<ArenaModuleElement<F::Element>, GroebnerError> {
    let terms = row
        .into_iter()
        .map(|(column, coefficient)| {
            value_columns
                .get(column as usize)
                .copied()
                .map(|term| (term, coefficient))
                .ok_or(GroebnerError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ArenaModuleElement::new(field, rank, terms, order, arena)
}

#[allow(clippy::too_many_arguments)]
fn split_augmented_row<F: Field>(
    field: &F,
    rank: usize,
    generator_count: usize,
    row: Vec<(u32, F::Element)>,
    value_columns: &[ModuleTerm],
    representation_columns: &[ModuleTerm],
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<TrackedParts<F::Element>, GroebnerError> {
    let mut value_terms = Vec::new();
    let mut representation_terms = vec![Vec::new(); generator_count];
    for (column, coefficient) in row {
        let column = column as usize;
        if column < value_columns.len() {
            value_terms.push((value_columns[column], coefficient));
        } else {
            let term = representation_columns
                .get(column - value_columns.len())
                .copied()
                .ok_or(GroebnerError::VerificationFailed)?;
            representation_terms[term.component()].push((term.monomial(), coefficient));
        }
    }
    let value = ArenaModuleElement::new(field, rank, value_terms, order, arena)?;
    let representation = representation_terms
        .into_iter()
        .map(|terms| ArenaPolynomial::new(arena, terms, field))
        .collect();
    Ok((value, representation))
}

fn observe_matrix_metrics<F, R>(
    field: &F,
    reducer: &R,
    matrix: &F4RowWork<F::Element>,
    columns: usize,
    arena: &MonomialArena,
    options: &GroebnerOptions,
    metrics: &mut GroebnerMetrics,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let nonzeros = matrix.nonzeros();
    metrics.matrix_rows = metrics.matrix_rows.saturating_add(matrix.len());
    metrics.matrix_columns = metrics.matrix_columns.saturating_add(columns);
    metrics.matrix_nonzeros = metrics.matrix_nonzeros.saturating_add(nonzeros);
    metrics.peak_matrix_rows = metrics.peak_matrix_rows.max(matrix.len());
    metrics.peak_matrix_columns = metrics.peak_matrix_columns.max(columns);
    metrics.peak_matrix_nonzeros = metrics.peak_matrix_nonzeros.max(nonzeros);
    let (structure, coefficients) = matrix.allocated_workspace_bytes();
    metrics.peak_monomial_bytes = metrics.peak_monomial_bytes.max(arena.allocated_bytes());
    metrics.peak_sparse_row_bytes = metrics.peak_sparse_row_bytes.max(structure);
    metrics.peak_coefficient_bytes = metrics.peak_coefficient_bytes.max(coefficients);
    let dense = reducer.f4_dense_workspace_bytes(field, columns, matrix);
    metrics.peak_dense_row_bytes = metrics.peak_dense_row_bytes.max(dense);
    metrics.peak_workspace_bytes = metrics.peak_workspace_bytes.max(
        arena
            .allocated_bytes()
            .saturating_add(structure)
            .saturating_add(coefficients)
            .saturating_add(dense),
    );
    if metrics.peak_workspace_bytes > options.max_workspace_bytes {
        Err(resource(
            "module F4 workspace bytes",
            options.max_workspace_bytes,
        ))
    } else {
        Ok(())
    }
}

fn normal_form_tracked<F: Field>(
    field: &F,
    mut value: Tracked<F::Element>,
    basis: &[Tracked<F::Element>],
    excluded_basis_index: Option<usize>,
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<Option<Tracked<F::Element>>, GroebnerError> {
    let rank = value.value.rank;
    let mut remainder = ArenaModuleElement::zero(rank);
    let mut steps = 0;
    while let Some((leading, coefficient)) = value.value.leading_term() {
        if steps == max_steps {
            return Err(resource("module normal-form steps", max_steps));
        }
        if let Some(reducer) = basis.iter().enumerate().find_map(|(index, entry)| {
            (Some(index) != excluded_basis_index
                && entry
                    .value
                    .leading_term()
                    .is_some_and(|(divisor, _)| module_term_divides(divisor, leading, arena)))
            .then_some(entry)
        }) {
            let (divisor, divisor_coefficient) = reducer
                .value
                .leading_term()
                .expect("a selected module reducer is nonzero");
            let multiplier = arena
                .quotient(leading.monomial(), divisor.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .ok_or(GroebnerError::VerificationFailed)?;
            let scalar = field.mul(coefficient, &field.inverse(divisor_coefficient));
            value.value = value.value.sub(
                field,
                &reducer
                    .value
                    .term_mul(field, multiplier, &scalar, order, arena)?,
                order,
                arena,
            )?;
            for (coordinate, reducer_coordinate) in
                value.representation.iter_mut().zip(&reducer.representation)
            {
                let correction = reducer_coordinate
                    .term_mul(multiplier, &scalar, arena, field)
                    .map_err(|_| GroebnerError::ExponentOverflow)?;
                *coordinate = coordinate.sub(&correction, arena, field);
            }
        } else {
            let term = ArenaModuleElement::new(
                field,
                rank,
                [(leading, coefficient.clone())],
                order,
                arena,
            )?;
            remainder = remainder.sub(
                field,
                &term.scalar_mul(field, &field.neg(&field.one())),
                order,
                arena,
            )?;
            value.value = value.value.sub(field, &term, order, arena)?;
        }
        steps += 1;
    }
    if remainder.is_zero() {
        Ok(None)
    } else {
        value.value = remainder;
        Ok(Some(value))
    }
}

fn reduces_to_zero<F: Field>(
    field: &F,
    mut value: ArenaModuleElement<F::Element>,
    basis: &[ArenaModuleElement<F::Element>],
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<bool, GroebnerError> {
    let mut steps = 0;
    while let Some((leading, coefficient)) = value.leading_term() {
        if steps == max_steps {
            return Err(resource("module membership reduction steps", max_steps));
        }
        let Some(reducer) = basis.iter().find(|candidate| {
            candidate
                .leading_term()
                .is_some_and(|(divisor, _)| module_term_divides(divisor, leading, arena))
        }) else {
            return Ok(false);
        };
        let (divisor, divisor_coefficient) = reducer
            .leading_term()
            .expect("a selected module reducer is nonzero");
        let multiplier = arena
            .quotient(leading.monomial(), divisor.monomial())
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .ok_or(GroebnerError::VerificationFailed)?;
        let scalar = field.mul(coefficient, &field.inverse(divisor_coefficient));
        value = value.sub(
            field,
            &reducer.term_mul(field, multiplier, &scalar, order, arena)?,
            order,
            arena,
        )?;
        steps += 1;
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn normal_form_value<F: Field>(
    field: &F,
    mut value: ArenaModuleElement<F::Element>,
    basis: &[ArenaModuleElement<F::Element>],
    additional_basis: &[ArenaModuleElement<F::Element>],
    excluded_basis_index: Option<usize>,
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<Option<ArenaModuleElement<F::Element>>, GroebnerError> {
    let rank = value.rank;
    let mut remainder = ArenaModuleElement::zero(rank);
    let mut steps = 0;
    while let Some((leading, coefficient)) = value.leading_term() {
        if steps == max_steps {
            return Err(resource("module normal-form steps", max_steps));
        }
        let reducer = basis
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                (Some(index) != excluded_basis_index
                    && candidate
                        .leading_term()
                        .is_some_and(|(divisor, _)| module_term_divides(divisor, leading, arena)))
                .then_some(candidate)
            })
            .or_else(|| {
                additional_basis.iter().find(|candidate| {
                    candidate
                        .leading_term()
                        .is_some_and(|(divisor, _)| module_term_divides(divisor, leading, arena))
                })
            });
        if let Some(reducer) = reducer {
            let (divisor, divisor_coefficient) = reducer
                .leading_term()
                .expect("a selected module reducer is nonzero");
            let multiplier = arena
                .quotient(leading.monomial(), divisor.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .ok_or(GroebnerError::VerificationFailed)?;
            let scalar = field.mul(coefficient, &field.inverse(divisor_coefficient));
            value = value.sub(
                field,
                &reducer.term_mul(field, multiplier, &scalar, order, arena)?,
                order,
                arena,
            )?;
        } else {
            let term = ArenaModuleElement::new(
                field,
                rank,
                [(leading, coefficient.clone())],
                order,
                arena,
            )?;
            remainder = remainder.sub(
                field,
                &term.scalar_mul(field, &field.neg(&field.one())),
                order,
                arena,
            )?;
            value = value.sub(field, &term, order, arena)?;
        }
        steps += 1;
    }
    Ok((!remainder.is_zero()).then_some(remainder))
}

fn make_monic_value<F: Field>(
    field: &F,
    value: ArenaModuleElement<F::Element>,
) -> ArenaModuleElement<F::Element> {
    let coefficient = value
        .leading_term()
        .expect("a monic candidate is nonzero")
        .1;
    if coefficient == &field.one() {
        value
    } else {
        value.scalar_mul(field, &field.inverse(coefficient))
    }
}

fn interreduce_values<F: Field>(
    field: &F,
    basis: Vec<ArenaModuleElement<F::Element>>,
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<Vec<ArenaModuleElement<F::Element>>, GroebnerError> {
    let mut minimal = Vec::new();
    for (index, entry) in basis.iter().enumerate() {
        let leading = entry
            .leading_term()
            .ok_or(GroebnerError::VerificationFailed)?
            .0;
        let redundant = basis.iter().enumerate().any(|(other, candidate)| {
            other != index
                && candidate.leading_term().is_some_and(|(candidate, _)| {
                    module_term_divides(candidate, leading, arena)
                        && (candidate != leading || other < index)
                })
        });
        if !redundant {
            minimal.push(entry.clone());
        }
    }
    let mut reduced = Vec::new();
    for index in 0..minimal.len() {
        if let Some(value) = normal_form_value(
            field,
            minimal[index].clone(),
            &minimal,
            &[],
            Some(index),
            order,
            max_steps,
            arena,
        )? {
            let value = make_monic_value(field, value);
            if !reduced.contains(&value) {
                reduced.push(value);
            }
        }
    }
    sort_module_values(&mut reduced, order, arena)?;
    Ok(reduced)
}

fn make_monic<F: Field>(
    field: &F,
    entry: Tracked<F::Element>,
    arena: &MonomialArena,
) -> Tracked<F::Element> {
    let coefficient = entry
        .value
        .leading_term()
        .expect("a monic candidate is nonzero")
        .1
        .clone();
    if coefficient == field.one() {
        return entry;
    }
    let inverse = field.inverse(&coefficient);
    Tracked {
        value: entry.value.scalar_mul(field, &inverse),
        representation: entry
            .representation
            .into_iter()
            .map(|value| value.scalar_mul(&inverse, arena, field))
            .collect(),
    }
}

fn interreduce_tracked<F: Field>(
    field: &F,
    basis: Vec<Tracked<F::Element>>,
    order: &ModuleOrder,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<Vec<Tracked<F::Element>>, GroebnerError> {
    let mut minimal = Vec::new();
    for (index, entry) in basis.iter().enumerate() {
        let leading = entry
            .value
            .leading_term()
            .ok_or(GroebnerError::VerificationFailed)?
            .0;
        let redundant = basis.iter().enumerate().any(|(other, candidate)| {
            if other == index {
                return false;
            }
            candidate
                .value
                .leading_term()
                .is_some_and(|(candidate, _)| {
                    module_term_divides(candidate, leading, arena)
                        && (candidate != leading || other < index)
                })
        });
        if !redundant {
            minimal.push(entry.clone());
        }
    }
    let mut reduced = Vec::new();
    for index in 0..minimal.len() {
        if let Some(entry) = normal_form_tracked(
            field,
            minimal[index].clone(),
            &minimal,
            Some(index),
            order,
            max_steps,
            arena,
        )? {
            let entry = make_monic(field, entry, arena);
            if !reduced
                .iter()
                .any(|known: &Tracked<F::Element>| known.value == entry.value)
            {
                reduced.push(entry);
            }
        }
    }
    sort_tracked(&mut reduced, order, arena)?;
    Ok(reduced)
}

fn verify_output<F: Field>(
    field: &F,
    inputs: &[ArenaModuleElement<F::Element>],
    basis: &[Tracked<F::Element>],
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for entry in basis {
        if entry.representation.len() != inputs.len() {
            return Err(GroebnerError::VerificationFailed);
        }
        let represented = ArenaModuleElement::sum_polynomial_products(
            field,
            inputs,
            &entry.representation,
            entry.value.rank,
            order,
            arena,
        )?;
        if represented != entry.value {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    let values = basis
        .iter()
        .map(|entry| entry.value.clone())
        .collect::<Vec<_>>();
    verify_module_basis(field, inputs, &values, order, options, arena)
}

fn verify_module_basis<F: Field>(
    field: &F,
    inputs: &[ArenaModuleElement<F::Element>],
    basis: &[ArenaModuleElement<F::Element>],
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    if basis.iter().any(|entry| {
        entry
            .leading_term()
            .is_none_or(|(_, coefficient)| coefficient != &field.one())
    }) {
        return Err(GroebnerError::VerificationFailed);
    }
    for (index, entry) in basis.iter().enumerate() {
        for term in entry.support.iter().copied() {
            if basis.iter().enumerate().any(|(other, candidate)| {
                other != index
                    && candidate
                        .leading_term()
                        .is_some_and(|(leading, _)| module_term_divides(leading, term, arena))
            }) {
                return Err(GroebnerError::VerificationFailed);
            }
        }
    }
    for input in inputs {
        let reduced = normal_form_value(
            field,
            input.clone(),
            basis,
            &[],
            None,
            order,
            options.max_reduction_steps,
            arena,
        )?;
        if reduced.is_some() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    for right in 1..basis.len() {
        let right_leading = basis[right].leading_term().expect("nonzero").0;
        for left in 0..right {
            let left_leading = basis[left].leading_term().expect("nonzero").0;
            if left_leading.component() != right_leading.component() {
                continue;
            }
            let lcm = arena
                .lcm(left_leading.monomial(), right_leading.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            if arena.total_degree(lcm)
                == arena
                    .total_degree(left_leading.monomial())
                    .saturating_add(arena.total_degree(right_leading.monomial()))
            {
                continue;
            }
            let s = s_element(field, &basis[left], &basis[right], lcm, order, arena)?;
            let reduced = normal_form_value(
                field,
                s,
                basis,
                &[],
                None,
                order,
                options.max_reduction_steps,
                arena,
            )?;
            if reduced.is_some() {
                return Err(GroebnerError::VerificationFailed);
            }
        }
    }
    Ok(())
}

fn s_element<F: Field>(
    field: &F,
    left: &ArenaModuleElement<F::Element>,
    right: &ArenaModuleElement<F::Element>,
    lcm: MonomialId,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<ArenaModuleElement<F::Element>, GroebnerError> {
    let (left_term, left_coefficient) = left.leading_term().expect("nonzero module basis");
    let (right_term, right_coefficient) = right.leading_term().expect("nonzero module basis");
    let left_multiplier = arena
        .quotient(lcm, left_term.monomial())
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .ok_or(GroebnerError::VerificationFailed)?;
    let right_multiplier = arena
        .quotient(lcm, right_term.monomial())
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .ok_or(GroebnerError::VerificationFailed)?;
    left.term_mul(
        field,
        left_multiplier,
        &field.inverse(left_coefficient),
        order,
        arena,
    )?
    .sub(
        field,
        &right.term_mul(
            field,
            right_multiplier,
            &field.inverse(right_coefficient),
            order,
            arena,
        )?,
        order,
        arena,
    )
}

#[derive(Clone, Copy)]
struct SchreyerPairWork {
    left: usize,
    right: usize,
    lcm: MonomialId,
}

struct SchreyerPairSchedule {
    cells: BTreeMap<u64, Vec<SchreyerPairWork>>,
    pairs_generated: usize,
    pairs_pruned: usize,
}

#[derive(Clone, Copy)]
struct SchreyerPairCandidate {
    pair: SchreyerPairWork,
    leading_multiplier: MonomialId,
}

fn schreyer_pair_cells(
    leaders: &[ModuleTerm],
    options: &GroebnerOptions,
    colon_criterion: bool,
    arena: &mut MonomialArena,
) -> Result<SchreyerPairSchedule, GroebnerError> {
    let mut cells = BTreeMap::<u64, Vec<SchreyerPairWork>>::new();
    let mut count = 0_usize;
    let mut pruned = 0_usize;
    for (left, left_leading) in leaders.iter().copied().enumerate() {
        let mut candidates = Vec::new();
        for (right, right_leading) in leaders.iter().copied().enumerate().skip(left + 1) {
            if left_leading.component() != right_leading.component() {
                continue;
            }
            if count == options.max_pending_pairs {
                return Err(resource(
                    "Schreyer critical pairs",
                    options.max_pending_pairs,
                ));
            }
            let lcm = arena
                .lcm(left_leading.monomial(), right_leading.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            let leading_multiplier = arena
                .quotient(lcm, left_leading.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .ok_or(GroebnerError::VerificationFailed)?;
            candidates.push(SchreyerPairCandidate {
                pair: SchreyerPairWork { left, right, lcm },
                leading_multiplier,
            });
            count += 1;
        }
        candidates.sort_unstable_by(|left, right| {
            arena
                .total_degree(left.leading_multiplier)
                .cmp(&arena.total_degree(right.leading_multiplier))
                .then_with(|| arena.compare(left.leading_multiplier, right.leading_multiplier))
                .then_with(|| left.pair.right.cmp(&right.pair.right))
        });
        let mut minimal_multipliers = Vec::new();
        for candidate in candidates {
            if colon_criterion
                && minimal_multipliers
                    .iter()
                    .any(|known| arena.divides(*known, candidate.leading_multiplier))
            {
                pruned = pruned.saturating_add(1);
                continue;
            }
            minimal_multipliers.push(candidate.leading_multiplier);
            cells
                .entry(arena.total_degree(candidate.pair.lcm))
                .or_default()
                .push(candidate.pair);
        }
    }
    Ok(SchreyerPairSchedule {
        cells,
        pairs_generated: count,
        pairs_pruned: pruned,
    })
}

fn representation_as_module<F: Field>(
    field: &F,
    representation: Representation<F::Element>,
    order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<ArenaModuleElement<F::Element>, GroebnerError> {
    let rank = representation.len();
    ArenaModuleElement::new(
        field,
        rank,
        representation
            .into_iter()
            .enumerate()
            .flat_map(|(component, polynomial)| {
                polynomial
                    .iter()
                    .map(|(monomial, coefficient)| {
                        (ModuleTerm::new(monomial, component), coefficient.clone())
                    })
                    .collect::<Vec<_>>()
            }),
        order,
        arena,
    )
}

fn monic_module_element<F: Field>(
    field: &F,
    relation: ArenaModuleElement<F::Element>,
) -> ArenaModuleElement<F::Element> {
    let coefficient = relation
        .leading_term()
        .expect("a normalized relation is nonzero")
        .1
        .clone();
    if coefficient == field.one() {
        relation
    } else {
        relation.scalar_mul(field, &field.inverse(&coefficient))
    }
}

fn verify_relation<F: Field>(
    field: &F,
    relation: &ArenaModuleElement<F::Element>,
    basis: &[ArenaModuleElement<F::Element>],
    target_order: &ModuleOrder,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    if relation.rank != basis.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let target_rank = basis.first().map_or(0, |value| value.rank);
    let mut image = ArenaModuleElement::zero(target_rank);
    let coordinates = relation
        .iter()
        .map(|(term, coefficient)| (term.component(), term.monomial(), coefficient.clone()))
        .collect::<Vec<_>>();
    for (component, monomial, coefficient) in coordinates {
        let term = basis
            .get(component)
            .ok_or(GroebnerError::VerificationFailed)?
            .term_mul(field, monomial, &coefficient, target_order, arena)?;
        image = image.sub(
            field,
            &term.scalar_mul(field, &field.neg(&field.one())),
            target_order,
            arena,
        )?;
    }
    if image.is_zero() {
        Ok(())
    } else {
        Err(GroebnerError::VerificationFailed)
    }
}

fn verify_reduced_schreyer_basis<F: Field>(
    field: &F,
    generators: &[ArenaModuleElement<F::Element>],
    basis: &[ArenaModuleElement<F::Element>],
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<usize, GroebnerError> {
    if basis.iter().any(|entry| {
        entry
            .leading_term()
            .is_none_or(|(_, coefficient)| coefficient != &field.one())
    }) {
        return Err(GroebnerError::VerificationFailed);
    }
    #[cfg(feature = "parallel")]
    if basis.len() >= MIN_PARALLEL_SCHREYER_BASIS_ENTRIES && rayon::current_num_threads() > 1 {
        if !basis
            .par_iter()
            .enumerate()
            .all(|(index, entry)| schreyer_entry_is_reduced(index, entry, basis, arena))
        {
            return Err(GroebnerError::VerificationFailed);
        }
    } else if !basis
        .iter()
        .enumerate()
        .all(|(index, entry)| schreyer_entry_is_reduced(index, entry, basis, arena))
    {
        return Err(GroebnerError::VerificationFailed);
    }
    #[cfg(not(feature = "parallel"))]
    if !basis
        .iter()
        .enumerate()
        .all(|(index, entry)| schreyer_entry_is_reduced(index, entry, basis, arena))
    {
        return Err(GroebnerError::VerificationFailed);
    }
    verify_schreyer_generator_sweep(field, generators, basis, order, options, arena)?;
    let mut pairs = Vec::new();
    for right in 1..basis.len() {
        let right_leading = basis[right]
            .leading_term()
            .ok_or(GroebnerError::VerificationFailed)?
            .0;
        for (left, left_entry) in basis.iter().enumerate().take(right) {
            let left_leading = left_entry
                .leading_term()
                .ok_or(GroebnerError::VerificationFailed)?
                .0;
            if left_leading.component() != right_leading.component() {
                continue;
            }
            let lcm = arena
                .lcm(left_leading.monomial(), right_leading.monomial())
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            pairs.push((left, right, lcm));
        }
    }
    verify_schreyer_pair_sweep(field, basis, &pairs, order, options, arena)
}

fn schreyer_entry_is_reduced<C: Clone + Eq>(
    index: usize,
    entry: &ArenaModuleElement<C>,
    basis: &[ArenaModuleElement<C>],
    arena: &MonomialArena,
) -> bool {
    entry.support.iter().copied().all(|term| {
        !basis.iter().enumerate().any(|(other, candidate)| {
            other != index
                && candidate
                    .leading_term()
                    .is_some_and(|(leading, _)| module_term_divides(leading, term, arena))
        })
    })
}

fn verify_schreyer_generator_sweep<F: Field>(
    field: &F,
    generators: &[ArenaModuleElement<F::Element>],
    basis: &[ArenaModuleElement<F::Element>],
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    #[cfg(feature = "parallel")]
    if generators.len() >= MIN_PARALLEL_SCHREYER_BASIS_ENTRIES && rayon::current_num_threads() > 1 {
        let workers = schreyer_verification_workers(options, arena, generators.len());
        if workers > 1 {
            let chunk_size = generators.len().div_ceil(workers);
            generators
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut worker_arena = arena.fork();
                    for generator in chunk {
                        if !reduces_to_zero(
                            field,
                            generator.clone(),
                            basis,
                            order,
                            options.max_reduction_steps,
                            &mut worker_arena,
                        )? {
                            return Err(GroebnerError::VerificationFailed);
                        }
                    }
                    Ok(())
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(());
        }
    }

    for generator in generators {
        if !reduces_to_zero(
            field,
            generator.clone(),
            basis,
            order,
            options.max_reduction_steps,
            arena,
        )? {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn verify_schreyer_pair_sweep<F: Field>(
    field: &F,
    basis: &[ArenaModuleElement<F::Element>],
    pairs: &[(usize, usize, MonomialId)],
    order: &ModuleOrder,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<usize, GroebnerError> {
    #[cfg(feature = "parallel")]
    if pairs.len() >= MIN_PARALLEL_SCHREYER_VERIFICATION_PAIRS && rayon::current_num_threads() > 1 {
        // Reductions may intern quotient and product monomials, so each
        // worker owns a fork of the frozen vocabulary. Existing numeric IDs
        // name the same exponent vectors in every fork; no worker result
        // escapes, making the parallel sweep deterministic and merge-free.
        let workers = schreyer_verification_workers(options, arena, pairs.len());
        if workers > 1 {
            let chunk_size = pairs.len().div_ceil(workers);
            pairs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut worker_arena = arena.fork();
                    verify_schreyer_pair_slice(
                        field,
                        basis,
                        chunk,
                        order,
                        options.max_reduction_steps,
                        &mut worker_arena,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(pairs.len());
        }
    }

    verify_schreyer_pair_slice(
        field,
        basis,
        pairs,
        order,
        options.max_reduction_steps,
        arena,
    )?;
    Ok(0)
}

#[cfg(feature = "parallel")]
fn schreyer_verification_workers(
    options: &GroebnerOptions,
    arena: &MonomialArena,
    tasks: usize,
) -> usize {
    let worker_bytes = arena.allocated_bytes().max(1);
    let memory_workers = (options.max_workspace_bytes / worker_bytes).max(1);
    rayon::current_num_threads().min(memory_workers).min(tasks)
}

fn verify_schreyer_pair_slice<F: Field>(
    field: &F,
    basis: &[ArenaModuleElement<F::Element>],
    pairs: &[(usize, usize, MonomialId)],
    order: &ModuleOrder,
    max_reduction_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for &(left, right, lcm) in pairs {
        let value = s_element(field, &basis[left], &basis[right], lcm, order, arena)?;
        if !reduces_to_zero(field, value, basis, order, max_reduction_steps, arena)? {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn module_term_divides(divisor: ModuleTerm, dividend: ModuleTerm, arena: &MonomialArena) -> bool {
    divisor.component() == dividend.component()
        && arena.divides(divisor.monomial(), dividend.monomial())
}

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryExtensionField16, BinaryField, PrimeField32};

    fn polynomial_over<F: Field>(
        field: &F,
        terms: impl IntoIterator<Item = (Vec<u32>, F::Element)>,
    ) -> SparsePolynomial<F::Element> {
        SparsePolynomial::new(
            2,
            MonomialOrder::GradedReverseLexicographic,
            terms.into_iter().map(|(exponents, coefficient)| {
                (Monomial::new(exponents.into_boxed_slice()), coefficient)
            }),
            field,
        )
    }

    fn polynomial(
        field: &PrimeField32,
        terms: impl IntoIterator<Item = (Vec<u32>, u32)>,
    ) -> SparsePolynomial<u32> {
        polynomial_over(field, terms)
    }

    #[test]
    fn monomial_module_lane_returns_the_monic_divisibility_antichain() {
        let field = PrimeField32::new(65_521).unwrap();
        let zero = || polynomial(&field, []);
        let two_x = || polynomial(&field, [(vec![1, 0], 2)]);
        let five_x2 = || polynomial(&field, [(vec![2, 0], 5)]);
        let minus_three_y = || polynomial(&field, [(vec![0, 1], 65_518)]);
        let generators = vec![
            SparseModuleElement::from_coordinates(vec![two_x(), zero()], &field),
            SparseModuleElement::from_coordinates(vec![five_x2(), zero()], &field),
            SparseModuleElement::from_coordinates(vec![zero(), minus_three_y()], &field),
        ];
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let output = module_groebner_basis_with_transform(
            &field,
            &generators,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap();
        let basis_only =
            module_groebner_basis(&field, &generators, &order, &GroebnerOptions::default())
                .unwrap();
        assert_eq!(basis_only.basis, output.basis);
        assert_eq!(
            basis_only.evidence,
            GroebnerEvidence::MonomialDivisibilityTheorem
        );
        assert_eq!(output.basis.len(), 2);
        assert_eq!(output.transformations.len(), 2);
        assert_eq!(
            output.evidence,
            GroebnerEvidence::MonomialDivisibilityTheorem
        );
        assert_eq!(output.metrics.rounds, 0);
        assert!(module_membership(
            &field,
            &output.basis,
            &generators,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap()
        .into_iter()
        .all(|member| member));
    }

    #[test]
    fn higher_rank_f4_returns_a_verified_change_matrix() {
        let field = PrimeField32::new(65_521).unwrap();
        let zero = || polynomial(&field, []);
        let x = || polynomial(&field, [(vec![1, 0], 1)]);
        let y = || polynomial(&field, [(vec![0, 1], 1)]);
        let x2 = || polynomial(&field, [(vec![2, 0], 1)]);
        let xy = || polynomial(&field, [(vec![1, 1], 1)]);
        let generators = vec![
            SparseModuleElement::from_coordinates(vec![x2(), y()], &field),
            SparseModuleElement::from_coordinates(vec![xy(), x()], &field),
            SparseModuleElement::from_coordinates(vec![zero(), x2()], &field),
        ];
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let output = module_groebner_basis_with_transform_prime(
            &field,
            &generators,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap();

        let basis_only =
            module_groebner_basis_prime(&field, &generators, &order, &GroebnerOptions::default())
                .unwrap();
        let generic_basis_only =
            module_groebner_basis(&field, &generators, &order, &GroebnerOptions::default())
                .unwrap();

        assert!(!output.basis.is_empty());
        assert_eq!(basis_only.basis, output.basis);
        assert_eq!(generic_basis_only.basis, output.basis);
        assert_eq!(
            basis_only.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
        assert_eq!(
            generic_basis_only.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
        assert!(basis_only.metrics.peak_matrix_columns < output.metrics.peak_matrix_columns);
        assert_eq!(output.basis.len(), output.transformations.len());
        assert!(output
            .transformations
            .iter()
            .all(|row| row.rank() == generators.len()));
        assert_eq!(
            output.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
        assert!(output.metrics.rounds > 0);
        assert_ne!(
            output.metrics.last_row_backend,
            Some(crate::RowBackend::GenericDense)
        );
    }

    #[test]
    fn rational_source_preconditioning_uses_fraction_free_bulk_rows() {
        let field = crate::rational::RationalCoefficientField;
        let blocks = 8_usize;
        let source_rank = 3 * blocks;
        let source_change = |row: usize, column: usize| {
            let Some(offset) = column.checked_sub(row) else {
                return Rational::zero();
            };
            if offset > 3 {
                return Rational::zero();
            }
            if offset == 0 {
                return Rational::one();
            }
            let numerator = ((row + 2) * (column + 3) + 5 * offset) % 11;
            let numerator = if numerator == 5 {
                if (row + column).is_multiple_of(2) {
                    1_i64
                } else {
                    -1_i64
                }
            } else {
                i64::try_from(numerator).unwrap() - 5
            };
            Rational::new(
                Integer::from(numerator),
                Integer::from([2_i64, 3, 5, 7][(2 * row + column) % 4]),
            )
            .unwrap()
        };
        let linear_form =
            |x: Rational, y: Rational| polynomial_over(&field, [(vec![1, 0], x), (vec![0, 1], y)]);
        let generators = (0..source_rank)
            .map(|column| {
                let coordinates = (0..blocks)
                    .flat_map(|block| {
                        let first = 3 * block;
                        [
                            linear_form(
                                source_change(first, column),
                                source_change(first + 1, column),
                            ),
                            linear_form(
                                source_change(first + 1, column),
                                source_change(first + 2, column),
                            ),
                        ]
                    })
                    .collect();
                SparseModuleElement::from_coordinates(coordinates, &field)
            })
            .collect::<Vec<_>>();
        let output = module_groebner_basis_with_transform_rational(
            &generators,
            &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert_eq!(output.basis.len(), output.transformations.len());
        assert!(output
            .transformations
            .iter()
            .all(|row| row.rank() == generators.len()));
        assert!(output.metrics.rational_dense_row_reductions > 0);
    }

    #[test]
    fn binary_module_f4_paths_match_the_generic_field_floor() {
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let options = GroebnerOptions::default();

        let f2 = BinaryField;
        let f2_polynomial = |terms: Vec<(Vec<u32>, u8)>| polynomial_over(&f2, terms);
        let f2_generators = vec![
            SparseModuleElement::from_coordinates(
                vec![
                    f2_polynomial(vec![(vec![2, 0], 1)]),
                    f2_polynomial(vec![(vec![0, 1], 1)]),
                ],
                &f2,
            ),
            SparseModuleElement::from_coordinates(
                vec![
                    f2_polynomial(vec![(vec![1, 1], 1)]),
                    f2_polynomial(vec![(vec![1, 0], 1)]),
                ],
                &f2,
            ),
            SparseModuleElement::from_coordinates(
                vec![f2_polynomial(vec![]), f2_polynomial(vec![(vec![2, 0], 1)])],
                &f2,
            ),
        ];
        let f2_packed =
            module_groebner_basis_with_transform_f2(&f2_generators, &order, &options).unwrap();
        let f2_generic =
            module_groebner_basis_with_transform(&f2, &f2_generators, &order, &options).unwrap();
        assert_eq!(f2_packed.basis, f2_generic.basis);
        assert_eq!(f2_packed.transformations, f2_generic.transformations);
        assert!(f2_packed.metrics.generic_row_reductions > 0);
        assert_eq!(
            module_groebner_basis_f2(&f2_generators, &order, &options)
                .unwrap()
                .basis,
            f2_packed.basis
        );
        let f2_transformations = f2_packed.transformations.clone();
        let f2_relations = module_relations_from_groebner_data_f2(
            &f2_packed.basis,
            &f2_transformations,
            &f2_generators,
            &order,
            &options,
        )
        .unwrap();
        let f2_generic_relations = module_relations_from_groebner_data(
            &f2,
            &f2_generic.basis,
            &f2_transformations,
            &f2_generators,
            &order,
            &options,
        )
        .unwrap();
        assert_eq!(f2_relations.generators, f2_generic_relations.generators);

        let extension = BinaryExtensionField16::new(0b111).unwrap();
        let extension_polynomial = |terms: Vec<(Vec<u32>, u16)>| polynomial_over(&extension, terms);
        let extension_generators = vec![
            SparseModuleElement::from_coordinates(
                vec![
                    extension_polynomial(vec![(vec![2, 0], 2)]),
                    extension_polynomial(vec![(vec![0, 1], 1)]),
                ],
                &extension,
            ),
            SparseModuleElement::from_coordinates(
                vec![
                    extension_polynomial(vec![(vec![1, 1], 3)]),
                    extension_polynomial(vec![(vec![1, 0], 2)]),
                ],
                &extension,
            ),
            SparseModuleElement::from_coordinates(
                vec![
                    extension_polynomial(vec![]),
                    extension_polynomial(vec![(vec![2, 0], 3)]),
                ],
                &extension,
            ),
        ];
        let extension_packed = module_groebner_basis_with_transform_binary_extension(
            &extension,
            &extension_generators,
            &order,
            &options,
        )
        .unwrap();
        let extension_generic = module_groebner_basis_with_transform(
            &extension,
            &extension_generators,
            &order,
            &options,
        )
        .unwrap();
        assert_eq!(extension_packed.basis, extension_generic.basis);
        assert_eq!(
            extension_packed.transformations,
            extension_generic.transformations
        );
        assert!(extension_packed.metrics.generic_row_reductions > 0);
        assert_eq!(
            module_groebner_basis_binary_extension(
                &extension,
                &extension_generators,
                &order,
                &options,
            )
            .unwrap()
            .basis,
            extension_packed.basis
        );
        let extension_transformations = extension_packed.transformations.clone();
        let extension_relations = module_relations_from_groebner_data_binary_extension(
            &extension,
            &extension_packed.basis,
            &extension_transformations,
            &extension_generators,
            &order,
            &options,
        )
        .unwrap();
        let extension_generic_relations = module_relations_from_groebner_data(
            &extension,
            &extension_generic.basis,
            &extension_transformations,
            &extension_generators,
            &order,
            &options,
        )
        .unwrap();
        assert_eq!(
            extension_relations.generators,
            extension_generic_relations.generators
        );
    }

    #[test]
    fn nested_schreyer_order_is_admitted() {
        let field = PrimeField32::new(101).unwrap();
        let x = polynomial(&field, [(vec![1, 0], 1)]);
        let y = polynomial(&field, [(vec![0, 1], 1)]);
        let zero = polynomial(&field, []);
        let generators = vec![
            SparseModuleElement::from_coordinates(vec![x.clone(), y.clone()], &field),
            SparseModuleElement::from_coordinates(vec![y, zero], &field),
        ];
        let base = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let order = SparseModuleOrder::schreyer(
            base,
            vec![
                SparseModuleTerm::new(Monomial::new(vec![1, 0].into_boxed_slice()), 0),
                SparseModuleTerm::new(Monomial::new(vec![0, 1].into_boxed_slice()), 0),
            ],
        );
        let output = module_groebner_basis_with_transform(
            &field,
            &generators,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert!(!output.basis.is_empty());
    }

    #[test]
    fn seeded_small_module_families_pass_the_independent_sweep() {
        let field = PrimeField32::new(101).unwrap();
        let mut state = 0x4D4F_4455_4C45_4634_u64;
        let mut next = || {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            state
        };
        for case in 0..12 {
            let rank = 2 + case % 2;
            let generators = (0..(2 + case % 3))
                .map(|_| {
                    let coordinates = (0..rank)
                        .map(|_| {
                            let term_count = (next() % 4) as usize;
                            polynomial(
                                &field,
                                (0..term_count).map(|_| {
                                    (
                                        vec![(next() % 3) as u32, (next() % 3) as u32],
                                        1 + (next() % 100) as u32,
                                    )
                                }),
                            )
                        })
                        .collect();
                    SparseModuleElement::from_coordinates(coordinates, &field)
                })
                .collect::<Vec<_>>();
            let order = if case % 2 == 0 {
                SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic)
            } else {
                SparseModuleOrder::PositionOverTerm(MonomialOrder::GradedReverseLexicographic)
            };
            let output = module_groebner_basis_with_transform_prime(
                &field,
                &generators,
                &order,
                &GroebnerOptions::default(),
            )
            .unwrap();
            let basis_only = module_groebner_basis_prime(
                &field,
                &generators,
                &order,
                &GroebnerOptions::default(),
            )
            .unwrap();
            assert_eq!(basis_only.basis, output.basis);
            assert_eq!(basis_only.evidence, output.evidence);
            assert_eq!(output.basis.len(), output.transformations.len());
            if !output.basis.is_empty() {
                let pruned = module_schreyer_syzygies_with_reducer(
                    &field,
                    &output.basis,
                    &order,
                    &GroebnerOptions::default(),
                    None,
                    SchreyerPairPolicy::CHECKED,
                    &PrimeReducer,
                )
                .unwrap();
                let complete = module_schreyer_syzygies_with_reducer(
                    &field,
                    &output.basis,
                    &order,
                    &GroebnerOptions::default(),
                    None,
                    SchreyerPairPolicy::UNPRUNED,
                    &PrimeReducer,
                )
                .unwrap();
                assert_eq!(pruned.generators, complete.generators);
            }
        }
    }

    #[test]
    fn degree_cells_return_verified_schreyer_relations() {
        let field = PrimeField32::new(101).unwrap();
        let x2 = Monomial::new(vec![2, 0].into_boxed_slice());
        let xy = Monomial::new(vec![1, 1].into_boxed_slice());
        let y2 = Monomial::new(vec![0, 2].into_boxed_slice());
        let basis = vec![
            SparseModuleElement::from_coordinates(
                vec![polynomial(&field, [(vec![2, 0], 1)])],
                &field,
            ),
            SparseModuleElement::from_coordinates(
                vec![polynomial(&field, [(vec![1, 1], 1)])],
                &field,
            ),
            SparseModuleElement::from_coordinates(
                vec![polynomial(&field, [(vec![0, 2], 1)])],
                &field,
            ),
        ];
        let base_order =
            SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let output = module_schreyer_syzygies_prime(
            &field,
            &basis,
            &base_order,
            &GroebnerOptions::default(),
        )
        .unwrap();

        assert_eq!(output.evidence, GroebnerEvidence::MonomialSchreyerTheorem);
        assert_eq!(output.metrics.pairs_generated, 3);
        assert_eq!(output.metrics.schreyer_colon_pairs, 1);
        assert_eq!(output.metrics.pairs_selected, 2);
        assert_eq!(output.metrics.rounds, 1);
        assert_eq!(output.metrics.matrix_rows, 0);
        assert_eq!(output.metrics.matrix_columns, 0);
        assert!(!output.generators.is_empty());
        assert!(output
            .generators
            .iter()
            .all(|relation| relation.rank() == basis.len()));

        let guided = module_schreyer_syzygies_prime_with_hilbert_target(
            &field,
            &basis,
            &base_order,
            &GroebnerOptions::default(),
            &StandardModuleHilbertTarget::new(
                vec![2, 2, 2],
                vec![
                    Integer::zero(),
                    Integer::zero(),
                    Integer::from(3),
                    Integer::from(-2),
                ],
            ),
        )
        .unwrap();
        assert_eq!(guided.generators, output.generators);
        assert_eq!(guided.metrics.rounds, 1);
        assert_eq!(guided.metrics.pairs_selected, 2);
        assert_eq!(guided.metrics.hilbert_target_checks, 0);
        assert_eq!(guided.metrics.hilbert_pruned_pairs, 0);

        let schreyer_order = SparseModuleOrder::schreyer(
            base_order,
            vec![
                SparseModuleTerm::new(x2, 0),
                SparseModuleTerm::new(xy, 0),
                SparseModuleTerm::new(y2, 0),
            ],
        );
        let recomputed = module_groebner_basis_with_transform_prime(
            &field,
            &output.generators,
            &schreyer_order,
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert_eq!(recomputed.basis, output.generators);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn large_schreyer_sweep_is_deterministic_across_worker_counts() {
        let field = PrimeField32::new(65_521).unwrap();
        let variables = 16;
        let basis = (0..variables)
            .map(|variable| {
                let mut exponent = vec![0_u32; variables];
                exponent[variable] = 1;
                SparseModuleElement::from_coordinates(
                    vec![SparsePolynomial::new(
                        variables,
                        MonomialOrder::GradedReverseLexicographic,
                        [(Monomial::new(exponent.into_boxed_slice()), 1_u32)],
                        &field,
                    )],
                    &field,
                )
            })
            .collect::<Vec<_>>();
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    module_schreyer_syzygies_with_reducer(
                        &field,
                        &basis,
                        &order,
                        &GroebnerOptions::default(),
                        None,
                        SchreyerPairPolicy::FORCE_F4,
                        &PrimeReducer,
                    )
                    .unwrap()
                })
        };

        let serial = run(1);
        let parallel = run(4);
        assert_eq!(parallel.generators, serial.generators);
        assert_eq!(parallel.evidence, serial.evidence);
        assert_eq!(serial.metrics.schreyer_parallel_verification_pairs, 0);
        assert_eq!(
            parallel.metrics.schreyer_parallel_verification_pairs,
            variables * (variables - 1) * (variables - 2) / 6
        );
        let mut parallel_metrics = parallel.metrics;
        parallel_metrics.schreyer_parallel_verification_pairs = 0;
        assert_eq!(parallel_metrics, serial.metrics);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn independent_degree_cells_merge_deterministically() {
        let field = PrimeField32::new(65_521).unwrap();
        let variables = 24;
        let degrees = (0..variables)
            .map(|variable| 1 + variable % 6)
            .collect::<Vec<_>>();
        let basis = degrees
            .iter()
            .enumerate()
            .map(|(variable, degree)| {
                let mut exponent = vec![0_u32; variables];
                exponent[variable] = *degree as u32;
                SparseModuleElement::from_coordinates(
                    vec![SparsePolynomial::new(
                        variables,
                        MonomialOrder::GradedReverseLexicographic,
                        [(Monomial::new(exponent.into_boxed_slice()), 1_u32)],
                        &field,
                    )],
                    &field,
                )
            })
            .collect::<Vec<_>>();
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    module_schreyer_syzygies_with_reducer(
                        &field,
                        &basis,
                        &order,
                        &GroebnerOptions::default(),
                        None,
                        SchreyerPairPolicy::FORCE_F4,
                        &PrimeReducer,
                    )
                    .unwrap()
                })
        };

        let serial = run(1);
        let parallel = run(4);
        assert_eq!(parallel.generators, serial.generators);
        assert_eq!(parallel.evidence, serial.evidence);
        assert_eq!(serial.metrics.schreyer_parallel_cells, 0);
        assert_eq!(
            parallel.metrics.schreyer_parallel_cells,
            serial.metrics.rounds
        );
        assert!(parallel.metrics.schreyer_parallel_cells > 1);
        assert_eq!(parallel.metrics.rounds, serial.metrics.rounds);
        assert_eq!(
            parallel.metrics.pairs_generated,
            serial.metrics.pairs_generated
        );
        assert_eq!(
            parallel.metrics.pairs_selected,
            serial.metrics.pairs_selected
        );
        assert_eq!(parallel.metrics.matrix_rows, serial.metrics.matrix_rows);
        assert_eq!(
            parallel.metrics.matrix_columns,
            serial.metrics.matrix_columns
        );
        assert_eq!(
            parallel.metrics.matrix_nonzeros,
            serial.metrics.matrix_nonzeros
        );
        assert_eq!(
            parallel.metrics.monomials_interned,
            serial.metrics.monomials_interned
        );

        let generic = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                module_schreyer_syzygies_with_reducer(
                    &field,
                    &basis,
                    &order,
                    &GroebnerOptions::default(),
                    None,
                    SchreyerPairPolicy::FORCE_F4,
                    &GenericReducer,
                )
                .unwrap()
            });
        assert_eq!(generic.generators, serial.generators);
        assert!(generic.metrics.schreyer_parallel_cells > 1);

        let leading_terms = basis
            .iter()
            .map(|generator| {
                let (term, _) = generator.terms().first().unwrap();
                (term.monomial().clone(), 0)
            })
            .collect::<Vec<_>>();
        let target = StandardModuleHilbertTarget::from_submodule_leading_terms(
            variables,
            &[0],
            degrees,
            &leading_terms,
        )
        .unwrap();
        let guided = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                module_schreyer_syzygies_with_reducer(
                    &field,
                    &basis,
                    &order,
                    &GroebnerOptions::default(),
                    Some(&target),
                    SchreyerPairPolicy::FORCE_F4,
                    &PrimeReducer,
                )
                .unwrap()
            });
        assert_eq!(guided.generators, serial.generators);
        assert_eq!(guided.metrics.schreyer_parallel_cells, 0);
        assert_eq!(guided.metrics.rounds, serial.metrics.rounds);

        let bounded_options = GroebnerOptions {
            max_workspace_bytes: serial.metrics.peak_workspace_bytes,
            ..GroebnerOptions::default()
        };
        let bounded = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                module_schreyer_syzygies_with_reducer(
                    &field,
                    &basis,
                    &order,
                    &bounded_options,
                    None,
                    SchreyerPairPolicy::FORCE_F4,
                    &PrimeReducer,
                )
                .unwrap()
            });
        assert_eq!(bounded.generators, serial.generators);
        assert_eq!(bounded.metrics.schreyer_parallel_cells, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn degree_cell_scheduler_balances_work_with_stable_ties() {
        let cells = vec![
            (2, vec![(); 9]),
            (3, vec![(); 8]),
            (4, vec![(); 7]),
            (5, vec![(); 6]),
            (6, vec![(); 5]),
        ];
        assert_eq!(
            schreyer_cell_assignments(&cells, 2),
            vec![vec![0, 3, 4], vec![1, 2]]
        );
    }

    #[test]
    fn packed_membership_handles_members_nonmembers_and_zero_together() {
        let field = PrimeField32::new(101).unwrap();
        let x2 = polynomial(&field, [(vec![2, 0], 1)]);
        let xy = polynomial(&field, [(vec![1, 1], 1)]);
        let zero = polynomial(&field, []);
        let basis = vec![
            SparseModuleElement::from_coordinates(vec![x2.clone(), zero.clone()], &field),
            SparseModuleElement::from_coordinates(vec![zero.clone(), xy.clone()], &field),
        ];
        let values = vec![
            SparseModuleElement::from_coordinates(
                vec![polynomial(&field, [(vec![3, 0], 1)]), zero.clone()],
                &field,
            ),
            SparseModuleElement::from_coordinates(
                vec![polynomial(&field, [(vec![1, 0], 1)]), zero.clone()],
                &field,
            ),
            SparseModuleElement::from_coordinates(vec![zero.clone(), zero], &field),
        ];

        assert_eq!(
            module_membership(
                &field,
                &basis,
                &values,
                &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic,),
                &GroebnerOptions::default(),
            )
            .unwrap(),
            vec![true, false, true]
        );

        let reductions = module_normal_forms(
            &field,
            &basis,
            &values,
            &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert!(reductions[0].remainder.is_zero());
        assert_eq!(
            reductions[0].quotients,
            vec![
                polynomial(&field, [(vec![1, 0], 1)]),
                polynomial(&field, [])
            ]
        );
        assert_eq!(reductions[1].remainder, values[1]);
        assert!(reductions[1]
            .quotients
            .iter()
            .all(SparsePolynomial::is_zero));
        assert!(reductions[2].remainder.is_zero());
        assert!(reductions[2]
            .quotients
            .iter()
            .all(SparsePolynomial::is_zero));
    }

    #[test]
    fn packed_original_relations_reconstruct_a_known_module_kernel() {
        let field = PrimeField32::new(101).unwrap();
        let x = polynomial(&field, [(vec![1, 0], 1)]);
        let y = polynomial(&field, [(vec![0, 1], 1)]);
        let zero = polynomial(&field, []);
        let values = vec![
            SparseModuleElement::from_coordinates(vec![x.clone(), zero.clone()], &field),
            SparseModuleElement::from_coordinates(vec![y.clone(), x.clone()], &field),
            SparseModuleElement::from_coordinates(vec![zero.clone(), y.clone()], &field),
        ];
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let basis_output = module_groebner_basis_with_transform_prime(
            &field,
            &values,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap();
        let transformations = basis_output.transformations;
        let kernel = module_relations_from_groebner_data_prime(
            &field,
            &basis_output.basis,
            &transformations,
            &values,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert!(kernel.generators.iter().all(|relation| {
            apply_sparse_module_map(
                &field,
                &values,
                relation,
                2,
                2,
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap()
            .is_zero()
        }));

        let expected = SparseModuleElement::from_coordinates(
            vec![
                polynomial(&field, [(vec![0, 2], 1)]),
                polynomial(&field, [(vec![1, 1], 100)]),
                polynomial(&field, [(vec![2, 0], 1)]),
            ],
            &field,
        );
        let actual_basis = module_groebner_basis_with_transform_prime(
            &field,
            &kernel.generators,
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap()
        .basis;
        let expected_basis = module_groebner_basis_with_transform_prime(
            &field,
            &[expected],
            &order,
            &GroebnerOptions::default(),
        )
        .unwrap()
        .basis;
        assert_eq!(actual_basis, expected_basis);
    }
}
