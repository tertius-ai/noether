// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Coefficient-generic polynomials over compact interned monomial identifiers.

use ahash::AHashMap as HashMap;
use core::cmp::Ordering;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    CoefficientRing, MonomialArena, MonomialArenaError, MonomialArenaId, MonomialId,
    SparsePolynomial, Truncation,
};

/// Reusable scratch storage for coefficient-generic polynomial operations.
#[derive(Clone, Debug)]
pub struct PolynomialWorkspace<C> {
    products: HashMap<MonomialId, C>,
    dense_products: Vec<Option<C>>,
    dense_touched: Vec<usize>,
    shifted_support: Vec<MonomialId>,
    second_shifted_support: Vec<MonomialId>,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

impl<C> Default for PolynomialWorkspace<C> {
    fn default() -> Self {
        Self {
            products: HashMap::new(),
            dense_products: Vec::new(),
            dense_touched: Vec::new(),
            shifted_support: Vec::new(),
            second_shifted_support: Vec::new(),
            left_indices: Vec::new(),
            right_indices: Vec::new(),
        }
    }
}

impl<C> PolynomialWorkspace<C> {
    /// Construct an empty reusable workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain allocations while removing accumulated coefficients.
    pub fn clear(&mut self) {
        self.products.clear();
        for index in self.dense_touched.drain(..) {
            self.dense_products[index] = None;
        }
        self.shifted_support.clear();
        self.second_shifted_support.clear();
        self.left_indices.clear();
        self.right_indices.clear();
    }
}

/// Canonical sparse polynomial whose support is stored as monomial IDs.
///
/// Support and coefficients occupy separate contiguous arrays. The support is
/// strictly descending in the arena's monomial order and contains no repeated
/// IDs; coefficients are nonzero. Values are compatible only with the arena
/// identity recorded at construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaPolynomial<C> {
    arena: MonomialArenaId,
    support: Vec<MonomialId>,
    coefficients: Vec<C>,
}

impl<C: Clone + Eq> ArenaPolynomial<C> {
    /// Construct the zero polynomial for an arena.
    pub fn zero(arena: &MonomialArena) -> Self {
        Self {
            arena: arena.identity(),
            support: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Canonicalize arbitrary ID-bearing terms.
    pub fn new<R>(
        arena: &MonomialArena,
        terms: impl IntoIterator<Item = (MonomialId, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut canonical = HashMap::<MonomialId, C>::new();
        for (monomial, coefficient) in terms {
            assert!(arena.contains(monomial), "monomial identifier out of range");
            if ring.is_zero(&coefficient) {
                continue;
            }
            accumulate(&mut canonical, monomial, coefficient, ring);
        }
        let mut terms = canonical.into_iter().collect::<Vec<_>>();
        terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        let (support, coefficients) = terms.into_iter().unzip();
        Self {
            arena: arena.identity(),
            support,
            coefficients,
        }
    }

    /// Construct from already canonical terms, checking the invariant once.
    ///
    /// Terms must be nonzero, contain distinct IDs, and appear greatest first
    /// in the arena's order.
    pub fn from_sorted_terms<R>(
        arena: &MonomialArena,
        terms: impl IntoIterator<Item = (MonomialId, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let terms = terms.into_iter().collect::<Vec<_>>();
        for (index, (monomial, coefficient)) in terms.iter().enumerate() {
            assert!(
                arena.contains(*monomial),
                "monomial identifier out of range"
            );
            assert!(!ring.is_zero(coefficient), "canonical coefficient is zero");
            if index != 0 {
                assert_eq!(
                    arena.compare(terms[index - 1].0, *monomial),
                    Ordering::Greater,
                    "polynomial terms are not strictly descending"
                );
            }
        }
        let (support, coefficients) = terms.into_iter().unzip();
        Self {
            arena: arena.identity(),
            support,
            coefficients,
        }
    }

    /// Intern and convert one standalone sparse polynomial.
    pub fn from_sparse<R>(
        arena: &mut MonomialArena,
        polynomial: &SparsePolynomial<C>,
        ring: &R,
    ) -> Result<Self, MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        assert_eq!(
            arena.arity(),
            polynomial.arity(),
            "polynomial arity mismatch"
        );
        assert_eq!(
            arena.order(),
            polynomial.order(),
            "polynomial order mismatch"
        );
        let mut output = Self::zero(arena);
        output.support.reserve(polynomial.terms().len());
        output.coefficients.reserve(polynomial.terms().len());
        for (monomial, coefficient) in polynomial.terms() {
            if !ring.is_zero(coefficient) {
                output.support.push(arena.intern(monomial.clone())?);
                output.coefficients.push(coefficient.clone());
            }
        }
        Ok(output)
    }

    /// Materialize a standalone sparse polynomial.
    pub fn to_sparse<R>(&self, arena: &MonomialArena, ring: &R) -> SparsePolynomial<C>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        SparsePolynomial::from_sorted_terms(
            arena.arity(),
            arena.order(),
            self.iter()
                .map(|(monomial, coefficient)| (arena.get(monomial), coefficient.clone())),
            ring,
        )
    }

    /// Identity of the monomial arena required by this polynomial.
    pub fn arena_id(&self) -> MonomialArenaId {
        self.arena
    }

    /// Number of nonzero terms.
    pub fn len(&self) -> usize {
        self.support.len()
    }

    /// Bytes reserved for the structure-of-arrays monomial support.
    pub fn allocated_support_bytes(&self) -> usize {
        self.support
            .capacity()
            .saturating_mul(core::mem::size_of::<MonomialId>())
    }

    /// Bytes reserved for inline coefficient slots.
    ///
    /// Coefficients that own allocations outside their inline Rust value may
    /// retain additional context-specific memory; the generic kernel cannot
    /// inspect that storage without weakening its coefficient contract.
    pub fn allocated_coefficient_bytes(&self) -> usize {
        self.coefficients
            .capacity()
            .saturating_mul(core::mem::size_of::<C>())
    }

    /// Total bytes reserved by this polynomial's two term buffers.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_support_bytes()
            .saturating_add(self.allocated_coefficient_bytes())
    }

    /// Whether the polynomial has no nonzero terms.
    pub fn is_empty(&self) -> bool {
        self.support.is_empty()
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.is_empty()
    }

    /// Canonical monomial support, greatest first.
    pub fn support(&self) -> &[MonomialId] {
        &self.support
    }

    /// Coefficients corresponding positionally to [`Self::support`].
    pub fn coefficients(&self) -> &[C] {
        &self.coefficients
    }

    /// Iterate through canonical terms, greatest first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (MonomialId, &C)> {
        self.support.iter().copied().zip(self.coefficients.iter())
    }

    /// Consume the polynomial into canonical arena terms without cloning
    /// coefficients.
    pub fn into_terms(self) -> impl ExactSizeIterator<Item = (MonomialId, C)> {
        self.support.into_iter().zip(self.coefficients)
    }

    /// Leading term, if nonzero.
    pub fn leading_term(&self) -> Option<(MonomialId, &C)> {
        self.support.first().copied().zip(self.coefficients.first())
    }

    /// Add two polynomials.
    pub fn add<R>(&self, other: &Self, arena: &MonomialArena, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut output = Self::zero(arena);
        self.add_into(other, arena, ring, &mut output);
        output
    }

    /// Add into reusable output storage.
    pub fn add_into<R>(&self, other: &Self, arena: &MonomialArena, ring: &R, output: &mut Self)
    where
        R: CoefficientRing<Element = C>,
    {
        self.merge_into(other, arena, ring, false, output);
    }

    /// Subtract two polynomials.
    pub fn sub<R>(&self, other: &Self, arena: &MonomialArena, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut output = Self::zero(arena);
        self.sub_into(other, arena, ring, &mut output);
        output
    }

    /// Subtract into reusable output storage.
    pub fn sub_into<R>(&self, other: &Self, arena: &MonomialArena, ring: &R, output: &mut Self)
    where
        R: CoefficientRing<Element = C>,
    {
        self.merge_into(other, arena, ring, true, output);
    }

    /// Negate every coefficient.
    pub fn neg<R>(&self, arena: &MonomialArena, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut output = Self::zero(arena);
        self.scalar_map_into(
            arena,
            &mut output,
            |coefficient| ring.neg(coefficient),
            ring,
        );
        output
    }

    /// Multiply every coefficient by one scalar.
    pub fn scalar_mul<R>(&self, scalar: &C, arena: &MonomialArena, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut output = Self::zero(arena);
        self.scalar_mul_into(scalar, arena, ring, &mut output);
        output
    }

    /// Multiply every coefficient into reusable output storage.
    pub fn scalar_mul_into<R>(&self, scalar: &C, arena: &MonomialArena, ring: &R, output: &mut Self)
    where
        R: CoefficientRing<Element = C>,
    {
        self.scalar_map_into(
            arena,
            output,
            |coefficient| ring.mul(coefficient, scalar),
            ring,
        );
    }

    /// Multiply by one coefficient-bearing monomial.
    pub fn term_mul<R>(
        &self,
        monomial: MonomialId,
        scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
    ) -> Result<Self, MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        let mut output = Self::zero(arena);
        self.term_mul_into(monomial, scalar, arena, ring, &mut output)?;
        Ok(output)
    }

    /// Multiply by a monomial while cloning coefficients unchanged.
    pub fn monomial_mul(
        &self,
        monomial: MonomialId,
        arena: &mut MonomialArena,
    ) -> Result<Self, MonomialArenaError> {
        let mut output = Self::zero(arena);
        self.monomial_mul_into(monomial, arena, &mut output)?;
        Ok(output)
    }

    /// Multiply by a monomial into reusable output storage.
    pub fn monomial_mul_into(
        &self,
        monomial: MonomialId,
        arena: &mut MonomialArena,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError> {
        self.require_arena(arena);
        assert!(arena.contains(monomial), "monomial identifier out of range");
        prepare_output(output, arena.identity(), self.len());
        for (term, coefficient) in self.iter() {
            output.support.push(arena.product(term, monomial)?);
            output.coefficients.push(coefficient.clone());
        }
        Ok(())
    }

    /// Multiply by a term into reusable output storage.
    pub fn term_mul_into<R>(
        &self,
        monomial: MonomialId,
        scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        assert!(arena.contains(monomial), "monomial identifier out of range");
        prepare_output(output, arena.identity(), self.len());
        if ring.is_zero(scalar) {
            return Ok(());
        }
        for (term, coefficient) in self.iter() {
            let coefficient = ring.mul(coefficient, scalar);
            if !ring.is_zero(&coefficient) {
                output.support.push(arena.product(term, monomial)?);
                output.coefficients.push(coefficient);
            }
        }
        Ok(())
    }

    /// Compute `left_scalar * self - right_scalar * monomial * other`.
    ///
    /// The merge is performed directly into one output rather than
    /// materializing both scaled operands. Multiplication by a monomial
    /// preserves every admitted monomial order, so the shifted right support
    /// remains canonical and can be merged linearly. The workspace retains
    /// that shifted support across calls.
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_sub_term_into<R>(
        &self,
        left_scalar: &C,
        other: &Self,
        monomial: MonomialId,
        right_scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.sub_term_with_left_map(
            other,
            monomial,
            right_scalar,
            arena,
            ring,
            workspace,
            output,
            |coefficient| ring.mul(coefficient, left_scalar),
        )
    }

    /// Compute `self - right_scalar * monomial * other`.
    ///
    /// Coefficients of `self` are cloned directly rather than multiplied by
    /// a separately constructed multiplicative identity. This is the common
    /// field-reduction specialization of [`Self::scaled_sub_term_into`].
    #[allow(clippy::too_many_arguments)]
    pub fn sub_term_into<R>(
        &self,
        other: &Self,
        monomial: MonomialId,
        right_scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.sub_term_with_left_map(
            other,
            monomial,
            right_scalar,
            arena,
            ring,
            workspace,
            output,
            Clone::clone,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sub_term_with_left_map<R>(
        &self,
        other: &Self,
        monomial: MonomialId,
        right_scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
        mut map_left: impl FnMut(&C) -> C,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_compatible(other, arena);
        assert!(arena.contains(monomial), "monomial identifier out of range");
        workspace.shifted_support.clear();
        workspace.shifted_support.reserve(other.len());
        for other_monomial in &other.support {
            workspace
                .shifted_support
                .push(arena.product(*other_monomial, monomial)?);
        }

        prepare_output(output, arena.identity(), self.len() + other.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.len() && right < other.len() {
            match arena.compare(self.support[left], workspace.shifted_support[right]) {
                Ordering::Greater => {
                    let coefficient = map_left(&self.coefficients[left]);
                    if !ring.is_zero(&coefficient) {
                        output.support.push(self.support[left]);
                        output.coefficients.push(coefficient);
                    }
                    left += 1;
                }
                Ordering::Less => {
                    let coefficient = ring.neg(&ring.mul(&other.coefficients[right], right_scalar));
                    if !ring.is_zero(&coefficient) {
                        output.support.push(workspace.shifted_support[right]);
                        output.coefficients.push(coefficient);
                    }
                    right += 1;
                }
                Ordering::Equal => {
                    let left_coefficient = map_left(&self.coefficients[left]);
                    let right_coefficient =
                        ring.neg(&ring.mul(&other.coefficients[right], right_scalar));
                    let coefficient = ring.add(&left_coefficient, &right_coefficient);
                    if !ring.is_zero(&coefficient) {
                        output.support.push(self.support[left]);
                        output.coefficients.push(coefficient);
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        for index in left..self.len() {
            let coefficient = map_left(&self.coefficients[index]);
            if !ring.is_zero(&coefficient) {
                output.support.push(self.support[index]);
                output.coefficients.push(coefficient);
            }
        }
        for index in right..other.len() {
            let coefficient = ring.neg(&ring.mul(&other.coefficients[index], right_scalar));
            if !ring.is_zero(&coefficient) {
                output.support.push(workspace.shifted_support[index]);
                output.coefficients.push(coefficient);
            }
        }
        Ok(())
    }

    /// Compute `left_scalar * left_monomial * self -
    /// right_scalar * right_monomial * other` into reusable storage.
    ///
    /// Both shifted supports are formed in the workspace and merged directly
    /// into one output. This is the two-sided form needed by polynomial
    /// critical-pair construction; it avoids materializing either shifted
    /// operand.
    #[allow(clippy::too_many_arguments)]
    pub fn shifted_scaled_sub_shifted_into<R>(
        &self,
        left_monomial: MonomialId,
        left_scalar: &C,
        other: &Self,
        right_monomial: MonomialId,
        right_scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_compatible(other, arena);
        assert!(
            arena.contains(left_monomial) && arena.contains(right_monomial),
            "monomial identifier out of range"
        );
        workspace.shifted_support.clear();
        workspace.shifted_support.reserve(self.len());
        for monomial in &self.support {
            workspace
                .shifted_support
                .push(arena.product(*monomial, left_monomial)?);
        }
        workspace.second_shifted_support.clear();
        workspace.second_shifted_support.reserve(other.len());
        for monomial in &other.support {
            workspace
                .second_shifted_support
                .push(arena.product(*monomial, right_monomial)?);
        }

        prepare_output(output, arena.identity(), self.len() + other.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.len() && right < other.len() {
            match arena.compare(
                workspace.shifted_support[left],
                workspace.second_shifted_support[right],
            ) {
                Ordering::Greater => {
                    let coefficient = ring.mul(&self.coefficients[left], left_scalar);
                    if !ring.is_zero(&coefficient) {
                        output.support.push(workspace.shifted_support[left]);
                        output.coefficients.push(coefficient);
                    }
                    left += 1;
                }
                Ordering::Less => {
                    let coefficient = ring.neg(&ring.mul(&other.coefficients[right], right_scalar));
                    if !ring.is_zero(&coefficient) {
                        output.support.push(workspace.second_shifted_support[right]);
                        output.coefficients.push(coefficient);
                    }
                    right += 1;
                }
                Ordering::Equal => {
                    let left_coefficient = ring.mul(&self.coefficients[left], left_scalar);
                    let right_coefficient =
                        ring.neg(&ring.mul(&other.coefficients[right], right_scalar));
                    let coefficient = ring.add(&left_coefficient, &right_coefficient);
                    if !ring.is_zero(&coefficient) {
                        output.support.push(workspace.shifted_support[left]);
                        output.coefficients.push(coefficient);
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        for index in left..self.len() {
            let coefficient = ring.mul(&self.coefficients[index], left_scalar);
            if !ring.is_zero(&coefficient) {
                output.support.push(workspace.shifted_support[index]);
                output.coefficients.push(coefficient);
            }
        }
        for index in right..other.len() {
            let coefficient = ring.neg(&ring.mul(&other.coefficients[index], right_scalar));
            if !ring.is_zero(&coefficient) {
                output.support.push(workspace.second_shifted_support[index]);
                output.coefficients.push(coefficient);
            }
        }
        Ok(())
    }

    /// Multiply two polynomials, optionally under a strict truncation.
    pub fn multiply<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
    ) -> Result<Self, MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        let mut workspace = PolynomialWorkspace::new();
        let mut output = Self::zero(arena);
        self.multiply_into(other, arena, ring, truncation, &mut workspace, &mut output)?;
        Ok(output)
    }

    /// Multiply using the measured serial/parallel crossover.
    ///
    /// The parallel lane is admitted only for a bounded dense exponent box.
    /// Worker-private dense accumulators cross over at 65,536 coefficient
    /// products on the reference benchmark and are capped at 32 MiB in total.
    /// Sparse-output and larger-workspace shapes retain [`Self::multiply`].
    #[cfg(feature = "parallel")]
    pub fn multiply_adaptive<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
    ) -> Result<Self, MonomialArenaError>
    where
        R: CoefficientRing<Element = C> + Sync,
        C: Send + Sync,
    {
        if rayon::current_num_threads() <= 1 || self.is_empty() || other.is_empty() {
            return self.multiply(other, arena, ring, truncation);
        }
        self.require_compatible(other, arena);
        if let Some(shape) = truncation {
            assert_eq!(shape.arity(), arena.arity(), "truncation arity mismatch");
        }
        let layout = DenseProductLayout::for_operands(self, other, arena);
        if let Some(layout) = &layout {
            let products = self.len().saturating_mul(other.len());
            let rows_per_chunk = parallel_rows_per_chunk(self.len());
            let chunks = self.len().div_ceil(rows_per_chunk);
            if products >= 65_536 && private_dense_accumulators_fit::<C>(layout.size, chunks) {
                return self.multiply_dense_parallel(other, arena, ring, truncation, layout);
            }
        }

        let mut workspace = PolynomialWorkspace::new();
        let mut output = Self::zero(arena);
        if let Some(layout) = &layout {
            self.multiply_dense_into(
                other,
                arena,
                ring,
                truncation,
                layout,
                &mut workspace,
                &mut output,
            )?;
        } else {
            self.multiply_sparse_into(other, arena, ring, truncation, &mut workspace, &mut output)?;
        }
        Ok(output)
    }

    #[cfg(feature = "parallel")]
    fn multiply_dense_parallel<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
        layout: &DenseProductLayout,
    ) -> Result<Self, MonomialArenaError>
    where
        R: CoefficientRing<Element = C> + Sync,
        C: Send + Sync,
    {
        let left_indices = self
            .support
            .iter()
            .map(|monomial| layout.encode(arena, *monomial))
            .collect::<Vec<_>>();
        let right_indices = other
            .support
            .iter()
            .map(|monomial| layout.encode(arena, *monomial))
            .collect::<Vec<_>>();
        let rows_per_chunk = parallel_rows_per_chunk(self.len());
        let mut products = vec![None; layout.size];
        let mut touched = Vec::new();
        let partials = self
            .support
            .par_chunks(rows_per_chunk)
            .zip(self.coefficients.par_chunks(rows_per_chunk))
            .zip(left_indices.par_chunks(rows_per_chunk))
            .map(|((left_support, left_coefficients), encoded_left)| {
                let mut local = DensePartial::new(layout.size);
                for ((&left_monomial, left_coefficient), &left_index) in
                    left_support.iter().zip(left_coefficients).zip(encoded_left)
                {
                    for (right_position, (&right_monomial, right_coefficient)) in
                        other.support.iter().zip(&other.coefficients).enumerate()
                    {
                        if let Some(shape) = truncation {
                            if !shape.retains_product_ids(arena, left_monomial, right_monomial)? {
                                continue;
                            }
                        }
                        let coefficient = ring.mul(left_coefficient, right_coefficient);
                        if !ring.is_zero(&coefficient) {
                            local.accumulate(
                                left_index + right_indices[right_position],
                                coefficient,
                                ring,
                            );
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Result<Vec<_>, MonomialArenaError>>()?;
        for mut partial in partials {
            for index in partial.touched {
                let Some(coefficient) = partial.products[index].take() else {
                    continue;
                };
                accumulate_dense(&mut products, &mut touched, index, coefficient, ring);
            }
        }

        let mut terms = Vec::with_capacity(touched.len());
        let mut exponents = vec![0_u32; arena.arity()];
        for index in touched {
            let Some(coefficient) = products[index].take() else {
                continue;
            };
            layout.decode(index, &mut exponents);
            terms.push((arena.intern_exponents(&exponents)?, coefficient));
        }
        terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        Ok(Self::from_sorted_terms(arena, terms, ring))
    }

    /// Multiply into reusable map and output allocations.
    pub fn multiply_into<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_compatible(other, arena);
        if let Some(shape) = truncation {
            assert_eq!(shape.arity(), arena.arity(), "truncation arity mismatch");
        }
        workspace.clear();
        if let Some(layout) = DenseProductLayout::for_operands(self, other, arena) {
            return self
                .multiply_dense_into(other, arena, ring, truncation, &layout, workspace, output);
        }
        self.multiply_sparse_into(other, arena, ring, truncation, workspace, output)
    }

    /// Sum polynomial products into one reusable accumulator and output.
    ///
    /// Each product contributes directly to the same coefficient map. The
    /// result is canonicalized once, avoiding one temporary polynomial and
    /// one support sort per summand. This is the polynomial analogue of a
    /// vector dot product and is useful for polynomial matrix arithmetic and
    /// exact transformation checks.
    pub fn sum_products_into<'a, R>(
        products: impl IntoIterator<Item = (&'a Self, &'a Self)>,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        C: 'a,
        R: CoefficientRing<Element = C>,
    {
        Self::accumulate_products(products, arena, ring, workspace)?;
        prepare_output(output, arena.identity(), workspace.products.len());
        let mut terms = workspace.products.drain().collect::<Vec<_>>();
        terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        for (monomial, coefficient) in terms {
            output.support.push(monomial);
            output.coefficients.push(coefficient);
        }
        Ok(())
    }

    /// Test whether this polynomial equals a sum of polynomial products.
    ///
    /// The products are accumulated exactly as in
    /// [`Self::sum_products_into`], but equality is checked directly against
    /// the coefficient map. No output support is materialized or sorted.
    pub fn equals_sum_products<'a, R>(
        &self,
        products: impl IntoIterator<Item = (&'a Self, &'a Self)>,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
    ) -> Result<bool, MonomialArenaError>
    where
        C: 'a,
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        Self::accumulate_products(products, arena, ring, workspace)?;
        Ok(workspace.products.len() == self.len()
            && self.iter().all(|(monomial, coefficient)| {
                workspace.products.get(&monomial) == Some(coefficient)
            }))
    }

    /// Test whether this polynomial equals a scaled sum of products.
    ///
    /// Every input triple contributes `scalar * left * right` to one
    /// reusable coefficient map. This supports fraction-free dot products
    /// and polynomial matrix identities without materializing any summand.
    pub fn equals_sum_scaled_products<'a, R>(
        &self,
        products: impl IntoIterator<Item = (&'a Self, &'a Self, &'a C)>,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
    ) -> Result<bool, MonomialArenaError>
    where
        C: 'a,
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        workspace.clear();
        for (left, right, scalar) in products {
            left.require_compatible(right, arena);
            if ring.is_zero(scalar) {
                continue;
            }
            workspace
                .products
                .reserve(left.len().saturating_mul(right.len()).min(1_048_576));
            for (left_monomial, left_coefficient) in left.iter() {
                for (right_monomial, right_coefficient) in right.iter() {
                    let monomial = arena.product(left_monomial, right_monomial)?;
                    let coefficient =
                        ring.mul(&ring.mul(left_coefficient, right_coefficient), scalar);
                    if !ring.is_zero(&coefficient) {
                        accumulate(&mut workspace.products, monomial, coefficient, ring);
                    }
                }
            }
        }
        Ok(workspace.products.len() == self.len()
            && self.iter().all(|(monomial, coefficient)| {
                workspace.products.get(&monomial) == Some(coefficient)
            }))
    }

    fn accumulate_products<'a, R>(
        products: impl IntoIterator<Item = (&'a Self, &'a Self)>,
        arena: &mut MonomialArena,
        ring: &R,
        workspace: &mut PolynomialWorkspace<C>,
    ) -> Result<(), MonomialArenaError>
    where
        C: 'a,
        R: CoefficientRing<Element = C>,
    {
        workspace.clear();
        for (left, right) in products {
            left.require_compatible(right, arena);
            workspace
                .products
                .reserve(left.len().saturating_mul(right.len()).min(1_048_576));
            for (left_monomial, left_coefficient) in left.iter() {
                for (right_monomial, right_coefficient) in right.iter() {
                    let monomial = arena.product(left_monomial, right_monomial)?;
                    let coefficient = ring.mul(left_coefficient, right_coefficient);
                    if !ring.is_zero(&coefficient) {
                        accumulate(&mut workspace.products, monomial, coefficient, ring);
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn multiply_sparse_into<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        workspace
            .products
            .reserve(self.len().saturating_mul(other.len()).min(1_048_576));
        for (left_monomial, left_coefficient) in self.iter() {
            for (right_monomial, right_coefficient) in other.iter() {
                if let Some(shape) = truncation {
                    if !shape.retains_product_ids(arena, left_monomial, right_monomial)? {
                        continue;
                    }
                }
                let monomial = arena.product(left_monomial, right_monomial)?;
                let coefficient = ring.mul(left_coefficient, right_coefficient);
                if !ring.is_zero(&coefficient) {
                    accumulate(&mut workspace.products, monomial, coefficient, ring);
                }
            }
        }
        prepare_output(output, arena.identity(), workspace.products.len());
        let mut terms = workspace.products.drain().collect::<Vec<_>>();
        terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        for (monomial, coefficient) in terms {
            output.support.push(monomial);
            output.coefficients.push(coefficient);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn multiply_dense_into<R>(
        &self,
        other: &Self,
        arena: &mut MonomialArena,
        ring: &R,
        truncation: Option<&Truncation>,
        layout: &DenseProductLayout,
        workspace: &mut PolynomialWorkspace<C>,
        output: &mut Self,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        workspace.dense_products.resize_with(layout.size, || None);
        workspace.left_indices.reserve(self.len());
        workspace.right_indices.reserve(other.len());
        workspace.left_indices.extend(
            self.support
                .iter()
                .map(|monomial| layout.encode(arena, *monomial)),
        );
        workspace.right_indices.extend(
            other
                .support
                .iter()
                .map(|monomial| layout.encode(arena, *monomial)),
        );

        for (left_index, (&left_monomial, left_coefficient)) in
            self.support.iter().zip(&self.coefficients).enumerate()
        {
            for (right_index, (&right_monomial, right_coefficient)) in
                other.support.iter().zip(&other.coefficients).enumerate()
            {
                if let Some(shape) = truncation {
                    if !shape.retains_product_ids(arena, left_monomial, right_monomial)? {
                        continue;
                    }
                }
                let coefficient = ring.mul(left_coefficient, right_coefficient);
                if ring.is_zero(&coefficient) {
                    continue;
                }
                let index =
                    workspace.left_indices[left_index] + workspace.right_indices[right_index];
                if let Some(previous) = &workspace.dense_products[index] {
                    workspace.dense_products[index] = Some(ring.add(previous, &coefficient));
                } else {
                    workspace.dense_products[index] = Some(coefficient);
                    workspace.dense_touched.push(index);
                }
            }
        }

        let mut indexed_terms = Vec::with_capacity(workspace.dense_touched.len());
        let mut exponents = vec![0_u32; arena.arity()];
        for &index in &workspace.dense_touched {
            let Some(coefficient) = workspace.dense_products[index].as_ref() else {
                continue;
            };
            if ring.is_zero(coefficient) {
                continue;
            }
            layout.decode(index, &mut exponents);
            indexed_terms.push((arena.intern_exponents(&exponents)?, index));
        }
        indexed_terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        prepare_output(output, arena.identity(), indexed_terms.len());
        for (monomial, index) in indexed_terms {
            let coefficient = workspace.dense_products[index]
                .take()
                .expect("a touched dense coefficient remains present");
            output.support.push(monomial);
            output.coefficients.push(coefficient);
        }
        for index in workspace.dense_touched.drain(..) {
            workspace.dense_products[index] = None;
        }
        Ok(())
    }

    fn merge_into<R>(
        &self,
        other: &Self,
        arena: &MonomialArena,
        ring: &R,
        subtract: bool,
        output: &mut Self,
    ) where
        R: CoefficientRing<Element = C>,
    {
        self.require_compatible(other, arena);
        prepare_output(output, arena.identity(), self.len() + other.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.len() && right < other.len() {
            match arena.compare(self.support[left], other.support[right]) {
                Ordering::Greater => {
                    output.support.push(self.support[left]);
                    output.coefficients.push(self.coefficients[left].clone());
                    left += 1;
                }
                Ordering::Less => {
                    let coefficient = if subtract {
                        ring.neg(&other.coefficients[right])
                    } else {
                        other.coefficients[right].clone()
                    };
                    if !ring.is_zero(&coefficient) {
                        output.support.push(other.support[right]);
                        output.coefficients.push(coefficient);
                    }
                    right += 1;
                }
                Ordering::Equal => {
                    let right_coefficient = if subtract {
                        ring.neg(&other.coefficients[right])
                    } else {
                        other.coefficients[right].clone()
                    };
                    let coefficient = ring.add(&self.coefficients[left], &right_coefficient);
                    if !ring.is_zero(&coefficient) {
                        output.support.push(self.support[left]);
                        output.coefficients.push(coefficient);
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        output.support.extend_from_slice(&self.support[left..]);
        output
            .coefficients
            .extend_from_slice(&self.coefficients[left..]);
        for index in right..other.len() {
            let coefficient = if subtract {
                ring.neg(&other.coefficients[index])
            } else {
                other.coefficients[index].clone()
            };
            if !ring.is_zero(&coefficient) {
                output.support.push(other.support[index]);
                output.coefficients.push(coefficient);
            }
        }
    }

    fn scalar_map_into<R>(
        &self,
        arena: &MonomialArena,
        output: &mut Self,
        mut map: impl FnMut(&C) -> C,
        ring: &R,
    ) where
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        prepare_output(output, arena.identity(), self.len());
        for (monomial, coefficient) in self.iter() {
            let coefficient = map(coefficient);
            if !ring.is_zero(&coefficient) {
                output.support.push(monomial);
                output.coefficients.push(coefficient);
            }
        }
    }

    fn require_compatible(&self, other: &Self, arena: &MonomialArena) {
        self.require_arena(arena);
        other.require_arena(arena);
    }

    fn require_arena(&self, arena: &MonomialArena) {
        assert_eq!(self.arena, arena.identity(), "polynomial arena mismatch");
    }
}

struct DenseProductLayout {
    radices: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
}

#[cfg(feature = "parallel")]
struct DensePartial<C> {
    products: Vec<Option<C>>,
    touched: Vec<usize>,
}

#[cfg(feature = "parallel")]
impl<C: Clone + Eq> DensePartial<C> {
    fn new(size: usize) -> Self {
        Self {
            products: vec![None; size],
            touched: Vec::new(),
        }
    }

    fn accumulate<R: CoefficientRing<Element = C>>(
        &mut self,
        index: usize,
        coefficient: C,
        ring: &R,
    ) {
        accumulate_dense(
            &mut self.products,
            &mut self.touched,
            index,
            coefficient,
            ring,
        );
    }
}

impl DenseProductLayout {
    const MIN_PRODUCTS: usize = 256;
    const MAX_CELLS: usize = 1_048_576;
    const MAX_CELLS_PER_PRODUCT: usize = 2;

    fn for_operands<C: Clone + Eq>(
        left: &ArenaPolynomial<C>,
        right: &ArenaPolynomial<C>,
        arena: &MonomialArena,
    ) -> Option<Self> {
        let products = left.len().checked_mul(right.len())?;
        if products < Self::MIN_PRODUCTS || arena.arity() == 0 {
            return None;
        }
        let mut left_maxima = vec![0_u32; arena.arity()];
        let mut right_maxima = vec![0_u32; arena.arity()];
        for &monomial in &left.support {
            for (variable, maximum) in left_maxima.iter_mut().enumerate() {
                *maximum = (*maximum).max(arena.exponent(monomial, variable));
            }
        }
        for &monomial in &right.support {
            for (variable, maximum) in right_maxima.iter_mut().enumerate() {
                *maximum = (*maximum).max(arena.exponent(monomial, variable));
            }
        }
        let radices = left_maxima
            .into_iter()
            .zip(right_maxima)
            .map(|(left, right)| {
                left.checked_add(right)
                    .and_then(|maximum| maximum.checked_add(1))
                    .and_then(|radix| usize::try_from(radix).ok())
            })
            .collect::<Option<Vec<_>>>()?;
        let mut strides = Vec::with_capacity(radices.len());
        let mut size = 1_usize;
        for &radix in &radices {
            strides.push(size);
            size = size.checked_mul(radix)?;
            if size > Self::MAX_CELLS || size > products.saturating_mul(Self::MAX_CELLS_PER_PRODUCT)
            {
                return None;
            }
        }
        Some(Self {
            radices,
            strides,
            size,
        })
    }

    fn encode(&self, arena: &MonomialArena, monomial: MonomialId) -> usize {
        self.strides
            .iter()
            .enumerate()
            .map(|(variable, stride)| {
                usize::try_from(arena.exponent(monomial, variable))
                    .expect("word exponent fits usize in an admitted dense layout")
                    * stride
            })
            .sum()
    }

    fn decode(&self, index: usize, output: &mut [u32]) {
        for (variable, exponent) in output.iter_mut().enumerate() {
            *exponent = u32::try_from((index / self.strides[variable]) % self.radices[variable])
                .expect("admitted dense radix fits a word exponent");
        }
    }
}

fn prepare_output<C>(output: &mut ArenaPolynomial<C>, arena: MonomialArenaId, capacity: usize) {
    output.arena = arena;
    output.support.clear();
    output.coefficients.clear();
    output.support.reserve(capacity);
    output.coefficients.reserve(capacity);
}

fn accumulate<C: Clone + Eq, R: CoefficientRing<Element = C>>(
    values: &mut HashMap<MonomialId, C>,
    monomial: MonomialId,
    coefficient: C,
    ring: &R,
) {
    use std::collections::hash_map::Entry;

    match values.entry(monomial) {
        Entry::Vacant(entry) => {
            entry.insert(coefficient);
        }
        Entry::Occupied(mut entry) => {
            let combined = ring.add(entry.get(), &coefficient);
            if ring.is_zero(&combined) {
                entry.remove();
            } else {
                *entry.get_mut() = combined;
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn parallel_rows_per_chunk(row_count: usize) -> usize {
    row_count.div_ceil(rayon::current_num_threads()).max(1)
}

#[cfg(feature = "parallel")]
fn private_dense_accumulators_fit<C>(cells: usize, chunks: usize) -> bool {
    const MAX_PRIVATE_DENSE_BYTES: usize = 32 * 1024 * 1024;
    cells
        .checked_mul(chunks)
        .and_then(|entries| {
            entries.checked_mul(
                core::mem::size_of::<Option<C>>().saturating_add(core::mem::size_of::<usize>()),
            )
        })
        .is_some_and(|bytes| bytes <= MAX_PRIVATE_DENSE_BYTES)
}

#[cfg(feature = "parallel")]
fn accumulate_dense<C: Clone + Eq, R: CoefficientRing<Element = C>>(
    products: &mut [Option<C>],
    touched: &mut Vec<usize>,
    index: usize,
    coefficient: C,
    ring: &R,
) {
    if let Some(previous) = &products[index] {
        let combined = ring.add(previous, &coefficient);
        products[index] = (!ring.is_zero(&combined)).then_some(combined);
    } else {
        products[index] = Some(coefficient);
        touched.push(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Monomial, MonomialOrder};

    struct Mod7;

    impl CoefficientRing for Mod7 {
        type Element = u32;

        fn zero(&self) -> Self::Element {
            0
        }

        fn is_zero(&self, value: &Self::Element) -> bool {
            *value == 0
        }

        fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
            (left + right) % 7
        }

        fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
            (left * right) % 7
        }

        fn neg(&self, value: &Self::Element) -> Self::Element {
            (7 - value) % 7
        }
    }

    #[test]
    fn arithmetic_matches_the_standalone_floor() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let left = SparsePolynomial::new(
            2,
            order,
            [
                (Monomial::new(vec![2, 0].into_boxed_slice()), 2),
                (Monomial::new(vec![0, 1].into_boxed_slice()), 4),
            ],
            &ring,
        );
        let right = SparsePolynomial::new(
            2,
            order,
            [
                (Monomial::new(vec![1, 1].into_boxed_slice()), 3),
                (Monomial::new(vec![0, 1].into_boxed_slice()), 3),
            ],
            &ring,
        );
        let truncation = Truncation::TotalDegree { arity: 2, bound: 4 };
        let mut arena = MonomialArena::new(2, order);
        let compact_left = ArenaPolynomial::from_sparse(&mut arena, &left, &ring).unwrap();
        let compact_right = ArenaPolynomial::from_sparse(&mut arena, &right, &ring).unwrap();

        assert_eq!(
            compact_left
                .add(&compact_right, &arena, &ring)
                .to_sparse(&arena, &ring),
            left.add(&right, &ring)
        );
        assert_eq!(
            compact_left
                .sub(&compact_right, &arena, &ring)
                .to_sparse(&arena, &ring),
            left.sub(&right, &ring)
        );
        let shift = Monomial::new(vec![1, 0].into_boxed_slice());
        let shift_id = arena.intern(shift.clone()).unwrap();
        let mut workspace = PolynomialWorkspace::new();
        let mut fused = ArenaPolynomial::zero(&arena);
        compact_left
            .scaled_sub_term_into(
                &3,
                &compact_right,
                shift_id,
                &5,
                &mut arena,
                &ring,
                &mut workspace,
                &mut fused,
            )
            .unwrap();
        assert_eq!(
            fused.to_sparse(&arena, &ring),
            left.scalar_mul(&3, &ring)
                .sub(&right.term_mul(&shift, &5, &ring), &ring)
        );
        compact_left
            .sub_term_into(
                &compact_right,
                shift_id,
                &5,
                &mut arena,
                &ring,
                &mut workspace,
                &mut fused,
            )
            .unwrap();
        assert_eq!(
            fused.to_sparse(&arena, &ring),
            left.sub(&right.term_mul(&shift, &5, &ring), &ring)
        );
        let left_shift = Monomial::new(vec![0, 1].into_boxed_slice());
        let left_shift_id = arena.intern(left_shift.clone()).unwrap();
        compact_left
            .shifted_scaled_sub_shifted_into(
                left_shift_id,
                &3,
                &compact_right,
                shift_id,
                &5,
                &mut arena,
                &ring,
                &mut workspace,
                &mut fused,
            )
            .unwrap();
        assert_eq!(
            fused.to_sparse(&arena, &ring),
            left.term_mul(&left_shift, &3, &ring)
                .sub(&right.term_mul(&shift, &5, &ring), &ring)
        );
        ArenaPolynomial::sum_products_into(
            [
                (&compact_left, &compact_right),
                (&compact_right, &compact_right),
            ],
            &mut arena,
            &ring,
            &mut workspace,
            &mut fused,
        )
        .unwrap();
        assert_eq!(
            fused.to_sparse(&arena, &ring),
            left.multiply(&right, &ring, None)
                .add(&right.multiply(&right, &ring, None), &ring)
        );
        assert!(fused
            .equals_sum_products(
                [
                    (&compact_left, &compact_right),
                    (&compact_right, &compact_right),
                ],
                &mut arena,
                &ring,
                &mut workspace,
            )
            .unwrap());
        assert!(!compact_left
            .equals_sum_products(
                [(&compact_left, &compact_right)],
                &mut arena,
                &ring,
                &mut workspace,
            )
            .unwrap());
        let scale_three = 3;
        let scale_five = 5;
        let scaled_expected = left
            .multiply(&right, &ring, None)
            .scalar_mul(&scale_three, &ring)
            .add(
                &right
                    .multiply(&right, &ring, None)
                    .scalar_mul(&scale_five, &ring),
                &ring,
            );
        let compact_scaled_expected =
            ArenaPolynomial::from_sparse(&mut arena, &scaled_expected, &ring).unwrap();
        assert!(compact_scaled_expected
            .equals_sum_scaled_products(
                [
                    (&compact_left, &compact_right, &scale_three),
                    (&compact_right, &compact_right, &scale_five),
                ],
                &mut arena,
                &ring,
                &mut workspace,
            )
            .unwrap());
        assert_eq!(
            compact_left
                .multiply(&compact_right, &mut arena, &ring, Some(&truncation))
                .unwrap()
                .to_sparse(&arena, &ring),
            left.multiply(&right, &ring, Some(&truncation))
        );
    }

    #[test]
    fn multiplication_workspace_and_output_are_reusable() {
        let ring = Mod7;
        let mut arena = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let x = arena.intern_exponents(&[1]).unwrap();
        let one = arena.intern_exponents(&[0]).unwrap();
        let polynomial = ArenaPolynomial::new(&arena, [(x, 1), (one, 1)], &ring);
        let mut workspace = PolynomialWorkspace::new();
        let mut output = ArenaPolynomial::zero(&arena);
        polynomial
            .multiply_into(
                &polynomial,
                &mut arena,
                &ring,
                None,
                &mut workspace,
                &mut output,
            )
            .unwrap();
        assert_eq!(output.coefficients(), &[1, 2, 1]);
    }

    #[test]
    fn allocated_bytes_split_support_and_coefficients() {
        let ring = Mod7;
        let mut arena = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let x = arena.intern_exponents(&[1]).unwrap();
        let one = arena.intern_exponents(&[0]).unwrap();
        let polynomial = ArenaPolynomial::new(&arena, [(x, 3_u32), (one, 2_u32)], &ring);

        assert!(polynomial.allocated_support_bytes() >= 2 * core::mem::size_of::<MonomialId>());
        assert!(polynomial.allocated_coefficient_bytes() >= 2 * core::mem::size_of::<u32>());
        assert_eq!(
            polynomial.allocated_bytes(),
            polynomial
                .allocated_support_bytes()
                .saturating_add(polynomial.allocated_coefficient_bytes())
        );
    }

    #[test]
    fn dense_kronecker_lane_matches_sparse_truncated_arithmetic() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let left = SparsePolynomial::new(
            2,
            order,
            (0..4).flat_map(|x| {
                (0..4).map(move |y| {
                    (
                        Monomial::new(vec![x, y].into_boxed_slice()),
                        (x + 2 * y) % 6 + 1,
                    )
                })
            }),
            &ring,
        );
        let right = SparsePolynomial::new(
            2,
            order,
            (0..4).flat_map(|x| {
                (0..4).map(move |y| {
                    (
                        Monomial::new(vec![x, y].into_boxed_slice()),
                        (3 * x + y) % 6 + 1,
                    )
                })
            }),
            &ring,
        );
        let truncation = Truncation::TotalDegree { arity: 2, bound: 6 };
        let mut arena = MonomialArena::new(2, order);
        let packed_left = ArenaPolynomial::from_sparse(&mut arena, &left, &ring).unwrap();
        let packed_right = ArenaPolynomial::from_sparse(&mut arena, &right, &ring).unwrap();

        let packed = packed_left
            .multiply(&packed_right, &mut arena, &ring, Some(&truncation))
            .unwrap()
            .to_sparse(&arena, &ring);
        assert_eq!(packed, left.multiply(&right, &ring, Some(&truncation)));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn adaptive_dense_multiplication_matches_serial_floor() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let polynomial = SparsePolynomial::new(
            4,
            order,
            (0..256).map(|index| {
                (
                    Monomial::new(
                        vec![index % 4, (index / 4) % 4, (index / 16) % 4, index / 64]
                            .into_boxed_slice(),
                    ),
                    index % 6 + 1,
                )
            }),
            &ring,
        );
        let truncation = Truncation::TotalDegree {
            arity: 4,
            bound: 10,
        };
        let mut arena = MonomialArena::new(4, order);
        let packed = ArenaPolynomial::from_sparse(&mut arena, &polynomial, &ring).unwrap();
        assert!(DenseProductLayout::for_operands(&packed, &packed, &arena).is_some());
        let serial = packed
            .multiply(&packed, &mut arena, &ring, Some(&truncation))
            .unwrap()
            .to_sparse(&arena, &ring);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| packed.multiply_adaptive(&packed, &mut arena, &ring, Some(&truncation)))
            .unwrap()
            .to_sparse(&arena, &ring);
        assert_eq!(parallel, serial);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn adaptive_sparse_multiplication_matches_serial_floor() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let polynomial = SparsePolynomial::new(
            8,
            order,
            (0..64_u32).map(|index| {
                let mut exponents = vec![0; 8];
                let index_usize = usize::try_from(index).unwrap();
                exponents[index_usize % 8] = 10_000 + index;
                exponents[(index_usize + 3) % 8] = index * 17;
                (Monomial::new(exponents.into_boxed_slice()), index % 6 + 1)
            }),
            &ring,
        );
        let mut arena = MonomialArena::new(8, order);
        let packed = ArenaPolynomial::from_sparse(&mut arena, &polynomial, &ring).unwrap();
        assert!(DenseProductLayout::for_operands(&packed, &packed, &arena).is_none());
        let serial = packed
            .multiply(&packed, &mut arena, &ring, None)
            .unwrap()
            .to_sparse(&arena, &ring);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| packed.multiply_adaptive(&packed, &mut arena, &ring, None))
            .unwrap()
            .to_sparse(&arena, &ring);
        assert_eq!(parallel, serial);
    }
}
