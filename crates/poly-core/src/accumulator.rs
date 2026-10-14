// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Reusable heap-backed sparse polynomial accumulation.

use ahash::AHashMap as HashMap;

use crate::{
    ArenaPolynomial, CoefficientRing, MonomialArena, MonomialArenaError, MonomialArenaId,
    MonomialId,
};

/// Mutable sparse polynomial accumulator with ordered leading-term access.
///
/// Coefficients live in a hash table while a monomial-order heap provides
/// leading terms. Cancelled heap entries are removed lazily. This avoids
/// rebuilding whole canonical polynomials during repeated reductions and is
/// generic over every coefficient adapter implementing [`CoefficientRing`].
#[derive(Clone, Debug)]
pub struct PolynomialAccumulator<C> {
    arena: MonomialArenaId,
    coefficients: HashMap<MonomialId, C>,
    heap: Vec<MonomialId>,
}

impl<C: Clone + Eq> PolynomialAccumulator<C> {
    /// Construct an empty accumulator for `arena`.
    pub fn new(arena: &MonomialArena) -> Self {
        Self {
            arena: arena.identity(),
            coefficients: HashMap::new(),
            heap: Vec::new(),
        }
    }

    /// Construct from one canonical polynomial.
    pub fn from_polynomial(polynomial: &ArenaPolynomial<C>, arena: &MonomialArena) -> Self {
        assert_eq!(
            polynomial.arena_id(),
            arena.identity(),
            "polynomial arena mismatch"
        );
        let mut output = Self::new(arena);
        output.coefficients.reserve(polynomial.len());
        output.heap.reserve(polynomial.len());
        for (monomial, coefficient) in polynomial.iter() {
            output.coefficients.insert(monomial, coefficient.clone());
            heap_push(&mut output.heap, monomial, arena);
        }
        output
    }

    /// Number of currently nonzero coefficients.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Whether every coefficient is zero.
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Remove all terms while retaining allocations.
    pub fn clear(&mut self, arena: &MonomialArena) {
        self.require_arena(arena);
        self.coefficients.clear();
        self.heap.clear();
    }

    /// Add one coefficient to a monomial, combining or cancelling in place.
    pub fn add_term<R>(
        &mut self,
        monomial: MonomialId,
        coefficient: C,
        arena: &MonomialArena,
        ring: &R,
    ) where
        R: CoefficientRing<Element = C>,
    {
        use std::collections::hash_map::Entry;

        self.require_arena(arena);
        assert!(arena.contains(monomial), "monomial identifier out of range");
        if ring.is_zero(&coefficient) {
            return;
        }
        match self.coefficients.entry(monomial) {
            Entry::Vacant(entry) => {
                entry.insert(coefficient);
                heap_push(&mut self.heap, monomial, arena);
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

    /// Current leading term, removing stale cancelled heap entries.
    pub fn leading_term(&mut self, arena: &MonomialArena) -> Option<(MonomialId, C)> {
        self.require_arena(arena);
        loop {
            let monomial = *self.heap.first()?;
            if let Some(coefficient) = self.coefficients.get(&monomial) {
                return Some((monomial, coefficient.clone()));
            }
            heap_pop(&mut self.heap, arena);
        }
    }

    /// Remove and return the current leading term.
    pub fn pop_leading(&mut self, arena: &MonomialArena) -> Option<(MonomialId, C)> {
        let (monomial, coefficient) = self.leading_term(arena)?;
        self.coefficients.remove(&monomial);
        heap_pop(&mut self.heap, arena);
        Some((monomial, coefficient))
    }

    /// Subtract `scalar * monomial * polynomial` in place.
    pub fn subtract_multiple<R>(
        &mut self,
        polynomial: &ArenaPolynomial<C>,
        monomial: MonomialId,
        scalar: &C,
        arena: &mut MonomialArena,
        ring: &R,
    ) -> Result<(), MonomialArenaError>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        assert_eq!(
            polynomial.arena_id(),
            arena.identity(),
            "polynomial arena mismatch"
        );
        if ring.is_zero(scalar) {
            return Ok(());
        }
        for (term, coefficient) in polynomial.iter() {
            let product = arena.product(term, monomial)?;
            let coefficient = ring.neg(&ring.mul(coefficient, scalar));
            self.add_term(product, coefficient, arena, ring);
        }
        Ok(())
    }

    /// Drain into a canonical polynomial while retaining allocated storage.
    pub fn drain<R>(&mut self, arena: &MonomialArena, ring: &R) -> ArenaPolynomial<C>
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_arena(arena);
        let mut terms = self.coefficients.drain().collect::<Vec<_>>();
        self.heap.clear();
        terms.sort_unstable_by(|(left, _), (right, _)| arena.compare(*right, *left));
        ArenaPolynomial::from_sorted_terms(arena, terms, ring)
    }

    fn require_arena(&self, arena: &MonomialArena) {
        assert_eq!(self.arena, arena.identity(), "polynomial arena mismatch");
    }
}

fn heap_push(heap: &mut Vec<MonomialId>, monomial: MonomialId, arena: &MonomialArena) {
    heap.push(monomial);
    let mut child = heap.len() - 1;
    while child != 0 {
        let parent = (child - 1) / 2;
        if arena.compare(heap[parent], heap[child]).is_ge() {
            break;
        }
        heap.swap(parent, child);
        child = parent;
    }
}

fn heap_pop(heap: &mut Vec<MonomialId>, arena: &MonomialArena) -> Option<MonomialId> {
    let removed = *heap.first()?;
    let last = heap.pop().expect("nonempty heap");
    if heap.is_empty() {
        return Some(removed);
    }
    heap[0] = last;
    let mut parent = 0;
    loop {
        let left = parent * 2 + 1;
        if left >= heap.len() {
            break;
        }
        let right = left + 1;
        let child = if right < heap.len() && arena.compare(heap[right], heap[left]).is_gt() {
            right
        } else {
            left
        };
        if arena.compare(heap[parent], heap[child]).is_ge() {
            break;
        }
        heap.swap(parent, child);
        parent = child;
    }
    Some(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MonomialOrder, SparsePolynomial};

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
    fn cancellation_and_multiple_subtraction_preserve_order() {
        let ring = Mod7;
        let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let x2 = arena.intern_exponents(&[2, 0]).unwrap();
        let y = arena.intern_exponents(&[0, 1]).unwrap();
        let x = arena.intern_exponents(&[1, 0]).unwrap();
        let reducer = ArenaPolynomial::new(&arena, [(x, 1), (y, 1)], &ring);
        let value = ArenaPolynomial::new(&arena, [(x2, 3), (y, 2)], &ring);
        let mut accumulator = PolynomialAccumulator::from_polynomial(&value, &arena);
        accumulator
            .subtract_multiple(&reducer, x, &3, &mut arena, &ring)
            .unwrap();
        let result = accumulator.drain(&arena, &ring).to_sparse(&arena, &ring);
        let xy = arena.product(x, y).unwrap();
        let expected = SparsePolynomial::new(
            2,
            MonomialOrder::GradedReverseLexicographic,
            [(arena.get(xy), 4), (arena.get(y), 2)],
            &ring,
        );
        assert_eq!(result, expected);
    }
}
