// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generic validated sparse polynomial arithmetic.

use ahash::AHashMap as HashMap;

use crate::{Monomial, MonomialOrder, Truncation};

/// Narrow validated coefficient contract for kernel arithmetic.
pub trait CoefficientRing {
    /// Coefficient value.
    type Element: Clone + Eq + Send + Sync;

    /// Additive identity.
    fn zero(&self) -> Self::Element;
    /// Whether a coefficient is zero.
    fn is_zero(&self, value: &Self::Element) -> bool;
    /// Addition.
    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element;
    /// Multiplication.
    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element;
    /// Additive inverse.
    fn neg(&self, value: &Self::Element) -> Self::Element;
}

/// Canonical sparse polynomial in one fixed monomial order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePolynomial<C> {
    arity: usize,
    order: MonomialOrder,
    terms: Vec<(Monomial, C)>,
}

impl<C: Clone + Eq> SparsePolynomial<C> {
    /// Canonicalize arbitrary terms, summing duplicates and dropping zeros.
    pub fn new<R>(
        arity: usize,
        order: MonomialOrder,
        terms: impl IntoIterator<Item = (Monomial, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let mut canonical: HashMap<Monomial, C> = HashMap::new();
        for (monomial, coefficient) in terms {
            assert_eq!(
                monomial.arity(),
                arity,
                "polynomial monomial arity mismatch"
            );
            if ring.is_zero(&coefficient) {
                continue;
            }
            let combined = canonical
                .remove(&monomial)
                .map_or(coefficient.clone(), |previous| {
                    ring.add(&previous, &coefficient)
                });
            if !ring.is_zero(&combined) {
                canonical.insert(monomial, combined);
            }
        }
        let mut terms = canonical.into_iter().collect::<Vec<_>>();
        terms.sort_by(|(left, _), (right, _)| right.compare(left, order));
        Self {
            arity,
            order,
            terms,
        }
    }

    /// Construct from already canonical terms, checking the invariant once.
    ///
    /// Terms must be nonzero, have the declared arity, contain no duplicate
    /// monomials, and appear greatest first in `order`.
    pub fn from_sorted_terms<R>(
        arity: usize,
        order: MonomialOrder,
        terms: impl IntoIterator<Item = (Monomial, C)>,
        ring: &R,
    ) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        let terms = terms.into_iter().collect::<Vec<_>>();
        for (index, (monomial, coefficient)) in terms.iter().enumerate() {
            assert_eq!(
                monomial.arity(),
                arity,
                "polynomial monomial arity mismatch"
            );
            assert!(!ring.is_zero(coefficient), "canonical coefficient is zero");
            if index != 0 {
                assert_eq!(
                    terms[index - 1].0.compare(monomial, order),
                    core::cmp::Ordering::Greater,
                    "polynomial terms are not strictly descending"
                );
            }
        }
        Self {
            arity,
            order,
            terms,
        }
    }

    /// Number of variables.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Monomial order.
    pub fn order(&self) -> MonomialOrder {
        self.order
    }

    /// Canonical nonzero terms from greatest to least.
    pub fn terms(&self) -> &[(Monomial, C)] {
        &self.terms
    }

    /// Consume this polynomial into its canonical nonzero terms.
    pub fn into_terms(self) -> Vec<(Monomial, C)> {
        self.terms
    }

    /// Whether the polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Leading monomial.
    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.terms.first().map(|(monomial, _)| monomial)
    }

    /// Leading term.
    pub fn leading_term(&self) -> Option<(&Monomial, &C)> {
        self.terms
            .first()
            .map(|(monomial, coefficient)| (monomial, coefficient))
    }

    /// Add two canonical polynomials.
    pub fn add<R>(&self, other: &Self, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_same_context(other);
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.terms.len() && right < other.terms.len() {
            match self.terms[left]
                .0
                .compare(&other.terms[right].0, self.order)
            {
                core::cmp::Ordering::Greater => {
                    terms.push(self.terms[left].clone());
                    left += 1;
                }
                core::cmp::Ordering::Less => {
                    terms.push(other.terms[right].clone());
                    right += 1;
                }
                core::cmp::Ordering::Equal => {
                    let coefficient = ring.add(&self.terms[left].1, &other.terms[right].1);
                    if !ring.is_zero(&coefficient) {
                        terms.push((self.terms[left].0.clone(), coefficient));
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        terms.extend_from_slice(&self.terms[left..]);
        terms.extend_from_slice(&other.terms[right..]);
        Self {
            arity: self.arity,
            order: self.order,
            terms,
        }
    }

    /// Negate every coefficient.
    pub fn neg<R>(&self, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        Self {
            arity: self.arity,
            order: self.order,
            terms: self
                .terms
                .iter()
                .filter_map(|(monomial, coefficient)| {
                    let coefficient = ring.neg(coefficient);
                    (!ring.is_zero(&coefficient)).then(|| (monomial.clone(), coefficient))
                })
                .collect(),
        }
    }

    /// Subtract two canonical polynomials.
    pub fn sub<R>(&self, other: &Self, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_same_context(other);
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.terms.len() && right < other.terms.len() {
            match self.terms[left]
                .0
                .compare(&other.terms[right].0, self.order)
            {
                core::cmp::Ordering::Greater => {
                    terms.push(self.terms[left].clone());
                    left += 1;
                }
                core::cmp::Ordering::Less => {
                    let coefficient = ring.neg(&other.terms[right].1);
                    if !ring.is_zero(&coefficient) {
                        terms.push((other.terms[right].0.clone(), coefficient));
                    }
                    right += 1;
                }
                core::cmp::Ordering::Equal => {
                    let coefficient =
                        ring.add(&self.terms[left].1, &ring.neg(&other.terms[right].1));
                    if !ring.is_zero(&coefficient) {
                        terms.push((self.terms[left].0.clone(), coefficient));
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        terms.extend_from_slice(&self.terms[left..]);
        terms.extend(
            other.terms[right..]
                .iter()
                .filter_map(|(monomial, coefficient)| {
                    let coefficient = ring.neg(coefficient);
                    (!ring.is_zero(&coefficient)).then(|| (monomial.clone(), coefficient))
                }),
        );
        Self {
            arity: self.arity,
            order: self.order,
            terms,
        }
    }

    /// Multiply every coefficient by one scalar.
    pub fn scalar_mul<R>(&self, scalar: &C, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        Self {
            arity: self.arity,
            order: self.order,
            terms: self
                .terms
                .iter()
                .filter_map(|(monomial, coefficient)| {
                    let coefficient = ring.mul(coefficient, scalar);
                    (!ring.is_zero(&coefficient)).then(|| (monomial.clone(), coefficient))
                })
                .collect(),
        }
    }

    /// Multiply by one coefficient-bearing monomial.
    pub fn term_mul<R>(&self, monomial: &Monomial, scalar: &C, ring: &R) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        assert_eq!(
            monomial.arity(),
            self.arity,
            "polynomial monomial arity mismatch"
        );
        Self {
            arity: self.arity,
            order: self.order,
            terms: self
                .terms
                .iter()
                .filter_map(|(term, coefficient)| {
                    let coefficient = ring.mul(coefficient, scalar);
                    (!ring.is_zero(&coefficient)).then(|| {
                        (
                            term.checked_product(monomial)
                                .expect("kernel exponent overflow must route to the generic floor"),
                            coefficient,
                        )
                    })
                })
                .collect(),
        }
    }

    /// Multiply, optionally discarding terms outside one truncation before
    /// coefficient multiplication.
    pub fn multiply<R>(&self, other: &Self, ring: &R, truncation: Option<&Truncation>) -> Self
    where
        R: CoefficientRing<Element = C>,
    {
        self.require_same_context(other);
        if let Some(truncation) = truncation {
            assert_eq!(truncation.arity(), self.arity, "truncation arity mismatch");
        }
        let mut output = HashMap::<Monomial, C>::new();
        for (left_monomial, left_coefficient) in &self.terms {
            for (right_monomial, right_coefficient) in &other.terms {
                let monomial = left_monomial
                    .checked_product(right_monomial)
                    .expect("kernel exponent overflow must route to the generic floor");
                if truncation.is_some_and(|shape| !shape.retains(&monomial)) {
                    continue;
                }
                let product = ring.mul(left_coefficient, right_coefficient);
                let coefficient = output
                    .remove(&monomial)
                    .map_or(product.clone(), |previous| ring.add(&previous, &product));
                if !ring.is_zero(&coefficient) {
                    output.insert(monomial, coefficient);
                }
            }
        }
        Self::new(self.arity, self.order, output, ring)
    }

    fn require_same_context(&self, other: &Self) {
        assert_eq!(self.arity, other.arity, "polynomial arity mismatch");
        assert_eq!(self.order, other.order, "polynomial order mismatch");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn truncated_multiplication_cuts_terms() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let polynomial = SparsePolynomial::new(
            1,
            order,
            [
                (Monomial::new(vec![0].into_boxed_slice()), 1),
                (Monomial::new(vec![1].into_boxed_slice()), 1),
            ],
            &ring,
        );
        let truncation = Truncation::TotalDegree { arity: 1, bound: 2 };
        let square = polynomial.multiply(&polynomial, &ring, Some(&truncation));
        assert_eq!(square.terms().len(), 2);
        assert_eq!(square.terms()[0].1, 2);
    }

    #[test]
    fn subtraction_merges_terms_without_losing_cancellation() {
        let ring = Mod7;
        let order = MonomialOrder::GradedReverseLexicographic;
        let left = SparsePolynomial::new(
            1,
            order,
            [
                (Monomial::new(vec![3].into_boxed_slice()), 2),
                (Monomial::new(vec![1].into_boxed_slice()), 4),
            ],
            &ring,
        );
        let right = SparsePolynomial::new(
            1,
            order,
            [
                (Monomial::new(vec![3].into_boxed_slice()), 2),
                (Monomial::new(vec![2].into_boxed_slice()), 3),
            ],
            &ring,
        );
        assert_eq!(
            left.sub(&right, &ring).terms(),
            &[
                (Monomial::new(vec![2].into_boxed_slice()), 4),
                (Monomial::new(vec![1].into_boxed_slice()), 4),
            ]
        );
    }
}
