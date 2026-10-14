// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Coefficient-free polynomial and packed truncated-series kernels over F₂.

use std::collections::HashSet;
use std::sync::Arc;

use crate::{DenseMonomialRegion, Monomial, MonomialOrder, RegionError, Truncation};

/// Polynomial-ring kernel context over F₂.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct F2PolynomialRing {
    arity: usize,
    order: MonomialOrder,
}

impl F2PolynomialRing {
    /// Construct a polynomial kernel context.
    pub fn new(arity: usize, order: MonomialOrder) -> Self {
        assert!(arity > 0, "an F2 polynomial ring needs a variable");
        Self { arity, order }
    }

    /// Number of variables.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Monomial order.
    pub fn order(&self) -> MonomialOrder {
        self.order
    }

    /// The zero polynomial.
    pub fn zero(&self) -> F2Polynomial {
        F2Polynomial {
            ring: self.clone(),
            support: Vec::new(),
        }
    }

    /// The constant one polynomial.
    pub fn one(&self) -> F2Polynomial {
        self.from_support([Monomial::one(self.arity)])
    }

    /// Canonicalize a support; duplicate monomials cancel in pairs.
    pub fn from_support(&self, monomials: impl IntoIterator<Item = Monomial>) -> F2Polynomial {
        let mut support = HashSet::new();
        for monomial in monomials {
            assert_eq!(monomial.arity(), self.arity, "F2 monomial arity mismatch");
            if !support.insert(monomial.clone()) {
                support.remove(&monomial);
            }
        }
        let mut support = support.into_iter().collect::<Vec<_>>();
        support.sort_by(|left, right| right.compare(left, self.order));
        F2Polynomial {
            ring: self.clone(),
            support,
        }
    }
}

/// Canonical coefficient-free sparse polynomial over F₂.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct F2Polynomial {
    ring: F2PolynomialRing,
    support: Vec<Monomial>,
}

impl F2Polynomial {
    /// Ring kernel context.
    pub fn ring(&self) -> &F2PolynomialRing {
        &self.ring
    }

    /// Canonical support from greatest to least.
    pub fn support(&self) -> &[Monomial] {
        &self.support
    }

    /// Number of nonzero terms.
    pub fn term_count(&self) -> usize {
        self.support.len()
    }

    /// Whether this polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.support.is_empty()
    }

    /// Leading monomial.
    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.support.first()
    }

    /// Add by a linear symmetric-difference merge.
    pub fn add(&self, other: &Self) -> Self {
        self.require_same_ring(other);
        let mut output = Vec::with_capacity(self.support.len() + other.support.len());
        let mut left = 0;
        let mut right = 0;
        while left < self.support.len() && right < other.support.len() {
            match self.support[left].compare(&other.support[right], self.ring.order) {
                core::cmp::Ordering::Greater => {
                    output.push(self.support[left].clone());
                    left += 1;
                }
                core::cmp::Ordering::Less => {
                    output.push(other.support[right].clone());
                    right += 1;
                }
                core::cmp::Ordering::Equal => {
                    left += 1;
                    right += 1;
                }
            }
        }
        output.extend_from_slice(&self.support[left..]);
        output.extend_from_slice(&other.support[right..]);
        Self {
            ring: self.ring.clone(),
            support: output,
        }
    }

    /// Multiply by parity-toggle accumulation.
    pub fn multiply(&self, other: &Self) -> Self {
        self.require_same_ring(other);
        if self == other {
            return self.square();
        }
        let mut support = HashSet::new();
        for left in &self.support {
            for right in &other.support {
                let product = left
                    .checked_product(right)
                    .expect("F2 kernel exponent overflow must route to the generic floor");
                if !support.insert(product.clone()) {
                    support.remove(&product);
                }
            }
        }
        self.ring.from_support(support)
    }

    /// Square using the characteristic-two Frobenius, doubling exponents
    /// without constructing cross terms.
    pub fn square(&self) -> Self {
        self.ring.from_support(self.support.iter().map(|monomial| {
            Monomial::new(
                monomial
                    .to_u32_vec()
                    .into_iter()
                    .map(|exponent| {
                        exponent
                            .checked_mul(2)
                            .expect("F2 kernel exponent overflow must route to the generic floor")
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        }))
    }

    /// Multiply by one monomial.
    pub fn monomial_mul(&self, monomial: &Monomial) -> Self {
        assert_eq!(
            monomial.arity(),
            self.ring.arity,
            "F2 monomial arity mismatch"
        );
        self.ring.from_support(self.support.iter().map(|term| {
            term.checked_product(monomial)
                .expect("F2 kernel exponent overflow must route to the generic floor")
        }))
    }

    /// Raise to a nonnegative power.
    pub fn pow(&self, mut exponent: u64) -> Self {
        let mut result = self.ring.one();
        let mut base = self.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.multiply(&base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square();
            }
        }
        result
    }

    /// Formal derivative in one variable.
    pub fn derivative(&self, variable: usize) -> Self {
        assert!(
            variable < self.ring.arity,
            "F2 derivative variable out of range"
        );
        self.ring
            .from_support(self.support.iter().filter_map(|monomial| {
                let exponent = monomial.exponent(variable);
                if exponent & 1 == 0 {
                    return None;
                }
                let mut values = monomial.to_u32_vec();
                values[variable] -= 1;
                Some(Monomial::new(values.into_boxed_slice()))
            }))
    }

    fn require_same_ring(&self, other: &Self) {
        assert_eq!(self.ring, other.ring, "F2 polynomial ring mismatch");
    }
}

/// Dense bit-packed fixed-truncation series context over F₂.
#[derive(Clone, Debug)]
pub struct F2SeriesRing {
    region: Arc<DenseMonomialRegion>,
}

impl F2SeriesRing {
    /// Enumerate a truncation region with an explicit term cap.
    pub fn new(truncation: Truncation, term_limit: usize) -> Result<Self, RegionError> {
        DenseMonomialRegion::new(truncation, term_limit).map(|region| Self {
            region: Arc::new(region),
        })
    }

    /// Dense monomial region.
    pub fn region(&self) -> &DenseMonomialRegion {
        &self.region
    }

    /// The zero series.
    pub fn zero(&self) -> F2Series {
        F2Series {
            ring: self.clone(),
            words: vec![0; self.region.len().div_ceil(64)],
        }
    }

    /// The constant one series.
    pub fn one(&self) -> F2Series {
        self.from_support([Monomial::one(self.region.truncation().arity())])
    }

    /// Construct from a support, dropping out-of-region terms and cancelling
    /// duplicates.
    pub fn from_support(&self, support: impl IntoIterator<Item = Monomial>) -> F2Series {
        let mut output = self.zero();
        for monomial in support {
            if let Some(index) = self.region.index_of(&monomial) {
                output.toggle(index);
            }
        }
        output
    }
}

impl PartialEq for F2SeriesRing {
    fn eq(&self, other: &Self) -> bool {
        self.region.truncation() == other.region.truncation()
    }
}

impl Eq for F2SeriesRing {}

/// Dense bit-packed fixed-truncation series over F₂.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct F2Series {
    ring: F2SeriesRing,
    words: Vec<u64>,
}

impl F2Series {
    /// Series context.
    pub fn ring(&self) -> &F2SeriesRing {
        &self.ring
    }

    /// Whether this series is zero.
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Number of retained nonzero coefficients.
    pub fn term_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Whether one retained monomial has coefficient one.
    pub fn contains(&self, monomial: &Monomial) -> bool {
        self.ring
            .region
            .index_of(monomial)
            .is_some_and(|index| self.bit(index))
    }

    /// Add by packed XOR.
    pub fn add(&self, other: &Self) -> Self {
        self.require_same_ring(other);
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| left ^ right)
            .collect();
        Self {
            ring: self.ring.clone(),
            words,
        }
    }

    /// Multiply with truncation checked before toggling an output bit.
    pub fn multiply(&self, other: &Self) -> Self {
        self.require_same_ring(other);
        if self == other {
            return self.square();
        }
        let left = self.nonzero_indices();
        let right = other.nonzero_indices();
        let mut output = self.ring.zero();
        for left_index in left {
            for &right_index in &right {
                if let Some(index) = self
                    .ring
                    .region
                    .product_index(left_index, right_index)
                    .expect("enumerated region exponent invariant")
                {
                    output.toggle(index);
                }
            }
        }
        output
    }

    /// Square through Frobenius, checking the truncation before setting each
    /// retained doubled monomial.
    pub fn square(&self) -> Self {
        let mut output = self.ring.zero();
        for index in self.nonzero_indices() {
            let monomial = self.ring.region.monomial(index);
            let doubled = Monomial::new(
                monomial
                    .to_u32_vec()
                    .into_iter()
                    .map(|exponent| {
                        exponent
                            .checked_mul(2)
                            .expect("F2 series exponent overflow must route to the generic floor")
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            if let Some(output_index) = self.ring.region.index_of(&doubled) {
                output.toggle(output_index);
            }
        }
        output
    }

    /// Raise to a nonnegative power.
    pub fn pow(&self, mut exponent: u64) -> Self {
        let mut result = self.ring.one();
        let mut base = self.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.multiply(&base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square();
            }
        }
        result
    }

    /// Invert a series with constant coefficient one.
    pub fn inverse(&self) -> Option<Self> {
        let one = Monomial::one(self.ring.region.truncation().arity());
        if !self.contains(&one) {
            return None;
        }
        let tail = self.add(&self.ring.one());
        let mut result = self.ring.one();
        let mut power = self.ring.one();
        for _ in 0..self.ring.region.len() {
            power = power.multiply(&tail);
            if power.is_zero() {
                return Some(result);
            }
            result = result.add(&power);
        }
        None
    }

    /// Return retained monomials with coefficient one.
    pub fn support(&self) -> Vec<Monomial> {
        self.nonzero_indices()
            .into_iter()
            .map(|index| self.ring.region.monomial(index).clone())
            .collect()
    }

    fn bit(&self, index: usize) -> bool {
        self.words[index / 64] & (1_u64 << (index % 64)) != 0
    }

    fn toggle(&mut self, index: usize) {
        self.words[index / 64] ^= 1_u64 << (index % 64);
    }

    fn nonzero_indices(&self) -> Vec<usize> {
        let mut output = Vec::with_capacity(self.term_count());
        for (word_index, &word) in self.words.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if index < self.ring.region.len() {
                    output.push(index);
                }
                remaining &= remaining - 1;
            }
        }
        output
    }

    fn require_same_ring(&self, other: &Self) {
        assert_eq!(self.ring, other.ring, "F2 series ring mismatch");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(values: &[u32]) -> Monomial {
        Monomial::new(values.to_vec().into_boxed_slice())
    }

    #[test]
    fn sparse_addition_and_product_use_parity() {
        let ring = F2PolynomialRing::new(2, MonomialOrder::GradedReverseLexicographic);
        let left = ring.from_support([m(&[1, 0]), m(&[0, 1])]);
        assert!(left.add(&left).is_zero());
        let square = left.multiply(&left);
        assert_eq!(square.support(), &[m(&[2, 0]), m(&[0, 2])]);
    }

    #[test]
    fn packed_series_multiply_and_invert() {
        let ring = F2SeriesRing::new(Truncation::TotalDegree { arity: 1, bound: 8 }, 100).unwrap();
        let value = ring.from_support([m(&[0]), m(&[1])]);
        let inverse = value.inverse().unwrap();
        assert_eq!(value.multiply(&inverse), ring.one());
        assert_eq!(inverse.term_count(), 8);
    }
}
