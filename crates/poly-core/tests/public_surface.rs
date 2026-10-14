// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use poly_core::{CoefficientRing, Monomial, MonomialOrder, SparsePolynomial, Truncation};

struct Integers;

impl CoefficientRing for Integers {
    type Element = i64;

    fn zero(&self) -> Self::Element {
        0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value == 0
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left + right
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left * right
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        -value
    }
}

fn monomial(exponents: &[u32]) -> Monomial {
    Monomial::new(exponents.to_vec().into_boxed_slice())
}

#[test]
fn canonical_sparse_arithmetic_respects_strict_truncation() {
    let ring = Integers;
    let order = MonomialOrder::GradedReverseLexicographic;
    let polynomial = SparsePolynomial::new(
        2,
        order,
        [
            (monomial(&[1, 0]), 2),
            (monomial(&[0, 1]), 3),
            (monomial(&[1, 0]), -2),
        ],
        &ring,
    );
    assert_eq!(polynomial.terms(), [(monomial(&[0, 1]), 3)]);

    let factor = SparsePolynomial::new(
        2,
        order,
        [(monomial(&[1, 0]), 1), (monomial(&[0, 0]), 1)],
        &ring,
    );
    let truncation = Truncation::TotalDegree { arity: 2, bound: 2 };
    let product = factor.multiply(&factor, &ring, Some(&truncation));
    assert_eq!(
        product.terms(),
        [(monomial(&[1, 0]), 2), (monomial(&[0, 0]), 1),]
    );
}
