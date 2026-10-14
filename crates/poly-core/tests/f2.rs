// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use poly_core::{
    f2::{F2PolynomialRing, F2SeriesRing},
    Monomial, MonomialOrder, Truncation,
};
use proptest::prelude::*;

fn monomial(exponents: &[u32]) -> Monomial {
    Monomial::new(exponents.to_vec().into_boxed_slice())
}

#[test]
fn characteristic_two_arithmetic_uses_support_parity_and_frobenius() {
    let ring = F2PolynomialRing::new(2, MonomialOrder::GradedReverseLexicographic);
    let polynomial = ring.from_support([monomial(&[1, 0]), monomial(&[0, 1])]);

    assert!(polynomial.add(&polynomial).is_zero());
    assert_eq!(
        polynomial.square().support(),
        [monomial(&[2, 0]), monomial(&[0, 2])]
    );

    let series_ring =
        F2SeriesRing::new(Truncation::TotalDegree { arity: 2, bound: 3 }, 32).unwrap();
    let series = series_ring.from_support([monomial(&[1, 0]), monomial(&[0, 1])]);
    assert_eq!(
        series.square().support(),
        vec![monomial(&[0, 2]), monomial(&[2, 0])]
    );
}

#[test]
fn sparse_api_covers_ordering_derivatives_monomials_and_powers() {
    let ring = F2PolynomialRing::new(1, MonomialOrder::Lexicographic);
    assert_eq!(ring.arity(), 1);
    assert_eq!(ring.order(), MonomialOrder::Lexicographic);
    assert!(ring.zero().is_zero());

    let polynomial = ring.from_support([
        monomial(&[0]),
        monomial(&[1]),
        monomial(&[2]),
        monomial(&[3]),
        monomial(&[2]),
        monomial(&[2]),
    ]);
    assert_eq!(polynomial.ring(), &ring);
    assert_eq!(polynomial.term_count(), 4);
    assert_eq!(polynomial.leading_monomial(), Some(&monomial(&[3])));
    assert_eq!(
        polynomial.derivative(0).support(),
        &[monomial(&[2]), monomial(&[0])]
    );
    assert_eq!(
        polynomial.monomial_mul(&monomial(&[1])).support(),
        &[
            monomial(&[4]),
            monomial(&[3]),
            monomial(&[2]),
            monomial(&[1]),
        ]
    );
    assert_eq!(polynomial.pow(0), ring.one());

    let left = ring.from_support([monomial(&[0]), monomial(&[1])]);
    let right = ring.from_support([monomial(&[0]), monomial(&[1]), monomial(&[2])]);
    assert_eq!(
        left.multiply(&right).support(),
        &[monomial(&[3]), monomial(&[0])]
    );

    let high_and_low = ring.from_support([monomial(&[3]), monomial(&[1])]);
    let middle_and_low = ring.from_support([monomial(&[2]), monomial(&[1])]);
    assert_eq!(
        high_and_low.add(&middle_and_low).support(),
        &[monomial(&[3]), monomial(&[2])]
    );
}

#[test]
fn packed_series_crosses_word_boundaries_without_losing_terms() {
    let truncation = Truncation::TotalDegree {
        arity: 1,
        bound: 130,
    };
    let ring = F2SeriesRing::new(truncation.clone(), 130).unwrap();
    let equal_context = F2SeriesRing::new(truncation, 130).unwrap();
    let different_context =
        F2SeriesRing::new(Truncation::TotalDegree { arity: 1, bound: 8 }, 8).unwrap();
    assert_eq!(ring, equal_context);
    assert_ne!(ring, different_context);
    assert_eq!(ring.region().len(), 130);

    let value = ring.from_support([
        monomial(&[0]),
        monomial(&[65]),
        monomial(&[65]),
        monomial(&[129]),
        monomial(&[130]),
        monomial(&[0, 0]),
    ]);
    assert_eq!(value.ring(), &ring);
    assert_eq!(value.term_count(), 2);
    assert!(value.contains(&monomial(&[0])));
    assert!(value.contains(&monomial(&[129])));
    assert!(!value.contains(&monomial(&[65])));
    assert!(!value.contains(&monomial(&[130])));
    assert!(!value.contains(&monomial(&[0, 0])));
    assert_eq!(value.support(), vec![monomial(&[0]), monomial(&[129])]);
    assert_eq!(value.pow(0), ring.one());
    assert!(ring.from_support([monomial(&[1])]).inverse().is_none());

    let left = ring.from_support([monomial(&[0]), monomial(&[65])]);
    let right = ring.from_support([monomial(&[0]), monomial(&[64])]);
    assert_eq!(
        left.multiply(&right).support(),
        vec![
            monomial(&[0]),
            monomial(&[64]),
            monomial(&[65]),
            monomial(&[129]),
        ]
    );
    assert_eq!(left.add(&right).term_count(), 2);
    assert_eq!(left.square(), ring.one());
}

#[test]
#[should_panic(expected = "an F2 polynomial ring needs a variable")]
fn polynomial_context_rejects_zero_variables() {
    let _ = F2PolynomialRing::new(0, MonomialOrder::Lexicographic);
}

#[test]
#[should_panic(expected = "F2 polynomial ring mismatch")]
fn sparse_arithmetic_rejects_different_monomial_orders() {
    let lex = F2PolynomialRing::new(1, MonomialOrder::Lexicographic).from_support([monomial(&[1])]);
    let grevlex = F2PolynomialRing::new(1, MonomialOrder::GradedReverseLexicographic)
        .from_support([monomial(&[1])]);
    let _ = lex.add(&grevlex);
}

#[test]
#[should_panic(expected = "F2 derivative variable out of range")]
fn derivative_rejects_an_out_of_range_variable() {
    let value =
        F2PolynomialRing::new(1, MonomialOrder::Lexicographic).from_support([monomial(&[1])]);
    let _ = value.derivative(1);
}

#[test]
#[should_panic(expected = "F2 monomial arity mismatch")]
fn monomial_multiplication_rejects_an_incompatible_monomial() {
    let value =
        F2PolynomialRing::new(1, MonomialOrder::Lexicographic).from_support([monomial(&[1])]);
    let _ = value.monomial_mul(&monomial(&[1, 0]));
}

#[test]
#[should_panic(expected = "F2 series ring mismatch")]
fn packed_arithmetic_rejects_different_truncations() {
    let short = F2SeriesRing::new(Truncation::TotalDegree { arity: 1, bound: 4 }, 4)
        .unwrap()
        .one();
    let long = F2SeriesRing::new(Truncation::TotalDegree { arity: 1, bound: 5 }, 5)
        .unwrap()
        .one();
    let _ = short.multiply(&long);
}

proptest! {
    #[test]
    fn sparse_product_is_distributive(
        left in proptest::collection::vec((0_u32..6, 0_u32..6), 0..16),
        middle in proptest::collection::vec((0_u32..6, 0_u32..6), 0..16),
        right in proptest::collection::vec((0_u32..6, 0_u32..6), 0..16),
    ) {
        let ring = F2PolynomialRing::new(2, MonomialOrder::GradedReverseLexicographic);
        let left = ring.from_support(left.into_iter().map(|(x, y)| monomial(&[x, y])));
        let middle = ring.from_support(middle.into_iter().map(|(x, y)| monomial(&[x, y])));
        let right = ring.from_support(right.into_iter().map(|(x, y)| monomial(&[x, y])));

        prop_assert_eq!(
            left.multiply(&middle.add(&right)),
            left.multiply(&middle).add(&left.multiply(&right)),
        );
    }

    #[test]
    fn packed_weighted_series_matches_sparse_truncation(
        left in proptest::collection::vec((0_u32..7, 0_u32..4), 0..16),
        right in proptest::collection::vec((0_u32..7, 0_u32..4), 0..16),
    ) {
        let truncation = Truncation::Weighted {
            weights: vec![1, 2].into_boxed_slice(),
            bound: 7,
        };
        let series_ring = F2SeriesRing::new(truncation.clone(), 64).unwrap();
        let polynomial_ring =
            F2PolynomialRing::new(2, MonomialOrder::GradedReverseLexicographic);
        let left = left.into_iter().map(|(x, y)| monomial(&[x, y])).collect::<Vec<_>>();
        let right = right.into_iter().map(|(x, y)| monomial(&[x, y])).collect::<Vec<_>>();

        let packed_product = series_ring
            .from_support(left.iter().cloned())
            .multiply(&series_ring.from_support(right.iter().cloned()));
        let sparse_product = polynomial_ring
            .from_support(left)
            .multiply(&polynomial_ring.from_support(right));
        let expected = series_ring.from_support(
            sparse_product
                .support()
                .iter()
                .filter(|monomial| truncation.retains(monomial))
                .cloned(),
        );

        prop_assert_eq!(packed_product, expected);
    }
}
