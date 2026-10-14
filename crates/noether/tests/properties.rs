// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property tests at the public Rust facade.
//!
//! Kernel suites test their storage and algorithms independently. These tests
//! keep the user-facing generic types honest across specialized FLINT storage,
//! generic coefficient contexts, serialization, and fixed truncation.

use std::collections::BTreeMap;

use noether::commalg::{
    MonomialOrder, Polynomial, PolynomialRing, PowerSeries, PowerSeriesRing, SeriesTruncation,
};
use noether::homalg::ChainComplex;
use noether::linalg::{Matrix, Vector};
use noether::{Integer, IntegerRing, Rational, Ring, Q, Z};
use proptest::collection::vec;
use proptest::prelude::*;

fn property_config(default_cases: u32) -> ProptestConfig {
    let configured = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_some() {
        configured
    } else {
        ProptestConfig {
            cases: default_cases,
            ..configured
        }
    }
}

fn square_matrix(size: usize, entries: &[i16]) -> Matrix<IntegerRing> {
    Matrix::from_rows(
        Z,
        entries
            .chunks(size)
            .map(|row| row.iter().copied().map(Integer::from).collect::<Vec<_>>()),
    )
    .expect("the property strategy always supplies a rectangular matrix")
}

fn square_matrix_case() -> impl Strategy<Value = (usize, Vec<i16>, Vec<i16>, Vec<i16>)> {
    (1_usize..=4).prop_flat_map(|size| {
        let entries = size * size;
        (
            Just(size),
            vec(-20_i16..=20, entries),
            vec(-20_i16..=20, entries),
            vec(-20_i16..=20, entries),
        )
    })
}

fn vector_pair_case() -> impl Strategy<Value = (Vec<i16>, Vec<i16>)> {
    (0_usize..=12).prop_flat_map(|dimension| {
        (
            vec(-100_i16..=100, dimension),
            vec(-100_i16..=100, dimension),
        )
    })
}

fn rational_matrix_case() -> impl Strategy<Value = (usize, usize, Vec<i16>)> {
    (0_usize..=5, 0_usize..=5).prop_flat_map(|(rows, columns)| {
        (Just(rows), Just(columns), vec(-8_i16..=8, rows * columns))
    })
}

fn integer_polynomial(
    ring: &PolynomialRing<IntegerRing>,
    coefficients: &[i16],
) -> Polynomial<IntegerRing> {
    ring.from_terms([
        (
            vec![Integer::from(2), Integer::zero()],
            Integer::from(coefficients[0]),
        ),
        (
            vec![Integer::one(), Integer::one()],
            Integer::from(coefficients[1]),
        ),
        (
            vec![Integer::zero(), Integer::one()],
            Integer::from(coefficients[2]),
        ),
        (
            vec![Integer::zero(), Integer::zero()],
            Integer::from(coefficients[3]),
        ),
    ])
    .expect("the property strategy supplies valid polynomial terms")
}

fn integer_series(
    ring: &PowerSeriesRing<IntegerRing>,
    coefficients: &[i16],
) -> PowerSeries<IntegerRing> {
    ring.from_terms(
        coefficients
            .iter()
            .enumerate()
            .map(|(exponent, value)| (vec![exponent as u64], Integer::from(*value))),
    )
    .expect("the property strategy supplies retained series terms")
}

proptest! {
    #![proptest_config(property_config(96))]

    #[test]
    fn integer_matrices_obey_ring_laws_and_round_trip(
        (size, left, middle, right) in square_matrix_case(),
    ) {
        let a = square_matrix(size, &left);
        let b = square_matrix(size, &middle);
        let c = square_matrix(size, &right);

        prop_assert_eq!(a.add(&b).unwrap().sub(&b).unwrap(), a.clone());
        prop_assert_eq!(a.transpose().transpose(), a.clone());
        prop_assert_eq!(a.matmul(&Matrix::identity(Z, size)).unwrap(), a.clone());
        prop_assert_eq!(
            a.matmul(&b.add(&c).unwrap()).unwrap(),
            a.matmul(&b).unwrap().add(&a.matmul(&c).unwrap()).unwrap(),
        );
        prop_assert_eq!(
            Matrix::<IntegerRing>::from_json(&a.to_json().unwrap()).unwrap(),
            a,
        );
    }

    #[test]
    fn vectors_reuse_matrix_arithmetic_without_changing_their_laws(
        (left, right) in vector_pair_case(),
    ) {
        let left = Vector::from_values(Z, left.into_iter().map(Integer::from)).unwrap();
        let right = Vector::from_values(Z, right.into_iter().map(Integer::from)).unwrap();

        prop_assert_eq!(left.add(&right).unwrap().sub(&right).unwrap(), left.clone());
        prop_assert_eq!(left.dot(&right).unwrap(), right.dot(&left).unwrap());
        prop_assert_eq!(
            Vector::<IntegerRing>::from_json(&left.to_json().unwrap()).unwrap(),
            left,
        );
    }

    #[test]
    fn invariants_only_betti_numbers_match_explicit_homology(
        (rows, columns, entries) in rational_matrix_case(),
    ) {
        let boundary = if rows == 0 || columns == 0 {
            Matrix::zeros(Q, rows, columns)
        } else {
            Matrix::from_rows(
                Q,
                entries.chunks(columns).map(|row| {
                    row.iter().copied().map(Rational::from).collect::<Vec<_>>()
                }),
            ).unwrap()
        };
        let complex = ChainComplex::from_boundaries(
            Q,
            BTreeMap::from([(1, boundary)]),
        ).unwrap();

        for degree in [0, 1] {
            prop_assert_eq!(
                complex.betti_number(degree).unwrap(),
                complex.homology(degree).unwrap().dimension(),
            );
        }
    }

    #[test]
    fn integral_homology_invariants_match_source_retaining_results(
        (rows, columns, entries) in rational_matrix_case(),
    ) {
        let boundary = if rows == 0 || columns == 0 {
            Matrix::zeros(Z, rows, columns)
        } else {
            Matrix::from_rows(
                Z,
                entries.chunks(columns).map(|row| {
                    row.iter().copied().map(Integer::from).collect::<Vec<_>>()
                }),
            ).unwrap()
        };
        let complex = ChainComplex::from_boundaries(
            Z,
            BTreeMap::from([(1, boundary)]),
        ).unwrap();

        for degree in [0, 1] {
            let lean = complex.homology_invariants(degree).unwrap();
            let checked = complex.homology(degree).unwrap();
            prop_assert_eq!(lean.free_rank(), checked.free_rank());
            prop_assert_eq!(
                lean.torsion_coefficients(),
                checked.torsion_coefficients(),
            );
            lean.verify().unwrap();
        }
    }

    #[test]
    fn polynomial_evaluation_is_a_ring_homomorphism(
        left in vec(-12_i16..=12, 4),
        right in vec(-12_i16..=12, 4),
        x in -8_i16..=8,
        y in -8_i16..=8,
    ) {
        let ring = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
            .unwrap();
        let left = integer_polynomial(&ring, &left);
        let right = integer_polynomial(&ring, &right);
        let values = [Integer::from(x), Integer::from(y)];

        let product = ring.mul(&left, &right).unwrap();
        prop_assert_eq!(
            product.evaluate(&values).unwrap(),
            left.evaluate(&values).unwrap() * right.evaluate(&values).unwrap(),
        );
        prop_assert_eq!(
            Polynomial::<IntegerRing>::from_json(&product.to_json().unwrap()).unwrap(),
            product,
        );
    }

    #[test]
    fn fixed_precision_series_multiplication_is_associative_and_serializable(
        precision in 1_u64..=8,
        left in vec(-8_i16..=8, 8),
        middle in vec(-8_i16..=8, 8),
        right in vec(-8_i16..=8, 8),
    ) {
        let ring = PowerSeriesRing::new(
            Z,
            ["t"],
            SeriesTruncation::univariate(precision).unwrap(),
        )
        .unwrap();
        let retained = precision as usize;
        let left = integer_series(&ring, &left[..retained]);
        let middle = integer_series(&ring, &middle[..retained]);
        let right = integer_series(&ring, &right[..retained]);

        let left_associated = ring.mul(&ring.mul(&left, &middle).unwrap(), &right).unwrap();
        let right_associated = ring.mul(&left, &ring.mul(&middle, &right).unwrap()).unwrap();
        prop_assert_eq!(left_associated, right_associated.clone());
        prop_assert_eq!(
            PowerSeries::<IntegerRing>::from_json(&right_associated.to_json().unwrap()).unwrap(),
            right_associated,
        );
    }
}
