// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Cross-context properties at the public arithmetic facade.
//!
//! These checks target the boundary where Noether routes one mathematical
//! operation through specialized FLINT storage.  Small, direct ring formulas
//! remain the oracle, so agreement does not depend on the implementation
//! selected by `Matrix<R>`.

use std::collections::BTreeMap;
use std::fmt::Debug;

use noether::commalg::{
    PowerSeries, PowerSeriesRing, ResidueRing, SeriesTruncation, WeightedGrading,
};
use noether::linalg::Matrix;
use noether::numthy::{
    FiniteField, FiniteFieldElement, PadicElement, PadicRing, QadicElement, QadicRing,
};
use noether::{CommutativeRing, Integer, Rational, Ring, Q, Z};
use proptest::collection::vec;
use proptest::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

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

fn matrix_case() -> impl Strategy<Value = (usize, usize, usize, Vec<i16>, Vec<i16>)> {
    (1_usize..=4, 1_usize..=4, 1_usize..=4).prop_flat_map(|(rows, inner, columns)| {
        (
            Just(rows),
            Just(inner),
            Just(columns),
            vec(-30_i16..=30, rows * inner),
            vec(-30_i16..=30, inner * columns),
        )
    })
}

fn matrix_from_values<R, F>(
    ring: &R,
    rows: usize,
    columns: usize,
    values: &[i16],
    lift: F,
) -> Matrix<R>
where
    R: Ring,
    F: Fn(i16) -> R::Element,
{
    debug_assert_eq!(values.len(), rows * columns);
    Matrix::from_rows(
        ring.clone(),
        values
            .chunks(columns)
            .map(|row| row.iter().copied().map(&lift).collect::<Vec<_>>()),
    )
    .unwrap()
}

fn reference_matmul<R: Ring>(left: &Matrix<R>, right: &Matrix<R>) -> Matrix<R> {
    let ring = left.ring();
    let (rows, inner) = left.shape();
    let (_, columns) = right.shape();
    let values = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| {
                    (0..inner).fold(ring.zero(), |sum, index| {
                        let product = ring
                            .mul(
                                &left.get(row, index).unwrap(),
                                &right.get(index, column).unwrap(),
                            )
                            .unwrap();
                        ring.add(&sum, &product).unwrap()
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Matrix::from_rows(ring.clone(), values).unwrap()
}

fn assert_matmul_route<R>(left: &Matrix<R>, right: &Matrix<R>)
where
    R: Ring + Serialize + DeserializeOwned,
    R::Element: Serialize + DeserializeOwned,
{
    let product = left.matmul(right).unwrap();
    assert_eq!(product, reference_matmul(left, right));
    let json = product.to_json().unwrap();
    assert_eq!(product.to_json().unwrap(), json);
    assert_eq!(Matrix::<R>::from_json(&json).unwrap(), product);
}

fn two_by_two_minor<R: CommutativeRing>(
    ring: &R,
    a: &R::Element,
    b: &R::Element,
    c: &R::Element,
    d: &R::Element,
) -> R::Element {
    ring.sub(&ring.mul(a, d).unwrap(), &ring.mul(b, c).unwrap())
        .unwrap()
}

fn determinant_three<R: CommutativeRing>(matrix: &Matrix<R>) -> R::Element {
    let ring = matrix.ring();
    let a = matrix.get(0, 0).unwrap();
    let b = matrix.get(0, 1).unwrap();
    let c = matrix.get(0, 2).unwrap();
    let d = matrix.get(1, 0).unwrap();
    let e = matrix.get(1, 1).unwrap();
    let f = matrix.get(1, 2).unwrap();
    let g = matrix.get(2, 0).unwrap();
    let h = matrix.get(2, 1).unwrap();
    let i = matrix.get(2, 2).unwrap();
    let first = ring
        .mul(&a, &two_by_two_minor(ring, &e, &f, &h, &i))
        .unwrap();
    let second = ring
        .mul(&b, &two_by_two_minor(ring, &d, &f, &g, &i))
        .unwrap();
    let third = ring
        .mul(&c, &two_by_two_minor(ring, &d, &e, &g, &h))
        .unwrap();
    ring.add(&ring.sub(&first, &second).unwrap(), &third)
        .unwrap()
}

fn assert_degree_three_characteristic_polynomial<R>(matrix: &Matrix<R>)
where
    R: CommutativeRing,
    R::Element: Debug,
{
    let ring = matrix.ring();
    let determinant = determinant_three(matrix);
    assert_eq!(matrix.det().unwrap(), determinant);

    let trace = ring
        .add(
            &ring
                .add(&matrix.get(0, 0).unwrap(), &matrix.get(1, 1).unwrap())
                .unwrap(),
            &matrix.get(2, 2).unwrap(),
        )
        .unwrap();
    let principal_two = [(0_usize, 1_usize), (0_usize, 2_usize), (1_usize, 2_usize)]
        .into_iter()
        .fold(ring.zero(), |sum, (left, right)| {
            let minor = two_by_two_minor(
                ring,
                &matrix.get(left, left).unwrap(),
                &matrix.get(left, right).unwrap(),
                &matrix.get(right, left).unwrap(),
                &matrix.get(right, right).unwrap(),
            );
            ring.add(&sum, &minor).unwrap()
        });
    let polynomial = matrix.charpoly().unwrap();
    assert_eq!(
        polynomial.coefficient(&[Integer::from(3)]).unwrap(),
        ring.one()
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::from(2)]).unwrap(),
        ring.neg(&trace).unwrap()
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::one()]).unwrap(),
        principal_two
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::zero()]).unwrap(),
        ring.neg(&determinant).unwrap()
    );
}

fn rational(value: i16) -> Rational {
    let denominator = i32::from(value).unsigned_abs() % 5 + 1;
    Rational::new(Integer::from(value), Integer::from(denominator)).unwrap()
}

fn extension_element(field: &FiniteField, value: i16) -> FiniteFieldElement {
    let value = i64::from(value);
    let characteristic = field.characteristic().to_string().parse::<i64>().unwrap();
    field
        .element_from_coefficients(&[
            value.rem_euclid(characteristic) as u64,
            value.div_euclid(characteristic).rem_euclid(characteristic) as u64,
        ])
        .unwrap()
}

fn assert_ring_laws<R>(ring: &R, left: &R::Element, middle: &R::Element, right: &R::Element)
where
    R: CommutativeRing,
    R::Element: Debug,
{
    assert_eq!(
        ring.add(&ring.add(left, middle).unwrap(), right).unwrap(),
        ring.add(left, &ring.add(middle, right).unwrap()).unwrap()
    );
    assert_eq!(
        ring.mul(&ring.mul(left, middle).unwrap(), right).unwrap(),
        ring.mul(left, &ring.mul(middle, right).unwrap()).unwrap()
    );
    assert_eq!(
        ring.mul(left, &ring.add(middle, right).unwrap()).unwrap(),
        ring.add(
            &ring.mul(left, middle).unwrap(),
            &ring.mul(left, right).unwrap()
        )
        .unwrap()
    );
    assert_eq!(ring.add(left, &ring.zero()).unwrap(), left.clone());
    assert_eq!(ring.mul(left, &ring.one()).unwrap(), left.clone());
    assert_eq!(
        ring.add(left, &ring.neg(left).unwrap()).unwrap(),
        ring.zero()
    );
}

fn qadic_element(ring: &QadicRing, first: i16, second: i16) -> QadicElement {
    let coefficients = [
        ring.coefficient_ring().element(first),
        ring.coefficient_ring().element(second),
    ];
    ring.element_from_coefficients(&coefficients).unwrap()
}

fn series_from_coefficients(
    ring: &PowerSeriesRing<noether::IntegerRing>,
    coefficients: &[i16],
) -> PowerSeries<noether::IntegerRing> {
    let bound = ring
        .truncation()
        .bound()
        .expect("weighted-degree truncation has one bound");
    let exponents = (0..bound)
        .flat_map(|first| (0..bound).map(move |second| vec![first, second]))
        .filter(|exponent| ring.truncation().contains_exponent(exponent));
    ring.from_terms(
        exponents
            .zip(coefficients.iter().copied())
            .map(|(exponent, coefficient)| (exponent, Integer::from(coefficient))),
    )
    .unwrap()
}

fn reference_series_product(
    ring: &PowerSeriesRing<noether::IntegerRing>,
    left: &PowerSeries<noether::IntegerRing>,
    right: &PowerSeries<noether::IntegerRing>,
) -> PowerSeries<noether::IntegerRing> {
    let mut terms = BTreeMap::<Vec<u64>, Integer>::new();
    for (left_exponent, left_coefficient) in left.terms() {
        for (right_exponent, right_coefficient) in right.terms() {
            let exponent = left_exponent
                .iter()
                .zip(right_exponent)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>();
            if !ring.truncation().contains_exponent(&exponent) {
                continue;
            }
            let product = left_coefficient * right_coefficient;
            let sum = terms.remove(&exponent).unwrap_or_else(Integer::zero) + product;
            if !sum.is_zero() {
                terms.insert(exponent, sum);
            }
        }
    }
    ring.from_terms(terms).unwrap()
}

proptest! {
    #![proptest_config(property_config(64))]

    #[test]
    fn flint_matrix_routes_match_direct_ring_arithmetic(
        (rows, inner, columns, left, right) in matrix_case(),
        modulus in prop::sample::select(vec![4_u64, 8, 9, 15, 17, 25, 64]),
    ) {
        let integer_left = matrix_from_values(&Z, rows, inner, &left, Integer::from);
        let integer_right = matrix_from_values(&Z, inner, columns, &right, Integer::from);
        assert_matmul_route(&integer_left, &integer_right);

        let rational_left = matrix_from_values(&Q, rows, inner, &left, rational);
        let rational_right = matrix_from_values(&Q, inner, columns, &right, rational);
        assert_matmul_route(&rational_left, &rational_right);

        let residue = ResidueRing::new(modulus).unwrap();
        let residue_left = matrix_from_values(&residue, rows, inner, &left, |value| residue.element(value));
        let residue_right = matrix_from_values(&residue, inner, columns, &right, |value| residue.element(value));
        assert_matmul_route(&residue_left, &residue_right);

        for finite in [
            FiniteField::extension(5, 2).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
        ] {
            let finite_left = matrix_from_values(&finite, rows, inner, &left, |value| extension_element(&finite, value));
            let finite_right = matrix_from_values(&finite, inner, columns, &right, |value| extension_element(&finite, value));
            assert_matmul_route(&finite_left, &finite_right);
        }

        let padic = PadicRing::new(5_u8, 4).unwrap();
        let padic_left = matrix_from_values(&padic, rows, inner, &left, |value| padic.element(value));
        let padic_right = matrix_from_values(&padic, inner, columns, &right, |value| padic.element(value));
        assert_matmul_route(&padic_left, &padic_right);
    }

    #[test]
    fn determinant_and_charpoly_routes_match_degree_three_formulas(
        entries in vec(-20_i16..=20, 9),
        modulus in prop::sample::select(vec![4_u64, 8, 9, 15, 17, 25]),
    ) {
        let integers = matrix_from_values(&Z, 3, 3, &entries, Integer::from);
        assert_degree_three_characteristic_polynomial(&integers);

        let rationals = matrix_from_values(&Q, 3, 3, &entries, rational);
        assert_degree_three_characteristic_polynomial(&rationals);

        let residue = ResidueRing::new(modulus).unwrap();
        let residues = matrix_from_values(&residue, 3, 3, &entries, |value| residue.element(value));
        assert_degree_three_characteristic_polynomial(&residues);

        for finite in [
            FiniteField::extension(5, 2).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
        ] {
            let finite_entries = matrix_from_values(&finite, 3, 3, &entries, |value| extension_element(&finite, value));
            assert_degree_three_characteristic_polynomial(&finite_entries);
        }

        let padic = PadicRing::new(5_u8, 4).unwrap();
        let padic_entries = matrix_from_values(&padic, 3, 3, &entries, |value| padic.element(value));
        assert_degree_three_characteristic_polynomial(&padic_entries);
    }

    #[test]
    fn fixed_precision_padic_arithmetic_matches_quotient_and_restriction_maps(
        prime in prop::sample::select(vec![2_u64, 3, 5, 7]),
        precision in 2_u64..=6,
        low_precision in 1_u64..=6,
        left in -10_000_i64..=10_000,
        middle in -10_000_i64..=10_000,
        right in -10_000_i64..=10_000,
    ) {
        let ring = PadicRing::new(prime, precision).unwrap();
        let left = ring.element(left);
        let middle = ring.element(middle);
        let right = ring.element(right);
        assert_ring_laws(&ring, &left, &middle, &right);

        let quotient = ResidueRing::new(ring.modulus().clone()).unwrap();
        let quotient_left = quotient.element(left.representative().unwrap());
        let quotient_middle = quotient.element(middle.representative().unwrap());
        assert_eq!(
            ring.add(&left, &middle).unwrap().representative().unwrap(),
            quotient.add(&quotient_left, &quotient_middle).unwrap().representative().clone()
        );
        assert_eq!(
            ring.mul(&left, &middle).unwrap().representative().unwrap(),
            quotient.mul(&quotient_left, &quotient_middle).unwrap().representative().clone()
        );
        if left.is_unit() {
            assert_eq!(ring.mul(&left, &left.inverse().unwrap()).unwrap(), ring.one());
        }

        let low_precision = low_precision.min(precision);
        let restricted_left = ring.restrict(&left, low_precision).unwrap();
        let restricted_middle = ring.restrict(&middle, low_precision).unwrap();
        let restricted_ring = restricted_left.ring().unwrap();
        assert_eq!(
            ring.restrict(&ring.add(&left, &middle).unwrap(), low_precision).unwrap(),
            restricted_ring.add(&restricted_left, &restricted_middle).unwrap()
        );
        assert_eq!(
            ring.restrict(&ring.mul(&left, &middle).unwrap(), low_precision).unwrap(),
            restricted_ring.mul(&restricted_left, &restricted_middle).unwrap()
        );
        assert_eq!(
            PadicElement::from_json(&left.to_json().unwrap()).unwrap(),
            left
        );
    }

    #[test]
    fn finite_and_qadic_extensions_preserve_field_and_precision_laws(
        coefficients in vec(-2_000_i16..=2_000, 6),
        low_precision in 1_u64..=5,
    ) {
        let residue_field = FiniteField::extension(3, 2).unwrap();
        let finite_values = coefficients
            .chunks(2)
            .map(|pair| extension_element(&residue_field, i16::wrapping_add(pair[0], pair[1].wrapping_mul(3))))
            .collect::<Vec<_>>();
        assert_ring_laws(&residue_field, &finite_values[0], &finite_values[1], &finite_values[2]);
        if !finite_values[0].is_zero() {
            assert_eq!(
                residue_field.mul(&finite_values[0], &finite_values[0].inverse().unwrap()).unwrap(),
                residue_field.one()
            );
        }
        assert_eq!(finite_values[0].frobenius(2), finite_values[0]);
        assert_eq!(
            residue_field.add(&finite_values[0], &finite_values[1]).unwrap().trace(),
            residue_field.add(&finite_values[0].trace(), &finite_values[1].trace()).unwrap()
        );
        assert_eq!(
            residue_field.mul(&finite_values[0], &finite_values[1]).unwrap().norm(),
            residue_field.mul(&finite_values[0].norm(), &finite_values[1].norm()).unwrap()
        );
        assert_eq!(
            FiniteFieldElement::from_json(&finite_values[0].to_json().unwrap()).unwrap(),
            finite_values[0]
        );

        let ring = QadicRing::new(residue_field.clone(), 5).unwrap();
        let left = qadic_element(&ring, coefficients[0], coefficients[1]);
        let middle = qadic_element(&ring, coefficients[2], coefficients[3]);
        let right = qadic_element(&ring, coefficients[4], coefficients[5]);
        assert_ring_laws(&ring, &left, &middle, &right);
        if left.is_unit() {
            assert_eq!(ring.mul(&left, &left.inverse().unwrap()).unwrap(), ring.one());
        }
        assert_eq!(left.frobenius(2), left);
        assert_eq!(
            ring.add(&left, &middle).unwrap().frobenius(1),
            ring.add(&left.frobenius(1), &middle.frobenius(1)).unwrap()
        );
        assert_eq!(
            ring.mul(&left, &middle).unwrap().frobenius(1),
            ring.mul(&left.frobenius(1), &middle.frobenius(1)).unwrap()
        );
        assert_eq!(
            ring.residue(&ring.add(&left, &middle).unwrap()).unwrap(),
            residue_field
                .add(&ring.residue(&left).unwrap(), &ring.residue(&middle).unwrap())
                .unwrap()
        );

        let restricted_left = ring.restrict(&left, low_precision).unwrap();
        let restricted_middle = ring.restrict(&middle, low_precision).unwrap();
        let restricted_ring = restricted_left.ring().unwrap();
        assert_eq!(
            ring.restrict(&ring.add(&left, &middle).unwrap(), low_precision).unwrap(),
            restricted_ring.add(&restricted_left, &restricted_middle).unwrap()
        );
        assert_eq!(
            ring.restrict(&ring.mul(&left, &middle).unwrap(), low_precision).unwrap(),
            restricted_ring.mul(&restricted_left, &restricted_middle).unwrap()
        );
        assert_eq!(QadicElement::from_json(&left.to_json().unwrap()).unwrap(), left);
    }

    #[test]
    fn weighted_series_multiplication_matches_sparse_convolution(
        first_weight in 1_u64..=3,
        second_weight in 1_u64..=3,
        bound in 1_u64..=7,
        left in vec(-5_i16..=5, 49),
        middle in vec(-5_i16..=5, 49),
        right in vec(-5_i16..=5, 49),
    ) {
        let grading = WeightedGrading::new(vec![first_weight, second_weight]).unwrap();
        let ring = PowerSeriesRing::new(
            Z,
            ["x", "y"],
            SeriesTruncation::graded(grading, bound).unwrap(),
        )
        .unwrap();
        let left = series_from_coefficients(&ring, &left);
        let middle = series_from_coefficients(&ring, &middle);
        let right = series_from_coefficients(&ring, &right);
        let product = ring.mul(&left, &middle).unwrap();
        assert_eq!(product, reference_series_product(&ring, &left, &middle));
        assert_ring_laws(&ring, &left, &middle, &right);
        assert_eq!(
            PowerSeries::<noether::IntegerRing>::from_json(&product.to_json().unwrap()).unwrap(),
            product
        );
    }
}
