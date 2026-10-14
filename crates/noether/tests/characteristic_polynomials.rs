// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing, ResidueRing};
use noether::linalg::{LinalgError, Matrix};
use noether::numthy::FiniteField;
use noether::{Integer, Rational, Ring, Q, Z};

fn rational(numerator: i64, denominator: i64) -> Rational {
    Rational::new(Integer::from(numerator), Integer::from(denominator)).unwrap()
}

#[test]
fn flint_integer_and_rational_charpolys_are_exact() {
    let large = "100000000000000000000000000000000000003"
        .parse::<Integer>()
        .unwrap();
    let integers = Matrix::from_rows(
        Z,
        [
            [large.clone(), Integer::zero()],
            [Integer::zero(), Integer::from(2)],
        ],
    )
    .unwrap();
    let polynomial = integers.charpoly().unwrap();
    assert_eq!(
        polynomial.coefficient(&[Integer::from(2)]).unwrap(),
        Integer::one()
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::one()]).unwrap(),
        -(&large + &Integer::from(2))
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::zero()]).unwrap(),
        &large * &Integer::from(2)
    );

    let rationals = Matrix::from_rows(
        Q,
        [
            [rational(1, 2), Rational::from(1)],
            [Rational::from(0), rational(2, 3)],
        ],
    )
    .unwrap();
    let polynomial = rationals.charpoly().unwrap();
    assert_eq!(
        polynomial.coefficient(&[Integer::from(2)]).unwrap(),
        Rational::one()
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::one()]).unwrap(),
        rational(-7, 6)
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::zero()]).unwrap(),
        rational(1, 3)
    );
}

#[test]
fn finite_ring_and_extension_field_charpolys_are_exact() {
    let residue_ring = ResidueRing::new(25).unwrap();
    let r = |value| residue_ring.element(value);
    let matrix = Matrix::from_rows(
        residue_ring.clone(),
        [[r(1), r(2), r(3)], [r(0), r(4), r(5)], [r(1), r(0), r(6)]],
    )
    .unwrap();
    let polynomial = matrix.charpoly().unwrap();
    let coefficients = (0..=3)
        .map(|degree| {
            polynomial
                .coefficient(&[Integer::from(degree)])
                .unwrap()
                .representative()
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coefficients,
        [3, 6, 14, 1]
            .into_iter()
            .map(Integer::from)
            .collect::<Vec<_>>()
    );

    let field = FiniteField::extension(2_u64, 4).unwrap();
    let a = field.generator();
    let matrix = Matrix::from_rows(
        field.clone(),
        [[a.clone(), field.one()], [field.one(), field.zero()]],
    )
    .unwrap();
    let polynomial = matrix.charpoly().unwrap();
    assert_eq!(
        polynomial.coefficient(&[Integer::from(2)]).unwrap(),
        field.one()
    );
    assert_eq!(polynomial.coefficient(&[Integer::one()]).unwrap(), a);
    assert_eq!(
        polynomial.coefficient(&[Integer::zero()]).unwrap(),
        field.one()
    );
}

#[test]
fn generic_polynomial_coefficients_need_no_new_matrix_family() {
    let coefficient_ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = coefficient_ring.generator(0).unwrap();
    let one = coefficient_ring.one();
    let x_plus_one = coefficient_ring.add(&x, &one).unwrap();
    let matrix = Matrix::from_rows(
        coefficient_ring.clone(),
        [
            [x.clone(), one.clone()],
            [coefficient_ring.zero(), x_plus_one.clone()],
        ],
    )
    .unwrap();

    let polynomial = matrix.charpoly().unwrap();
    let two_x_plus_one = coefficient_ring
        .add(&coefficient_ring.add(&x, &x).unwrap(), &one)
        .unwrap();
    assert_eq!(polynomial.coefficient(&[Integer::from(2)]).unwrap(), one);
    assert_eq!(
        polynomial.coefficient(&[Integer::one()]).unwrap(),
        coefficient_ring.neg(&two_x_plus_one).unwrap()
    );
    assert_eq!(
        polynomial.coefficient(&[Integer::zero()]).unwrap(),
        coefficient_ring.mul(&x, &x_plus_one).unwrap()
    );
}

#[test]
fn empty_and_nonsquare_conventions_are_explicit() {
    assert!(Matrix::zeros(Z, 0, 0).charpoly().unwrap().is_one());
    assert!(matches!(
        Matrix::zeros(Z, 2, 3).charpoly(),
        Err(LinalgError::NotSquare {
            operation: "characteristic polynomial",
            shape: (2, 3),
        })
    ));
}
