// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::numthy::{FractionalIdeal, FractionalIdealError, NumberField};
use noether::{Rational, Ring, SemanticType, Q};

fn quadratic(constant: i64, name: &str) -> NumberField {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let polynomial = (&x.pow(2).unwrap() + &ring.constant(constant.into()).unwrap()).unwrap();
    NumberField::new(polynomial, Some(name.to_owned())).unwrap()
}

#[test]
fn principal_fractional_ideals_have_exact_lattice_identity_and_norm() {
    let field = quadratic(1, "i");
    let order = field.maximal_order().unwrap();
    let one_plus_i = field.one().add(&field.generator()).unwrap();
    let ideal = order
        .fractional_ideal(std::slice::from_ref(&one_plus_i))
        .unwrap();
    let same = order
        .fractional_ideal(&[
            one_plus_i.clone(),
            one_plus_i.mul(&field.scalar(3.into())).unwrap(),
        ])
        .unwrap();

    assert_eq!(ideal, same);
    assert_eq!(ideal.order(), &order);
    assert_eq!(ideal.norm(), Rational::from(2));
    assert!(ideal.is_integral());
    assert!(ideal.contains(&one_plus_i));
    assert!(ideal.contains(&field.scalar(2.into())));
    assert!(!ideal.contains(&field.one()));
    assert_eq!(FractionalIdeal::TYPE_ID, "noether/numthy/FractionalIdeal@1");
}

#[test]
fn ideal_sum_product_and_fractional_scaling_are_canonical() {
    let field = quadratic(1, "i");
    let order = field.maximal_order().unwrap();
    let two = order.fractional_ideal(&[field.scalar(2.into())]).unwrap();
    let three = order.fractional_ideal(&[field.scalar(3.into())]).unwrap();
    let unit = order.fractional_ideal(&[field.one()]).unwrap();

    assert_eq!(two.add(&three).unwrap(), unit);
    assert_eq!(two.mul(&three).unwrap().norm(), Rational::from(36));

    let half = Rational::new(1.into(), 2.into()).unwrap();
    let inverse_two = order.fractional_ideal(&[field.scalar(half)]).unwrap();
    assert_eq!(
        inverse_two.norm(),
        Rational::new(1.into(), 4.into()).unwrap()
    );
    assert!(!inverse_two.is_integral());
    assert_eq!((&two * &inverse_two).unwrap(), unit);
}

#[test]
fn ideal_construction_and_arithmetic_fail_closed() {
    let gaussian = quadratic(1, "i");
    let gaussian_order = gaussian.maximal_order().unwrap();
    assert!(matches!(
        gaussian_order.fractional_ideal(&[]),
        Err(FractionalIdealError::ZeroIdeal)
    ));
    assert!(matches!(
        gaussian_order.fractional_ideal(&[gaussian.zero()]),
        Err(FractionalIdealError::ZeroIdeal)
    ));

    let other = quadratic(-5, "r");
    let other_order = other.maximal_order().unwrap();
    assert!(matches!(
        gaussian_order.fractional_ideal(&[other.one()]),
        Err(FractionalIdealError::ContextMismatch(_))
    ));
    let left = gaussian_order.fractional_ideal(&[gaussian.one()]).unwrap();
    let right = other_order.fractional_ideal(&[other.one()]).unwrap();
    assert!(matches!(
        left.add(&right),
        Err(FractionalIdealError::ContextMismatch(_))
    ));
}

#[test]
fn fractional_ideal_json_round_trip_revalidates_the_lattice() {
    let field = quadratic(-5, "r");
    let order = field.maximal_order().unwrap();
    let ideal = order
        .fractional_ideal(&[field.scalar(Rational::new(3.into(), 2.into()).unwrap())])
        .unwrap();

    assert_eq!(
        FractionalIdeal::from_json(&ideal.to_json().unwrap()).unwrap(),
        ideal
    );
    assert!(FractionalIdeal::from_json(r#"{"order":{},"basis":[]}"#).is_err());
}
