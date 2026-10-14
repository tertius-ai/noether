// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::linalg::Matrix;
use noether::numthy::{NumberField, NumberFieldOrder, NumberFieldOrderError};
use noether::{Integer, Rational, Ring, SemanticType, UnitInversion, Q};

fn quadratic(constant: i64, name: &str) -> NumberField {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let polynomial = (&x.pow(2).unwrap() + &ring.constant(constant.into()).unwrap()).unwrap();
    NumberField::new(polynomial, Some(name.to_owned())).unwrap()
}

#[test]
fn equation_orders_have_exact_integral_bases_and_discriminants() {
    let gaussian = quadratic(1, "i");
    let order = gaussian.equation_order().unwrap();

    assert_eq!(order.field(), &gaussian);
    assert_eq!(order.basis(), vec![gaussian.one(), gaussian.generator()]);
    assert_eq!(order.discriminant().to_string(), "-4");
    assert!(order.contains(&gaussian.element(&[2.into(), 3.into()]).unwrap()));
    assert!(!order.contains(
        &gaussian
            .element(&[Rational::new(1.into(), 2.into()).unwrap()])
            .unwrap()
    ));
    assert_eq!(
        NumberFieldOrder::TYPE_ID,
        "noether/numthy/NumberFieldOrder@1"
    );
}

#[test]
fn explicit_orders_are_validated_and_canonicalized_as_lattices() {
    let field = quadratic(-5, "r");
    let one = field.one();
    let root = field.generator();
    let two = field.scalar(2.into());
    let phi = one.add(&root).unwrap().checked_div(&two).unwrap();
    let maximal = field.order(&[one.clone(), phi.clone()]).unwrap();
    let reordered = field.order(&[phi.add(&one).unwrap(), one.clone()]).unwrap();

    assert_eq!(maximal, reordered);
    assert_eq!(maximal.discriminant(), Integer::from(5));
    assert_eq!(
        field.equation_order().unwrap().index(&maximal).unwrap(),
        Integer::from(2)
    );

    assert!(matches!(
        field.order(std::slice::from_ref(&one)),
        Err(NumberFieldOrderError::WrongBasisLength { .. })
    ));
    assert!(matches!(
        field.order(&[one.clone(), one]),
        Err(NumberFieldOrderError::NotFullRank)
    ));
    assert!(matches!(
        field.order(&[
            field.scalar(2.into()),
            field.scalar(2.into()).mul(&root).unwrap()
        ]),
        Err(NumberFieldOrderError::MissingOne)
    ));
    let half_root = root.checked_div(&two).unwrap();
    assert!(matches!(
        field.order(&[field.one(), half_root]),
        Err(NumberFieldOrderError::NotClosed)
    ));
}

#[test]
fn nonintegral_generators_do_not_masquerade_as_equation_orders() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let half = Rational::new(1.into(), 2.into()).unwrap();
    let polynomial = (&x.pow(2).unwrap() + &ring.constant(half).unwrap()).unwrap();
    let field = NumberField::new(polynomial, Some("a".to_owned())).unwrap();

    assert!(matches!(
        field.equation_order(),
        Err(NumberFieldOrderError::GeneratorNotIntegral)
    ));
    let maximal = field.maximal_order().unwrap();
    assert_eq!(maximal.discriminant(), Integer::from(-8));
    assert_eq!(field.discriminant().unwrap(), Integer::from(-8));
    assert!(maximal.is_maximal().unwrap());
}

#[test]
fn round_two_computes_true_field_discriminants() {
    let gaussian = quadratic(1, "i");
    let gaussian_order = gaussian.maximal_order().unwrap();
    assert_eq!(gaussian_order, gaussian.equation_order().unwrap());
    assert_eq!(gaussian.discriminant().unwrap(), Integer::from(-4));
    assert!(gaussian_order.is_maximal().unwrap());

    let golden = quadratic(-5, "r");
    let equation = golden.equation_order().unwrap();
    let maximal = golden.maximal_order().unwrap();
    assert_eq!(equation.discriminant(), Integer::from(20));
    assert_eq!(maximal.discriminant(), Integer::from(5));
    assert_eq!(equation.index(&maximal).unwrap(), Integer::from(2));
    assert!(!equation.is_maximal().unwrap());
    assert!(maximal.is_maximal().unwrap());

    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let polynomial = ring
        .from_u64_terms([
            (vec![0], (-8).into()),
            (vec![1], (-2).into()),
            (vec![2], (-1).into()),
            (vec![3], 1.into()),
        ])
        .unwrap();
    let cubic = NumberField::new(polynomial, Some("a".to_owned())).unwrap();
    assert_eq!(
        cubic.equation_order().unwrap().discriminant(),
        Integer::from(-2012)
    );
    assert_eq!(cubic.discriminant().unwrap(), Integer::from(-503));
}

#[test]
fn repeated_maximal_order_queries_retain_the_same_checked_order() {
    let field = quadratic(-5, "r");
    let first = field.maximal_order().unwrap();
    let second = field.maximal_order().unwrap();
    assert_eq!(first, second);
    assert_eq!(field.discriminant().unwrap(), first.discriminant());
    assert!(first.is_maximal().unwrap());
    assert!(second.is_maximal().unwrap());

    let restored = NumberField::from_json(&field.to_json().unwrap()).unwrap();
    let restored_order = restored.maximal_order().unwrap();
    assert_eq!(restored_order.discriminant(), first.discriminant());
    assert!(restored_order.is_maximal().unwrap());
}

#[test]
fn orders_are_ring_contexts_for_generic_matrices_and_units() {
    let field = quadratic(1, "i");
    let order = field.equation_order().unwrap();
    let i = field.generator();
    let matrix = Matrix::from_rows(
        order.clone(),
        [[i.clone(), field.one()], [field.one(), i.neg()]],
    )
    .unwrap();

    assert_eq!(
        matrix.matmul(&matrix).unwrap(),
        Matrix::zeros(order.clone(), 2, 2)
    );
    assert_eq!(order.inverse_unit(&i).unwrap(), i.neg());
    assert!(order.inverse_unit(&field.scalar(2.into())).is_err());
}

#[test]
fn order_json_round_trip_revalidates_the_ring() {
    let field = quadratic(-5, "r");
    let phi = field
        .one()
        .add(&field.generator())
        .unwrap()
        .checked_div(&field.scalar(2.into()))
        .unwrap();
    let order = field.order(&[field.one(), phi]).unwrap();

    assert_eq!(
        NumberFieldOrder::from_json(&order.to_json().unwrap()).unwrap(),
        order
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&order.to_json().unwrap()).unwrap()["type"],
        NumberFieldOrder::TYPE_ID
    );
    assert!(NumberFieldOrder::from_json(r#"{"field":{},"basis":[]}"#).is_err());
}
