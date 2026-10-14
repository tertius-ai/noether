// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, Polynomial, PolynomialRing};
use noether::linalg::Matrix;
use noether::numthy::{NumberField, NumberFieldElement, NumberFieldError};
use noether::{ComplexBallField, Rational, Ring, SemanticType, Q};

fn polynomial(coefficients: &[Rational], variable: &str) -> Polynomial<noether::RationalField> {
    let ring = PolynomialRing::new(Q, [variable], MonomialOrder::Lexicographic).unwrap();
    ring.from_u64_terms(
        coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| (vec![index as u64], coefficient.clone())),
    )
    .unwrap()
}

fn qi(name: &str) -> NumberField {
    NumberField::new(
        polynomial(&[1.into(), 0.into(), 1.into()], "x"),
        Some(name.to_owned()),
    )
    .unwrap()
}

#[test]
fn defining_polynomial_is_validated_and_generator_names_are_metadata() {
    let first = qi("i");
    let second = qi("j");

    assert_eq!(first, second);
    assert_eq!(first.degree(), 2);
    assert_eq!(first.generator_name(), "i");
    assert_eq!(first.defining_polynomial().ring().variables(), &["i"]);
    assert_eq!(first.generator().coefficients(), vec![0.into(), 1.into()]);
    assert_eq!(NumberField::TYPE_ID, "noether/numthy/NumberField@1");

    let sum = first.generator().add(&second.generator()).unwrap();
    assert_eq!(sum.coefficients(), vec![0.into(), 2.into()]);
    assert_eq!(sum.field(), &first);

    assert!(matches!(
        NumberField::new(
            polynomial(&[Rational::from(-1), 0.into(), 1.into()], "x"),
            None,
        ),
        Err(NumberFieldError::ReducibleDefiningPolynomial)
    ));
    assert!(matches!(
        NumberField::new(polynomial(&[1.into(), 0.into(), 2.into()], "x"), None),
        Err(NumberFieldError::DefiningPolynomialNotMonic)
    ));

    let multivariate_ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let x = multivariate_ring.generator(0).unwrap();
    let multivariate = (&x.pow(2).unwrap() + &multivariate_ring.one()).unwrap();
    assert!(matches!(
        NumberField::new(multivariate, None),
        Err(NumberFieldError::DefiningPolynomialNotUnivariate { variables: 2 })
    ));
}

#[test]
fn arithmetic_norm_trace_and_polynomials_are_exact() {
    let field = qi("i");
    let i = field.generator();
    let value = field.element(&[2.into(), 3.into()]).unwrap();

    assert_eq!(i.mul(&i).unwrap(), field.scalar(Rational::from(-1)));
    assert_eq!(value.trace(), Rational::from(4));
    assert_eq!(value.norm(), Rational::from(13));
    assert_eq!(value.trace(), Rational::from(4));
    assert_eq!(value.norm(), Rational::from(13));
    assert_eq!(value.inverse().unwrap().mul(&value).unwrap(), field.one());
    assert_eq!(value.pow(0).unwrap(), field.one());
    assert_eq!(
        value.pow(3).unwrap(),
        value.mul(&value).unwrap().mul(&value).unwrap()
    );
    assert!(field.zero().inverse().is_err());

    let minimal = i.minimal_polynomial().unwrap();
    let characteristic = i.characteristic_polynomial().unwrap();
    assert_eq!(minimal, polynomial(&[1.into(), 0.into(), 1.into()], "t"));
    assert_eq!(characteristic, minimal);
    assert_eq!(i.minimal_polynomial().unwrap(), minimal);
    assert_eq!(i.characteristic_polynomial().unwrap(), minimal);

    let rational_field = NumberField::new(
        polynomial(
            &[
                Rational::new(1.into(), 2.into()).unwrap(),
                0.into(),
                1.into(),
            ],
            "x",
        ),
        Some("a".to_owned()),
    )
    .unwrap();
    assert_eq!(rational_field.generator().norm().to_string(), "1/2");
}

#[test]
fn signatures_embeddings_and_json_are_certified_and_self_contained() {
    let imaginary = qi("i");
    assert_eq!(imaginary.signature().unwrap(), (0, 1));

    let real = NumberField::new(
        polynomial(&[Rational::from(-2), 0.into(), 1.into()], "x"),
        Some("r".to_owned()),
    )
    .unwrap();
    assert_eq!(real.signature().unwrap(), (2, 0));

    let balls = imaginary
        .generator()
        .conjugates(ComplexBallField::new(160).unwrap())
        .unwrap();
    assert_eq!(balls.len(), 2);
    for image in balls {
        assert!(image
            .pow(2)
            .overlaps(&ComplexBallField::new(160).unwrap().element(-1))
            .unwrap());
    }

    let encoded_field = imaginary.to_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&encoded_field).unwrap()["type"],
        NumberField::TYPE_ID
    );
    let restored_field = NumberField::from_json(&encoded_field).unwrap();
    assert_eq!(restored_field, imaginary);
    assert_eq!(restored_field.generator_name(), "i");

    let value = imaginary
        .element(&["7/3".parse().unwrap(), (-5).into()])
        .unwrap();
    let restored = NumberFieldElement::from_json(&value.to_json().unwrap()).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&value.to_json().unwrap()).unwrap()["type"],
        NumberFieldElement::TYPE_ID
    );
    assert_eq!(restored, value);
    assert_eq!(restored.field(), &imaginary);
    assert!(NumberField::from_json(r#"{"coefficients":["1","0","1"],"generator":""}"#).is_err());
}

#[test]
fn generic_matrices_exist_over_number_fields_without_another_matrix_type() {
    let field = qi("i");
    let i = field.generator();
    let matrix = Matrix::from_rows(
        field.clone(),
        [[i.clone(), field.one()], [field.one(), i.neg()]],
    )
    .unwrap();
    let square = matrix.matmul(&matrix).unwrap();

    assert_eq!(square, Matrix::zeros(field, 2, 2));
}
