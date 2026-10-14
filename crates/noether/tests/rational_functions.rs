// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    MonomialOrder, PolynomialRing, RationalFunction, RationalFunctionError, RationalFunctionField,
};
use noether::linalg::Matrix;
use noether::numthy::FiniteField;
use noether::{Integer, Rational, RationalField, Ring, SemanticType, Q};

#[test]
fn fractions_have_one_canonical_reduced_representation() {
    let field = RationalFunctionField::new(Q, "x").unwrap();
    let polynomials = field.polynomial_ring();
    let x = polynomials.generator(0).unwrap();
    let one = polynomials.one();
    let numerator = (&x.pow(2).unwrap() - &one).unwrap();
    let denominator = (&x - &one).unwrap();
    let value = field.fraction(numerator, denominator).unwrap();
    let expected = (&x + &one).unwrap();

    assert_eq!(value.numerator(), &expected);
    assert!(value.denominator().is_one());
    assert_eq!(value, field.from_polynomial(expected).unwrap());
    assert_eq!(
        value.canonical_bytes().unwrap(),
        value.clone().canonical_bytes().unwrap()
    );
}

#[test]
fn zero_denominators_division_by_zero_and_poles_are_distinct() {
    let field = RationalFunctionField::new(Q, "t").unwrap();
    let polynomials = field.polynomial_ring();
    let t = polynomials.generator(0).unwrap();

    assert!(matches!(
        field.fraction(polynomials.one(), polynomials.zero()),
        Err(RationalFunctionError::ZeroDenominator)
    ));
    assert!(matches!(
        field.one().checked_div(&field.zero()),
        Err(RationalFunctionError::DivisionByZero)
    ));
    assert!(matches!(
        field
            .fraction(polynomials.one(), t)
            .unwrap()
            .specialize(&Rational::from(0_i8)),
        Err(RationalFunctionError::Pole)
    ));
}

#[test]
fn arithmetic_obeys_field_identities_and_cancels_before_growth() {
    let field = RationalFunctionField::new(Q, "x").unwrap();
    let polynomials = field.polynomial_ring();
    let x = field.generator();
    let one = field.one();
    let x_plus_one = (&x + &one).unwrap();
    let x_minus_one = (&x - &one).unwrap();
    let quotient = (&x_plus_one / &x_minus_one).unwrap();

    assert_eq!((&quotient * &x_minus_one).unwrap(), x_plus_one);
    assert_eq!((&quotient + &(-&quotient).unwrap()).unwrap(), field.zero());
    assert_eq!(quotient.pow(0).unwrap(), field.one());
    assert_eq!(
        quotient.pow(3).unwrap(),
        (&(&quotient * &quotient).unwrap() * &quotient).unwrap()
    );
    assert_eq!(
        quotient.specialize(&Rational::from(2_i8)).unwrap(),
        Rational::from(3_i8)
    );

    let numerator = (&polynomials.generator(0).unwrap() + &polynomials.one()).unwrap();
    assert_eq!(field.from_polynomial(numerator).unwrap(), x_plus_one);
}

#[test]
fn unequal_variable_and_coefficient_contexts_fail_closed() {
    let x_field = RationalFunctionField::new(Q, "x").unwrap();
    let t_field = RationalFunctionField::new(Q, "t").unwrap();
    assert!(matches!(
        &x_field.one() + &t_field.one(),
        Err(RationalFunctionError::ContextMismatch(_))
    ));
    assert!(matches!(
        x_field.fraction(
            t_field.polynomial_ring().one(),
            t_field.polynomial_ring().one()
        ),
        Err(RationalFunctionError::ContextMismatch(_))
    ));

    let first = FiniteField::with_modulus(3, vec![1, 0, 1]).unwrap();
    let second = FiniteField::with_modulus(3, vec![2, 1, 1]).unwrap();
    let first_functions = RationalFunctionField::new(first, "u").unwrap();
    let second_functions = RationalFunctionField::new(second, "u").unwrap();
    assert_ne!(first_functions, second_functions);
}

#[test]
fn finite_field_coefficients_and_generic_composition_need_no_new_matrix_type() {
    let coefficients = FiniteField::extension(3, 2).unwrap();
    let functions = RationalFunctionField::new(coefficients, "t").unwrap();
    let t = functions.generator();
    let matrix = Matrix::from_rows(
        functions.clone(),
        [[functions.one(), t.clone()], [t.clone(), functions.one()]],
    )
    .unwrap();
    let square = matrix.matmul(&matrix).unwrap();

    assert_eq!(square.shape(), (2, 2));
    assert_eq!(square.get(0, 1).unwrap(), (&t + &t).unwrap());

    let nested_polynomials =
        PolynomialRing::new(functions, ["y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    assert_eq!(
        nested_polynomials
            .generator(0)
            .unwrap()
            .degree(None)
            .unwrap(),
        Some(Integer::one())
    );
}

#[test]
fn typed_json_revalidates_fields_and_reduced_fractions() {
    let field = RationalFunctionField::new(Q, "x").unwrap();
    let polynomials = field.polynomial_ring();
    let x = polynomials.generator(0).unwrap();
    let numerator = (&x.pow(2).unwrap() - &polynomials.one()).unwrap();
    let denominator = (&x - &polynomials.one()).unwrap();
    let value = field
        .fraction(numerator.clone(), denominator.clone())
        .unwrap();

    assert_eq!(
        RationalFunctionField::<RationalField>::from_json(&field.to_json().unwrap()).unwrap(),
        field
    );
    assert_eq!(
        RationalFunction::<RationalField>::from_json(&value.to_json().unwrap()).unwrap(),
        value
    );

    let mut noncanonical: serde_json::Value =
        serde_json::from_str(&value.to_json().unwrap()).unwrap();
    noncanonical["numerator"] = serde_json::from_str(&numerator.to_json().unwrap()).unwrap();
    noncanonical["denominator"] = serde_json::from_str(&denominator.to_json().unwrap()).unwrap();
    let error = RationalFunction::<RationalField>::from_json(&noncanonical.to_string())
        .unwrap_err()
        .to_string();
    assert!(error.contains("must already be canonical"));

    let coefficients = FiniteField::extension(3, 2).unwrap();
    let finite_field = RationalFunctionField::new(coefficients, "c").unwrap();
    let finite_value = (&finite_field.generator() + &finite_field.one()).unwrap();
    assert_eq!(
        RationalFunctionField::<FiniteField>::from_json(&finite_field.to_json().unwrap()).unwrap(),
        finite_field
    );
    assert_eq!(
        RationalFunction::<FiniteField>::from_json(&finite_value.to_json().unwrap()).unwrap(),
        finite_value
    );
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn rational_function_facade_is_thread_safe() {
    assert_send_sync::<RationalFunctionField<noether::RationalField>>();
    assert_send_sync::<RationalFunction<noether::RationalField>>();
}
