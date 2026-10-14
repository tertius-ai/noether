// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    MonomialOrder, Polynomial, PolynomialError, PolynomialRing, ResidueRing, WeightedGrading,
};
use noether::linalg::Matrix;
use noether::numthy::{FiniteField, PadicRing, QadicRing};
use noether::{CommutativeRing, Integer, Ring, SemanticType, UnitInversion, Q, Z};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn integer_polynomial_ring(variables: &[&str]) -> PolynomialRing<noether::IntegerRing> {
    PolynomialRing::new(
        Z,
        variables.iter().copied(),
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap()
}

#[test]
fn context_identity_carries_order_variables_and_grading() {
    let standard = integer_polynomial_ring(&["x", "y"]);
    let lex = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let weighted = PolynomialRing::with_grading(
        Z,
        ["x", "y"],
        MonomialOrder::GradedReverseLexicographic,
        WeightedGrading::new(vec![1, 2]).unwrap(),
    )
    .unwrap();

    assert_eq!(standard.variables(), &["x", "y"]);
    assert_eq!(standard.term_order().name(), "grevlex");
    assert_eq!(
        standard
            .weighted_grading()
            .expect("standard grading has a weighted view")
            .weights(),
        &[1, 1]
    );
    assert_ne!(
        standard.canonical_bytes().unwrap(),
        lex.canonical_bytes().unwrap()
    );
    assert_ne!(
        standard.canonical_bytes().unwrap(),
        weighted.canonical_bytes().unwrap()
    );
    assert!(PolynomialRing::new(Z, ["x", "x"], MonomialOrder::default()).is_err());
    assert!(PolynomialRing::with_grading(
        Z,
        ["x", "y"],
        MonomialOrder::default(),
        WeightedGrading::standard(1).unwrap(),
    )
    .is_err());
}

#[test]
fn monomial_orders_have_revalidated_typed_json() {
    for order in [
        MonomialOrder::Lexicographic,
        MonomialOrder::GradedLexicographic,
        MonomialOrder::GradedReverseLexicographic,
    ] {
        assert_eq!(
            MonomialOrder::from_json(&order.to_json().unwrap()).unwrap(),
            order
        );
    }
    assert!(MonomialOrder::from_json(
        r#"{"type":"noether/commalg/MonomialOrder@1","name":"unknown"}"#,
    )
    .is_err());
}

#[test]
fn block_and_elimination_orders_are_exact_context_identity() {
    let block = MonomialOrder::block([
        (vec![0], MonomialOrder::Lexicographic),
        (vec![1, 2], MonomialOrder::GradedReverseLexicographic),
    ])
    .unwrap();
    assert_eq!(block.name(), "block");
    assert_eq!(block.blocks().unwrap().len(), 2);
    assert_eq!(
        MonomialOrder::from_json(&block.to_json().unwrap()).unwrap(),
        block
    );
    assert_eq!(MonomialOrder::elimination(3, [0]).unwrap(), block);

    let ring = PolynomialRing::new(Z, ["x", "y", "z"], block.clone()).unwrap();
    let polynomial = ring
        .from_terms([
            (vec![1.into(), 0.into(), 0.into()], 1.into()),
            (vec![0.into(), 0.into(), 20.into()], 1.into()),
        ])
        .unwrap();
    assert_eq!(
        polynomial.terms().next().unwrap().0,
        [Integer::one(), Integer::zero(), Integer::zero()]
    );
    assert_eq!(ring.term_order(), block);

    let incomplete = MonomialOrder::block([(vec![0], MonomialOrder::Lexicographic)]).unwrap();
    assert!(PolynomialRing::new(Z, ["x", "y"], incomplete).is_err());
}

fn assert_polynomial_json_round_trip<R>(ring: &PolynomialRing<R>, value: &Polynomial<R>)
where
    R: CommutativeRing + Serialize + DeserializeOwned,
    R::Element: Serialize + DeserializeOwned,
{
    assert_eq!(
        PolynomialRing::<R>::from_json(&ring.to_json().unwrap()).unwrap(),
        *ring
    );
    assert_eq!(
        Polynomial::<R>::from_json(&value.to_json().unwrap()).unwrap(),
        *value
    );
}

#[test]
fn generic_polynomial_json_covers_every_accepted_coefficient_context() {
    let integers = PolynomialRing::with_grading(
        Z,
        ["x", "y"],
        MonomialOrder::GradedReverseLexicographic,
        WeightedGrading::new(vec![1, 2]).unwrap(),
    )
    .unwrap();
    let integer_value = integers
        .from_u64_terms([(vec![2, 0], 3.into()), (vec![0, 1], 5.into())])
        .unwrap();
    assert_polynomial_json_round_trip(&integers, &integer_value);

    let rationals = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic).unwrap();
    let rational_value = rationals
        .from_u64_terms([(vec![1], noether::Rational::from(2_i8))])
        .unwrap();
    assert_polynomial_json_round_trip(&rationals, &rational_value);

    let residue = ResidueRing::new(12_u8).unwrap();
    let residues = PolynomialRing::new(residue.clone(), ["t"], MonomialOrder::default()).unwrap();
    let residue_value = residues
        .from_u64_terms([(vec![1], residue.element(5_u8))])
        .unwrap();
    assert_polynomial_json_round_trip(&residues, &residue_value);

    let finite = FiniteField::extension(3, 2).unwrap();
    let finite_polynomials =
        PolynomialRing::new(finite.clone(), ["t"], MonomialOrder::default()).unwrap();
    let finite_value = finite_polynomials
        .from_u64_terms([(vec![1], finite.generator())])
        .unwrap();
    assert_polynomial_json_round_trip(&finite_polynomials, &finite_value);

    let padic = PadicRing::new(3_u8, 4).unwrap();
    let padic_polynomials =
        PolynomialRing::new(padic.clone(), ["t"], MonomialOrder::default()).unwrap();
    let padic_value = padic_polynomials
        .from_u64_terms([(vec![1], padic.element(-1_i8))])
        .unwrap();
    assert_polynomial_json_round_trip(&padic_polynomials, &padic_value);

    let qadic = QadicRing::new(finite, 4).unwrap();
    let qadic_polynomials =
        PolynomialRing::new(qadic.clone(), ["t"], MonomialOrder::default()).unwrap();
    let qadic_value = qadic_polynomials
        .from_u64_terms([(vec![1], qadic.generator())])
        .unwrap();
    assert_polynomial_json_round_trip(&qadic_polynomials, &qadic_value);

    let mut unordered: serde_json::Value =
        serde_json::from_str(&integer_value.to_json().unwrap()).unwrap();
    unordered["terms"].as_array_mut().unwrap().reverse();
    assert!(Polynomial::<noether::IntegerRing>::from_json(&unordered.to_string()).is_err());

    let mut zero_term: serde_json::Value =
        serde_json::from_str(&integer_value.to_json().unwrap()).unwrap();
    zero_term["terms"][0]["coefficient"] = serde_json::Value::String("0".to_owned());
    assert!(Polynomial::<noether::IntegerRing>::from_json(&zero_term.to_string()).is_err());
}

#[test]
fn sparse_arithmetic_derivative_evaluation_and_substitution_are_generic() {
    let ring = integer_polynomial_ring(&["x", "y"]);
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let f = (&x.pow(2).unwrap() + &y).unwrap();
    let one = ring.one();
    let product = (&(&x + &one).unwrap() * &(&y + &one).unwrap()).unwrap();

    assert_eq!(f.degree(None).unwrap(), Some(Integer::from(2_u8)));
    assert_eq!(f.degree(Some(1)).unwrap(), Some(Integer::from(1_u8)));
    assert_eq!(
        f.derivative(0)
            .unwrap()
            .evaluate(&[3.into(), 5.into()])
            .unwrap(),
        Integer::from(6_u8)
    );
    assert_eq!(
        f.evaluate(&[3.into(), 5.into()]).unwrap(),
        Integer::from(14_u8)
    );
    assert_eq!(
        f.substitute(&[y.clone(), x.clone()]).unwrap(),
        (&y.pow(2).unwrap() + &x).unwrap()
    );
    assert_eq!(product.nonzero_term_count(), 4);
    assert_eq!(product.constant_coefficient(), Integer::one());
    assert!(matches!(
        f.evaluate(&[Integer::one()]),
        Err(PolynomialError::ImageArity { .. })
    ));
}

#[test]
fn machine_word_construction_canonicalizes_duplicates_before_native_storage() {
    let ring = integer_polynomial_ring(&["x", "y"]);
    let value = ring
        .from_u64_terms([
            (vec![1, 0], Integer::from(2)),
            (vec![0, 1], Integer::from(5)),
            (vec![1, 0], Integer::from(3)),
            (vec![0, 1], Integer::from(-5)),
        ])
        .unwrap();

    assert_eq!(
        value,
        ring.generator(0)
            .unwrap()
            .scalar_mul(&Integer::from(5))
            .unwrap()
    );
}

#[test]
fn exponent_storage_is_exact_beyond_machine_word_arithmetic() {
    let ring = integer_polynomial_ring(&["x"]);
    let huge = Integer::from(u64::MAX);
    let monomial = ring
        .from_terms([(vec![huge.clone()], Integer::one())])
        .unwrap();
    let x = ring.generator(0).unwrap();
    let product = (&monomial * &x).unwrap();
    let expected = &huge + &Integer::one();

    assert_eq!(product.degree(Some(0)).unwrap(), Some(expected.clone()));
    assert_eq!(product.coefficient(&[expected]).unwrap(), Integer::one());
}

#[test]
fn one_grading_splits_polynomials_and_bounds_series_compatibly() {
    let ring = PolynomialRing::with_grading(
        Z,
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
        WeightedGrading::new(vec![1, 2, 3]).unwrap(),
    )
    .unwrap();
    let [x, y, z] = ring.generators().unwrap().try_into().unwrap();
    let homogeneous = (&(&x.pow(2).unwrap() + &y).unwrap() + &z).unwrap();
    let mixed = (&homogeneous + &ring.one()).unwrap();

    assert_eq!(homogeneous.homogeneous_degree().unwrap(), None);
    let degree_two = (&x.pow(2).unwrap() + &y).unwrap();
    assert_eq!(
        degree_two.homogeneous_degree().unwrap(),
        Some(Integer::from(2_u8))
    );
    let components = mixed.homogeneous_components().unwrap();
    assert_eq!(components.len(), 3);
    assert_eq!(components[0].0, Integer::zero());
    assert_eq!(components[1].0, Integer::from(2_u8));
    assert_eq!(components[2].0, Integer::from(3_u8));
}

#[test]
fn matrices_exist_automatically_over_polynomial_contexts() {
    let field = FiniteField::extension(3_u64, 2).unwrap();
    let ring =
        PolynomialRing::new(field, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let matrix = Matrix::from_rows(
        ring.clone(),
        [[ring.one(), x.clone()], [x.clone(), ring.one()]],
    )
    .unwrap();
    let square = matrix.matmul(&matrix).unwrap();

    assert_eq!(square.shape(), (2, 2));
    assert_eq!(square.get(0, 1).unwrap(), (&x + &x).unwrap());
}

#[test]
fn accepted_local_coefficients_and_constant_unit_inversion_need_no_new_type() {
    let padics = PadicRing::new(3_u8, 4).unwrap();
    let padic_polynomials = PolynomialRing::new(
        padics.clone(),
        ["t"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let value = padic_polynomials
        .from_u64_terms([
            (vec![0], padics.element(2_u8)),
            (vec![1], padics.element(3_u8)),
        ])
        .unwrap();
    assert_eq!(
        value
            .evaluate(&[padics.element(2_u8)])
            .unwrap()
            .representative(),
        Some(Integer::from(8_u8))
    );

    let residue_field = FiniteField::extension(3_u64, 2).unwrap();
    let qadic = QadicRing::new(residue_field, 3).unwrap();
    let qadic_polynomials =
        PolynomialRing::new(qadic, ["u"], MonomialOrder::GradedReverseLexicographic).unwrap();
    assert_eq!(
        qadic_polynomials
            .generator(0)
            .unwrap()
            .degree(None)
            .unwrap(),
        Some(1_u8.into())
    );

    let rational_polynomials =
        PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let unit = rational_polynomials
        .constant(noether::Rational::from(2_i8))
        .unwrap();
    assert_eq!(
        rational_polynomials
            .mul(&unit, &rational_polynomials.inverse_unit(&unit).unwrap())
            .unwrap(),
        rational_polynomials.one()
    );
}

#[test]
fn univariate_field_division_is_exact_and_context_checked() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let dividend = (&x.pow(3).unwrap() - &ring.one()).unwrap();
    let divisor = (&x - &ring.one()).unwrap();
    let expected = (&(&x.pow(2).unwrap() + &x).unwrap() + &ring.one()).unwrap();

    let (quotient, remainder) = dividend.div_rem(&divisor).unwrap();
    assert_eq!(quotient, expected);
    assert!(remainder.is_zero());
    assert_eq!(dividend.exact_div(&divisor).unwrap(), expected);
    assert!(matches!(
        dividend.div_rem(&ring.zero()),
        Err(PolynomialError::DivisionByZero)
    ));

    let other_ring =
        PolynomialRing::new(Q, ["t"], MonomialOrder::GradedReverseLexicographic).unwrap();
    assert!(matches!(
        dividend.div_rem(&other_ring.one()),
        Err(PolynomialError::ContextMismatch(_))
    ));
}

#[test]
fn univariate_gcd_is_monic_over_rational_and_finite_fields() {
    let rational_ring =
        PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = rational_ring.generator(0).unwrap();
    let shared = (&x - &rational_ring.one()).unwrap();
    let left = (&shared * &(&x + &rational_ring.one()).unwrap()).unwrap();
    let right = (&shared * &(&x + &rational_ring.constant(2.into()).unwrap()).unwrap()).unwrap();
    assert_eq!(left.gcd(&right).unwrap(), shared);
    assert!(rational_ring
        .zero()
        .gcd(&rational_ring.zero())
        .unwrap()
        .is_zero());

    let field = FiniteField::extension(3_u64, 2).unwrap();
    let finite_ring =
        PolynomialRing::new(field, ["t"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let t = finite_ring.generator(0).unwrap();
    let factor = (&t + &finite_ring.one()).unwrap();
    let repeated = (&factor.pow(2).unwrap() * &t).unwrap();
    assert_eq!(repeated.gcd(&factor).unwrap(), factor);
}

#[test]
fn univariate_integer_exact_division_and_gcd_include_content() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let common = (&x.scalar_mul(&2.into()).unwrap() + &ring.constant(2.into()).unwrap()).unwrap();
    let left_cofactor = (&x.scalar_mul(&3.into()).unwrap() + &ring.one()).unwrap();
    let right_cofactor =
        (&x.scalar_mul(&5.into()).unwrap() - &ring.constant(3.into()).unwrap()).unwrap();
    let left = (&common * &left_cofactor).unwrap();
    let right = (&common * &right_cofactor).unwrap();

    assert_eq!(left.exact_div(&common).unwrap(), left_cofactor);
    assert_eq!(left.gcd(&right).unwrap(), common);
    assert_eq!(
        x.scalar_mul(&6.into())
            .unwrap()
            .gcd(&x.scalar_mul(&10.into()).unwrap())
            .unwrap(),
        x.scalar_mul(&2.into()).unwrap()
    );
    assert!(ring.zero().gcd(&ring.zero()).unwrap().is_zero());
    assert!(matches!(
        x.exact_div(&(&x.scalar_mul(&2.into()).unwrap() + &ring.one()).unwrap()),
        Err(PolynomialError::NonExactDivision)
    ));
}

#[test]
fn native_field_gcd_accepts_multivariate_contexts_and_exact_division_rejects_remainders() {
    let multivariate =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let [x, y] = multivariate.generators().unwrap().try_into().unwrap();
    let shared = (&x + &y).unwrap();
    let left = (&shared * &(&x + &multivariate.one()).unwrap()).unwrap();
    let right = (&shared * &(&y + &multivariate.one()).unwrap()).unwrap();
    assert_eq!(left.gcd(&right).unwrap(), shared);

    let univariate =
        PolynomialRing::new(Q, ["t"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let t = univariate.generator(0).unwrap();
    assert!(matches!(
        t.exact_div(&(&t + &univariate.one()).unwrap()),
        Err(PolynomialError::NonExactDivision)
    ));
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn polynomial_facade_is_thread_safe() {
    assert_send_sync::<PolynomialRing<noether::IntegerRing>>();
    assert_send_sync::<noether::commalg::Polynomial<noether::IntegerRing>>();
}
