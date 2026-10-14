// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::linalg::Matrix;
use noether::numthy::{HenselLiftError, PadicElement, PadicRing, PadicValuation};
use noether::{CommutativeRing, FiniteContext, Integer, PrecisionContext, Ring, ValuedRing};

fn assert_context<R: CommutativeRing + PrecisionContext + ValuedRing + FiniteContext>(_ring: &R) {}
fn assert_thread_safe<T: Send + Sync>() {}

#[test]
fn fixed_cap_context_bounds_storage_and_arithmetic() {
    let ring = PadicRing::new(5_u8, 3).unwrap();
    let minus_one = ring.element(-1_i8);
    let two = ring.element(2_u8);

    assert_context(&ring);
    assert_eq!(ring.prime(), &Integer::from(5));
    assert_eq!(ring.absolute_precision(), 3);
    assert_eq!(ring.precision(), &3);
    assert_eq!(ring.modulus(), &Integer::from(125));
    assert_eq!(ring.cardinality(), Integer::from(125));
    assert_eq!(minus_one.representative(), Some(Integer::from(124)));
    assert_eq!(ring.mul(&minus_one, &minus_one).unwrap(), ring.one());
    assert_eq!(ring.mul(&two, &two.inverse().unwrap()).unwrap(), ring.one());
    assert!(ring.element(5_u8).inverse().is_err());
    assert_thread_safe::<PadicRing>();
    assert_thread_safe::<PadicElement>();
}

#[test]
fn valuations_distinguish_exact_values_from_precision_zero() {
    let ring = PadicRing::new(5_u8, 3).unwrap();

    assert_eq!(ring.element(2_u8).valuation(), PadicValuation::Exact(0));
    assert_eq!(ring.element(10_u8).valuation(), PadicValuation::Exact(1));
    assert_eq!(ring.element(25_u8).valuation(), PadicValuation::Exact(2));
    assert_eq!(ring.zero().valuation(), PadicValuation::AtLeast(3));
    assert_eq!(
        ring.element(125_u16).valuation(),
        PadicValuation::AtLeast(3)
    );
    assert_eq!(ring.valuation_lower_bound(&ring.element(25_u8)), Some(2));
    assert_eq!(ring.valuation_lower_bound(&ring.zero()), None);
}

#[test]
fn residue_teichmuller_and_precision_restriction_are_explicit() {
    let ring = PadicRing::new(5_u8, 3).unwrap();
    let residue_two = ring.residue_field().element(2_u8);
    let teichmuller = ring.teichmuller(&residue_two).unwrap();

    assert_eq!(teichmuller.representative(), Some(Integer::from(57)));
    assert_eq!(teichmuller.pow(5), teichmuller);
    assert_eq!(ring.residue(&teichmuller).unwrap(), residue_two);

    let restricted = ring.restrict(&ring.element(124_u8), 2).unwrap();
    assert_eq!(restricted.ring().unwrap().absolute_precision(), 2);
    assert_eq!(restricted.representative(), Some(Integer::from(24)));
    assert_ne!(restricted.ring(), Some(&ring));
    assert!(ring.restrict(&ring.one(), 0).is_err());
    assert!(ring.restrict(&ring.one(), 4).is_err());
}

#[test]
fn simple_roots_hensel_lift_to_the_full_fixed_precision() {
    let ring = PadicRing::new(5_u8, 6).unwrap();
    let polynomials = PolynomialRing::new(ring.clone(), ["x"], MonomialOrder::default()).unwrap();
    let x = polynomials.generator(0).unwrap();
    let six = polynomials.constant(ring.element(6_u8)).unwrap();
    let polynomial = (&x.pow(2).unwrap() - &six).unwrap();
    let residue_root = ring.residue_field().element(1_u8);

    let root = ring.hensel_lift(&polynomial, &residue_root).unwrap();

    assert_eq!(root.representative(), Some(Integer::from(4_891_u16)));
    assert_eq!(root.residue(), Some(residue_root));
    assert_eq!(polynomial.evaluate(&[root]).unwrap(), ring.zero());
}

#[test]
fn hensel_lifting_checks_every_simple_root_hypothesis() {
    let ring = PadicRing::new(5_u8, 4).unwrap();
    let other_ring = PadicRing::new(5_u8, 3).unwrap();
    let other_residue_ring = PadicRing::new(7_u8, 4).unwrap();
    let polynomials = PolynomialRing::new(ring.clone(), ["x"], MonomialOrder::default()).unwrap();
    let x = polynomials.generator(0).unwrap();

    assert!(matches!(
        ring.hensel_lift(
            &(&x.pow(2).unwrap() - &polynomials.one()).unwrap(),
            &ring.residue_field().element(2_u8)
        ),
        Err(HenselLiftError::NotAResidueRoot)
    ));
    assert!(matches!(
        ring.hensel_lift(&x.pow(2).unwrap(), &ring.residue_field().zero()),
        Err(HenselLiftError::SingularDerivative)
    ));
    assert!(matches!(
        ring.hensel_lift(&x, &other_residue_ring.residue_field().zero()),
        Err(HenselLiftError::ContextMismatch(_))
    ));

    let bivariate =
        PolynomialRing::new(ring.clone(), ["x", "y"], MonomialOrder::default()).unwrap();
    assert!(matches!(
        ring.hensel_lift(
            &bivariate.generator(0).unwrap(),
            &ring.residue_field().zero()
        ),
        Err(HenselLiftError::NotUnivariate { arity: 2 })
    ));

    let wrong_coefficients =
        PolynomialRing::new(other_ring, ["x"], MonomialOrder::default()).unwrap();
    assert!(matches!(
        ring.hensel_lift(
            &wrong_coefficients.generator(0).unwrap(),
            &ring.residue_field().zero()
        ),
        Err(HenselLiftError::ContextMismatch(_))
    ));
}

#[test]
fn generic_matrices_exist_over_fixed_cap_padic_integers() {
    let ring = PadicRing::new(3_u8, 4).unwrap();
    let matrix = Matrix::from_rows(
        ring.clone(),
        [
            [ring.element(1_u8), ring.element(2_u8)],
            [ring.element(3_u8), ring.element(4_u8)],
        ],
    )
    .unwrap();
    assert_eq!(matrix.matmul(&matrix).unwrap().shape(), (2, 2));
    assert_eq!(
        matrix.matmul(&matrix).unwrap().get(0, 0).unwrap(),
        ring.element(7_u8)
    );
}

#[test]
fn invalid_contexts_fail_before_claiming_padic_semantics() {
    assert!(PadicRing::new(4_u8, 5).is_err());
    assert!(PadicRing::new(5_u8, 0).is_err());
}

#[test]
fn ring_and_integral_elements_have_revalidated_typed_json() {
    let prime: Integer = "170141183460469231731687303715884105727".parse().unwrap();
    let ring = PadicRing::new(prime, 2).unwrap();
    let value = ring.element(-1_i8);

    assert_eq!(
        PadicRing::from_json(&ring.to_json().unwrap()).unwrap(),
        ring
    );
    assert_eq!(
        PadicElement::from_json(&value.to_json().unwrap()).unwrap(),
        value
    );

    let invalid_prime = ring
        .to_json()
        .unwrap()
        .replace(ring.prime().to_string().as_str(), "4");
    assert!(PadicRing::from_json(&invalid_prime).is_err());

    let noncanonical = value.to_json().unwrap().replace(
        value.representative().unwrap().to_string().as_str(),
        ring.modulus().to_string().as_str(),
    );
    assert!(PadicElement::from_json(&noncanonical).is_err());
}
