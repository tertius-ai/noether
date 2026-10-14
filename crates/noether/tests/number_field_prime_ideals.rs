// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{Ideal, MonomialOrder, PolynomialRing, PrimeIdeal};
use noether::numthy::{
    FractionalIdeal, IdealFactorizationError, NumberField, NumberFieldOrder,
    PrimeDecompositionError,
};
use noether::{Integer, Rational, Ring, SemanticType, Q};

fn field_from_coefficients(coefficients: &[i64], name: &str) -> NumberField {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let polynomial = ring
        .from_u64_terms(
            coefficients
                .iter()
                .enumerate()
                .map(|(exponent, coefficient)| (vec![exponent as u64], (*coefficient).into())),
        )
        .unwrap();
    NumberField::new(polynomial, Some(name.to_owned())).unwrap()
}

fn decomposition_data(primes: &[PrimeIdeal<NumberFieldOrder>]) -> Vec<(usize, usize, Integer)> {
    primes
        .iter()
        .map(|prime| {
            (
                prime.ramification_index(),
                prime.residue_degree(),
                prime.norm(),
            )
        })
        .collect()
}

fn scale_ideal(ideal: &FractionalIdeal, numerator: i64, denominator: i64) -> FractionalIdeal {
    let scalar = ideal
        .order()
        .field()
        .scalar(Rational::new(numerator.into(), denominator.into()).expect("nonzero denominator"));
    let basis = ideal
        .basis()
        .iter()
        .map(|element| element.mul(&scalar).unwrap())
        .collect::<Vec<_>>();
    ideal.order().fractional_ideal(&basis).unwrap()
}

#[test]
fn quadratic_split_inert_and_ramified_primes_are_checked() {
    let field = field_from_coefficients(&[1, 0, 1], "i");
    let order = field.maximal_order().unwrap();

    let split = order.prime_ideals_above(&Integer::from(5)).unwrap();
    assert_eq!(
        decomposition_data(&split),
        vec![(1, 1, 5.into()), (1, 1, 5.into())]
    );
    assert_ne!(split[0], split[1]);

    let inert = order.prime_ideals_above(&Integer::from(3)).unwrap();
    assert_eq!(decomposition_data(&inert), vec![(1, 2, 9.into())]);

    let ramified = order.prime_ideals_above(&Integer::from(2)).unwrap();
    assert_eq!(decomposition_data(&ramified), vec![(2, 1, 2.into())]);

    for prime in split.iter().chain(&inert).chain(&ramified) {
        prime.verify().unwrap();
        assert!(prime.contains(&field.scalar(prime.rational_prime().into())));
        assert_eq!(
            PrimeIdeal::<NumberFieldOrder>::TYPE_ID,
            "noether/commalg/PrimeIdeal@1"
        );
    }
}

#[test]
fn repeated_prime_decomposition_reconstructs_the_same_verified_ideals() {
    let field = field_from_coefficients(&[1, 0, 1], "i");
    let order = field.maximal_order().unwrap();
    let first = order.prime_ideals_above(&5.into()).unwrap();
    let second = order.prime_ideals_above(&5.into()).unwrap();

    assert_eq!(first, second);
    for prime in first.iter().chain(&second) {
        prime.verify().unwrap();
    }
    assert_eq!(
        order.prime_ideals_above(&3.into()).unwrap().len(),
        1,
        "the cache is keyed by the rational prime"
    );
}

#[test]
fn algebra_split_handles_a_prime_dividing_the_equation_order_index() {
    // Dedekind's cubic x^3 - x^2 - 2x - 8 has equation-order index 2.
    let field = field_from_coefficients(&[-8, -2, -1, 1], "a");
    let order = field.maximal_order().unwrap();
    let primes = order.prime_ideals_above(&Integer::from(2)).unwrap();

    assert_eq!(primes.len(), 3);
    assert_eq!(
        decomposition_data(&primes),
        vec![(1, 1, 2.into()), (1, 1, 2.into()), (1, 1, 2.into())]
    );
    assert!(primes.iter().all(|prime| prime.verify().is_ok()));
}

#[test]
fn algebra_split_handles_a_nonintegral_field_presentation() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let half = Rational::new(1.into(), 2.into()).unwrap();
    let polynomial = (&x.pow(2).unwrap() + &ring.constant(half).unwrap()).unwrap();
    let field = NumberField::new(polynomial, Some("a".to_owned())).unwrap();
    let primes = field
        .maximal_order()
        .unwrap()
        .prime_ideals_above(&3.into())
        .unwrap();

    assert_eq!(
        decomposition_data(&primes),
        vec![(1, 1, 3.into()), (1, 1, 3.into())]
    );
    assert!(primes.iter().all(|prime| prime.verify().is_ok()));
}

#[test]
fn prime_ideal_json_round_trip_reverifies_the_field_quotient() {
    let field = field_from_coefficients(&[-5, 0, 1], "r");
    let prime = field
        .maximal_order()
        .unwrap()
        .prime_ideals_above(&3.into())
        .unwrap()
        .remove(0);
    let encoded = prime.to_json().unwrap();
    let restored = PrimeIdeal::<NumberFieldOrder>::from_json(&encoded).unwrap();

    assert_eq!(restored, prime);
    assert_eq!(restored.rational_prime(), Integer::from(3));
    restored.verify().unwrap();
    assert_eq!(
        Ideal::<NumberFieldOrder>::from_json(&prime.ideal().to_json().unwrap()).unwrap(),
        *prime.ideal()
    );
    prime.ideal().verify().unwrap();
}

#[test]
fn invalid_primes_and_nonmaximal_orders_fail_closed() {
    let field = field_from_coefficients(&[-5, 0, 1], "r");
    let equation_order = field.equation_order().unwrap();
    assert!(matches!(
        equation_order.prime_ideals_above(&2.into()),
        Err(PrimeDecompositionError::NonMaximalOrder)
    ));
    let maximal = field.maximal_order().unwrap();
    assert!(matches!(
        maximal.prime_ideals_above(&1.into()),
        Err(PrimeDecompositionError::InvalidRationalPrime(_))
    ));
    assert!(matches!(
        maximal.prime_ideals_above(&6.into()),
        Err(PrimeDecompositionError::InvalidRationalPrime(_))
    ));
}

#[test]
fn higher_degree_and_totally_ramified_cases_preserve_sum_ef() {
    let pure_cubic = field_from_coefficients(&[-2, 0, 0, 1], "c");
    let above_two = pure_cubic
        .maximal_order()
        .unwrap()
        .prime_ideals_above(&2.into())
        .unwrap();
    assert_eq!(decomposition_data(&above_two), vec![(3, 1, 2.into())]);

    for (coefficients, primes) in [
        (&[1, -1, 1, -1, 1][..], &[2, 3, 5, 11][..]),
        (&[1, -1, 1, -1, 1, -1, 1][..], &[2, 3, 7, 29][..]),
    ] {
        let field = field_from_coefficients(coefficients, "z");
        let order = field.maximal_order().unwrap();
        for rational_prime in primes {
            let decomposition = order
                .prime_ideals_above(&Integer::from(*rational_prime))
                .unwrap();
            assert_eq!(
                decomposition
                    .iter()
                    .map(|prime| prime.ramification_index() * prime.residue_degree())
                    .sum::<usize>(),
                field.degree()
            );
            assert!(decomposition.iter().all(|prime| prime.verify().is_ok()));
        }
    }
}

#[test]
fn fractional_ideal_valuations_and_factorization_are_signed_and_reconstructed() {
    let field = field_from_coefficients(&[1, 0, 1], "i");
    let order = field.maximal_order().unwrap();
    let primes = order.prime_ideals_above(&5.into()).unwrap();
    let left_prime = primes[0].as_fractional_ideal();
    let right_prime = primes[1].as_fractional_ideal();

    let integral = left_prime
        .mul(&left_prime)
        .unwrap()
        .mul(&right_prime)
        .unwrap();
    assert_eq!(integral.valuation(&primes[0]).unwrap(), 2);
    assert_eq!(integral.valuation(&primes[1]).unwrap(), 1);
    assert_eq!(
        integral
            .factor()
            .unwrap()
            .iter()
            .map(|(prime, exponent)| (prime.clone(), *exponent))
            .collect::<Vec<_>>(),
        vec![(primes[0].clone(), 2), (primes[1].clone(), 1)]
    );

    // P^2 / (5) = P / Q has norm one, so factoring the rational norm
    // alone would miss both nonzero valuations.
    let norm_one = scale_ideal(&left_prime.mul(&left_prime).unwrap(), 1, 5);
    assert_eq!(norm_one.norm(), Rational::one());
    assert_eq!(norm_one.valuation(&primes[0]).unwrap(), 1);
    assert_eq!(norm_one.valuation(&primes[1]).unwrap(), -1);
    assert_eq!(
        norm_one.factor().unwrap(),
        vec![(primes[0].clone(), 1), (primes[1].clone(), -1)]
    );

    let ramified = order.prime_ideals_above(&2.into()).unwrap().remove(0);
    let inverse_two = order
        .fractional_ideal(&[field.scalar(Rational::new(1.into(), 2.into()).unwrap())])
        .unwrap();
    assert_eq!(inverse_two.valuation(&ramified).unwrap(), -2);
    assert_eq!(inverse_two.factor().unwrap(), vec![(ramified, -2)]);
}

#[test]
fn ideal_factorization_rejects_nonmaximal_and_mismatched_orders() {
    let field = field_from_coefficients(&[-5, 0, 1], "r");
    let equation_order = field.equation_order().unwrap();
    let equation_ideal = equation_order.fractional_ideal(&[field.one()]).unwrap();
    assert!(matches!(
        equation_ideal.factor(),
        Err(IdealFactorizationError::NonMaximalOrder)
    ));

    let gaussian = field_from_coefficients(&[1, 0, 1], "i");
    let gaussian_order = gaussian.maximal_order().unwrap();
    let gaussian_prime = gaussian_order
        .prime_ideals_above(&3.into())
        .unwrap()
        .remove(0);
    let maximal_ideal = field
        .maximal_order()
        .unwrap()
        .fractional_ideal(&[field.one()])
        .unwrap();
    assert!(matches!(
        maximal_ideal.valuation(&gaussian_prime),
        Err(IdealFactorizationError::ContextMismatch(_))
    ));
}
