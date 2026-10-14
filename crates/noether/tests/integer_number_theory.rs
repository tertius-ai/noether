// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::numthy::{
    cyclotomic_polynomial, factor, is_prime, partial_factor, CofactorClassification,
    IntegerFactorization, PartialIntegerFactorization,
};
use noether::{Integer, SemanticType};

#[test]
fn factorization_preserves_sign_and_reconstructs() {
    let value = Integer::from(-360);
    let result = factor(&value).unwrap();
    assert_eq!(result.unit(), -1);
    assert_eq!(
        result
            .factors()
            .iter()
            .map(|(prime, exponent)| (prime.to_string(), *exponent))
            .collect::<Vec<_>>(),
        [
            ("2".to_owned(), 3),
            ("3".to_owned(), 2),
            ("5".to_owned(), 1)
        ]
    );
    assert_eq!(result.value(), value);
    assert!(result.verify(&Integer::from(-360)));
    assert!(!result.verify(&Integer::from(360)));
}

#[test]
fn units_and_zero_have_honest_factorization_behavior() {
    for unit in [-1, 1] {
        let result = factor(&Integer::from(unit)).unwrap();
        assert_eq!(result.unit(), unit as i8);
        assert!(result.factors().is_empty());
        assert_eq!(result.value(), Integer::from(unit));
    }
    assert!(factor(&Integer::zero()).is_err());
}

#[test]
fn factorization_json_is_typed_and_validated() {
    let result = factor(&Integer::from(-360)).unwrap();
    let json = result.to_json().unwrap();
    assert_eq!(
        json,
        r#"{"unit":-1,"factors":[{"prime":"2","exponent":3},{"prime":"3","exponent":2},{"prime":"5","exponent":1}]}"#
    );
    assert_eq!(IntegerFactorization::from_json(&json).unwrap(), result);
    assert_eq!(
        IntegerFactorization::TYPE_ID,
        "noether/numthy/IntegerFactorization@1"
    );
    assert!(IntegerFactorization::from_json(
        r#"{"unit":1,"factors":[{"prime":"4","exponent":1}]}"#
    )
    .is_err());
    assert!(IntegerFactorization::from_json(
        r#"{"unit":1,"factors":[{"prime":"3","exponent":1},{"prime":"2","exponent":1}]}"#
    )
    .is_err());
}

#[test]
fn primality_is_proved_not_probabilistic_status() {
    assert!(is_prime(&Integer::from(2)));
    assert!(is_prime(&Integer::from(104_729)));
    assert!(!is_prime(&Integer::from(1)));
    assert!(!is_prime(&Integer::from(-2)));
    assert!(!is_prime(&Integer::from(1_105)));
}

#[test]
fn cyclotomic_polynomials_use_arbitrary_precision_integer_coefficients() {
    let phi_twelve = cyclotomic_polynomial(12).unwrap();
    let terms = phi_twelve
        .terms()
        .map(|(exponent, coefficient)| (exponent[0].to_string(), coefficient.to_string()))
        .collect::<Vec<_>>();
    assert_eq!(
        terms,
        [
            ("4".to_owned(), "1".to_owned()),
            ("2".to_owned(), "-1".to_owned()),
            ("0".to_owned(), "1".to_owned())
        ]
    );

    let phi_105 = cyclotomic_polynomial(105).unwrap();
    assert!(phi_105
        .terms()
        .any(|(_, coefficient)| coefficient == &Integer::from(-2)));
    assert!(cyclotomic_polynomial(0).is_err());
}

#[test]
fn partial_factorization_reports_only_proved_information() {
    let p127 = &Integer::from(2).pow(127) - &Integer::one();
    let p107 = &Integer::from(2).pow(107) - &Integer::one();
    let hard_composite = &p127 * &p107;
    let value = &(&Integer::from(-96) * &hard_composite);
    let result = partial_factor(value, 10, 64).unwrap();
    assert_eq!(result.unit(), -1);
    assert_eq!(
        result.factors(),
        &[(Integer::from(2), 5), (Integer::from(3), 1)]
    );
    assert_eq!(result.cofactor(), &hard_composite);
    assert_eq!(
        result.classification(),
        CofactorClassification::ProvedComposite
    );
    assert_eq!(result.value(), *value);
    assert!(result.verify());

    let unclassified = partial_factor(&p127, 100, 64).unwrap();
    assert!(unclassified.factors().is_empty());
    assert_eq!(
        unclassified.classification(),
        CofactorClassification::Unclassified
    );

    let proved_prime = partial_factor(&Integer::from(101), 5, 16).unwrap();
    assert_eq!(
        proved_prime.classification(),
        CofactorClassification::ProvedPrime
    );
}

#[test]
fn partial_factorization_json_recomputes_the_bounded_result() {
    let result = partial_factor(&Integer::from(-360), 5, 64).unwrap();
    let json = result.to_json().unwrap();
    assert_eq!(
        PartialIntegerFactorization::from_json(&json).unwrap(),
        result
    );
    let tampered = json.replacen("\"cofactor\":\"1\"", "\"cofactor\":\"9\"", 1);
    assert!(PartialIntegerFactorization::from_json(&tampered).is_err());
    assert!(partial_factor(&Integer::zero(), 100, 64).is_err());
    assert!(partial_factor(&Integer::one(), 1_000_000_001, 64).is_err());
}
