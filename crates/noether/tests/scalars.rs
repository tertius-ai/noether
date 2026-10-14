// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::{
    BezoutIdentity, EuclideanDomain, Field, Integer, IntegerRing, PrincipalIdealDomain, Rational,
    RationalField, Ring, SemanticType, Q, Z,
};

fn assert_thread_safe<T: Send + Sync>() {}

#[test]
fn exact_scalars_are_thread_safe_flint_values() {
    assert_thread_safe::<Integer>();
    assert_thread_safe::<Rational>();
    assert_eq!(Integer::TYPE_ID, "noether/root/Integer@1");
    assert_eq!(Rational::TYPE_ID, "noether/root/Rational@1");
    assert_eq!(Integer::describe().python_path, Some("builtins.int"));
    assert_eq!(Rational::describe().python_path, Some("fractions.Fraction"));
}

#[test]
fn shared_flint_scalars_support_concurrent_reads() {
    let integer = "100000000000000000000000000000000000003"
        .parse::<Integer>()
        .unwrap();
    let rational = "100000000000000000003/100000000000000000019"
        .parse::<Rational>()
        .unwrap();

    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                assert_eq!((&integer + &Integer::from(7)) - &Integer::from(7), integer);
                assert_eq!(
                    (&rational * &Rational::from(3)) - &rational - &rational,
                    rational
                );
            });
        }
    });
}

#[test]
fn integer_arithmetic_is_arbitrary_precision() {
    let large = "100000000000000000000000000000000000000"
        .parse::<Integer>()
        .expect("valid integer");
    assert_eq!(
        (&large + &Integer::from(7)).to_string(),
        "100000000000000000000000000000000000007"
    );
    assert_eq!(Integer::from(-12).abs(), Integer::from(12));
    assert_eq!(Integer::from(-3).pow(5), Integer::from(-243));
    assert_eq!(Integer::from(123).pow(0), Integer::one());
    assert_eq!(
        large.canonical_bytes().unwrap(),
        large.to_string().as_bytes()
    );

    let hexadecimal = "-abcdef1234567890abcdef1234567890";
    let parsed = Integer::from_str_radix(hexadecimal, 16).unwrap();
    assert_eq!(parsed.to_string_radix(16).unwrap(), hexadecimal);
    assert!(Integer::from_str_radix("1", 1).is_err());
    assert!(parsed.to_string_radix(63).is_err());
}

#[test]
fn integer_division_has_nonnegative_euclidean_remainder() {
    for (a, b, expected_q, expected_r) in [
        (-17, 5, -4, 3),
        (-17, -5, 4, 3),
        (17, -5, -3, 2),
        (17, 5, 3, 2),
    ] {
        let a = Integer::from(a);
        let b = Integer::from(b);
        let (quotient, remainder) = Z.div_rem(&a, &b).unwrap();
        assert_eq!(quotient, Integer::from(expected_q));
        assert_eq!(remainder, Integer::from(expected_r));
        assert_eq!(&(&quotient * &b) + &remainder, a);
        assert!(remainder >= Integer::zero());
        assert!(remainder < b.abs());
    }
    assert!(Z.div_rem(&Integer::one(), &Integer::zero()).is_err());
}

#[test]
fn integer_bezout_identity_is_normalized_and_verified() {
    let a = Integer::from(30);
    let b = Integer::from(18);
    let identity = Z.extended_gcd(&a, &b);
    assert_eq!(identity.gcd, Integer::from(6));
    identity.verify(&Z, &a, &b).unwrap();
    assert_eq!(
        identity.to_string(),
        "gcd = 6; coefficient_a = -1; coefficient_b = 2"
    );
    assert_eq!(
        BezoutIdentity::<IntegerRing>::from_json(&identity.to_json().unwrap()).unwrap(),
        identity
    );

    let invalid = BezoutIdentity {
        gcd: Integer::from(7),
        coefficient_a: Integer::from(-1),
        coefficient_b: Integer::from(2),
    };
    assert!(invalid.verify(&Z, &a, &b).is_err());

    let zero = Integer::zero();
    let zero_identity = Z.extended_gcd(&zero, &zero);
    assert_eq!(zero_identity.gcd, zero);
    zero_identity.verify(&Z, &zero, &zero).unwrap();
}

#[test]
fn rationals_are_reduced_and_have_positive_denominator() {
    let value = Rational::new(Integer::from(6), Integer::from(-8)).unwrap();
    assert_eq!(value.to_string(), "-3/4");
    assert_eq!(value.numerator(), Integer::from(-3));
    assert_eq!(value.denominator(), Integer::from(4));
    assert_eq!("10/15".parse::<Rational>().unwrap().to_string(), "2/3");
    assert!(Rational::new(Integer::one(), Integer::zero()).is_err());
    assert!("1/0".parse::<Rational>().is_err());
}

#[test]
fn rational_field_operations_fail_closed_on_zero() {
    let two_thirds = Q.fraction(Integer::from(2), Integer::from(3)).unwrap();
    let five_sevenths = "5/7".parse::<Rational>().unwrap();
    assert_eq!(
        Q.add(&two_thirds, &five_sevenths).unwrap().to_string(),
        "29/21"
    );
    assert_eq!(Q.inverse(&two_thirds).unwrap().to_string(), "3/2");
    assert!(Q.inverse(&Rational::zero()).is_err());
    assert!(two_thirds.checked_div(&Rational::zero()).is_err());
}

#[test]
fn field_bezout_identity_uses_the_canonical_unit_gcd() {
    let a = "2/3".parse::<Rational>().unwrap();
    let b = "5/7".parse::<Rational>().unwrap();
    let identity = Q.extended_gcd(&a, &b);
    assert_eq!(identity.gcd, Rational::one());
    identity.verify(&Q, &a, &b).unwrap();

    let zero = Rational::zero();
    let zero_identity = Q.extended_gcd(&zero, &zero);
    assert_eq!(zero_identity.gcd, zero);
    zero_identity.verify(&Q, &zero, &zero).unwrap();
}

#[test]
fn singleton_contexts_have_the_declared_traits() {
    fn integer_context<R: Ring<Element = Integer> + PrincipalIdealDomain + EuclideanDomain>(_: R) {}
    fn rational_context<
        R: Ring<Element = Rational> + Field + PrincipalIdealDomain + EuclideanDomain,
    >(
        _: R,
    ) {
    }

    integer_context(IntegerRing);
    rational_context(RationalField);
    assert_eq!(Z.to_string(), "Z");
    assert_eq!(Q.to_string(), "Q");
}

#[test]
fn exact_scalars_and_singleton_contexts_have_canonical_serde() {
    let integer: Integer = "100000000000000000000000000000000000003".parse().unwrap();
    let rational = Rational::new(integer.clone(), Integer::from(7_u8)).unwrap();

    let integer_json = serde_json::to_string(&integer).unwrap();
    assert_eq!(
        serde_json::from_str::<Integer>(&integer_json).unwrap(),
        integer
    );
    let rational_json = serde_json::to_string(&rational).unwrap();
    assert_eq!(
        serde_json::from_str::<Rational>(&rational_json).unwrap(),
        rational
    );
    assert_eq!(IntegerRing::from_json(&Z.to_json().unwrap()).unwrap(), Z);
    assert_eq!(RationalField::from_json(&Q.to_json().unwrap()).unwrap(), Q);

    assert!(serde_json::from_str::<Rational>(r#"{"numerator":"2","denominator":"4"}"#).is_err());
    assert!(IntegerRing::from_json(r#"{"type":"noether/root/RationalField@1"}"#).is_err());
}
