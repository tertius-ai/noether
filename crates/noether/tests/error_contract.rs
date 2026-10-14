// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use std::error::Error as _;

use noether::{
    ContextMismatchError, DivisionByZeroError, Integer, InversionError, NoetherError, Rational,
    UnitError, UnsupportedOperationError, VerificationError,
};

#[test]
fn capability_and_context_errors_have_stable_structured_messages() {
    let unsupported = UnsupportedOperationError::new("linalg.solve", "IntegerRing")
        .with_detail("the coefficient matrix is not square")
        .with_supported_contexts(["RationalField", "FiniteField"]);
    assert_eq!(
        unsupported.to_string(),
        "operation linalg.solve is unsupported for context IntegerRing: the coefficient matrix \
         is not square; supported contexts: RationalField, FiniteField"
    );
    assert!(unsupported.source().is_none());

    let bare = UnsupportedOperationError::new("numthy.inverse", "IntegerRing");
    assert_eq!(
        bare.to_string(),
        "operation numthy.inverse is unsupported for context IntegerRing"
    );

    let mismatch = ContextMismatchError {
        expected: "FiniteField(9)".to_owned(),
        actual: "FiniteField(25)".to_owned(),
    };
    assert_eq!(
        mismatch.to_string(),
        "context mismatch: expected FiniteField(9), received FiniteField(25)"
    );
    assert!(mismatch.source().is_none());
}

#[test]
fn inversion_and_unit_errors_preserve_their_sources() {
    let division = InversionError::from(DivisionByZeroError);
    assert_eq!(division.to_string(), "division by zero");
    assert_eq!(division.source().unwrap().to_string(), "division by zero");
    assert_eq!(
        UnitError::from(division),
        UnitError::DivisionByZero(DivisionByZeroError)
    );

    let mismatch = ContextMismatchError {
        expected: "RationalField".to_owned(),
        actual: "FiniteField".to_owned(),
    };
    let inversion = InversionError::from(mismatch.clone());
    assert_eq!(
        inversion.source().unwrap().to_string(),
        mismatch.to_string()
    );
    assert_eq!(
        UnitError::from(inversion),
        UnitError::ContextMismatch(mismatch.clone())
    );

    let nonunit = UnitError::NotAUnit {
        context: "ResidueRing(12)",
    };
    assert_eq!(
        nonunit.to_string(),
        "element is not a unit in ResidueRing(12)"
    );
    assert!(nonunit.source().is_none());
    assert_eq!(
        UnitError::from(mismatch.clone())
            .source()
            .unwrap()
            .to_string(),
        mismatch.to_string()
    );
}

#[test]
fn verification_and_facade_errors_keep_typed_causes() {
    let mismatch = ContextMismatchError {
        expected: "IntegerRing".to_owned(),
        actual: "RationalField".to_owned(),
    };
    let verification = VerificationError::from(mismatch.clone());
    assert_eq!(
        verification.source().unwrap().to_string(),
        mismatch.to_string()
    );

    let invalid = VerificationError::InvalidWitness {
        claim: "the transformation does not reconstruct the input",
    };
    assert_eq!(
        invalid.to_string(),
        "witness verification failed: the transformation does not reconstruct the input"
    );
    assert!(invalid.source().is_none());

    let errors = [
        NoetherError::Serialization("bad payload".to_owned()),
        NoetherError::from(mismatch),
        NoetherError::from(invalid),
        NoetherError::from(UnsupportedOperationError::new("factor", "PowerSeriesRing")),
    ];
    assert_eq!(errors[0].to_string(), "serialization failed: bad payload");
    assert!(errors[0].source().is_none());
    for error in &errors[1..] {
        assert!(error.source().is_some());
    }

    let integer_parse: NoetherError = "not-an-integer".parse::<Integer>().unwrap_err().into();
    let rational_parse: NoetherError = "1/not-an-integer".parse::<Rational>().unwrap_err().into();
    assert!(matches!(integer_parse, NoetherError::Serialization(_)));
    assert!(matches!(rational_parse, NoetherError::Serialization(_)));
}
