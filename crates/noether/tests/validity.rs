// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::{Assumption, Integer, MathematicalValidity, SemanticType};

#[test]
fn assumptions_have_stable_identity_and_canonical_json() {
    let examples = [
        (
            Assumption::CompatiblePadicComplexLift {
                prime: Integer::from(3),
                precision: 8,
            },
            r#"{"kind":"compatible_padic_complex_lift","prime":"3","precision":8}"#,
        ),
        (
            Assumption::ExactPadicDifferentialRanks {
                outgoing: 2,
                incoming: 3,
            },
            r#"{"kind":"exact_padic_differential_ranks","outgoing":2,"incoming":3}"#,
        ),
        (
            Assumption::GeneralizedRiemannHypothesis,
            r#"{"kind":"generalized_riemann_hypothesis"}"#,
        ),
        (
            Assumption::TorsionOrderIsSmooth {
                bound: Integer::from(97),
            },
            r#"{"kind":"torsion_order_is_smooth","bound":"97"}"#,
        ),
        (
            Assumption::Squarefree {
                cofactor: Integer::from(1_u128 << 100),
            },
            r#"{"kind":"squarefree","cofactor":"1267650600228229401496703205376"}"#,
        ),
    ];

    assert_eq!(Assumption::TYPE_ID, "noether/root/Assumption@1");
    assert_eq!(Assumption::describe().rust_path, "noether::Assumption");
    assert_eq!(
        Assumption::describe().python_path,
        Some("noether.Assumption")
    );

    for (assumption, expected) in examples {
        assert_eq!(assumption.to_json().unwrap(), expected);
        assert_eq!(Assumption::from_json(expected).unwrap(), assumption);
        assert_eq!(assumption.canonical_bytes().unwrap(), expected.as_bytes());
    }
}

#[test]
fn conditional_validity_is_a_canonical_set() {
    let grh = Assumption::GeneralizedRiemannHypothesis;
    let smooth = Assumption::TorsionOrderIsSmooth {
        bound: Integer::from(13),
    };
    let validity = MathematicalValidity::conditional([smooth.clone(), grh.clone(), smooth, grh]);

    assert_eq!(validity.assumptions().len(), 2);
    assert_eq!(
        validity.to_json().unwrap(),
        r#"{"kind":"conditional","assumptions":[{"kind":"generalized_riemann_hypothesis"},{"kind":"torsion_order_is_smooth","bound":"13"}]}"#
    );
    assert_eq!(
        MathematicalValidity::from_json(&validity.to_json().unwrap()).unwrap(),
        validity
    );
}

#[test]
fn empty_conditional_normalizes_to_unconditional() {
    let normalized =
        MathematicalValidity::from_json(r#"{"kind":"conditional","assumptions":[]}"#).unwrap();
    assert!(normalized.is_unconditional());
    assert_eq!(normalized.to_json().unwrap(), r#"{"kind":"unconditional"}"#);
    assert_eq!(
        MathematicalValidity::TYPE_ID,
        "noether/root/MathematicalValidity@1"
    );
}

#[test]
fn deserialization_rejects_free_form_and_malformed_assumptions() {
    assert!(Assumption::from_json(r#"{"kind":"probably"}"#).is_err());
    assert!(Assumption::from_json(
        r#"{"kind":"compatible_padic_complex_lift","prime":"4","precision":3}"#
    )
    .is_err());
    assert!(Assumption::from_json(r#"{"kind":"squarefree","cofactor":"not-an-integer"}"#).is_err());
    assert!(MathematicalValidity::from_json(r#"{"kind":"unverified"}"#).is_err());
}
