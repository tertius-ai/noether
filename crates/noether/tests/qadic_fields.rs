// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::Matrix;
use noether::numthy::{FiniteField, PadicValuation, QadicElement, QadicField, QadicRing};
use noether::{PrecisionContext, Ring, SemanticType, ValuedRing};

#[test]
fn unramified_fraction_context_supports_negative_valuation_and_base_field_coefficients() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let field = QadicField::new(residue_field, 8).unwrap();
    let three = field.element(3_u8);
    let inverse = field.inverse(&three).unwrap();

    assert_eq!(*field.precision(), 8);
    assert_eq!(field.valuation_lower_bound(&three), Some(1));
    assert_eq!(inverse.valuation(), PadicValuation::Exact(-1));
    assert_eq!(field.mul(&three, &inverse).unwrap(), field.one());
    assert_eq!(inverse.ring(), None);
    assert_eq!(inverse.field(), Some(&field));
    assert_eq!(inverse.residue(), None);
    assert!(inverse
        .coefficients()
        .iter()
        .all(|coefficient| coefficient.field() == Some(field.coefficient_field())));

    let qadic_inverse = field.inverse(&field.element(12_u8)).unwrap();
    let padic_inverse = field
        .coefficient_field()
        .inverse(&field.coefficient_field().element(12_u8))
        .unwrap();
    assert_eq!(qadic_inverse.coefficients()[0], padic_inverse);
}

#[test]
fn fixed_precision_equality_does_not_claim_exact_field_laws() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let field = QadicField::new(residue_field, 4).unwrap();
    let twenty_seven = field.element(27_u8);
    let three = field.element(3_u8);

    assert_ne!(twenty_seven, field.zero());
    assert_ne!(three, field.zero());
    assert_eq!(field.mul(&twenty_seven, &three).unwrap(), field.zero());
}

#[test]
fn ring_and_field_contexts_do_not_alias() {
    let residue_field = FiniteField::extension(5, 2).unwrap();
    let ring = QadicRing::new(residue_field.clone(), 6).unwrap();
    let field = QadicField::new(residue_field, 6).unwrap();

    assert!(!ring.contains(&field.one()));
    assert!(!field.contains(&ring.one()));
    assert!((&ring.one() + &field.one()).is_err());
}

#[test]
fn frobenius_trace_and_norm_use_the_unramified_field_structure() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let field = QadicField::new(residue_field, 8).unwrap();
    let generator = field.generator();

    assert_ne!(generator.frobenius(1), generator);
    assert_eq!(generator.frobenius(2), generator);
    assert_eq!(field.one().trace(), field.coefficient_field().element(2_u8));
    assert_eq!(
        field.element(3_u8).norm(),
        field.coefficient_field().element(9_u8)
    );
}

#[test]
fn generic_matrices_exist_over_fixed_cap_qadic_fields() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let field = QadicField::new(residue_field, 8).unwrap();
    let inverse_three = field.inverse(&field.element(3_u8)).unwrap();
    let matrix = Matrix::from_rows(
        field.clone(),
        [
            [field.one(), inverse_three],
            [field.element(3_u8), field.one()],
        ],
    )
    .unwrap();

    let square = matrix.matmul(&matrix).unwrap();
    assert_eq!(square.shape(), (2, 2));
    assert_eq!(square.get(0, 0).unwrap(), field.element(2_u8));
}

#[test]
fn field_coordinates_round_trip_negative_valuation_and_full_unit_precision() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let field = QadicField::new(residue_field, 8).unwrap();
    let unit = field.add(&field.generator(), &field.one()).unwrap();
    let value = field
        .inverse(&field.mul(&field.element(3_u8), &unit).unwrap())
        .unwrap();

    assert_eq!(value.valuation(), PadicValuation::Exact(-1));
    assert_eq!(
        QadicField::from_json(&field.to_json().unwrap()).unwrap(),
        field
    );
    assert_eq!(
        QadicElement::from_json(&value.to_json().unwrap()).unwrap(),
        value
    );
    assert_eq!(
        QadicElement::from_json(&field.zero().to_json().unwrap()).unwrap(),
        field.zero()
    );

    let mut noncanonical: serde_json::Value =
        serde_json::from_str(&value.to_json().unwrap()).unwrap();
    noncanonical["value"]["unit_coefficients"] = serde_json::json!(["3", "3"]);
    assert!(QadicElement::from_json(&noncanonical.to_string()).is_err());

    let mut wrong_context: serde_json::Value =
        serde_json::from_str(&value.to_json().unwrap()).unwrap();
    wrong_context["value"]["context"]["type"] =
        serde_json::Value::String(QadicRing::TYPE_ID.to_owned());
    assert!(QadicElement::from_json(&wrong_context.to_string()).is_err());
}
