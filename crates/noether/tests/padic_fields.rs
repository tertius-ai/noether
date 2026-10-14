// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::Matrix;
use noether::numthy::{PadicElement, PadicField, PadicRing, PadicValuation};
use noether::{PrecisionContext, Ring, SemanticType, ValuedRing};

#[test]
fn fixed_cap_fraction_context_supports_negative_valuation() {
    let field = PadicField::new(5_u8, 8).unwrap();
    let five = field.element(5_u8);
    let inverse = field.inverse(&five).unwrap();

    assert_eq!(*field.precision(), 8);
    assert_eq!(field.valuation_lower_bound(&five), Some(1));
    assert_eq!(inverse.valuation(), PadicValuation::Exact(-1));
    assert_eq!(field.mul(&five, &inverse).unwrap(), field.one());
    assert_eq!(inverse.ring(), None);
    assert_eq!(inverse.field(), Some(&field));
    assert_eq!(inverse.representative(), None);
    assert_eq!(inverse.residue(), None);
}

#[test]
fn fixed_precision_equality_does_not_claim_exact_field_laws() {
    let field = PadicField::new(3_u8, 4).unwrap();
    let twenty_seven = field.element(27_u8);
    let three = field.element(3_u8);

    assert_ne!(twenty_seven, field.zero());
    assert_ne!(three, field.zero());
    assert_eq!(field.mul(&twenty_seven, &three).unwrap(), field.zero());
}

#[test]
fn ring_and_field_contexts_do_not_alias() {
    let ring = PadicRing::new(3_u8, 6).unwrap();
    let field = PadicField::new(3_u8, 6).unwrap();

    assert!(!ring.contains(&field.one()));
    assert!(!field.contains(&ring.one()));
    assert!((&ring.one() + &field.one()).is_err());
}

#[test]
fn generic_matrices_exist_over_fixed_cap_padic_fields() {
    let field = PadicField::new(3_u8, 8).unwrap();
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
fn field_elements_preserve_negative_valuation_in_typed_json() {
    let field = PadicField::new(5_u8, 8).unwrap();
    let inverse = field.inverse(&field.element(5_u8)).unwrap();

    assert_eq!(
        PadicField::from_json(&field.to_json().unwrap()).unwrap(),
        field
    );
    assert_eq!(
        PadicElement::from_json(&inverse.to_json().unwrap()).unwrap(),
        inverse
    );
    assert_eq!(
        PadicElement::from_json(&field.zero().to_json().unwrap()).unwrap(),
        field.zero()
    );

    let noncanonical = inverse
        .to_json()
        .unwrap()
        .replace("\"unit\":\"1\"", "\"unit\":\"5\"");
    assert!(PadicElement::from_json(&noncanonical).is_err());

    let wrong_context = inverse
        .to_json()
        .unwrap()
        .replace(PadicField::TYPE_ID, PadicRing::TYPE_ID);
    assert!(PadicElement::from_json(&wrong_context).is_err());
}
