// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::numthy::{FiniteField, PadicRing, PadicValuation, QadicElement, QadicRing};
use noether::{CommutativeRing, Integer, PrecisionContext, Ring, UnitInversion, ValuedRing};

fn assert_context<R: CommutativeRing + PrecisionContext + ValuedRing + UnitInversion>(_ring: &R) {}
fn assert_thread_safe<T: Send + Sync>() {}

#[test]
fn fixed_cap_witt_ring_uses_the_residue_field_presentation() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let ring = QadicRing::new(residue_field.clone(), 4).unwrap();

    assert_context(&ring);
    assert_eq!(ring.residue_field(), &residue_field);
    assert_eq!(ring.prime(), &Integer::from(3));
    assert_eq!(ring.residue_degree(), 2);
    assert_eq!(ring.coefficient_ring(), &PadicRing::new(3_u8, 4).unwrap());
    assert_eq!(ring.absolute_precision(), 4);
    assert_eq!(ring.cardinality(), Integer::from(3_u8).pow(8));
    assert_thread_safe::<QadicRing>();
    assert_thread_safe::<QadicElement>();
}

#[test]
fn teichmuller_lifts_and_frobenius_reduce_correctly() {
    let residue_field = FiniteField::extension(5, 2).unwrap();
    let ring = QadicRing::new(residue_field.clone(), 5).unwrap();
    let generator = residue_field.generator();
    let one = residue_field.one();
    let residue = &generator + &one;
    let residue = residue.unwrap();
    let lift = ring.teichmuller(&residue).unwrap();

    assert_eq!(ring.residue(&lift).unwrap(), residue);
    assert_eq!(
        lift.pow(residue_field.order().to_string().parse().unwrap()),
        lift
    );
    assert_eq!(
        ring.residue(&lift.frobenius(1)).unwrap(),
        residue.frobenius(1)
    );
}

#[test]
fn arithmetic_stays_integral_and_only_units_invert() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let ring = QadicRing::new(residue_field, 4).unwrap();
    let generator = ring.generator();
    let unit = ring.add(&generator, &ring.one()).unwrap();
    let uniformizer = ring.uniformizer();

    assert_eq!(
        ring.mul(&unit, &unit.inverse().unwrap()).unwrap(),
        ring.one()
    );
    assert!(generator
        .coefficients()
        .iter()
        .all(|coefficient| coefficient.ring() == Some(ring.coefficient_ring())));
    assert_eq!(ring.one().trace(), ring.coefficient_ring().element(2_u8));
    assert_eq!(
        ring.element(3_u8).norm(),
        ring.coefficient_ring().element(9_u8)
    );
    assert!(uniformizer.inverse().is_err());
    assert_eq!(uniformizer.valuation(), PadicValuation::Exact(1));
    assert_eq!(ring.zero().valuation(), PadicValuation::AtLeast(4));
}

#[test]
fn precision_restriction_discards_digits_without_changing_residue() {
    let residue_field = FiniteField::extension(5, 2).unwrap();
    let ring = QadicRing::new(residue_field, 4).unwrap();
    let value = ring.add(&ring.generator(), &ring.element(126_u16)).unwrap();
    let restricted = ring.restrict(&value, 2).unwrap();

    assert_eq!(restricted.ring().unwrap().absolute_precision(), 2);
    assert_eq!(
        ring.residue(&value).unwrap(),
        restricted.ring().unwrap().residue(&restricted).unwrap()
    );

    let other = QadicRing::new(FiniteField::extension(3, 2).unwrap(), 4).unwrap();
    assert!(ring.restrict(&other.one(), 2).is_err());
}

#[test]
fn invalid_contexts_fail_before_entering_flint() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    assert!(QadicRing::new(residue_field, 0).is_err());
}

#[test]
fn witt_contexts_and_integral_coordinates_have_typed_json() {
    let residue_field = FiniteField::extension(3, 2).unwrap();
    let ring = QadicRing::new(residue_field, 4).unwrap();
    let value = ring.add(&ring.generator(), &ring.element(17_u8)).unwrap();

    assert_eq!(
        QadicRing::from_json(&ring.to_json().unwrap()).unwrap(),
        ring
    );
    assert_eq!(
        QadicElement::from_json(&value.to_json().unwrap()).unwrap(),
        value
    );

    let mut noncanonical: serde_json::Value =
        serde_json::from_str(&value.to_json().unwrap()).unwrap();
    noncanonical["value"]["coefficients"][0] = serde_json::Value::String("81".to_owned());
    assert!(QadicElement::from_json(&noncanonical.to_string()).is_err());

    let mut reducible: serde_json::Value = serde_json::from_str(&ring.to_json().unwrap()).unwrap();
    reducible["residue_field"]["modulus"] = serde_json::json!([2, 0, 1]);
    assert!(QadicRing::from_json(&reducible.to_string()).is_err());
}
