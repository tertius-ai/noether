// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::ResidueRing;
use noether::numthy::{FiniteField, PadicRing};
use noether::{Integer, Rational, Ring, UnitError, UnitInversion, Q, Z};

fn inverse<R: UnitInversion>(ring: &R, value: &R::Element) -> Result<R::Element, UnitError> {
    ring.inverse_unit(value)
}

#[test]
fn fields_expose_unit_inversion_without_losing_field_semantics() {
    let rational = Rational::new(Integer::from(2), Integer::from(3)).unwrap();
    assert_eq!(inverse(&Q, &rational).unwrap().to_string(), "3/2");
    assert!(matches!(
        inverse(&Q, &Q.zero()),
        Err(UnitError::DivisionByZero(_))
    ));

    let field = FiniteField::extension(5_u64, 2).unwrap();
    let generator = field.generator();
    assert_eq!(
        field
            .mul(&generator, &inverse(&field, &generator).unwrap())
            .unwrap(),
        field.one()
    );
}

#[test]
fn rings_invert_exactly_their_units() {
    assert_eq!(inverse(&Z, &Integer::from(-1)).unwrap(), Integer::from(-1));
    assert!(matches!(
        inverse(&Z, &Integer::from(2)),
        Err(UnitError::NotAUnit { .. })
    ));

    let residues = ResidueRing::new(12_u8).unwrap();
    assert_eq!(
        inverse(&residues, &residues.element(7_u8)).unwrap(),
        residues.element(7_u8)
    );
    assert!(matches!(
        inverse(&residues, &residues.element(6_u8)),
        Err(UnitError::NotAUnit { .. })
    ));

    let padics = PadicRing::new(5_u8, 3).unwrap();
    let two_inverse = inverse(&padics, &padics.element(2_u8)).unwrap();
    assert_eq!(
        padics.mul(&padics.element(2_u8), &two_inverse).unwrap(),
        padics.one()
    );
    assert!(matches!(
        inverse(&padics, &padics.element(5_u8)),
        Err(UnitError::NotAUnit { .. })
    ));
}

#[test]
fn context_mismatches_fail_before_unit_claims() {
    let z12 = ResidueRing::new(12_u8).unwrap();
    let z13 = ResidueRing::new(13_u8).unwrap();
    assert!(matches!(
        inverse(&z12, &z13.element(1_u8)),
        Err(UnitError::ContextMismatch(_))
    ));

    let low = PadicRing::new(5_u8, 2).unwrap();
    let high = PadicRing::new(5_u8, 3).unwrap();
    assert!(matches!(
        inverse(&low, &high.one()),
        Err(UnitError::ContextMismatch(_))
    ));
}
