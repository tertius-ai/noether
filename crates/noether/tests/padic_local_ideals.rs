// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::numthy::{
    FiniteField, PadicLocalField, PadicLocalFractionalIdeal, PadicLocalIdealError,
    PadicLocalPrecision, QadicRing,
};
use noether::{Ring, SemanticType};

fn local_field(residue_degree: usize) -> PadicLocalField {
    let residue = if residue_degree == 1 {
        FiniteField::prime(3_u64).unwrap()
    } else {
        FiniteField::extension(3_u64, residue_degree).unwrap()
    };
    let coefficients = QadicRing::new(residue, 6).unwrap();
    let polynomials =
        PolynomialRing::new(coefficients.clone(), ["T"], MonomialOrder::Lexicographic).unwrap();
    let eisenstein = polynomials
        .from_u64_terms([
            (vec![0], coefficients.element(3)),
            (vec![2], coefficients.one()),
        ])
        .unwrap();
    PadicLocalField::from_eisenstein(
        coefficients,
        eisenstein,
        PadicLocalPrecision::new(11).unwrap(),
        Some("pi".to_owned()),
    )
    .unwrap()
}

#[test]
fn maximal_order_ideals_are_verified_uniformizer_powers() {
    let field = local_field(2);
    let order = field.maximal_order().unwrap();
    let unit = order.fractional_ideal(&[field.one()]).unwrap();
    let maximal = order.maximal_ideal().unwrap();

    assert_eq!(
        PadicLocalFractionalIdeal::TYPE_ID,
        "noether/numthy/PadicLocalFractionalIdeal@1"
    );
    assert!(unit.is_integral());
    assert_eq!(unit.norm_valuation().unwrap(), 0);
    assert_eq!(unit.maximal_ideal_exponent().unwrap(), 0);
    assert_eq!(maximal.norm_valuation().unwrap(), 2);
    assert_eq!(maximal.maximal_ideal_exponent().unwrap(), 1);
    assert!(maximal.contains(&field.uniformizer()));
    assert!(!maximal.contains(&field.one()));

    let square = maximal.mul(&maximal).unwrap();
    assert_eq!(square.maximal_ideal_exponent().unwrap(), 2);
    assert_eq!(maximal.add(&unit).unwrap(), unit);
    let inverse = maximal.inverse().unwrap();
    assert_eq!(inverse.maximal_ideal_exponent().unwrap(), -1);
    assert!(!inverse.is_integral());
    let product = maximal.mul(&inverse).unwrap();
    assert_eq!(product.maximal_ideal_exponent().unwrap(), 0);
    assert!(product.contains(&field.one()));
    assert!(product.certified_precision() < unit.certified_precision());
    assert_eq!(
        PadicLocalFractionalIdeal::from_json(&inverse.to_json().unwrap()).unwrap(),
        inverse
    );
}

#[test]
fn principal_scaling_has_the_expected_signed_norm_valuation() {
    let field = local_field(1);
    let order = field.maximal_order().unwrap();
    let unit = order.fractional_ideal(&[field.one()]).unwrap();
    let pi = field.uniformizer();

    assert_eq!(unit.scale(&pi.pow(3)).unwrap().norm_valuation().unwrap(), 3);
    assert_eq!(
        unit.scale(&pi.inverse().unwrap().pow(2))
            .unwrap()
            .norm_valuation()
            .unwrap(),
        -2
    );
}

#[test]
fn nonmaximal_orders_keep_ideal_arithmetic_without_dvr_claims() {
    let field = local_field(1);
    let maximal = field.maximal_order().unwrap();
    let p = field
        .integral_element(&[field.coefficient_ring().element(3)])
        .unwrap();
    let p_pi = field.mul(&p, &field.uniformizer()).unwrap();
    let order = field.order(&[field.one(), p_pi]).unwrap();
    let identity = order.fractional_ideal(&[field.one()]).unwrap();
    let principal = order.fractional_ideal(std::slice::from_ref(&p)).unwrap();

    assert_eq!(identity.norm_valuation().unwrap(), 0);
    assert_eq!(principal.norm_valuation().unwrap(), 2);
    assert!(principal.is_integral());
    assert_eq!(principal.mul(&identity).unwrap(), principal);
    assert_eq!(
        principal.inverse(),
        Err(PadicLocalIdealError::NonMaximalOrder)
    );
    assert_eq!(
        order.maximal_ideal(),
        Err(PadicLocalIdealError::NonMaximalOrder)
    );

    let other_identity = maximal.fractional_ideal(&[field.one()]).unwrap();
    assert!(matches!(
        identity.add(&other_identity),
        Err(PadicLocalIdealError::ContextMismatch(_))
    ));
}
