// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::numthy::{
    FiniteField, PadicLocalField, PadicLocalFieldCoefficientFrobenius, PadicLocalFieldElement,
    PadicLocalFieldRestriction, PadicLocalPrecision, PadicValuation, QadicRing,
};
use noether::{Ring, SemanticType};

fn ramified_unramified_field() -> PadicLocalField {
    let residue = FiniteField::extension(3, 2).unwrap();
    let coefficients = QadicRing::new(residue, 4).unwrap();
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
        PadicLocalPrecision::new(7).unwrap(),
        Some("pi".to_owned()),
    )
    .unwrap()
}

#[test]
fn public_local_context_has_typed_staggered_precision() {
    let field = ramified_unramified_field();

    assert_eq!((field.ramification_index(), field.residue_degree()), (2, 2));
    assert_eq!(field.degree(), 4);
    assert_eq!(field.coordinate_precisions(), [4, 3]);
    assert_eq!(PadicLocalField::TYPE_ID, "noether/numthy/PadicLocalField@1");
    assert_eq!(
        PadicLocalFieldElement::TYPE_ID,
        "noether/numthy/PadicLocalFieldElement@1"
    );
    assert_eq!(
        PadicLocalFieldCoefficientFrobenius::TYPE_ID,
        "noether/numthy/PadicLocalFieldCoefficientFrobenius@1"
    );
}

#[test]
fn public_local_context_and_integral_elements_round_trip() {
    let field = ramified_unramified_field();
    let element = field
        .integral_element(&[
            field.coefficient_ring().element(4),
            field.coefficient_ring().element(5),
        ])
        .unwrap();

    assert_eq!(
        PadicLocalField::from_json(&field.to_json().unwrap()).unwrap(),
        field
    );
    assert_eq!(
        PadicLocalFieldElement::from_json(&element.to_json().unwrap()).unwrap(),
        element
    );
    assert_eq!(
        element.normalized_coordinates().len(),
        field.ramification_index()
    );
}

#[test]
fn public_local_field_is_a_precision_aware_coefficient_ring() {
    let field = ramified_unramified_field();
    let uniformizer = field.uniformizer();
    let inverse = uniformizer.inverse().unwrap();

    assert_eq!(uniformizer.valuation(), PadicValuation::Exact(1));
    assert_eq!(inverse.valuation(), PadicValuation::Exact(-1));
    assert_eq!(inverse.known_absolute_precision(), 5);
    assert_eq!(
        field.mul(&uniformizer, &inverse).unwrap().valuation(),
        PadicValuation::Exact(0)
    );

    let relation = field.add(
        &uniformizer.pow(2),
        &field
            .integral_element(&[field.coefficient_ring().element(3)])
            .unwrap(),
    );
    assert!(relation.unwrap().is_zero());

    let inverse_p = field
        .coefficient_field()
        .inverse(&field.coefficient_field().element(3))
        .unwrap();
    let embedded = field.element_from_qadic(&inverse_p).unwrap();
    assert_eq!(embedded.valuation(), PadicValuation::Exact(-2));
    assert_eq!(embedded.relative_precision(), Some(6));
    assert_eq!(embedded.known_absolute_precision(), 4);

    let residue = field.residue_field().generator();
    let lift = field.teichmuller(&residue).unwrap();
    assert_eq!(field.residue(&lift).unwrap(), Some(residue));

    let restriction = field
        .precision_restriction(PadicLocalPrecision::new(4).unwrap())
        .unwrap();
    assert_eq!(
        restriction.apply(&uniformizer).unwrap(),
        restriction.target().uniformizer()
    );
    assert_eq!(
        PadicLocalFieldRestriction::from_json(&restriction.to_json().unwrap()).unwrap(),
        restriction
    );

    let trace = uniformizer.trace().unwrap();
    let base = trace.field().unwrap();
    assert_eq!(trace, base.zero());
    assert_eq!(uniformizer.norm().unwrap(), base.element(9));
    assert_eq!(
        uniformizer
            .characteristic_polynomial()
            .unwrap()
            .coefficient(&[4_u64.into()])
            .unwrap(),
        base.one()
    );

    let frobenius = field.coefficient_frobenius_conjugate(1).unwrap();
    assert!(frobenius.is_endomorphism());
    assert_eq!(frobenius.apply(&uniformizer).unwrap(), uniformizer);
}
