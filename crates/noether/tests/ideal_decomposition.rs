// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{IdealDecomposition, MonomialOrder, PolynomialRing, RationalFunctionField};
use noether::numthy::FiniteField;
use noether::{Ring, SemanticType, Q};

#[test]
fn conjugate_lines_split_after_constant_extension_with_verified_frobenius() {
    let prime = FiniteField::prime(3_u64).unwrap();
    let functions = RationalFunctionField::new(prime, "c").unwrap();
    let ring = PolynomialRing::new(
        functions,
        ["x", "y"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let variables = ring.generators().unwrap();
    let source = ring
        .ideal([(&variables[0].pow(2).unwrap() + &variables[1].pow(2).unwrap()).unwrap()])
        .unwrap();

    let extension = FiniteField::extension(3, 2).unwrap();
    let changed = source
        .base_change_constant_field(extension.clone())
        .unwrap();
    assert_eq!(
        changed.ring().coefficient_ring().coefficient_ring(),
        &extension
    );

    let decomposition = changed.minimal_primes().unwrap();
    assert_eq!(decomposition.components().len(), 2);
    assert_eq!(decomposition.frobenius_permutation(), Some(&[1, 0][..]));
    let hilbert_ring =
        PolynomialRing::new(Q, ["m"], MonomialOrder::GradedReverseLexicographic).unwrap();
    for component in decomposition.components() {
        assert_eq!(component.dimension().unwrap(), 1);
        assert_eq!(component.degree().unwrap().to_string(), "1");
        assert_eq!(component.hilbert_polynomial().unwrap(), hilbert_ring.one());
    }
    decomposition.verify().unwrap();
    decomposition.verify_frobenius().unwrap();

    let encoded = decomposition.to_json().unwrap();
    assert_eq!(
        IdealDecomposition::from_json(&encoded).unwrap(),
        decomposition
    );
    let mut payload: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    payload["frobenius_permutation"] = serde_json::json!([0, 1]);
    let tampered = serde_json::to_string(&payload).unwrap();
    assert!(IdealDecomposition::from_json(&tampered).is_err());

    assert_eq!(
        IdealDecomposition::<RationalFunctionField<FiniteField>>::TYPE_ID,
        "noether/commalg/IdealDecomposition@1"
    );
}

#[test]
fn decomposition_floor_fails_closed_before_the_splitting_field() {
    let prime = FiniteField::prime(3_u64).unwrap();
    let functions = RationalFunctionField::new(prime, "c").unwrap();
    let ring = PolynomialRing::new(functions, ["x", "y"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let source = ring
        .ideal([(&variables[0].pow(2).unwrap() + &variables[1].pow(2).unwrap()).unwrap()])
        .unwrap();

    let error = source.minimal_primes().unwrap_err();
    assert!(error.to_string().contains("split completely"));
}
