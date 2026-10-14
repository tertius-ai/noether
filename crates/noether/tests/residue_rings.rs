// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{CrtDecomposition, CrtError, ResidueClass, ResidueRing};
use noether::linalg::Matrix;
use noether::{CommutativeRing, FiniteContext, Integer, Ring, SemanticType};

fn assert_commutative_ring<R: CommutativeRing>(_ring: &R) {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn residue_classes_are_canonical_and_context_checked() {
    let ring = ResidueRing::new(12_u8).unwrap();
    let seven = ring.element(-5_i8);
    let ten = ring.element(22_u8);

    assert_commutative_ring(&ring);
    assert_eq!(ring.modulus(), &Integer::from(12));
    assert_eq!(seven.representative(), &Integer::from(7));
    assert_eq!(ten.representative(), &Integer::from(10));
    assert_eq!(
        ring.add(&seven, &ten).unwrap().representative(),
        &Integer::from(5)
    );
    assert_eq!(
        ring.sub(&seven, &ten).unwrap().representative(),
        &Integer::from(9)
    );
    assert_eq!(
        ring.mul(&seven, &ten).unwrap().representative(),
        &Integer::from(10)
    );
    assert_eq!(
        ring.neg(&seven).unwrap().representative(),
        &Integer::from(5)
    );
    assert!(!ten.is_unit());
    assert!(ten.inverse().is_err());
    assert_eq!(seven.inverse().unwrap().representative(), &Integer::from(7));

    let other = ResidueRing::new(13_u8).unwrap().element(7_u8);
    assert!((&seven + &other).is_err());
    assert_ne!(seven, other);
}

#[test]
fn arbitrary_moduli_and_matrices_remain_exact() {
    let modulus: Integer = "1361129467683753853853498429727072845867".parse().unwrap();
    let ring = ResidueRing::new(modulus.clone()).unwrap();
    let near_modulus = ring.element(&modulus - &Integer::one());
    let matrix = Matrix::from_rows(
        ring.clone(),
        [
            [near_modulus.clone(), ring.element(2_u8)],
            [ring.element(3_u8), near_modulus],
        ],
    )
    .unwrap();
    let square = matrix.matmul(&matrix).unwrap();

    assert_eq!(ring.cardinality(), modulus);
    assert_eq!(
        square.get(0, 0).unwrap().representative(),
        &Integer::from(7)
    );
    assert_eq!(
        square.get(0, 1).unwrap().representative(),
        &(&Integer::from(-4_i8) + ring.modulus())
    );
}

#[test]
fn finite_iteration_is_lazy_and_deterministic() {
    let ring = ResidueRing::new(5_u8).unwrap();
    let representatives = ring
        .elements()
        .map(|value| value.representative().clone())
        .collect::<Vec<_>>();

    assert_eq!(
        representatives,
        [0_u8, 1, 2, 3, 4].map(Integer::from).to_vec()
    );
}

#[test]
fn construction_and_semantic_identity_are_explicit() {
    assert!(ResidueRing::new(1_u8).is_err());
    assert!(ResidueRing::new(-2_i8).is_err());
    assert_eq!(ResidueRing::TYPE_ID, "noether/commalg/ResidueRing@1");
    assert_eq!(ResidueClass::TYPE_ID, "noether/commalg/ResidueClass@1");
    assert_eq!(
        ResidueClass::describe().python_path,
        Some("noether.commalg.ResidueClass")
    );
    assert_send_sync::<ResidueRing>();
    assert_send_sync::<ResidueClass>();
}

#[test]
fn typed_json_round_trips_and_revalidates_canonical_classes() {
    let modulus: Integer = "1361129467683753853853498429727072845867".parse().unwrap();
    let ring = ResidueRing::new(modulus.clone()).unwrap();
    let value = ring.element(-11_i8);

    assert_eq!(
        ResidueRing::from_json(&ring.to_json().unwrap()).unwrap(),
        ring
    );
    assert_eq!(
        ResidueClass::from_json(&value.to_json().unwrap()).unwrap(),
        value
    );

    let invalid_modulus = ring.to_json().unwrap().replace(&modulus.to_string(), "1");
    assert!(ResidueRing::from_json(&invalid_modulus).is_err());

    let noncanonical = value.to_json().unwrap().replace(
        value.representative().to_string().as_str(),
        &modulus.to_string(),
    );
    assert!(ResidueClass::from_json(&noncanonical).is_err());

    let wrong_type = value
        .to_json()
        .unwrap()
        .replace(ResidueClass::TYPE_ID, ResidueRing::TYPE_ID);
    assert!(ResidueClass::from_json(&wrong_type).is_err());
}

#[test]
fn canonical_crt_projection_and_reconstruction_are_inverse() {
    let ring = ResidueRing::new(360_u16).unwrap();
    let decomposition = ring.crt().unwrap();
    let value = ring.element(347_u16);
    let components = decomposition.project(&value).unwrap();

    assert_eq!(
        decomposition
            .factors()
            .iter()
            .map(ResidueRing::modulus)
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            Integer::from(8_u8),
            Integer::from(9_u8),
            Integer::from(5_u8)
        ]
    );
    assert_eq!(
        components
            .iter()
            .map(ResidueClass::representative)
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            Integer::from(3_u8),
            Integer::from(5_u8),
            Integer::from(2_u8)
        ]
    );
    assert_eq!(decomposition.reconstruct(&components).unwrap(), value);
    decomposition.verify().unwrap();
    assert_eq!(
        CrtDecomposition::from_json(&decomposition.to_json().unwrap()).unwrap(),
        decomposition
    );
}

#[test]
fn crt_fails_closed_on_wrong_components_and_noncanonical_json() {
    let ring = ResidueRing::new(12_u8).unwrap();
    let decomposition = ring.crt().unwrap();

    assert!(matches!(
        decomposition.reconstruct(&[ResidueRing::new(4_u8).unwrap().one()]),
        Err(CrtError::ComponentCount {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        decomposition.reconstruct(&[
            ResidueRing::new(3_u8).unwrap().one(),
            ResidueRing::new(4_u8).unwrap().one(),
        ]),
        Err(CrtError::ComponentContext { index: 0, .. })
    ));
    assert!(matches!(
        decomposition.project(&ResidueRing::new(13_u8).unwrap().one()),
        Err(CrtError::SourceContext(_))
    ));

    let mut payload: serde_json::Value =
        serde_json::from_str(&decomposition.to_json().unwrap()).unwrap();
    payload["factors"].as_array_mut().unwrap().reverse();
    let error = CrtDecomposition::from_json(&payload.to_string())
        .unwrap_err()
        .to_string();
    assert!(error.contains("canonical increasing-prime powers"));
}
