// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::ResidueRing;
use noether::linalg::Matrix;
use noether::{Integer, Ring, Z};

#[test]
fn integer_hermite_form_is_canonical_and_idempotent() {
    let matrix = Matrix::from_rows(
        Z,
        [
            [Integer::from(2), Integer::from(4)],
            [Integer::from(6), Integer::from(8)],
        ],
    )
    .unwrap();
    let expected = Matrix::from_rows(
        Z,
        [
            [Integer::from(2), Integer::zero()],
            [Integer::zero(), Integer::from(4)],
        ],
    )
    .unwrap();

    let hermite = matrix.hermite_normal_form();

    assert_eq!(hermite, expected);
    assert_eq!(hermite.hermite_normal_form(), hermite);
}

#[test]
fn residue_howell_form_preserves_the_hand_checked_z25_row_module() {
    let ring = ResidueRing::new(25).unwrap();
    let matrix = Matrix::from_rows(
        ring.clone(),
        [
            [ring.element(5), ring.element(10)],
            [ring.zero(), ring.element(5)],
        ],
    )
    .unwrap();
    let expected = Matrix::from_rows(
        ring.clone(),
        [
            [ring.element(5), ring.zero()],
            [ring.zero(), ring.element(5)],
        ],
    )
    .unwrap();

    let howell = matrix.howell_normal_form().unwrap();

    assert_eq!(howell, expected);
    assert_eq!(howell.howell_normal_form().unwrap(), howell);
}

#[test]
fn residue_howell_form_explicitly_completes_a_wide_matrix() {
    let ring = ResidueRing::new(25).unwrap();
    let matrix = Matrix::from_rows(ring.clone(), [[ring.element(5), ring.element(10)]]).unwrap();
    let expected = Matrix::from_rows(
        ring.clone(),
        [
            [ring.element(5), ring.element(10)],
            [ring.zero(), ring.zero()],
        ],
    )
    .unwrap();

    let howell = matrix.howell_normal_form().unwrap();

    assert_eq!(howell, expected);
    assert_eq!(howell.howell_normal_form().unwrap(), howell);
}

#[test]
fn residue_howell_form_uses_arbitrary_modulus_flint_storage() {
    let modulus = Integer::from(u64::MAX) + &Integer::from(2);
    let ring = ResidueRing::new(modulus).unwrap();
    let matrix = Matrix::identity(ring.clone(), 1);

    assert_eq!(
        matrix.howell_normal_form().unwrap(),
        Matrix::identity(ring, 1)
    );
}
