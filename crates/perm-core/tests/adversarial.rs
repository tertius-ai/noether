// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial cases: every checked error path returns a typed
//! [`GroupError`], not a panic, not silent garbage.

use perm_core::{element_membership_test, GroupError, NotAPermutationReason, Perm, PermGroup};

#[test]
fn mismatched_generator_degree_is_error() {
    // First generator has degree 5, second degree 4.
    let g1 = Perm::from_images(&[1, 0, 2, 3, 4]).unwrap();
    let g2 = Perm::from_images(&[1, 0, 2, 3]).unwrap();
    let err = PermGroup::from_generators(5, &[g1, g2]).unwrap_err();
    assert!(
        matches!(
            err,
            GroupError::DegreeMismatch {
                got: 4,
                expected: 5
            }
        ),
        "expected DegreeMismatch, got {err:?}"
    );
}

#[test]
fn non_permutation_image_out_of_range_is_error() {
    let bad = Perm::from_images(&[0, 1, 2, 7]);
    let err = bad.unwrap_err();
    assert!(matches!(
        err,
        GroupError::NotAPermutation {
            generator: 0,
            reason: NotAPermutationReason::OutOfRange { .. }
        }
    ));
}

#[test]
fn non_permutation_repeated_image_is_error() {
    let bad = Perm::from_images(&[0, 1, 2, 2]);
    let err = bad.unwrap_err();
    assert!(matches!(
        err,
        GroupError::NotAPermutation {
            generator: 0,
            reason: NotAPermutationReason::RepeatedImage { image: 2 }
        }
    ));
}

#[test]
fn trivial_group_from_identity_only_is_order_one_not_panic() {
    let id = Perm::identity(4);
    let g = PermGroup::from_generators(4, &[id]).unwrap();
    assert!(g.base().is_empty());
    use perm_core::BigUInt;
    assert_eq!(g.order(), BigUInt::one());
}

#[test]
fn membership_test_on_wrong_degree_is_error_not_silent_false() {
    let g1 = Perm::from_images(&[1, 0, 2, 3, 4]).unwrap();
    let g2 = Perm::from_images(&[1, 2, 3, 4, 0]).unwrap();
    let s5 = PermGroup::from_generators(5, &[g1, g2]).unwrap();
    let wrong = Perm::from_images(&[1, 0, 2, 3]).unwrap(); // degree 4
    let err = element_membership_test(&s5, &wrong).unwrap_err();
    assert!(matches!(
        err,
        GroupError::DegreeMismatch {
            got: 4,
            expected: 5
        }
    ));
}

#[test]
fn orbit_request_on_out_of_range_point_is_error() {
    let g1 = Perm::from_images(&[1, 0, 2]).unwrap();
    let g3 = PermGroup::from_generators(3, &[g1]).unwrap();
    let err = g3.orbit(7).unwrap_err();
    assert!(matches!(
        err,
        GroupError::DegreeMismatch {
            got: 7,
            expected: 3
        }
    ));
}
