// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::{Matrix, SmithNormalForm, SmithNormalFormWithTransforms};
use noether::{Assumption, Integer, SemanticType, Z};

fn integer(value: i64) -> Integer {
    Integer::from(value)
}

#[test]
fn smith_form_uses_the_fast_no_transform_contract() {
    let matrix =
        Matrix::from_rows(Z, [[integer(2), integer(4)], [integer(6), integer(8)]]).unwrap();
    let result = matrix.smith_normal_form();

    // gcd(entries) = 2 and |det| = 8, hence the invariants are 2 and 4.
    assert_eq!(result.invariant_factors(), &[integer(2), integer(4)]);
    assert_eq!(result.rank(), 2);
    assert_eq!(
        result.form().rows().collect::<Vec<_>>(),
        vec![vec![integer(2), integer(0)], vec![integer(0), integer(4)]]
    );
    assert_eq!(result.source(), &matrix);
    assert!(result.validity().is_unconditional());
    assert_eq!(result.verification().strategy(), "exact");
    assert!(result.verification().checked_primes().is_empty());
    result.verify().unwrap();
    assert_eq!(SmithNormalForm::TYPE_ID, "noether/linalg/SmithNormalForm@1");
}

#[test]
fn rectangular_and_rank_deficient_forms_keep_exact_shape() {
    let rectangular = Matrix::from_rows(
        Z,
        [
            [integer(2), integer(4), integer(4)],
            [integer(6), integer(6), integer(12)],
        ],
    )
    .unwrap();
    let rectangular_result = rectangular.smith_normal_form();
    assert_eq!(rectangular_result.form().shape(), (2, 3));
    assert_eq!(
        rectangular_result.invariant_factors(),
        &[integer(2), integer(6)]
    );
    rectangular_result.verify().unwrap();

    let deficient =
        Matrix::from_rows(Z, [[integer(2), integer(4)], [integer(1), integer(2)]]).unwrap();
    let deficient_result = deficient.smith_normal_form();
    assert_eq!(deficient_result.invariant_factors(), &[integer(1)]);
    assert_eq!(deficient_result.rank(), 1);
    assert_eq!(
        deficient_result.form().rows().collect::<Vec<_>>(),
        vec![vec![integer(1), integer(0)], vec![integer(0), integer(0)]]
    );
    deficient_result.verify().unwrap();
}

#[test]
fn zero_dimensions_and_arbitrary_precision_are_preserved() {
    let empty = Matrix::zeros(Z, 0, 3).smith_normal_form();
    assert_eq!(empty.form().shape(), (0, 3));
    assert!(empty.invariant_factors().is_empty());
    empty.verify().unwrap();

    let large = "100000000000000000000000000000000000003"
        .parse::<Integer>()
        .unwrap();
    let matrix = Matrix::from_rows(
        Z,
        [
            [large.clone(), Integer::zero()],
            [Integer::zero(), &large * &integer(2)],
        ],
    )
    .unwrap();
    let result = matrix.smith_normal_form();
    assert_eq!(
        result.invariant_factors(),
        &[large.clone(), &large * &integer(2)]
    );
    result.verify().unwrap();
    assert_eq!(
        result.canonical_bytes().unwrap(),
        matrix.smith_normal_form().canonical_bytes().unwrap()
    );
}

#[test]
fn explicit_transform_result_verifies_both_unimodular_witnesses() {
    let matrix = Matrix::from_rows(
        Z,
        [
            [integer(2), integer(4), integer(4)],
            [integer(6), integer(6), integer(12)],
        ],
    )
    .unwrap();
    let result = matrix.smith_normal_form_with_transforms().unwrap();

    assert_eq!(result.smith().source(), &matrix);
    assert_eq!(
        result.smith().invariant_factors(),
        &[integer(2), integer(6)]
    );
    assert_eq!(result.left().shape(), (2, 2));
    assert_eq!(result.right().shape(), (3, 3));
    assert_eq!(
        result
            .left()
            .matmul(&matrix)
            .unwrap()
            .matmul(result.right())
            .unwrap(),
        *result.smith().form()
    );
    assert!(result.smith().validity().is_unconditional());
    result.verify().unwrap();
    assert_eq!(
        SmithNormalFormWithTransforms::TYPE_ID,
        "noether/linalg/SmithNormalFormWithTransforms@1"
    );
    assert_eq!(
        result.canonical_bytes().unwrap(),
        matrix
            .smith_normal_form_with_transforms()
            .unwrap()
            .canonical_bytes()
            .unwrap()
    );
}

#[test]
fn explicit_transform_result_handles_empty_rectangles() {
    let matrix = Matrix::zeros(Z, 0, 3);
    let result = matrix.smith_normal_form_with_transforms().unwrap();
    assert_eq!(result.smith().form().shape(), (0, 3));
    assert_eq!(result.left().shape(), (0, 0));
    assert_eq!(result.right().shape(), (3, 3));
    result.verify().unwrap();
}

#[test]
fn smooth_local_strategy_records_its_exact_assumption_and_prime_checks() {
    let matrix = Matrix::from_rows(
        Z,
        [
            [integer(2), integer(0), integer(0)],
            [integer(0), integer(6), integer(0)],
        ],
    )
    .unwrap();
    let result = matrix.smith_normal_form_assuming_smooth(3).unwrap();

    assert_eq!(result.invariant_factors(), &[integer(2), integer(6)]);
    assert_eq!(result.verification().strategy(), "smooth_local");
    assert_eq!(
        result.verification().checked_primes(),
        &[integer(2), integer(3)]
    );
    assert_eq!(result.verification().smooth_bound(), Some(&integer(3)));
    assert_eq!(
        result.verification().deterministic_prime_bound(),
        Some(&integer(72))
    );
    let checks = result.verification().local_checks().collect::<Vec<_>>();
    assert_eq!(checks[0].2, &[1, 1]);
    assert_eq!(checks[1].2, &[0, 1]);
    assert_eq!(
        result.validity().assumptions().iter().collect::<Vec<_>>(),
        vec![&Assumption::TorsionOrderIsSmooth { bound: integer(3) }]
    );
    result.verify().unwrap();
}

#[test]
fn a_false_smoothness_hypothesis_is_visible_instead_of_claimed_as_exact() {
    let matrix =
        Matrix::from_rows(Z, [[integer(2), integer(0)], [integer(0), integer(10)]]).unwrap();
    let conditional = matrix.smith_normal_form_assuming_smooth(3).unwrap();
    let exact = matrix.smith_normal_form();

    assert_eq!(conditional.invariant_factors(), &[integer(2), integer(2)]);
    assert_eq!(exact.invariant_factors(), &[integer(2), integer(10)]);
    assert!(!conditional.validity().is_unconditional());
    conditional.verify().unwrap();
}

#[test]
fn covering_the_deterministic_bound_upgrades_smooth_smith_to_unconditional() {
    let matrix = Matrix::identity(Z, 2);
    let result = matrix.smith_normal_form_assuming_smooth(2).unwrap();

    assert_eq!(result.invariant_factors(), &[integer(1), integer(1)]);
    assert_eq!(
        result.verification().deterministic_prime_bound(),
        Some(&integer(2))
    );
    assert!(result.validity().is_unconditional());
    result.verify().unwrap();
}

#[test]
fn smooth_smith_fails_closed_at_invalid_bounds_and_word_precision() {
    let matrix = Matrix::identity(Z, 1);
    assert!(matrix.smith_normal_form_assuming_smooth(1).is_err());
    assert!(matrix.smith_normal_form_assuming_smooth(100_001).is_err());

    let high_valuation = Matrix::from_rows(Z, [[integer(2).pow(63)]]).unwrap();
    assert!(high_valuation.smith_normal_form_assuming_smooth(2).is_err());
}

#[test]
fn smooth_local_strategy_matches_exact_smith_on_all_small_two_by_two_matrices() {
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
                    // Every nonzero determinant in this box has absolute value
                    // at most 8; rank-one invariant factors divide entries of
                    // size at most 2. Thus every torsion prime is at most 7.
                    let matrix =
                        Matrix::from_rows(Z, [[integer(a), integer(b)], [integer(c), integer(d)]])
                            .unwrap();
                    let smooth = matrix.smith_normal_form_assuming_smooth(7).unwrap();
                    let exact = matrix.smith_normal_form();
                    assert_eq!(
                        smooth.form(),
                        exact.form(),
                        "local/exact mismatch for [[{a}, {b}], [{c}, {d}]]"
                    );
                }
            }
        }
    }
}

#[test]
fn smooth_local_strategy_matches_exact_smith_on_rectangular_rank_profiles() {
    let mut state = 0x5eed_u64;
    for _ in 0..128 {
        let mut rows = Vec::new();
        for _ in 0..3 {
            let mut row = Vec::new();
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                row.push(integer(((state >> 32) % 5) as i64 - 2));
            }
            rows.push(row);
        }
        // Hadamard bounds every 3x3 minor in this box by 42, while
        // lower-rank minors are smaller, so bound 47 covers all torsion primes.
        let matrix = Matrix::from_rows(Z, rows).unwrap();
        let smooth = matrix.smith_normal_form_assuming_smooth(47).unwrap();
        let exact = matrix.smith_normal_form();
        assert_eq!(smooth.form(), exact.form());
    }
}
