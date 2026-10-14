// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Correctness against exhaustive exact references, adversarial behavior, and
//! ill-conditioned positive-definite forms.

use crate::cvp::cvp_enumerate;
use crate::error::EnumError;
use crate::gso::exact_quadratic;
use crate::pruned::{pruned_enumeration, PruningMode};
use crate::radius::{enumeration_radius_set, gaussian_heuristic_bound};
use crate::svp::svp_enumerate;
use crate::theta::representation_vectors_collect;
use flint_core::Integer;

// ── brute-force reference ────────────────────────────────────────────

fn canon(v: &[i64]) -> Vec<i64> {
    for &x in v {
        if x != 0 {
            if x < 0 {
                return v.iter().map(|&e| -e).collect();
            }
            break;
        }
    }
    v.to_vec()
}

/// Every ± pair rep with `0 < vᵀGv ≤ bound`, by exhaustive box search.
fn brute_svp(gram: &[Vec<Integer>], bound: i64, half: i64) -> Vec<(Integer, Vec<i64>)> {
    let n = gram.len();
    let mut idx = vec![-half; n];
    let mut out = std::collections::BTreeSet::new();
    loop {
        let norm = exact_quadratic(gram, &idx).unwrap();
        if norm.is_positive() && norm <= bound {
            let c = canon(&idx);
            out.insert((norm, c));
        }
        // increment mixed-radix odometer over [-half, half]^n
        let mut k = 0;
        loop {
            if k == n {
                let mut v: Vec<_> = out.into_iter().collect();
                v.sort();
                return v;
            }
            idx[k] += 1;
            if idx[k] > half {
                idx[k] = -half;
                k += 1;
            } else {
                break;
            }
        }
    }
}

fn i_n(n: usize) -> Vec<Vec<Integer>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| Integer::from(if i == j { 1 } else { 0 }))
                .collect()
        })
        .collect()
}

fn a_n(n: usize) -> Vec<Vec<Integer>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    Integer::from(if i == j {
                        2
                    } else if i.abs_diff(j) == 1 {
                        -1
                    } else {
                        0
                    })
                })
                .collect()
        })
        .collect()
}

fn integer_matrix(rows: Vec<Vec<i64>>) -> Vec<Vec<Integer>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(Integer::from).collect())
        .collect()
}

fn e8() -> Vec<Vec<Integer>> {
    integer_matrix(vec![
        vec![2, -1, 0, 0, 0, 0, 0, 0],
        vec![-1, 2, -1, 0, 0, 0, 0, 0],
        vec![0, -1, 2, -1, 0, 0, 0, 0],
        vec![0, 0, -1, 2, -1, 0, 0, 0],
        vec![0, 0, 0, -1, 2, -1, 0, -1],
        vec![0, 0, 0, 0, -1, 2, -1, 0],
        vec![0, 0, 0, 0, 0, -1, 2, 0],
        vec![0, 0, 0, 0, -1, 0, 0, 2],
    ])
}

// ── SVP parity vs brute force ────────────────────────────────────────

fn svp_set(gram: &[Vec<Integer>], bound: i64) -> Vec<(Integer, Vec<i64>)> {
    let sv = svp_enumerate(gram, Some(Integer::from(bound)), None).unwrap();
    let mut out: Vec<(Integer, Vec<i64>)> = sv
        .vectors
        .iter()
        .map(|v| (exact_quadratic(gram, v).unwrap(), v.clone()))
        .collect();
    out.sort();
    out
}

#[test]
fn svp_matches_brute_i2() {
    let g = i_n(2);
    assert_eq!(svp_set(&g, 4), brute_svp(&g, 4, 3));
}

#[test]
fn svp_diagonal_fast_path_does_not_assume_sorted_lengths() {
    let g = integer_matrix(vec![vec![25, 0, 0], vec![0, 1, 0], vec![0, 0, 9]]);
    assert_eq!(svp_set(&g, 25), brute_svp(&g, 25, 5));
}

#[test]
fn svp_matches_brute_a2() {
    let g = a_n(2);
    assert_eq!(svp_set(&g, 8), brute_svp(&g, 8, 4));
}

#[test]
fn svp_matches_brute_a4() {
    let g = a_n(4);
    assert_eq!(svp_set(&g, 6), brute_svp(&g, 6, 3));
}

#[test]
fn svp_matches_brute_i3() {
    let g = i_n(3);
    assert_eq!(svp_set(&g, 9), brute_svp(&g, 9, 4));
}

#[test]
fn e8_theta_series_coefficients() {
    // E8's theta series is 1 + 240q + 2160q² + 6720q³ + …; in ± pairs that
    // is 120 / 1080 / 3360 vectors of norm 2 / 4 / 6. A brute box can't be
    // used here — the E8 highest root is (2,3,4,6,5,4,3,2) in the Cartan
    // basis, so even minimal vectors have coordinates up to 6.
    let g = e8();
    assert_eq!(
        representation_vectors_collect(&g, Integer::from(2))
            .unwrap()
            .len(),
        120
    );
    assert_eq!(
        representation_vectors_collect(&g, Integer::from(4))
            .unwrap()
            .len(),
        1080
    );
    assert_eq!(
        representation_vectors_collect(&g, Integer::from(6))
            .unwrap()
            .len(),
        3360
    );
    // svp with a cumulative bound returns the union of the shells.
    let sv = svp_enumerate(&g, Some(Integer::from(6)), None).unwrap();
    assert_eq!(sv.vectors.len(), 120 + 1080 + 3360);
}

#[test]
fn e8_min_shell_is_240() {
    // E8 kissing number 240 = 120 pairs, all of norm 2.
    let sv = svp_enumerate(&e8(), None, None).unwrap();
    assert_eq!(sv.vectors.len(), 120);
    assert_eq!(sv.count, 240);
    assert_eq!(sv.max_norm, 2);
}

#[test]
fn a2_min_shell() {
    // A2 has 6 roots = 3 pairs, norm 2.
    let sv = svp_enumerate(&a_n(2), None, None).unwrap();
    assert_eq!(sv.vectors.len(), 3);
    assert_eq!(sv.max_norm, 2);
}

// ── ill-conditioned corpus ───────────────────────────────────────────

/// Highly skewed PD Gram: `B B^T` for a near-degenerate integer basis. The
/// condition number is large enough to expose cancellation in a floating
/// factorization.
fn ill_conditioned(scale: i64) -> Vec<Vec<Integer>> {
    // basis rows: e1*scale and e1*scale + e2  → very acute angle. G = B Bᵀ.
    let b = [[scale, 0], [scale, 1]];
    let dot = |r: usize, c: usize| b[r][0] * b[c][0] + b[r][1] * b[c][1];
    integer_matrix(vec![vec![dot(0, 0), dot(0, 1)], vec![dot(1, 0), dot(1, 1)]])
}

#[test]
fn ill_conditioned_runs_and_matches_brute() {
    for scale in [10_i64, 100, 1000] {
        let g = ill_conditioned(scale);
        // shortest is (1,-1) with norm 1 (e2). Enumerate a modest bound.
        let bound = 5;
        let got = svp_set(&g, bound);
        let want = brute_svp(&g, bound, 2 * scale + 3);
        assert_eq!(got, want, "ill-conditioned scale={scale}");
        assert!(!got.is_empty());
    }
}

#[test]
fn ill_conditioned_no_bound_min() {
    // The true minimum is norm 1 (the short e2 direction), found without a
    // complete even though the diagonal is huge (scale²).
    let g = ill_conditioned(1000);
    let sv = svp_enumerate(&g, None, None).unwrap();
    assert_eq!(sv.max_norm, 1);
}

// ── extreme ill-conditioned band ─────────────────────────────────────
//
// scale=1e8 → Gram entry s²=1e16 > 2⁵³, cond ≈ 1e16; scale=1e9 → s²=1e18,
// cond ≈ 1e18. An f64 Cholesky cancels the smallest pivot here (G[1][1] − s²
// loses the +1) and falsely rejects a genuinely PD form; the exact LDLᵀ GSO
// enumerates correctly.
//
// Reference: an exact exhaustive box. For `ill_conditioned(scale)` the inverse
// Gram diagonal is ≈ 1 (G⁻¹ = [[(s²+1)/s², −1], [−1, 1]]), so every v with
// vᵀGv ≤ B satisfies |vᵢ| ≤ √(B·(G⁻¹)ᵢᵢ) < √B + 1 ≤ 4 at B=5 — a 9×9 box the
// brute reference walks with an arbitrary-precision exact norm.

#[test]
fn extreme_ill_conditioned_enumerates_not_errors() {
    let bound = 5;
    for scale in [100_000_000_i64, 1_000_000_000] {
        let g = ill_conditioned(scale);
        // A bounded query must enumerate rather than misclassify the form.
        let got = svp_set(&g, bound);
        let want = brute_svp(&g, bound, 4);
        assert_eq!(got, want, "extreme ill-conditioned scale={scale}");
        // The two canonical short vectors: (1,-1) norm 1 and (2,-2) norm 4.
        assert_eq!(
            got,
            vec![
                (Integer::from(1), vec![1, -1]),
                (Integer::from(4), vec![2, -2]),
            ],
            "scale={scale}"
        );
        // Exact LLL pre-reduction makes the no-bound minimum-shell query use a
        // short reduced diagonal instead of the original s² diagonal.
        let minimum = svp_enumerate(&g, None, None).unwrap();
        assert!(minimum.complete);
        assert_eq!(minimum.max_norm, Integer::one());
        assert_eq!(minimum.vectors, vec![vec![1, -1]]);
    }
}

// ── CVP ──────────────────────────────────────────────────────────────

fn brute_cvp(gram: &[Vec<Integer>], num: &[i64], den: i64, half: i64) -> (Integer, Vec<Vec<i64>>) {
    let n = gram.len();
    let mut idx = vec![-half; n];
    let mut best: Option<Integer> = None;
    let mut winners: Vec<Vec<i64>> = Vec::new();
    loop {
        let w: Vec<i64> = (0..n).map(|i| den * idx[i] - num[i]).collect();
        let d = exact_quadratic(gram, &w).unwrap(); // = q²·dist²
        match &best {
            None => {
                best = Some(d.clone());
                winners.push(idx.clone());
            }
            Some(b) if d < *b => {
                best = Some(d.clone());
                winners.clear();
                winners.push(idx.clone());
            }
            Some(b) if d == *b => winners.push(idx.clone()),
            _ => {}
        }
        let mut k = 0;
        loop {
            if k == n {
                winners.sort();
                return (best.unwrap(), winners);
            }
            idx[k] += 1;
            if idx[k] > half {
                idx[k] = -half;
                k += 1;
            } else {
                break;
            }
        }
    }
}

#[test]
fn cvp_lattice_point_target_is_exact() {
    let g = i_n(3);
    let cv = cvp_enumerate(&g, &[3, -1, 2], 1, None).unwrap();
    assert_eq!(cv.distance_sq_num, 0);
    assert_eq!(cv.distance_sq_den, 1);
    assert_eq!(cv.vectors, vec![vec![3, -1, 2]]);
}

#[test]
fn cvp_matches_brute_i2_rational() {
    let g = i_n(2);
    // target (1/2, 1/2): four equidistant closest lattice points.
    let cv = cvp_enumerate(&g, &[1, 1], 2, None).unwrap();
    let (bn, bw) = brute_cvp(&g, &[1, 1], 2, 4);
    assert_eq!(cv.distance_sq_num, bn);
    assert_eq!(cv.distance_sq_den, 4);
    let mut got = cv.vectors.clone();
    got.sort();
    assert_eq!(got, bw);
    assert_eq!(got.len(), 4);
}

#[test]
fn cvp_matches_brute_a3_deep_hole() {
    let g = a_n(3);
    for (num, den) in [(vec![1, 1, 1], 3), (vec![5, -2, 3], 4), (vec![7, 0, -3], 2)] {
        let cv = cvp_enumerate(&g, &num, den, None).unwrap();
        let (bn, bw) = brute_cvp(&g, &num, den, 8);
        assert_eq!(cv.distance_sq_num, bn, "num={num:?} den={den}");
        let mut got = cv.vectors.clone();
        got.sort();
        assert_eq!(got, bw, "winners num={num:?} den={den}");
    }
}

// ── theta / representation vectors ───────────────────────────────────

#[test]
fn theta_e8_norm2_is_120() {
    let reps = representation_vectors_collect(&e8(), Integer::from(2)).unwrap();
    assert_eq!(reps.len(), 120);
    for v in &reps {
        assert_eq!(exact_quadratic(&e8(), v).unwrap(), 2);
    }
}

#[test]
fn theta_zero_is_empty() {
    assert!(representation_vectors_collect(&i_n(3), Integer::from(0))
        .unwrap()
        .is_empty());
}

// ── Gaussian heuristic + radius ──────────────────────────────────────

#[test]
fn gaussian_heuristic_positive() {
    let gh = gaussian_heuristic_bound(&e8()).unwrap();
    assert!(gh > 0.0 && gh.is_finite());
}

#[test]
fn radius_set_grows_with_count() {
    let g = i_n(4);
    let b10 = enumeration_radius_set(&g, 10).unwrap();
    let b1000 = enumeration_radius_set(&g, 1000).unwrap();
    assert!(b1000 > b10);
    // Sanity: the bound should actually admit a comparable number of points.
    let sv = svp_enumerate(&g, Some(b1000), None).unwrap();
    assert!(sv.count > 100);
}

// ── pruning ──────────────────────────────────────────────────────────

#[test]
fn pruning_all_ones_equals_svp() {
    let g = a_n(4);
    let ones = vec![1.0; 4];
    let pr = pruned_enumeration(&g, Integer::from(6), &ones).unwrap();
    let sv = svp_enumerate(&g, Some(Integer::from(6)), None).unwrap();
    assert_eq!(pr.vectors, sv.vectors);
}

#[test]
fn pruning_generated_curves_are_subset() {
    let g = a_n(4);
    let bound = 8;
    let full = svp_enumerate(&g, Some(Integer::from(bound)), None).unwrap();
    let full_set: std::collections::BTreeSet<_> = full.vectors.iter().cloned().collect();
    for mode in [PruningMode::Linear, PruningMode::Extreme] {
        let p = mode.pruning_vector(4);
        let pr = pruned_enumeration(&g, Integer::from(bound), &p).unwrap();
        for v in &pr.vectors {
            assert!(
                full_set.contains(v),
                "{mode:?} produced a spurious vector {v:?}"
            );
            assert!(exact_quadratic(&g, v).unwrap() <= bound);
        }
    }
}

// ── adversarial / graceful failure ───────────────────────────────────

#[test]
fn non_square_gram() {
    let g = integer_matrix(vec![vec![1, 0, 0], vec![0, 1, 0]]);
    assert!(matches!(
        svp_enumerate(&g, Some(Integer::from(4)), None),
        Err(EnumError::InvalidShape { .. })
    ));
}

#[test]
fn asymmetric_gram() {
    let g = integer_matrix(vec![vec![2, 1], vec![-1, 2]]);
    assert!(matches!(
        svp_enumerate(&g, Some(Integer::from(4)), None),
        Err(EnumError::Asymmetric { .. })
    ));
}

#[test]
fn non_pd_gram() {
    let g = integer_matrix(vec![vec![1, 2], vec![2, 1]]);
    assert_eq!(
        svp_enumerate(&g, Some(Integer::from(4)), None),
        Err(EnumError::NotPositiveDefinite)
    );
}

#[test]
fn negative_bound() {
    assert_eq!(
        svp_enumerate(&i_n(2), Some(Integer::from(-1)), None),
        Err(EnumError::InvalidBound {
            bound: Integer::from(-1)
        })
    );
}

#[test]
fn enumeration_overflow_cap() {
    // Large bound, no max_count, moderate dimension → estimate over 2e7.
    let g = i_n(8);
    match svp_enumerate(&g, Some(Integer::from(1_000_000)), None) {
        Err(EnumError::EnumerationOverflow { estimate }) => assert!(estimate > 2.0e7),
        other => panic!("expected overflow, got {other:?}"),
    }
    // …but with a max_count the cap is bypassed.
    let capped = svp_enumerate(&g, Some(Integer::from(1_000_000)), Some(50)).unwrap();
    assert_eq!(capped.vectors.len(), 50);
}

#[test]
fn cvp_target_shape_mismatch() {
    assert!(matches!(
        cvp_enumerate(&i_n(3), &[1, 2], 1, None),
        Err(EnumError::InvalidShape { .. })
    ));
}

#[test]
fn cvp_bad_denominator() {
    assert!(matches!(
        cvp_enumerate(&i_n(2), &[1, 1], 0, None),
        Err(EnumError::InvalidShape { .. })
    ));
}

#[test]
fn pruning_shape_mismatch() {
    assert!(matches!(
        pruned_enumeration(&i_n(3), Integer::from(4), &[1.0, 1.0]),
        Err(EnumError::PruningShape { .. })
    ));
}

#[test]
fn pruning_non_monotone_rejected() {
    assert!(matches!(
        pruned_enumeration(&i_n(3), Integer::from(4), &[1.0, 0.5, 1.0]),
        Err(EnumError::PruningShape { .. })
    ));
}

#[test]
fn max_count_caps_pairs() {
    let g = i_n(2);
    let sv = svp_enumerate(&g, Some(Integer::from(100)), Some(3)).unwrap();
    assert_eq!(sv.vectors.len(), 3);
    assert_eq!(sv.count, 6);
    assert_eq!(sv.bound, Some(Integer::from(100)));
    assert!(!sv.complete);
    // every returned vector is a genuine short vector
    for v in &sv.vectors {
        let no = exact_quadratic(&g, v).unwrap();
        assert!(no > 0 && no <= 100);
    }
}

#[test]
fn zero_max_count_is_empty() {
    let sv = svp_enumerate(&a_n(2), None, Some(0)).unwrap();
    assert!(sv.vectors.is_empty());
    assert_eq!(sv.count, 0);
    assert!(!sv.complete);
}

#[test]
fn no_bound_cap_is_a_prefix_of_the_true_minimum_shell() {
    let full = svp_enumerate(&a_n(4), None, None).unwrap();
    let capped = svp_enumerate(&a_n(4), None, Some(2)).unwrap();
    assert_eq!(capped.vectors, full.vectors[..2]);
    assert_eq!(capped.max_norm, full.max_norm);
    assert!(full.complete);
    assert!(!capped.complete);
}

#[test]
fn cap_at_or_above_the_exact_result_is_complete() {
    let exact = svp_enumerate(&i_n(2), Some(Integer::one()), Some(2)).unwrap();
    assert_eq!(exact.vectors.len(), 2);
    assert!(exact.complete);

    let roomy = svp_enumerate(&i_n(2), Some(Integer::one()), Some(3)).unwrap();
    assert_eq!(roomy.vectors, exact.vectors);
    assert!(roomy.complete);
}

#[test]
fn empty_gram_has_bounded_empty_shell_but_no_minimum_shell() {
    let g: Vec<Vec<Integer>> = Vec::new();
    assert_eq!(
        svp_enumerate(&g, None, None),
        Err(EnumError::NoNonzeroVectors)
    );
    let sv = svp_enumerate(&g, Some(Integer::zero()), None).unwrap();
    assert_eq!(sv.count, 0);
    assert!(sv.vectors.is_empty());
    assert!(sv.complete);
    assert_eq!(
        svp_enumerate(&g, Some(Integer::from(-1_i64)), None),
        Err(EnumError::InvalidBound {
            bound: Integer::from(-1_i64)
        })
    );
}
