// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrZ::snf`] and [`CsrZ::rank`].
//!
//! Coverage:
//!
//! 1. **Hand-checked SNF cases** — empty / zero / identity / known
//!    diagonal / Cohen-style worked examples.
//! 2. **Algebraic invariants** — `d_0 | d_1 | ... | d_{r-1}`, all
//!    `d_i ≥ 0`, padding length matches `min(rows, cols)`,
//!    `nonzero_count == rank`.
//! 3. **Cross-checks** — `∏ d_i == |det(A)|` for square nonsingular
//!    (using the existing [`CsrZ::det`]); rank matches between the
//!    HNF-based `CsrZ::rank` and the SNF-derived nonzero-diagonal
//!    count.

use std::cmp::Ordering;

use sparla::storage::{CsrZ, SparseRingStorage};
use sparla::Integer;

fn z(v: i64) -> Integer {
    Integer::from(v)
}

fn csr_from_dense(dense: &[Vec<i64>]) -> CsrZ {
    let rows = dense.len();
    let cols = if rows == 0 { 0 } else { dense[0].len() };
    let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
    for (r, row) in dense.iter().enumerate() {
        assert_eq!(row.len(), cols);
        for (c, &v) in row.iter().enumerate() {
            if v != 0 {
                triples.push((r, c, z(v)));
            }
        }
    }
    CsrZ::from_coo(rows, cols, &triples)
}

fn assert_snf_invariants(diag: &[Integer], rows: usize, cols: usize) {
    assert_eq!(
        diag.len(),
        rows.min(cols),
        "SNF length should equal min(rows, cols)"
    );
    for d in diag {
        assert!(d.cmp0() != Ordering::Less, "SNF entries must be ≥ 0: {}", d);
    }
    // Find rank: number of leading nonzero entries.
    let rank = diag
        .iter()
        .take_while(|d| d.cmp0() != Ordering::Equal)
        .count();
    // Trailing entries are zero.
    for (i, d) in diag.iter().enumerate().skip(rank) {
        assert_eq!(
            *d, 0,
            "diag[{}] should be zero (rank={}): got {}",
            i, rank, d
        );
    }
    // Divisibility chain on the nonzero prefix.
    for win in diag.iter().take(rank).collect::<Vec<_>>().windows(2) {
        assert!(
            win[1].is_divisible(win[0]),
            "divisibility chain failed: {} ∤ {}",
            win[0],
            win[1]
        );
    }
}

// =====================================================================
// SNF: hand-checked cases
// =====================================================================

#[test]
fn snf_of_empty_matrix() {
    assert!(CsrZ::zeros(0, 0).snf().is_empty());
}

#[test]
fn snf_of_zero_matrix() {
    let s = CsrZ::zeros(3, 4).snf();
    assert_eq!(s, vec![z(0), z(0), z(0)]);
}

#[test]
fn snf_of_identity() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    assert_eq!(a.snf(), vec![z(1), z(1), z(1)]);
}

#[test]
fn snf_of_diagonal_in_order() {
    // diag(2, 6): 2 | 6, already in SNF order.
    let a = csr_from_dense(&[vec![2, 0], vec![0, 6]]);
    assert_eq!(a.snf(), vec![z(2), z(6)]);
}

#[test]
fn snf_of_diagonal_out_of_order_gets_reordered() {
    // diag(6, 2): 6 ∤ 2, so the SNF reorders to (2, 6).
    let a = csr_from_dense(&[vec![6, 0], vec![0, 2]]);
    assert_eq!(a.snf(), vec![z(2), z(6)]);
}

#[test]
fn snf_of_coprime_diagonal() {
    // diag(2, 3): gcd 1, lcm 6 → SNF (1, 6).
    let a = csr_from_dense(&[vec![2, 0], vec![0, 3]]);
    assert_eq!(a.snf(), vec![z(1), z(6)]);
}

#[test]
fn snf_2x2_known_case() {
    // [[2, 4], [6, 8]] — HNF is [[2, 0], [0, 4]] (hand-verified
    // earlier). SNF of that is (2, 4): 2 | 4, both positive.
    let a = csr_from_dense(&[vec![2, 4], vec![6, 8]]);
    assert_eq!(a.snf(), vec![z(2), z(4)]);
}

#[test]
fn snf_3x3_coprime_diagonal() {
    // diag(2, 3, 5): gcd-of-all = 1, gcd-of-2x2-minors = 1, full
    // det = 30. SNF = (1, 1, 30).
    let a = csr_from_dense(&[vec![2, 0, 0], vec![0, 3, 0], vec![0, 0, 5]]);
    assert_eq!(a.snf(), vec![z(1), z(1), z(30)]);
}

#[test]
fn snf_singular_padded_with_zeros() {
    // [[2, 4], [4, 8]]: row 1 = 2 · row 0, rank 1. gcd of entries 2,
    // so SNF = (2, 0).
    let a = csr_from_dense(&[vec![2, 4], vec![4, 8]]);
    assert_eq!(a.snf(), vec![z(2), z(0)]);
}

#[test]
fn snf_rectangular_wide() {
    // [[2, 0, 4], [0, 3, 6]]: gcd of entries 1; rank 2. SNF (1, 6).
    let a = csr_from_dense(&[vec![2, 0, 4], vec![0, 3, 6]]);
    let s = a.snf();
    assert_eq!(s, vec![z(1), z(6)]);
}

#[test]
fn snf_rectangular_tall_pads_to_min_dim() {
    let a = csr_from_dense(&[vec![1, 2], vec![3, 4], vec![5, 6]]);
    let s = a.snf();
    // 3 × 2 matrix → SNF length = min(3, 2) = 2.
    assert_eq!(s.len(), 2);
    assert_snf_invariants(&s, 3, 2);
}

// =====================================================================
// SNF: algebraic invariants on random inputs
// =====================================================================

#[test]
fn snf_invariants_random_3x3() {
    let mut state: u64 = 0x5071_1A0E;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 3]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 9 - 4;
            }
        }
        let a = csr_from_dense(&dense);
        let s = a.snf();
        assert_snf_invariants(&s, a.rows(), a.cols());
    }
}

#[test]
fn snf_invariants_random_4x3_tall() {
    let mut state: u64 = 0xBADD_DEC0;
    for _ in 0..10 {
        let mut dense = vec![vec![0i64; 3]; 4];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7 - 3;
            }
        }
        let a = csr_from_dense(&dense);
        let s = a.snf();
        assert_snf_invariants(&s, a.rows(), a.cols());
    }
}

#[test]
fn snf_invariants_random_3x4_wide() {
    let mut state: u64 = 0xF1F1_BEEF;
    for _ in 0..10 {
        let mut dense = vec![vec![0i64; 4]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7 - 3;
            }
        }
        let a = csr_from_dense(&dense);
        let s = a.snf();
        assert_snf_invariants(&s, a.rows(), a.cols());
    }
}

// =====================================================================
// Cross-checks: SNF ↔ det, SNF ↔ rank, SNF ↔ HNF
// =====================================================================

#[test]
fn snf_product_equals_abs_det_for_square_nonsingular() {
    let cases = [
        vec![vec![2, 3], vec![1, 4]],
        vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 10]],
        vec![vec![3, 1, 4], vec![1, 5, 9], vec![2, 6, 5]],
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        let det = a.det();
        let abs_det: Integer = if det.cmp0() == Ordering::Less {
            -det
        } else {
            det
        };
        let s = a.snf();
        let product: Integer = s.iter().fold(Integer::from(1), |acc, d| acc * d);
        assert_eq!(
            product, abs_det,
            "∏ SNF diag ≠ |det| for {:?}; got {:?}",
            dense, s
        );
    }
}

#[test]
fn snf_nonzero_count_matches_rank() {
    let cases = [
        vec![vec![1i64, 2, 3], vec![2, 4, 6], vec![3, 6, 9]], // rank 1
        vec![vec![1, 2, 0], vec![0, 0, 1], vec![1, 2, 1]],    // rank 2
        vec![vec![3, 0, 1, 0], vec![0, 5, 0, 7], vec![1, 1, 1, 1]], // rank 3
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        let r = a.rank();
        let s = a.snf();
        let nz = s.iter().take_while(|d| d.cmp0() != Ordering::Equal).count();
        assert_eq!(nz, r, "SNF nonzero count ≠ rank on {:?}", dense);
    }
}

// =====================================================================
// rank: hand-checked + parity vs HNF pivot count
// =====================================================================

#[test]
fn rank_of_empty_is_zero() {
    assert_eq!(CsrZ::zeros(0, 0).rank(), 0);
}

#[test]
fn rank_of_zero_matrix_is_zero() {
    assert_eq!(CsrZ::zeros(4, 5).rank(), 0);
}

#[test]
fn rank_of_identity() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    assert_eq!(a.rank(), 3);
}

#[test]
fn rank_of_singular_matrix() {
    let a = csr_from_dense(&[vec![1, 2, 3], vec![2, 4, 6], vec![3, 6, 9]]);
    assert_eq!(a.rank(), 1);
}

#[test]
fn rank_rectangular() {
    let a = csr_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]]);
    assert_eq!(a.rank(), 2);
}

#[test]
fn rank_matches_nonzero_hnf_rows() {
    let cases = [
        vec![vec![1i64, 2, 3], vec![2, 4, 6], vec![3, 6, 9]],
        vec![vec![1, 2, 0], vec![0, 0, 1], vec![1, 2, 1]],
        vec![vec![3, -1, 0, 4], vec![-2, 5, 1, 0], vec![0, 0, -3, 6]],
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        let rank = a.rank();
        let h = a.hnf().to_dense();
        let nonzero_rows = h
            .iter()
            .filter(|row| row.iter().any(|v| v.cmp0() != Ordering::Equal))
            .count();
        assert_eq!(rank, nonzero_rows, "rank ≠ nonzero HNF rows on {:?}", dense);
    }
}

#[test]
fn rank_full_rank_random_3x3() {
    let mut state: u64 = 0x1234_5678_DEAD_C0DE;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 3]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7;
            }
        }
        let a = csr_from_dense(&dense);
        // For any matrix: rank == number of nonzero HNF rows.
        let rank = a.rank();
        let h = a.hnf().to_dense();
        let nz = h
            .iter()
            .filter(|row| row.iter().any(|v| v.cmp0() != Ordering::Equal))
            .count();
        assert_eq!(rank, nz);
    }
}
