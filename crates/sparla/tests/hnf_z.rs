// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrZ::hnf`].
//!
//! Same three layers as [`tests/rref_fp.rs`]:
//!
//! 1. **Hand-checked cases.** Tiny matrices with HNF derivable by
//!    hand or known by inspection (identity, zero, single row/col,
//!    diagonal, classic worked examples from Cohen's "A Course in
//!    Computational Algebraic Number Theory").
//! 2. **Canonical HNF invariants.** Pivots positive, pivot columns
//!    strictly ascending, above-pivot entries in `[0, pivot_val)`,
//!    rows below the last pivot empty.
//! 3. **Dense-reference parity.** A textbook HNF algorithm written
//!    inline using FLINT-backed `sparla::Integer` (so values can grow without
//!    overflow) is the oracle. HNF is unique, so two correct
//!    implementations must agree bit-for-bit.

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

fn dense_from_i64(mat: &[Vec<i64>]) -> Vec<Vec<Integer>> {
    mat.iter()
        .map(|row| row.iter().map(|&v| z(v)).collect())
        .collect()
}

/// Textbook dense Hermite Normal Form, the oracle for parity.
///
/// Same algorithm as the sparse kernel: column-by-column Euclidean
/// reduction on rows `[pivot_row..rows)`, then sign-normalise and
/// reduce above. Uses [`sparla::Integer`] throughout — values can grow
/// past i64 during reduction and we never want to clip.
// The dense oracle is most readable in textbook index style: the
// broadcast row op reads `m[pivot_row][j]` and writes `m[r][j]` in
// lockstep. Iterators here would bury the algorithm in
// `split_at_mut` plumbing.
#[allow(clippy::needless_range_loop)]
fn dense_hnf(mat: &[Vec<i64>]) -> Vec<Vec<Integer>> {
    let rows = mat.len();
    let cols = if rows == 0 { 0 } else { mat[0].len() };
    let mut m: Vec<Vec<Integer>> = dense_from_i64(mat);

    let mut pivot_row = 0usize;
    for c in 0..cols {
        if pivot_row >= rows {
            break;
        }

        // Euclidean-drive column c on rows [pivot_row..rows).
        loop {
            let nz: Vec<usize> = (pivot_row..rows)
                .filter(|&r| m[r][c].cmp0() != Ordering::Equal)
                .collect();
            if nz.len() <= 1 {
                break;
            }
            // Sort indices by |m[r][c]| ascending.
            let mut sorted = nz.clone();
            sorted.sort_by(|&a, &b| {
                let av = m[a][c].abs();
                let bv = m[b][c].abs();
                av.cmp(&bv)
            });
            let p = sorted[0];
            let o = sorted[1];
            let q = m[o][c].div_floor(&m[p][c]);
            if q.cmp0() != Ordering::Equal {
                for j in 0..cols {
                    let prod = &q * &m[p][j];
                    m[o][j] -= prod;
                }
            }
        }

        let nz_idx = (pivot_row..rows).find(|&r| m[r][c].cmp0() != Ordering::Equal);
        if let Some(r) = nz_idx {
            if r != pivot_row {
                m.swap(pivot_row, r);
            }
            // Sign-normalise the pivot row.
            if m[pivot_row][c].cmp0() == Ordering::Less {
                for j in 0..cols {
                    let v = std::mem::take(&mut m[pivot_row][j]);
                    m[pivot_row][j] = -v;
                }
            }
            // Reduce above.
            let pivot_val: Integer = m[pivot_row][c].clone();
            for r in 0..pivot_row {
                let q = m[r][c].div_floor(&pivot_val);
                if q.cmp0() != Ordering::Equal {
                    for j in 0..cols {
                        let prod = &q * &m[pivot_row][j];
                        m[r][j] -= prod;
                    }
                }
            }
            pivot_row += 1;
        }
    }

    m
}

/// Pick out the canonical pivot structure (column per pivot row) and
/// rank from a matrix already known to be in canonical HNF.
fn canonical_pivot_cols(dense: &[Vec<Integer>]) -> Vec<usize> {
    let mut out = Vec::new();
    for row in dense {
        if let Some(pos) = row.iter().position(|v| v.cmp0() != Ordering::Equal) {
            out.push(pos);
        }
    }
    out
}

/// Verify the canonical HNF invariants on a sparse output.
fn assert_canonical_hnf(matrix: &CsrZ) {
    let dense = matrix.to_dense();
    let pivot_cols = canonical_pivot_cols(&dense);
    let rank = pivot_cols.len();

    // Strictly ascending pivot columns (upper triangular).
    for win in pivot_cols.windows(2) {
        assert!(
            win[0] < win[1],
            "pivot columns not strictly ascending: {:?}",
            pivot_cols
        );
    }
    // Pivots positive.
    for (i, &pc) in pivot_cols.iter().enumerate() {
        assert!(
            dense[i][pc].cmp0() == Ordering::Greater,
            "pivot at row {}, col {} is non-positive: {}",
            i,
            pc,
            dense[i][pc]
        );
    }
    // Above-pivot entries in [0, pivot_val).
    for (k, &pc) in pivot_cols.iter().enumerate() {
        let pivot_val = &dense[k][pc];
        for (i, row) in dense.iter().enumerate().take(k) {
            let v = &row[pc];
            assert!(
                v.cmp0() != Ordering::Less,
                "above-pivot entry ({}, {}) is negative: {}",
                i,
                pc,
                v
            );
            assert!(
                v.cmp(pivot_val) == Ordering::Less,
                "above-pivot entry ({}, {}) = {} not less than pivot {}",
                i,
                pc,
                v,
                pivot_val
            );
        }
    }
    // Rows below the last pivot are zero.
    for (i, row) in dense.iter().enumerate().skip(rank) {
        for v in row {
            assert_eq!(*v, 0, "row {} should be zero in canonical HNF", i);
        }
    }
}

fn assert_sparse_matches_dense(input: &[Vec<i64>]) {
    let sparse_in = csr_from_dense(input);
    let sparse_out = sparse_in.hnf();
    let dense_out = dense_hnf(input);
    assert_eq!(
        sparse_out.to_dense(),
        dense_out,
        "sparse and dense HNF disagree on input {:?}",
        input
    );
    assert_canonical_hnf(&sparse_out);
}

// -- Hand-checked tiny cases -------------------------------------------

#[test]
fn empty_matrix() {
    let h = CsrZ::zeros(0, 0).hnf();
    assert_eq!(h.rows(), 0);
    assert_eq!(h.cols(), 0);
}

#[test]
fn zero_matrix() {
    let h = CsrZ::zeros(3, 4).hnf();
    assert_eq!(h.nnz(), 0);
    assert_eq!(h.rows(), 3);
    assert_eq!(h.cols(), 4);
    assert_canonical_hnf(&h);
}

#[test]
fn identity_passes_through() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let h = a.hnf();
    assert_eq!(
        h.to_dense(),
        dense_from_i64(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
    );
    assert_canonical_hnf(&h);
}

#[test]
fn diagonal_with_negatives_gets_sign_normalised() {
    let a = csr_from_dense(&[vec![-3, 0], vec![0, -5]]);
    let h = a.hnf();
    assert_eq!(h.to_dense(), dense_from_i64(&[vec![3, 0], vec![0, 5]]));
    assert_canonical_hnf(&h);
}

#[test]
fn two_rows_common_factor_known_answer() {
    // [[2, 4], [6, 8]] → [[2, 0], [0, 4]] worked out in
    // src/hnf/direct.rs's inline tests.
    let a = csr_from_dense(&[vec![2, 4], vec![6, 8]]);
    let h = a.hnf();
    assert_eq!(h.to_dense(), dense_from_i64(&[vec![2, 0], vec![0, 4]]));
    assert_canonical_hnf(&h);
}

#[test]
fn single_row_passes_through_with_pivot_positive() {
    let a = csr_from_dense(&[vec![0, -7, 14]]);
    let h = a.hnf();
    // pivot at col 1 is -7; after sign flip becomes 7. col 2 was 14
    // → -14 after the flip. Above-pivot reduction doesn't apply
    // (no earlier rows).
    assert_eq!(h.to_dense(), dense_from_i64(&[vec![0, 7, -14]]));
    // Note: this is NOT in canonical HNF (col 2 entry -14 is to the
    // right of the pivot, no above-pivot constraint applies). The
    // canonical invariant check passes because col 2 has no pivot.
    assert_canonical_hnf(&h);
}

#[test]
fn cohen_example_2x2() {
    // From Cohen §2.4 worked example. Input [[1, 2], [3, 4]]:
    //   Reduce col 0 with Euclidean:
    //     q = floor(3/1) = 3; row 1 -= 3*row 0 = [0, -2].
    //   col 0 pivot = row 0 with value 1.
    //   col 1 pivot search among (still-active) row 1: val -2.
    //     Sign-flip: row 1 = [0, 2].
    //     Reduce above: row 0[col 1] = 2. q = floor(2/2) = 1.
    //       row 0 -= 1 * row 1 = [1, 0].
    //   → [[1, 0], [0, 2]].
    let a = csr_from_dense(&[vec![1, 2], vec![3, 4]]);
    let h = a.hnf();
    assert_eq!(h.to_dense(), dense_from_i64(&[vec![1, 0], vec![0, 2]]));
    assert_canonical_hnf(&h);
}

#[test]
fn singular_two_by_two() {
    // Linearly dependent rows: rank 1, single pivot row.
    let a = csr_from_dense(&[vec![2, 4], vec![3, 6]]);
    let h = a.hnf();
    // gcd(2,3) = 1; the pivot row has [1, 2], the other row is 0.
    assert_eq!(h.to_dense(), dense_from_i64(&[vec![1, 2], vec![0, 0]]));
    assert_canonical_hnf(&h);
}

// -- Rectangular --------------------------------------------------------

#[test]
fn wide_rank_two() {
    assert_sparse_matches_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]]);
}

#[test]
fn tall_rank_one_with_zero_rows() {
    assert_sparse_matches_dense(&[vec![0, 0, 0], vec![1, 2, 3], vec![0, 0, 0], vec![2, 4, 6]]);
}

#[test]
fn rectangular_3x5() {
    assert_sparse_matches_dense(&[
        vec![3, 0, 1, 0, 2],
        vec![0, 5, 0, 7, 0],
        vec![1, 1, 1, 1, 1],
    ]);
}

// -- Larger / mixed-sign / parity-only ---------------------------------

#[test]
fn parity_mixed_sign_4x4() {
    assert_sparse_matches_dense(&[
        vec![3, -1, 0, 4],
        vec![-2, 5, 1, 0],
        vec![0, 0, -3, 6],
        vec![7, 0, 2, -1],
    ]);
}

#[test]
fn parity_singular_4x4() {
    assert_sparse_matches_dense(&[
        vec![1, 2, 3, 4],
        vec![2, 4, 6, 8],
        vec![1, 0, 1, 0],
        vec![0, 1, 0, 1],
    ]);
}

#[test]
fn parity_5x5_with_zero_columns() {
    assert_sparse_matches_dense(&[
        vec![0, 2, 0, 3, 0],
        vec![0, 4, 0, 5, 0],
        vec![0, 0, 0, 7, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 1, 0, 1, 0],
    ]);
}

#[test]
fn parity_random_pseudo_3x3_set() {
    // Deterministic LCG inputs. HNF is unique, so dense and sparse
    // must agree on every one.
    let mut state: u64 = 0xC0FF_EE42;
    for _ in 0..20 {
        let mut dense = vec![vec![0i64; 3]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Range [-5, 5].
                *cell = (state >> 32) as i64 % 11 - 5;
            }
        }
        assert_sparse_matches_dense(&dense);
    }
}

#[test]
fn parity_random_pseudo_4x6_wide() {
    let mut state: u64 = 0xFEED_FACE;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 6]; 4];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7 - 3;
            }
        }
        assert_sparse_matches_dense(&dense);
    }
}

#[test]
fn parity_random_pseudo_6x4_tall() {
    let mut state: u64 = 0xBADC_0DE5;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 4]; 6];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7 - 3;
            }
        }
        assert_sparse_matches_dense(&dense);
    }
}

#[test]
fn parity_sparse_density_pressure() {
    // Sparse pattern with one entry per row (a permutation-ish
    // structure). Should compress to a clean diagonal-shaped HNF.
    let dense = vec![
        vec![0, 0, 5, 0, 0],
        vec![0, 3, 0, 0, 0],
        vec![7, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 11],
        vec![0, 0, 0, 2, 0],
    ];
    assert_sparse_matches_dense(&dense);
}

#[test]
fn parity_with_large_integer_entries() {
    // Entries larger than i32 to exercise multi-limb GMP arithmetic
    // inside the kernel. Values fit in i64 but the intermediate
    // products in q*src grow further.
    assert_sparse_matches_dense(&[
        vec![1_000_000_000, 2_000_000_000, 0],
        vec![3_000_000_000, 0, 5],
        vec![0, 4_000_000_000, 7],
    ]);
}

#[test]
fn rank_is_pivot_count() {
    // Sanity: number of nonzero rows in the HNF equals the rank.
    let a = csr_from_dense(&[vec![1, 2, 3], vec![2, 4, 6], vec![3, 6, 9]]);
    let h = a.hnf();
    let dense = h.to_dense();
    let nonzero_rows = dense
        .iter()
        .filter(|r| r.iter().any(|v| v.cmp0() != Ordering::Equal))
        .count();
    assert_eq!(nonzero_rows, 1, "rank-1 input should have one pivot row");
}
