// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrFp::kernel`], [`CsrFp::image`], and
//! [`CsrFp::solve`].
//!
//! Coverage:
//!
//! 1. **Hand-checked cases** — empty matrix, identity, zero matrix,
//!    classic rank-deficient inputs.
//! 2. **Algebraic invariants** — `A · K = 0`, image is in column
//!    span of `A`, `A · x = b` for solve output, dimension counts
//!    (`nullity + rank = n`).
//! 3. **Round-trips** — kernel basis is full rank (free columns
//!    embed as identity rows); image columns are exactly `A`'s
//!    pivot columns.

use sparla::storage::{CsrFp, SparseRingStorage};

const P: u64 = 7;

fn csr_from_dense(dense: &[Vec<i64>], prime: u64) -> CsrFp {
    let rows = dense.len();
    let cols = if rows == 0 { 0 } else { dense[0].len() };
    let mut triples = Vec::new();
    for (r, row) in dense.iter().enumerate() {
        assert_eq!(row.len(), cols);
        for (c, &v) in row.iter().enumerate() {
            if v.rem_euclid(prime as i64) != 0 {
                triples.push((r, c, v));
            }
        }
    }
    CsrFp::from_coo(rows, cols, &triples, prime)
}

/// Sparse `m × k` · sparse `k × q` → dense `m × q` over F_p.
/// Used for matmul checks in the algebraic invariants.
fn matmul_dense(a: &CsrFp, b: &CsrFp) -> Vec<Vec<u32>> {
    assert_eq!(a.cols(), b.rows());
    let p = a.modulus();
    let m = a.rows();
    let q = b.cols();
    let mut out = vec![vec![0u64; q]; m];
    for (r, out_row) in out.iter_mut().enumerate() {
        for (k, av) in a.iter_row(r) {
            for (col, bv) in b.iter_row(k) {
                out_row[col] = (out_row[col] + av as u64 * bv as u64) % p;
            }
        }
    }
    out.into_iter()
        .map(|row| row.into_iter().map(|v| v as u32).collect())
        .collect()
}

/// All-zero `m × q` dense matrix for comparison.
fn zeros_dense(m: usize, q: usize) -> Vec<Vec<u32>> {
    vec![vec![0u32; q]; m]
}

// =====================================================================
// kernel
// =====================================================================

#[test]
fn kernel_of_identity_is_empty() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], P);
    let k = a.kernel();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 0);
    assert_eq!(k.nnz(), 0);
}

#[test]
fn kernel_of_zero_is_full_identity() {
    let a = CsrFp::zeros(2, 3, P);
    let k = a.kernel();
    // nullity == 3, rank == 0.
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 3);
    // K is the 3 × 3 identity: each free column maps to a standard
    // basis vector.
    assert_eq!(
        k.to_dense(),
        vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
    );
}

#[test]
fn kernel_of_rank_one_three_columns() {
    // A = [[1, 2, 3]]: pivot col 0, free cols 1 and 2.
    // K column 0 (for free col 1): x = (-2, 1, 0).
    // K column 1 (for free col 2): x = (-3, 0, 1).
    let a = csr_from_dense(&[vec![1, 2, 3]], P);
    let k = a.kernel();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 2);
    // -2 mod 7 = 5; -3 mod 7 = 4.
    let dense = k.to_dense();
    assert_eq!(dense, vec![vec![5, 4], vec![1, 0], vec![0, 1]]);
    // A · K = 0.
    let prod = matmul_dense(&a, &k);
    assert_eq!(prod, zeros_dense(1, 2));
}

#[test]
fn kernel_dimension_matches_nullity() {
    // 3 × 4 matrix of rank 2 → nullity 2.
    let a = csr_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4], vec![1, 2, 1, 7]], P);
    let k = a.kernel();
    let rank = a.rank();
    assert_eq!(rank + k.cols(), a.cols(), "rank-nullity should sum to n");
}

#[test]
fn kernel_satisfies_a_dot_k_is_zero_for_random_inputs() {
    // Several deterministic LCG-driven inputs.
    let mut state: u64 = 0xDEAD_BEEF;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 5]; 4];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7;
            }
        }
        let a = csr_from_dense(&dense, P);
        let k = a.kernel();
        let prod = matmul_dense(&a, &k);
        assert_eq!(
            prod,
            zeros_dense(a.rows(), k.cols()),
            "A · K ≠ 0 for {:?}",
            dense
        );
    }
}

#[test]
fn kernel_is_full_rank() {
    // The free-column rows of K embed as the nullity × nullity
    // identity, so K has rank == nullity.
    let a = csr_from_dense(&[vec![1, 2, 0, 3, 5], vec![0, 0, 1, 4, 6]], P);
    let k = a.kernel();
    assert_eq!(
        k.rank(),
        k.cols(),
        "kernel basis should be linearly independent"
    );
}

// =====================================================================
// image
// =====================================================================

#[test]
fn image_of_identity_is_identity() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], P);
    let img = a.image();
    assert_eq!(
        img.to_dense(),
        vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
    );
}

#[test]
fn image_of_zero_is_empty() {
    let a = CsrFp::zeros(3, 4, P);
    let img = a.image();
    assert_eq!(img.rows(), 3);
    assert_eq!(img.cols(), 0);
    assert_eq!(img.nnz(), 0);
}

#[test]
fn image_extracts_pivot_columns() {
    // Pivot cols of A = [[1, 2, 0], [0, 0, 1]] are {0, 2}.
    // Image = columns 0 and 2 of A = [[1, 0], [0, 1]].
    let a = csr_from_dense(&[vec![1, 2, 0], vec![0, 0, 1]], P);
    let img = a.image();
    assert_eq!(img.to_dense(), vec![vec![1, 0], vec![0, 1]]);
}

#[test]
fn image_dim_equals_rank() {
    let a = csr_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4], vec![2, 4, 1, 10]], P);
    assert_eq!(a.image().cols(), a.rank());
}

#[test]
fn image_columns_are_full_rank() {
    // The pivot columns of A are linearly independent, so the
    // image matrix has rank == rank(A) == its column count.
    let a = csr_from_dense(
        &[
            vec![3, 0, 1, 0, 2],
            vec![0, 5, 0, 7, 0],
            vec![1, 1, 1, 1, 1],
        ],
        P,
    );
    let img = a.image();
    assert_eq!(img.rank(), img.cols());
}

// =====================================================================
// solve
// =====================================================================

fn b_from_dense(dense: &[i64], prime: u64) -> CsrFp {
    let triples: Vec<(usize, usize, i64)> = dense
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.rem_euclid(prime as i64) != 0)
        .map(|(r, &v)| (r, 0, v))
        .collect();
    CsrFp::from_coo(dense.len(), 1, &triples, prime)
}

#[test]
fn solve_identity_returns_b() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], P);
    let b = b_from_dense(&[5, 3, 2], P);
    let x = a.solve(&b).expect("identity system is consistent");
    assert_eq!(x.to_dense(), vec![vec![5], vec![3], vec![2]]);
}

#[test]
fn solve_inconsistent_returns_none() {
    // A has rank 1 (rows linearly dependent) but b is not in the
    // image.
    let a = csr_from_dense(&[vec![1, 2], vec![2, 4]], P);
    let b = b_from_dense(&[1, 3], P);
    assert!(a.solve(&b).is_none());
}

#[test]
fn solve_consistent_underdetermined() {
    // A = [[1, 2, 3]], b = [6] → many solutions; we return one.
    // The particular solution from RREF sets free vars to 0, so
    // x = (6, 0, 0).
    let a = csr_from_dense(&[vec![1, 2, 3]], P);
    let b = b_from_dense(&[6], P);
    let x = a.solve(&b).expect("system is consistent");
    assert_eq!(x.to_dense(), vec![vec![6], vec![0], vec![0]]);
}

#[test]
fn solve_two_by_two_known() {
    // A = [[2, 3], [1, 4]] over F_7. det = 8 - 3 = 5 ≠ 0.
    // For b = [1, 2]: solve via inverse.
    //   inv(A) · b = (1/5) · [[4, -3], [-1, 2]] · [1, 2]
    //              = inv_5 · [4 - 6, -1 + 4]
    //              = 3 · [-2, 3]   (inv(5) mod 7 = 3 since 5·3 = 15 ≡ 1)
    //              = [-6, 9] ≡ [1, 2]   (mod 7).
    let a = csr_from_dense(&[vec![2, 3], vec![1, 4]], P);
    let b = b_from_dense(&[1, 2], P);
    let x = a.solve(&b).expect("nonsingular system");
    assert_eq!(x.to_dense(), vec![vec![1], vec![2]]);
}

#[test]
fn solve_verifies_a_dot_x_equals_b_for_random_inputs() {
    let mut state: u64 = 0xCAFE_F00D;
    for _ in 0..15 {
        let mut a_dense = vec![vec![0i64; 4]; 4];
        for row in a_dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7;
            }
        }
        let mut b_dense = vec![0i64; 4];
        for cell in b_dense.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *cell = (state >> 32) as i64 % 7;
        }
        let a = csr_from_dense(&a_dense, P);
        let b = b_from_dense(&b_dense, P);
        if let Some(x) = a.solve(&b) {
            let ax = matmul_dense(&a, &x);
            let b_dense_u32: Vec<u32> = (0..a.rows()).map(|r| b.get(r, 0).unwrap_or(0)).collect();
            let ax_flat: Vec<u32> = ax.into_iter().map(|row| row[0]).collect();
            assert_eq!(
                ax_flat, b_dense_u32,
                "A · x ≠ b for A = {:?}, b = {:?}",
                a_dense, b_dense
            );
        }
        // If None, the system is inconsistent — we don't check that
        // here, but the random sweep covers both cases.
    }
}

#[test]
fn solve_zero_matrix_zero_b_returns_zero_vector() {
    let a = CsrFp::zeros(3, 4, P);
    let b = b_from_dense(&[0, 0, 0], P);
    let x = a.solve(&b).expect("0 · x = 0 is always consistent");
    assert_eq!(x.rows(), 4);
    assert_eq!(x.cols(), 1);
    assert_eq!(x.nnz(), 0);
}

#[test]
fn solve_zero_matrix_nonzero_b_is_inconsistent() {
    let a = CsrFp::zeros(3, 4, P);
    let b = b_from_dense(&[1, 0, 0], P);
    assert!(a.solve(&b).is_none());
}

// =====================================================================
// Rank-nullity + invariants together
// =====================================================================

#[test]
fn rank_nullity_holds_across_random_inputs() {
    let mut state: u64 = 0xC001_C0DE;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 5]; 5];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7;
            }
        }
        let a = csr_from_dense(&dense, P);
        let rank = a.rank();
        let nullity = a.kernel().cols();
        assert_eq!(
            rank + nullity,
            a.cols(),
            "rank-nullity failed on {:?}",
            dense
        );
        assert_eq!(a.image().cols(), rank, "image cols ≠ rank on {:?}", dense);
    }
}
