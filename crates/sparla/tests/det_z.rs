// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrZ::det`].
//!
//! Three layers:
//!
//! 1. **Hand-checked cases.** Tiny matrices with `det` known by
//!    inspection — empty, zero, identity, diagonal, permutation
//!    matrices with predictable sign.
//! 2. **Sign cases.** Matrices designed to exercise the symmetric-
//!    lift path: negative determinants, mixed-sign entries, matrices
//!    where a row-permutation parity matters.
//! 3. **Dense-reference parity.** A recursive cofactor-expansion
//!    determinant using FLINT-backed `sparla::Integer` is the oracle. `det` is
//!    uniquely determined, so two correct implementations must
//!    agree on every input. Cofactor expansion is `O(n!)`, so the
//!    parity sweep stays in `n ≤ 5`.

use std::cmp::Ordering;

use sparla::storage::CsrZ;
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

/// Recursive cofactor expansion. `O(n!)` — keep `n ≤ 6` to avoid
/// pathological runtime in tests.
fn dense_det(mat: &[Vec<Integer>]) -> Integer {
    let n = mat.len();
    if n == 0 {
        return Integer::from(1);
    }
    if n == 1 {
        return mat[0][0].clone();
    }
    let mut det = Integer::from(0);
    for j in 0..n {
        // Build the minor M_{0,j}: rows 1..n, cols ≠ j.
        let minor: Vec<Vec<Integer>> = (1..n)
            .map(|r| {
                (0..n)
                    .filter(|&c| c != j)
                    .map(|c| mat[r][c].clone())
                    .collect()
            })
            .collect();
        let cofactor = dense_det(&minor);
        let signed: Integer = if j.is_multiple_of(2) {
            &mat[0][j] * &cofactor
        } else {
            let p = &mat[0][j] * &cofactor;
            -p
        };
        det += signed;
    }
    det
}

fn dense_from_i64(mat: &[Vec<i64>]) -> Vec<Vec<Integer>> {
    mat.iter()
        .map(|row| row.iter().map(|&v| z(v)).collect())
        .collect()
}

fn assert_sparse_matches_dense(input: &[Vec<i64>]) {
    let sparse = csr_from_dense(input);
    let sparse_det = sparse.det();
    let dense = dense_from_i64(input);
    let dense_det_v = dense_det(&dense);
    assert_eq!(
        sparse_det, dense_det_v,
        "sparse det disagrees with dense cofactor det on {:?}",
        input
    );
}

// -- Hand-checked tiny cases -------------------------------------------

#[test]
fn empty_matrix_has_det_one() {
    assert_eq!(CsrZ::zeros(0, 0).det(), z(1));
}

#[test]
fn zero_matrix_has_det_zero() {
    assert_eq!(CsrZ::zeros(3, 3).det(), z(0));
}

#[test]
fn identity_has_det_one() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    assert_eq!(a.det(), z(1));
}

#[test]
fn one_by_one() {
    assert_eq!(csr_from_dense(&[vec![7]]).det(), z(7));
    assert_eq!(csr_from_dense(&[vec![-13]]).det(), z(-13));
    assert_eq!(csr_from_dense(&[vec![0]]).det(), z(0));
}

#[test]
fn diagonal_det_is_product() {
    let a = csr_from_dense(&[vec![2, 0, 0], vec![0, 3, 0], vec![0, 0, 5]]);
    assert_eq!(a.det(), z(30));
    let b = csr_from_dense(&[vec![-2, 0, 0], vec![0, 3, 0], vec![0, 0, 5]]);
    assert_eq!(b.det(), z(-30));
}

#[test]
fn singular_row_dependence() {
    // Row 1 = 2 * row 0.
    let a = csr_from_dense(&[vec![1, 2, 3], vec![2, 4, 6], vec![0, 1, 0]]);
    assert_eq!(a.det(), z(0));
}

#[test]
fn two_by_two_known() {
    // [[3, 4], [5, 6]] → 18 - 20 = -2.
    let a = csr_from_dense(&[vec![3, 4], vec![5, 6]]);
    assert_eq!(a.det(), z(-2));
}

#[test]
fn three_by_three_known() {
    // [[1, 2, 3], [4, 5, 6], [7, 8, 10]] → -3 (a well-known case).
    let a = csr_from_dense(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 10]]);
    assert_eq!(a.det(), z(-3));
}

// -- Sign cases --------------------------------------------------------

#[test]
fn swap_matrix_has_det_minus_one() {
    let a = csr_from_dense(&[vec![0, 1], vec![1, 0]]);
    assert_eq!(a.det(), z(-1));
}

#[test]
fn three_cycle_has_det_plus_one() {
    // Even permutation.
    let a = csr_from_dense(&[vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]]);
    assert_eq!(a.det(), z(1));
}

#[test]
fn reverse_identity_4x4_has_det_plus_one() {
    // n=4 reverse-identity → 6 inversions → even → det +1.
    let a = csr_from_dense(&[
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 0],
        vec![0, 1, 0, 0],
        vec![1, 0, 0, 0],
    ]);
    assert_eq!(a.det(), z(1));
}

#[test]
fn reverse_identity_5x5_has_det_plus_one() {
    // n=5: 10 inversions → even → det +1.
    let a = csr_from_dense(&[
        vec![0, 0, 0, 0, 1],
        vec![0, 0, 0, 1, 0],
        vec![0, 0, 1, 0, 0],
        vec![0, 1, 0, 0, 0],
        vec![1, 0, 0, 0, 0],
    ]);
    assert_eq!(a.det(), z(1));
}

#[test]
fn mixed_sign_diagonal() {
    let a = csr_from_dense(&[vec![-1, 0], vec![0, -1]]);
    assert_eq!(a.det(), z(1)); // (-1)·(-1)
    let b = csr_from_dense(&[vec![-1, 0, 0], vec![0, -1, 0], vec![0, 0, -1]]);
    assert_eq!(b.det(), z(-1)); // (-1)³
}

// -- Rectangular rejection --------------------------------------------

#[test]
#[should_panic(expected = "det requires a square matrix")]
fn rejects_rectangular() {
    let _ = csr_from_dense(&[vec![1, 2, 3], vec![4, 5, 6]]).det();
}

// -- Dense-reference parity -------------------------------------------

#[test]
fn parity_random_3x3() {
    // Deterministic LCG inputs in range [-4, 4].
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for _ in 0..25 {
        let mut dense = vec![vec![0i64; 3]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 9 - 4;
            }
        }
        assert_sparse_matches_dense(&dense);
    }
}

#[test]
fn parity_random_4x4() {
    let mut state: u64 = 0xCAFE_BABE;
    for _ in 0..15 {
        let mut dense = vec![vec![0i64; 4]; 4];
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
fn parity_random_5x5_sparse() {
    // Bigger n with sparsity (most entries zero) to exercise the
    // sparse path more aggressively.
    let mut state: u64 = 0xDEAD_FEED;
    for _ in 0..10 {
        let mut dense = vec![vec![0i64; 5]; 5];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // ~30% density, values in [-3, 3].
                let bits = state >> 32;
                if bits.is_multiple_of(3) {
                    *cell = (bits >> 8) as i64 % 7 - 3;
                }
            }
        }
        assert_sparse_matches_dense(&dense);
    }
}

#[test]
fn parity_with_large_entries() {
    // Entries on the order of 10^8 so the determinant easily
    // exceeds a single u32 prime — exercises multi-prime CRT.
    let dense = vec![
        vec![100_000_007, 200_000_033, 0],
        vec![0, 300_000_007, 400_000_011],
        vec![500_000_003, 0, 600_000_001],
    ];
    assert_sparse_matches_dense(&dense);
    // The result is on the order of 10^25 — verify it required
    // multi-prime reconstruction (not just one 31-bit prime).
    let sparse_det: Integer = csr_from_dense(&dense).det();
    let abs_det: Integer = if sparse_det.cmp0() == Ordering::Less {
        -sparse_det.clone()
    } else {
        sparse_det.clone()
    };
    assert!(
        abs_det > (1u64 << 31),
        "test inputs chosen so |det| > 2^31 to exercise multi-prime CRT"
    );
}

#[test]
fn det_consistent_with_hnf_pivot_product_for_nonsingular_square() {
    // For nonsingular square A, the HNF is diagonal-of-pivots with
    // some structure; |det(A)| = product of HNF diagonal entries.
    // Cross-check det against the HNF pivot product.
    let dense = vec![vec![3, 1, 4], vec![1, 5, 9], vec![2, 6, 5]];
    let a = csr_from_dense(&dense);
    let det = a.det();
    let h = a.hnf().to_dense();
    let mut pivot_product = Integer::from(1);
    for (i, row) in h.iter().enumerate() {
        if i < row.len() {
            pivot_product *= &row[i];
        }
    }
    // |det| equals the HNF pivot product. Sign is independent of HNF.
    let abs_det: Integer = if det.cmp0() == Ordering::Less {
        -det
    } else {
        det
    };
    assert_eq!(abs_det, pivot_product);
}
