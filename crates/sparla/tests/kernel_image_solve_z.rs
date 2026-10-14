// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrZ::kernel`], [`CsrZ::image`], and
//! [`CsrZ::solve_left`].
//!
//! The load-bearing invariants:
//!
//! - **kernel**: `A · K = 0` (matmul check) and `rank(A) + cols(K)
//!   = n` (rank-nullity).
//! - **image**: every column of `image(A)` lies in the Z-column-span
//!   of `A` (it IS a column-span basis by construction; we sanity-
//!   check shape and rank).
//! - **solve_left**: when a solution exists, `A · x = b` holds
//!   exactly; when none exists, `None` is returned.

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

fn b_from_dense(dense: &[i64]) -> CsrZ {
    let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
    for (r, &v) in dense.iter().enumerate() {
        if v != 0 {
            triples.push((r, 0, z(v)));
        }
    }
    CsrZ::from_coo(dense.len(), 1, &triples)
}

// =====================================================================
// kernel
// =====================================================================

#[test]
fn kernel_of_identity_is_empty() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let k = a.kernel();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 0);
    assert_eq!(k.nnz(), 0);
}

#[test]
fn kernel_of_zero_is_identity() {
    let a = CsrZ::zeros(2, 3);
    let k = a.kernel();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 3);
    // Kernel basis spans all of Z^3 — represented by some unimodular
    // 3×3 matrix (typically just I_3 from the transform).
    // The exact entries depend on HNF's transform; we verify the
    // algebraic invariant `A · K = 0`, not specific entries.
    let prod = a.matmul(&k);
    assert_eq!(prod.nnz(), 0);
    // And the determinant of K is ±1 (unimodular basis).
    let det_k = k.det();
    assert!(det_k == z(1) || det_k == z(-1));
}

#[test]
fn kernel_a_dot_k_is_zero_hand_case() {
    // [[1, 2, 3]]: rank 1, nullity 2.
    let a = csr_from_dense(&[vec![1, 2, 3]]);
    let k = a.kernel();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 2);
    let prod = a.matmul(&k);
    assert_eq!(prod.nnz(), 0, "A · kernel(A) should be zero");
}

#[test]
fn kernel_a_dot_k_is_zero_random_inputs() {
    let cases = [
        vec![vec![1i64, 2, 3], vec![2, 4, 6], vec![3, 6, 9]], // rank 1
        vec![vec![1, 2, 0, 3], vec![0, 0, 1, 4]],             // rank 2
        vec![vec![3, -1, 0, 4], vec![-2, 5, 1, 0], vec![0, 0, -3, 6]],
        vec![vec![1, 2], vec![3, 4], vec![5, 6]], // 3×2 rank 2
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        let k = a.kernel();
        let prod = a.matmul(&k);
        assert_eq!(prod.nnz(), 0, "A · K ≠ 0 for {:?}", dense);
    }
}

#[test]
fn kernel_rank_nullity_holds() {
    let cases = [
        vec![vec![1i64, 2, 3], vec![2, 4, 6], vec![3, 6, 9]],
        vec![vec![1, 2, 0, 3], vec![0, 0, 1, 4]],
        vec![vec![3, -1, 0, 4], vec![-2, 5, 1, 0], vec![0, 0, -3, 6]],
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        let r = a.rank();
        let k = a.kernel();
        assert_eq!(r + k.cols(), a.cols(), "rank-nullity on {:?}", dense);
    }
}

// =====================================================================
// image
// =====================================================================

#[test]
fn image_of_identity_is_identity_shape() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let img = a.image();
    assert_eq!(img.rows(), 3);
    assert_eq!(img.cols(), 3);
    // Identity has trivial HNF, so image = identity columns = I_3.
    assert_eq!(
        img.to_dense(),
        vec![
            vec![z(1), z(0), z(0)],
            vec![z(0), z(1), z(0)],
            vec![z(0), z(0), z(1)]
        ]
    );
}

#[test]
fn image_of_zero_is_empty() {
    let img = CsrZ::zeros(3, 4).image();
    assert_eq!(img.rows(), 3);
    assert_eq!(img.cols(), 0);
    assert_eq!(img.nnz(), 0);
}

#[test]
fn image_dim_equals_rank() {
    let cases = [
        vec![vec![1i64, 2, 3], vec![2, 4, 6]], // rank 1
        vec![vec![1, 2, 0, 3], vec![0, 0, 1, 4]],
        vec![vec![1, 0], vec![0, 1], vec![1, 1]], // rank 2
    ];
    for dense in cases {
        let a = csr_from_dense(&dense);
        assert_eq!(a.image().cols(), a.rank());
    }
}

#[test]
fn image_columns_have_full_rank() {
    let a = csr_from_dense(&[
        vec![3, 0, 1, 0, 2],
        vec![0, 5, 0, 7, 0],
        vec![1, 1, 1, 1, 1],
    ]);
    let img = a.image();
    // Image basis columns are linearly independent: rank(image) == cols.
    assert_eq!(img.rank(), img.cols());
}

// =====================================================================
// solve_left
// =====================================================================

#[test]
fn solve_identity_returns_b() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let b = b_from_dense(&[5, 3, 2]);
    let x = a.solve_left(&b).expect("identity always solvable");
    assert_eq!(x.to_dense(), vec![vec![z(5)], vec![z(3)], vec![z(2)]]);
}

#[test]
fn solve_no_integer_solution() {
    // 2 · x = 1 over Z has no solution. A = [[2]], b = [[1]].
    let a = csr_from_dense(&[vec![2]]);
    let b = b_from_dense(&[1]);
    assert!(a.solve_left(&b).is_none());
}

#[test]
fn solve_integer_solution_via_divisibility() {
    // 2 · x = 6 → x = 3.
    let a = csr_from_dense(&[vec![2]]);
    let b = b_from_dense(&[6]);
    let x = a.solve_left(&b).expect("6 is divisible by 2");
    assert_eq!(x.to_dense(), vec![vec![z(3)]]);
}

#[test]
fn solve_inconsistent_in_image_check() {
    // A = [[1, 2], [2, 4]] has rank 1; image = Z·(1, 2). b = (1, 3)
    // is not on this line → inconsistent.
    let a = csr_from_dense(&[vec![1, 2], vec![2, 4]]);
    let b = b_from_dense(&[1, 3]);
    assert!(a.solve_left(&b).is_none());
}

#[test]
fn solve_consistent_underdetermined_gives_particular_solution() {
    // A = [[1, 2, 3]], b = [[6]] — many solutions; we return one.
    let a = csr_from_dense(&[vec![1, 2, 3]]);
    let b = b_from_dense(&[6]);
    let x = a.solve_left(&b).expect("system is consistent");
    // Verify A · x = b regardless of which particular solution we got.
    let ax = a.matmul(&x);
    assert_eq!(ax.to_dense(), b.to_dense());
}

#[test]
fn solve_verifies_a_dot_x_equals_b_on_random_inputs() {
    // For consistent random systems, A · x = b should hold.
    let cases: &[(Vec<Vec<i64>>, Vec<i64>)] = &[
        (
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
            vec![3, -2, 7],
        ),
        (vec![vec![2, 1], vec![1, 3]], vec![5, 7]),
        (vec![vec![1, 2, 3], vec![4, 5, 6]], vec![6, 15]),
    ];
    for (dense_a, dense_b) in cases {
        let a = csr_from_dense(dense_a);
        let b = b_from_dense(dense_b);
        if let Some(x) = a.solve_left(&b) {
            let ax = a.matmul(&x);
            assert_eq!(
                ax.to_dense(),
                b.to_dense(),
                "A · x ≠ b for A = {:?}, b = {:?}",
                dense_a,
                dense_b
            );
        }
    }
}

#[test]
fn solve_zero_matrix_zero_b_returns_zero_vector() {
    let a = CsrZ::zeros(3, 4);
    let b = b_from_dense(&[0, 0, 0]);
    let x = a.solve_left(&b).expect("0 · x = 0 is consistent");
    assert_eq!(x.rows(), 4);
    assert_eq!(x.cols(), 1);
    assert_eq!(x.nnz(), 0);
}

#[test]
fn solve_zero_matrix_nonzero_b_is_inconsistent() {
    let a = CsrZ::zeros(3, 4);
    let b = b_from_dense(&[1, 0, 0]);
    assert!(a.solve_left(&b).is_none());
}

// =====================================================================
// Cross-checks among kernel / image / solve_left
// =====================================================================

#[test]
fn solve_plus_kernel_enumerates_solutions() {
    // For A = [[1, 2, 3]] and b = [[6]], one solution is (6, 0, 0).
    // For any kernel basis vector k, (6, 0, 0) + k should also satisfy
    // A · y = b.
    let a = csr_from_dense(&[vec![1, 2, 3]]);
    let b = b_from_dense(&[6]);
    let x = a.solve_left(&b).expect("consistent");
    let k = a.kernel();
    // x + first column of k.
    let one_basis = csr_from_dense(&[
        vec![k.get(0, 0).cloned().unwrap_or(z(0)).to_i64().unwrap_or(0) as i64],
        vec![k.get(1, 0).cloned().unwrap_or(z(0)).to_i64().unwrap_or(0) as i64],
        vec![k.get(2, 0).cloned().unwrap_or(z(0)).to_i64().unwrap_or(0) as i64],
    ]);
    // y = x + k_col_0. Implement add via dense, then check A · y = b.
    let x_dense = x.to_dense();
    let k_dense = one_basis.to_dense();
    let y_dense: Vec<Vec<Integer>> = (0..x_dense.len())
        .map(|i| vec![&x_dense[i][0] + &k_dense[i][0]])
        .collect();
    // Build y as CsrZ.
    let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
    for (i, row) in y_dense.iter().enumerate() {
        if row[0].cmp0() != Ordering::Equal {
            triples.push((i, 0, row[0].clone()));
        }
    }
    let y = CsrZ::from_coo(3, 1, &triples);
    let ay = a.matmul(&y);
    assert_eq!(
        ay.to_dense(),
        b.to_dense(),
        "x + kernel_basis should also solve"
    );
}

#[test]
fn image_columns_are_in_a_column_span() {
    // Every column c of image(A) should itself be solvable: there
    // exists y ∈ Z^n with A · y = c.
    let a = csr_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4], vec![2, 4, 1, 10]]);
    let img = a.image();
    for j in 0..img.cols() {
        // Build b = j-th column of img.
        let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
        for r_idx in 0..img.rows() {
            if let Some(v) = img.get(r_idx, j) {
                triples.push((r_idx, 0, v.clone()));
            }
        }
        let b_col = CsrZ::from_coo(img.rows(), 1, &triples);
        let x = a.solve_left(&b_col);
        assert!(
            x.is_some(),
            "image column {} should be in column-span of A",
            j
        );
    }
}
