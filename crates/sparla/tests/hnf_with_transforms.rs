// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrZ::hnf_with_transforms`].
//!
//! The load-bearing invariants we check:
//!
//! 1. `U · A = H` — bit-for-bit equality after sparse matmul.
//! 2. `H == A.hnf()` — the canonical HNF matches the non-tracking
//!    kernel, so the transform tracking didn't corrupt anything.
//! 3. `U` is unimodular: `|det(U)| = 1`.
//! 4. `U` has shape `m × m`; `H` has shape `m × n`.

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

fn assert_hnf_with_transforms_invariants(a: &CsrZ) {
    let (h, u) = a.hnf_with_transforms();
    // Shape sanity.
    assert_eq!(h.rows(), a.rows());
    assert_eq!(h.cols(), a.cols());
    assert_eq!(u.rows(), a.rows());
    assert_eq!(u.cols(), a.rows());

    // (1) U · A = H, bit-for-bit.
    let prod = u.matmul(a);
    assert_eq!(
        prod.to_dense(),
        h.to_dense(),
        "U · A != H on {:?}",
        a.to_dense()
    );

    // (2) H matches the non-tracking hnf.
    let h_check = a.hnf();
    assert_eq!(h.to_dense(), h_check.to_dense(), "tracked H ≠ plain hnf");

    // (3) U is unimodular: det(U) ∈ {±1}. Holds iff U is m × m
    // square (which we've already checked).
    let det_u = u.det();
    assert!(
        det_u == z(1) || det_u == z(-1),
        "det(U) should be ±1, got {}",
        det_u
    );
}

// =====================================================================
// Hand-checked cases
// =====================================================================

#[test]
fn identity_yields_identity_transform() {
    let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    let (h, u) = a.hnf_with_transforms();
    assert_eq!(h.to_dense(), a.to_dense());
    // U is identity for identity input (no row ops needed).
    let mut expected = vec![vec![z(0); 3]; 3];
    for (i, row) in expected.iter_mut().enumerate() {
        row[i] = z(1);
    }
    assert_eq!(u.to_dense(), expected);
}

#[test]
fn zero_matrix_yields_identity_transform() {
    let a = CsrZ::zeros(3, 4);
    let (h, u) = a.hnf_with_transforms();
    assert_eq!(h.nnz(), 0);
    // No row ops needed; U = I_3.
    let mut expected = vec![vec![z(0); 3]; 3];
    for (i, row) in expected.iter_mut().enumerate() {
        row[i] = z(1);
    }
    assert_eq!(u.to_dense(), expected);
}

#[test]
fn empty_matrix() {
    let (h, u) = CsrZ::zeros(0, 0).hnf_with_transforms();
    assert_eq!(h.rows(), 0);
    assert_eq!(u.rows(), 0);
}

#[test]
fn cohen_2x2_example() {
    // [[1, 2], [3, 4]] → ([[1, 0], [0, 2]], some unimodular U).
    let a = csr_from_dense(&[vec![1, 2], vec![3, 4]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn two_rows_common_factor() {
    // [[2, 4], [6, 8]] → ([[2, 0], [0, 4]], U).
    let a = csr_from_dense(&[vec![2, 4], vec![6, 8]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn singular_with_zero_row() {
    let a = csr_from_dense(&[vec![1, 2, 3], vec![2, 4, 6], vec![3, 6, 9]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn negative_pivot_gets_sign_normalised() {
    let a = csr_from_dense(&[vec![-3, 6]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn rectangular_wide() {
    let a = csr_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn rectangular_tall_with_zero_rows() {
    let a = csr_from_dense(&[vec![0, 0, 0], vec![1, 2, 3], vec![0, 0, 0], vec![2, 4, 6]]);
    assert_hnf_with_transforms_invariants(&a);
}

// =====================================================================
// Random sweep — load-bearing parity verification
// =====================================================================

fn gen_dense_z(rows: usize, cols: usize, density: f64, seed: u64) -> Vec<Vec<i64>> {
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 32
        }
    }
    let mut rng = Lcg(seed);
    let mut dense = vec![vec![0i64; cols]; rows];
    for row in dense.iter_mut() {
        for cell in row.iter_mut() {
            let r = rng.next();
            // density-driven mask.
            if (r % 1000) < ((density * 1000.0) as u64) {
                *cell = (rng.next() as i64 % 19) - 9;
                if *cell == 0 {
                    *cell = 1;
                }
            }
        }
    }
    dense
}

#[test]
fn invariants_random_3x3() {
    for seed in 0..20u64 {
        let dense = gen_dense_z(3, 3, 0.5, 0xABCD ^ seed);
        let a = csr_from_dense(&dense);
        assert_hnf_with_transforms_invariants(&a);
    }
}

#[test]
fn invariants_random_4x6_wide() {
    for seed in 0..10u64 {
        let dense = gen_dense_z(4, 6, 0.4, 0xBEEF ^ seed);
        let a = csr_from_dense(&dense);
        assert_hnf_with_transforms_invariants(&a);
    }
}

#[test]
fn invariants_random_6x4_tall() {
    for seed in 0..10u64 {
        let dense = gen_dense_z(6, 4, 0.4, 0xCAFE ^ seed);
        let a = csr_from_dense(&dense);
        assert_hnf_with_transforms_invariants(&a);
    }
}

#[test]
fn invariants_sparse_5x5() {
    // Low density, exercises the sparse path more aggressively.
    for seed in 0..10u64 {
        let dense = gen_dense_z(5, 5, 0.2, 0xD00D ^ seed);
        let a = csr_from_dense(&dense);
        assert_hnf_with_transforms_invariants(&a);
    }
}

// =====================================================================
// Edge cases
// =====================================================================

#[test]
fn single_row() {
    let a = csr_from_dense(&[vec![3, -6, 9]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn single_column() {
    let a = csr_from_dense(&[vec![2], vec![4], vec![6]]);
    assert_hnf_with_transforms_invariants(&a);
}

#[test]
fn det_of_u_sign_observed() {
    // Verify we see both signs of det(U) across random inputs —
    // sanity-check that the algorithm isn't accidentally always
    // producing det(U) = +1.
    let mut saw_pos = false;
    let mut saw_neg = false;
    for seed in 0..30u64 {
        let dense = gen_dense_z(3, 3, 0.6, 0x1234 ^ seed);
        let a = csr_from_dense(&dense);
        let (_h, u) = a.hnf_with_transforms();
        let det = u.det();
        if det.cmp0() == Ordering::Greater {
            saw_pos = true;
        }
        if det.cmp0() == Ordering::Less {
            saw_neg = true;
        }
        if saw_pos && saw_neg {
            return;
        }
    }
    // It's allowed in principle that we see only one sign (some
    // structural reason), but in practice both should appear in 30
    // random 3×3 inputs.
    assert!(saw_pos, "expected to see det(U) = +1 at least once");
    assert!(saw_neg, "expected to see det(U) = -1 at least once");
}
