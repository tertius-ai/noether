// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial / edge-case tests. Catches the inputs that random
//! proptest sampling tends to miss: empty shapes, density extremes,
//! very thin / very tall matrices, and structured singular cases
//! that exercise the pivot-search and consistency-check paths.

use mzs::Mzs;

// ─── empty / degenerate shapes ──────────────────────────────────────

#[test]
fn zero_by_zero_works() {
    let m = Mzs::new(0, 0);
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 0);
    assert_eq!(m.nnz(), 0);
    assert_eq!(m.rank(), 0);
    assert_eq!(m.nullspace().cols(), 0);
    assert_eq!(m.transpose(), m);
    let (rref, rank) = m.rref();
    assert_eq!(rank, 0);
    assert_eq!(rref, m);
}

#[test]
fn zero_rows_keeps_columns() {
    let m = Mzs::new(0, 5);
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 5);
    let t = m.transpose();
    assert_eq!(t.rows(), 5);
    assert_eq!(t.cols(), 0);
    assert_eq!(m.rank(), 0);
    // Nullspace of empty m × 5 is full I_5: every v ∈ F_2^5 is in ker(0).
    let k = m.nullspace();
    assert_eq!(k.rows(), 5);
    assert_eq!(k.cols(), 5);
}

#[test]
fn zero_cols_keeps_rows() {
    let m = Mzs::new(7, 0);
    assert_eq!(m.rows(), 7);
    assert_eq!(m.cols(), 0);
    assert_eq!(m.nnz(), 0);
    assert_eq!(m.rank(), 0);
    assert_eq!(m.nullspace().cols(), 0);
}

#[test]
fn empty_matmul() {
    let a = Mzs::new(3, 0);
    let b = Mzs::new(0, 4);
    let c = a.matmul(&b);
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 4);
    assert_eq!(c.nnz(), 0);
}

// ─── density extremes ──────────────────────────────────────────────

#[test]
fn all_zeros_rank_is_zero() {
    for &(m, n) in &[(8usize, 8usize), (10, 4), (4, 10)] {
        let z = Mzs::new(m, n);
        assert_eq!(z.rank(), 0);
        let (rref, rank) = z.rref();
        assert_eq!(rank, 0);
        assert_eq!(rref, z);
    }
}

#[test]
fn all_ones_matrix() {
    // Every cell = 1. rank is 1 because every row is the same.
    let m = 5;
    let n = 7;
    let mut a = Mzs::new(m, n);
    for i in 0..m {
        for j in 0..n {
            a.set(i, j, 1);
        }
    }
    assert_eq!(a.nnz(), m * n);
    assert_eq!(a.rank(), 1);
    // Nullspace has dimension n - 1 = 6.
    let k = a.nullspace();
    assert_eq!(k.cols(), n - 1);
    // A · k must be zero.
    let ak = a.matmul(&k);
    assert_eq!(ak.nnz(), 0);
}

#[test]
fn density_one_matmul_with_identity() {
    let m = 6;
    let mut a = Mzs::new(m, m);
    for i in 0..m {
        for j in 0..m {
            a.set(i, j, 1);
        }
    }
    let i = {
        let mut x = Mzs::new(m, m);
        for k in 0..m {
            x.set(k, k, 1);
        }
        x
    };
    let c = a.matmul(&i);
    assert_eq!(c, a);
}

// ─── thin / tall ────────────────────────────────────────────────────

#[test]
fn one_by_n_vector() {
    let n = 100;
    let mut row = Mzs::new(1, n);
    for j in (0..n).step_by(7) {
        row.set(0, j, 1);
    }
    // Rank is 1 (one nonzero row).
    assert_eq!(row.rank(), 1);
    let k = row.nullspace();
    assert_eq!(k.rows(), n);
    assert_eq!(k.cols(), n - 1);
    let rk = row.matmul(&k);
    assert_eq!(rk.nnz(), 0);
}

#[test]
fn m_by_1_column() {
    let m = 100;
    let mut col = Mzs::new(m, 1);
    for i in (0..m).step_by(11) {
        col.set(i, 0, 1);
    }
    // Rank is 1 (one nonzero column).
    assert_eq!(col.rank(), 1);
    let k = col.nullspace();
    assert_eq!(k.cols(), 0);
}

#[test]
fn thin_matmul_dot_product() {
    // 1×k by k×1 = 1×1 — F_2 dot product of two vectors.
    let k = 50;
    let a = {
        let mut m = Mzs::new(1, k);
        for j in (0..k).step_by(3) {
            m.set(0, j, 1);
        }
        m
    };
    let b = {
        let mut m = Mzs::new(k, 1);
        for i in (0..k).step_by(5) {
            m.set(i, 0, 1);
        }
        m
    };
    let c = a.matmul(&b);
    // Compare to a manual dot product: count of i in {0,3,6,9,...} ∩ {0,5,10,...} mod 2.
    let count = (0..k).filter(|&i| i % 3 == 0 && i % 5 == 0).count();
    let expected = (count & 1) as u8;
    assert_eq!(c.get(0, 0), expected);
}

// ─── structured matrices ────────────────────────────────────────────

#[test]
fn permutation_matrix_is_full_rank() {
    // Cycle permutation: row i has a 1 at column (i+1) mod n.
    let n = 7;
    let p = Mzs::from_triples(n, n, &(0..n).map(|i| (i, (i + 1) % n)).collect::<Vec<_>>());
    assert_eq!(p.rank(), n);
    assert_eq!(p.nullspace().cols(), 0);
}

#[test]
fn upper_triangular_unit_diagonal_full_rank() {
    let n = 10;
    let mut u = Mzs::new(n, n);
    for i in 0..n {
        u.set(i, i, 1);
        // Add some upper-triangular structure for spice.
        for j in (i + 1)..n {
            if (i + j) % 3 == 0 {
                u.set(i, j, 1);
            }
        }
    }
    assert_eq!(u.rank(), n);
    // Inverse via solve_left(U, I):
    let id = {
        let mut x = Mzs::new(n, n);
        for k in 0..n {
            x.set(k, k, 1);
        }
        x
    };
    let inv = Mzs::solve_left(&u, &id).expect("U is invertible");
    let check = u.matmul(&inv);
    assert_eq!(check, id);
}

#[test]
fn anti_diagonal_full_rank() {
    let n = 6;
    let a = Mzs::from_triples(n, n, &(0..n).map(|i| (i, n - 1 - i)).collect::<Vec<_>>());
    assert_eq!(a.rank(), n);
}

#[test]
fn rank_deficient_known_kernel() {
    // A's rows are: [1 0 1], [0 1 1], [1 1 0]. The third is the XOR of
    // the first two, so rank is 2; the column [1 1 1]^T should land in
    // the right nullspace.
    let a = Mzs::from_triples(3, 3, &[(0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1)]);
    assert_eq!(a.rank(), 2);
    let k = a.nullspace();
    assert_eq!(k.cols(), 1);
    // The single basis vector must be [1, 1, 1]^T.
    for i in 0..3 {
        assert_eq!(k.get(i, 0), 1);
    }
}

// ─── byte-buffer edge cases ─────────────────────────────────────────

#[test]
fn byte_buffer_with_high_bits() {
    // High bits of input bytes are ignored — only `byte & 1` matters.
    let bytes_a = vec![0u8, 1, 0, 1];
    let bytes_b = vec![0xFE, 0xFF, 0x10, 0x11];
    let a = Mzs::from_byte_buffer(2, 2, &bytes_a);
    let b = Mzs::from_byte_buffer(2, 2, &bytes_b);
    assert_eq!(a, b);
}

#[test]
fn byte_buffer_empty() {
    let m = Mzs::from_byte_buffer(0, 0, &[]);
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 0);
    let bytes = m.to_byte_buffer();
    assert!(bytes.is_empty());
}

#[test]
#[should_panic(expected = "from_byte_buffer")]
fn byte_buffer_wrong_length_panics() {
    let _ = Mzs::from_byte_buffer(3, 4, &[1, 0, 1]); // expected 12, got 3
}

// ─── set/get edge bits ──────────────────────────────────────────────

#[test]
fn set_at_max_column() {
    let cols = 1000;
    let mut m = Mzs::new(1, cols);
    m.set(0, 0, 1);
    m.set(0, cols - 1, 1);
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(0, cols - 1), 1);
    assert_eq!(m.nnz(), 2);
}

#[test]
fn set_clear_set_pattern() {
    let mut m = Mzs::new(1, 10);
    for j in 0..10 {
        m.set(0, j, 1);
    }
    assert_eq!(m.nnz(), 10);
    for j in 0..10 {
        m.set(0, j, 0);
    }
    assert_eq!(m.nnz(), 0);
    for j in (0..10).step_by(2) {
        m.set(0, j, 1);
    }
    assert_eq!(m.nnz(), 5);
}
