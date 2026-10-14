// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::transposed` (out-of-place, any shape) and
//! `MatZpn::transpose_inplace` (square only) cross-validated against
//! the definition plus the algebraic identity `(A·B)ᵀ = Bᵀ·Aᵀ`.

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..m))
}

fn mats_equal(a: &MatZpn, b: &MatZpn) -> bool {
    if (a.p, a.n, a.rows, a.cols) != (b.p, b.n, b.rows, b.cols) {
        return false;
    }
    for i in 0..a.rows {
        for j in 0..a.cols {
            if a.get(i, j) != b.get(i, j) {
                return false;
            }
        }
    }
    true
}

// ─── transposed (out-of-place) ────────────────────────────────────

#[test]
fn transposed_hand_computed_2x3_z7() {
    // A = [[1, 2, 3], [4, 5, 6]]  →  Aᵀ = [[1, 4], [2, 5], [3, 6]].
    let a = MatZpn::from_fn(7, 1, 2, 3, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (0, 2) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (1, 2) => 6,
        _ => unreachable!(),
    });
    let at = a.transposed();
    assert_eq!(at.rows, 3);
    assert_eq!(at.cols, 2);
    assert_eq!(at.get(0, 0), 1);
    assert_eq!(at.get(0, 1), 4);
    assert_eq!(at.get(1, 0), 2);
    assert_eq!(at.get(1, 1), 5);
    assert_eq!(at.get(2, 0), 3);
    assert_eq!(at.get(2, 1), 6);
}

#[test]
fn transposed_matches_definition_rectangular() {
    for &(rows, cols) in &[(5usize, 7), (7, 5), (1, 20), (20, 1), (8, 8)] {
        let a = rand_matzpn(7, 2, rows, cols, 10_000 + rows as u64 + cols as u64);
        let at = a.transposed();
        assert_eq!(at.rows, cols);
        assert_eq!(at.cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(at.get(j, i), a.get(i, j));
            }
        }
    }
}

#[test]
fn transposed_twice_is_identity() {
    let a = rand_matzpn(65537, 1, 6, 9, 20_000);
    let att = a.transposed().transposed();
    assert!(mats_equal(&att, &a));
}

#[test]
fn transposed_of_product_identity() {
    // (A · B)ᵀ = Bᵀ · Aᵀ.
    let p = 11u32;
    let a = rand_matzpn(p, 1, 3, 5, 30_000);
    let b = rand_matzpn(p, 1, 5, 4, 30_001);
    let lhs = a.matmul(&b).transposed();
    let rhs = b.transposed().matmul(&a.transposed());
    assert!(mats_equal(&lhs, &rhs));
}

#[test]
fn transposed_preserves_identity_matrix() {
    let i_mat = MatZpn::from_fn(5, 2, 7, 7, |i, j| if i == j { 1 } else { 0 });
    let it = i_mat.transposed();
    assert!(mats_equal(&it, &i_mat));
}

// ─── transpose_inplace (square only) ──────────────────────────────

#[test]
fn transpose_inplace_hand_computed_3x3_z7() {
    // A =  [[1, 2, 3],
    //       [4, 5, 6],
    //       [7, 0, 2]]    (last row uses 7=0, 9=2 mod 7)
    // Expected in-place transpose:
    //      [[1, 4, 7],
    //       [2, 5, 0],
    //       [3, 6, 2]] (with 7 → 0 mod 7 applied by MatZpn::set)
    let mut a = MatZpn::from_fn(7, 1, 3, 3, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (0, 2) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (1, 2) => 6,
        (2, 0) => 7,
        (2, 1) => 0,
        (2, 2) => 2,
        _ => unreachable!(),
    });
    a.transpose_inplace();
    // (2, 0) held 7 ≡ 0 before the swap; it moves to (0, 2) as 0.
    let expected = [
        [1u64, 4, 0], // original (0,2)=3 is now (2,0); original (2,0)=7≡0 is now (0,2)
        [2, 5, 0],
        [3, 6, 2],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(a.get(i, j), expected[i][j], "cell ({i},{j})");
        }
    }
}

#[test]
fn transpose_inplace_matches_out_of_place() {
    for &sz in &[1usize, 2, 3, 8, 15, 32] {
        let original = rand_matzpn(7, 2, sz, sz, 40_000 + sz as u64);
        let mut a = original.clone();
        a.transpose_inplace();
        let b = original.transposed();
        assert!(mats_equal(&a, &b), "size {sz}");
    }
}

#[test]
fn transpose_inplace_twice_is_identity() {
    let mut a = rand_matzpn(11, 1, 10, 10, 50_000);
    let original = a.clone();
    a.transpose_inplace();
    a.transpose_inplace();
    assert!(mats_equal(&a, &original));
}

#[test]
#[should_panic(expected = "transpose_inplace: matrix must be square")]
fn transpose_inplace_panics_on_rectangular() {
    let mut a = rand_matzpn(5, 2, 3, 4, 60_000);
    a.transpose_inplace();
}

#[test]
fn transpose_inplace_symmetric_matrix_is_fixed() {
    let n = 6;
    let a_orig = MatZpn::from_fn(11, 1, n, n, |i, j| {
        // Symmetric: A[i,j] = A[j,i] = (i + j) mod 11.
        ((i + j) as u64) % 11
    });
    let mut a = a_orig.clone();
    a.transpose_inplace();
    assert!(mats_equal(&a, &a_orig));
}
