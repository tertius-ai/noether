// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse · sparse F_2 matmul correctness.
//!
//! Two test strategies:
//! - Hand-computed small cases that pin specific results bit-for-bit.
//! - Algebraic invariants on parametric random inputs:
//!     - `I · A == A`, `A · I == A`
//!     - `0 · A == 0`, `A · 0 == 0`
//!     - `(A · B)ᵀ == Bᵀ · Aᵀ`
//!     - `(A · B) · C == A · (B · C)` (associativity)
//!     - Schoolbook reference matmul matches sparse matmul on random
//!       inputs (the closest thing to a "densify-via-m4rst" oracle
//!       we get to without the m4rst path-dep).

use mzs::Mzs;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn schoolbook(a: &Mzs, b: &Mzs) -> Mzs {
    assert_eq!(a.cols(), b.rows());
    let mut c = Mzs::new(a.rows(), b.cols());
    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut acc = 0u8;
            for k in 0..a.cols() {
                acc ^= a.get(i, k) & b.get(k, j);
            }
            if acc != 0 {
                c.set(i, j, 1);
            }
        }
    }
    c
}

fn random_mzs(rows: usize, cols: usize, density: f64, seed: u64) -> Mzs {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut m = Mzs::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if rng.random::<f64>() < density {
                m.set(i, j, 1);
            }
        }
    }
    m
}

fn identity(n: usize) -> Mzs {
    let mut m = Mzs::new(n, n);
    for i in 0..n {
        m.set(i, i, 1);
    }
    m
}

#[test]
fn matmul_hand_2x2() {
    // A = [[1 1], [0 1]], B = [[1 0], [1 1]]
    // A·B = [[1+1, 0+1], [0+1, 0+1]] = [[0, 1], [1, 1]] over F_2.
    let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    let b = Mzs::from_triples(2, 2, &[(0, 0), (1, 0), (1, 1)]);
    let c = a.matmul(&b);
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(0, 1), 1);
    assert_eq!(c.get(1, 0), 1);
    assert_eq!(c.get(1, 1), 1);
}

#[test]
fn matmul_hand_3x3() {
    // A = identity except (0, 2) = 1.
    // B = identity except (2, 0) = 1.
    // A·B at (i, j): sum_k A[i,k]*B[k,j].
    //   (0,0): A[0,0]*B[0,0] + A[0,2]*B[2,0] = 1 + 1 = 0.
    //   (0,1): A[0,1]*B[1,1] = 0. (no contribution; only (0,2)*(2,1)=0).
    //   (0,2): A[0,0]*B[0,2] + A[0,2]*B[2,2] = 0 + 1 = 1.
    //   (1,1): A[1,1]*B[1,1] = 1.
    //   (2,0): A[2,2]*B[2,0] = 1.
    //   (2,2): A[2,2]*B[2,2] = 1.
    let a = Mzs::from_triples(3, 3, &[(0, 0), (0, 2), (1, 1), (2, 2)]);
    let b = Mzs::from_triples(3, 3, &[(0, 0), (1, 1), (2, 0), (2, 2)]);
    let c = a.matmul(&b);
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(0, 1), 0);
    assert_eq!(c.get(0, 2), 1);
    assert_eq!(c.get(1, 0), 0);
    assert_eq!(c.get(1, 1), 1);
    assert_eq!(c.get(1, 2), 0);
    assert_eq!(c.get(2, 0), 1);
    assert_eq!(c.get(2, 1), 0);
    assert_eq!(c.get(2, 2), 1);
}

#[test]
fn matmul_identity_left() {
    let a = random_mzs(7, 5, 0.3, 100);
    let i = identity(7);
    let c = i.matmul(&a);
    assert_eq!(c, a);
}

#[test]
fn matmul_identity_right() {
    let a = random_mzs(7, 5, 0.3, 101);
    let i = identity(5);
    let c = a.matmul(&i);
    assert_eq!(c, a);
}

#[test]
fn matmul_zero_left() {
    let a = random_mzs(7, 5, 0.4, 102);
    let z = Mzs::new(3, 7);
    let c = z.matmul(&a);
    assert_eq!(c.nnz(), 0);
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 5);
}

#[test]
fn matmul_zero_right() {
    let a = random_mzs(7, 5, 0.4, 103);
    let z = Mzs::new(5, 3);
    let c = a.matmul(&z);
    assert_eq!(c.nnz(), 0);
    assert_eq!(c.rows(), 7);
    assert_eq!(c.cols(), 3);
}

#[test]
fn matmul_matches_schoolbook() {
    // Several shapes × densities. The schoolbook reference is the
    // bit-level oracle here.
    for &(m, k, n, density, seed) in &[
        (5usize, 7usize, 11usize, 0.1, 200u64),
        (8, 8, 8, 0.3, 201),
        (3, 13, 5, 0.5, 202),
        (10, 4, 12, 0.05, 203),
        (1, 20, 1, 0.4, 204),
        (20, 1, 20, 0.4, 205),
    ] {
        let a = random_mzs(m, k, density, seed);
        let b = random_mzs(k, n, density, seed + 1000);
        let c_fast = a.matmul(&b);
        let c_ref = schoolbook(&a, &b);
        assert_eq!(
            c_fast, c_ref,
            "matmul mismatch at m={m} k={k} n={n} density={density}"
        );
    }
}

#[test]
fn matmul_associativity() {
    for &(m, p, q, n, seed) in &[
        (4usize, 5usize, 6usize, 7usize, 300u64),
        (3, 4, 3, 2, 301),
        (6, 3, 5, 4, 302),
    ] {
        let a = random_mzs(m, p, 0.3, seed);
        let b = random_mzs(p, q, 0.3, seed + 1000);
        let c = random_mzs(q, n, 0.3, seed + 2000);
        let ab_c = a.matmul(&b).matmul(&c);
        let a_bc = a.matmul(&b.matmul(&c));
        assert_eq!(ab_c, a_bc, "associativity at m={m} p={p} q={q} n={n}");
    }
}

#[test]
fn matmul_transpose_identity() {
    // (A·B)ᵀ == Bᵀ·Aᵀ
    for &(m, k, n, seed) in &[
        (4usize, 5usize, 6usize, 400u64),
        (3, 7, 5, 401),
        (8, 4, 8, 402),
    ] {
        let a = random_mzs(m, k, 0.3, seed);
        let b = random_mzs(k, n, 0.3, seed + 1000);
        let ab_t = a.matmul(&b).transpose();
        let bt_at = b.transpose().matmul(&a.transpose());
        assert_eq!(ab_t, bt_at, "(A·B)ᵀ ≠ Bᵀ·Aᵀ at m={m} k={k} n={n}");
    }
}

#[test]
fn transpose_is_involution() {
    let a = random_mzs(7, 11, 0.25, 500);
    let att = a.transpose().transpose();
    assert_eq!(a, att);
}

#[test]
fn transpose_swaps_dims() {
    let a = Mzs::from_triples(2, 5, &[(0, 0), (0, 3), (1, 1), (1, 4)]);
    let t = a.transpose();
    assert_eq!(t.rows(), 5);
    assert_eq!(t.cols(), 2);
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            assert_eq!(a.get(i, j), t.get(j, i));
        }
    }
}

#[test]
fn matmul_thin_tall_inputs() {
    // Stress shape handling: 1×k by k×1 → 1×1.
    let a = random_mzs(1, 50, 0.3, 600);
    let b = random_mzs(50, 1, 0.3, 601);
    let c = a.matmul(&b);
    assert_eq!(c.rows(), 1);
    assert_eq!(c.cols(), 1);
    // Compare against schoolbook.
    let cr = schoolbook(&a, &b);
    assert_eq!(c, cr);
}

#[test]
fn matmul_high_density() {
    // Density near 1 — all rows of A end up touching all of B; we're
    // testing the algorithm doesn't fall off cliffs here, not that
    // it's fast.
    let a = random_mzs(8, 8, 0.9, 700);
    let b = random_mzs(8, 8, 0.9, 701);
    let c = a.matmul(&b);
    let cr = schoolbook(&a, &b);
    assert_eq!(c, cr);
}

#[test]
#[should_panic(expected = "matmul")]
fn matmul_panics_on_shape_mismatch() {
    let a = Mzs::new(3, 4);
    let b = Mzs::new(5, 3);
    let _ = a.matmul(&b);
}

#[cfg(feature = "parallel")]
#[test]
fn matmul_parallel_matches_serial() {
    // Bit-identical to `matmul` at every (shape, density) we test.
    for &(m, k, n, density, seed) in &[
        (5usize, 7usize, 11usize, 0.1, 800u64),
        (8, 8, 8, 0.3, 801),
        (3, 13, 5, 0.5, 802),
        (32, 32, 32, 0.05, 803),
        (64, 32, 48, 0.2, 804),
    ] {
        let a = random_mzs(m, k, density, seed);
        let b = random_mzs(k, n, density, seed + 1000);
        let serial = a.matmul(&b);
        let parallel = a.matmul_parallel(&b);
        assert_eq!(
            serial, parallel,
            "parallel matmul disagreed with serial at m={m} k={k} n={n} density={density}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn matmul_parallel_empty_shapes() {
    let a = Mzs::new(0, 5);
    let b = Mzs::new(5, 3);
    let c = a.matmul_parallel(&b);
    assert_eq!(c.rows(), 0);
    assert_eq!(c.cols(), 3);

    let a = Mzs::new(3, 0);
    let b = Mzs::new(0, 4);
    let c = a.matmul_parallel(&b);
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 4);
    assert_eq!(c.nnz(), 0);
}
