// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! F_p matrix multiplication through `MatZpn` with n = 1.
//!
//! F_p matmul is a special case of Z/p^n matmul; this file exercises
//! field-specific properties (invertibility of any non-singular matrix,
//! idempotency of projectors, etc.).

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn identity_fp(p: u32, size: usize) -> MatZpn {
    let mut i = MatZpn::zeros(p, 1, size, size);
    for k in 0..size {
        i.set(k, k, 1);
    }
    i
}

fn random_fp(p: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    MatZpn::from_fn(p, 1, rows, cols, |_, _| rng.random_range(0..p as u64))
}

#[test]
fn fp_matmul_identity() {
    let p = 13;
    let a = random_fp(p, 4, 4, 1);
    assert_eq!(a.matmul(&identity_fp(p, 4)), a);
    assert_eq!(identity_fp(p, 4).matmul(&a), a);
}

#[test]
fn fp_matmul_associative() {
    let p = 17;
    let a = random_fp(p, 3, 4, 10);
    let b = random_fp(p, 4, 5, 11);
    let c = random_fp(p, 5, 3, 12);
    assert_eq!(a.matmul(&b).matmul(&c), a.matmul(&b.matmul(&c)));
}

#[test]
fn fp_matmul_distributes_over_add() {
    let p = 11;
    let a = random_fp(p, 3, 3, 20);
    let b = random_fp(p, 3, 3, 21);
    let c = random_fp(p, 3, 3, 22);
    let lhs = a.matmul(&b.add(&c));
    let rhs = a.matmul(&b).add(&a.matmul(&c));
    assert_eq!(lhs, rhs);
}

#[test]
fn fp_matrix_inverse_round_trip() {
    // Over F_p, any matrix with non-zero determinant is invertible.
    let p = 7;
    let size = 5;
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    for attempt in 0..20 {
        let a = MatZpn::from_fn(p, 1, size, size, |_, _| rng.random_range(0..p as u64));
        if let Some(inv) = a.inverse() {
            assert_eq!(a.matmul(&inv), identity_fp(p, size));
            return;
        }
        let _ = attempt;
    }
    panic!("couldn't find invertible random matrix over F_7 in 20 tries");
}

#[test]
fn fp_projector_idempotent() {
    // For P = [[1, 0], [0, 0]] in F_p, P·P = P (projection).
    let p = 19;
    let proj = MatZpn::from_fn(p, 1, 2, 2, |i, j| if i == 0 && j == 0 { 1 } else { 0 });
    assert_eq!(proj.matmul(&proj), proj);
}

#[test]
fn fp_matmul_different_primes() {
    // Exercise the Barrett reducer across several primes.
    for &p in &[3u32, 5, 7, 11, 13, 101, 257, 65521] {
        let a = random_fp(p, 3, 3, p as u64 * 7);
        let b = random_fp(p, 3, 3, p as u64 * 11);
        let c = a.matmul(&b);
        // Sanity: all entries in [0, p).
        for i in 0..3 {
            for j in 0..3 {
                assert!(c.get(i, j) < p as u64, "entry out of range at p={p}");
            }
        }
    }
}

#[test]
fn fp_zero_times_a_is_zero() {
    let p = 11;
    let a = random_fp(p, 3, 4, 50);
    let z = MatZpn::zeros(p, 1, 4, 5);
    let result = a.matmul(&z);
    for i in 0..3 {
        for j in 0..5 {
            assert_eq!(result.get(i, j), 0);
        }
    }
}

#[test]
fn fp_rotation_like_f5() {
    // In F_5, the "rotation-by-2" matrix [[0, 1], [-1, 0]] = [[0, 1], [4, 0]].
    // Its square is -I = [[4, 0], [0, 4]].
    let p = 5;
    let r = MatZpn::from_fn(p, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 0) => 4,
        (1, 1) => 0,
        _ => unreachable!(),
    });
    let r2 = r.matmul(&r);
    assert_eq!(r2.get(0, 0), 4);
    assert_eq!(r2.get(0, 1), 0);
    assert_eq!(r2.get(1, 0), 0);
    assert_eq!(r2.get(1, 1), 4);
}
