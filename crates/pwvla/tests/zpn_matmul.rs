// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Z/p^n matrix multiplication.

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

/// Reference matmul via u128 arithmetic (no Barrett).
fn reference_matmul(a: &MatZpn, b: &MatZpn) -> MatZpn {
    assert_eq!(a.cols, b.rows);
    let p_pow_n = a.data.p_pow_n as u128;
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut c = MatZpn::zeros(a.p, a.n, m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s: u128 = 0;
            for kk in 0..k {
                s = (s + (a.get(i, kk) as u128) * (b.get(kk, j) as u128)) % p_pow_n;
            }
            c.set(i, j, s as u64);
        }
    }
    c
}

#[test]
fn matmul_zero_matrices() {
    let a = MatZpn::zeros(7, 1, 4, 5);
    let b = MatZpn::zeros(7, 1, 5, 3);
    let c = a.matmul(&b);
    assert_eq!(c.rows, 4);
    assert_eq!(c.cols, 3);
    for i in 0..4 {
        for j in 0..3 {
            assert_eq!(c.get(i, j), 0);
        }
    }
}

#[test]
fn matmul_identity_is_identity() {
    // A · I = A.
    let p = 11;
    let n = 1;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let a = MatZpn::from_fn(p, n, 5, 5, |_, _| rng.random_range(0..11));
    let mut i5 = MatZpn::zeros(p, n, 5, 5);
    for k in 0..5 {
        i5.set(k, k, 1);
    }
    assert_eq!(a.matmul(&i5), a);
    assert_eq!(i5.matmul(&a), a);
}

#[test]
fn matmul_matches_reference_small() {
    let p = 7;
    let n = 1;
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let a = MatZpn::from_fn(p, n, 3, 4, |_, _| rng.random_range(0..p as u64));
    let b = MatZpn::from_fn(p, n, 4, 3, |_, _| rng.random_range(0..p as u64));
    let via_fast = a.matmul(&b);
    let via_ref = reference_matmul(&a, &b);
    assert_eq!(via_fast, via_ref);
}

#[test]
fn matmul_matches_reference_z_p_squared() {
    // Z/25 = Z/5^2.
    let p = 5;
    let n = 2;
    let p_pow_n = 25u64;
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let a = MatZpn::from_fn(p, n, 4, 4, |_, _| rng.random_range(0..p_pow_n));
    let b = MatZpn::from_fn(p, n, 4, 4, |_, _| rng.random_range(0..p_pow_n));
    assert_eq!(a.matmul(&b), reference_matmul(&a, &b));
}

#[test]
fn matmul_matches_reference_larger_prime() {
    let p = 257;
    let n = 1;
    let mut rng = ChaCha20Rng::seed_from_u64(300);
    let a = MatZpn::from_fn(p, n, 5, 6, |_, _| rng.random_range(0..p as u64));
    let b = MatZpn::from_fn(p, n, 6, 5, |_, _| rng.random_range(0..p as u64));
    assert_eq!(a.matmul(&b), reference_matmul(&a, &b));
}

#[test]
fn matmul_associativity() {
    let p = 11;
    let mut rng = ChaCha20Rng::seed_from_u64(400);
    let a = MatZpn::from_fn(p, 1, 3, 4, |_, _| rng.random_range(0..p as u64));
    let b = MatZpn::from_fn(p, 1, 4, 3, |_, _| rng.random_range(0..p as u64));
    let c = MatZpn::from_fn(p, 1, 3, 2, |_, _| rng.random_range(0..p as u64));
    let lhs = a.matmul(&b).matmul(&c);
    let rhs = a.matmul(&b.matmul(&c));
    assert_eq!(lhs, rhs);
}

#[test]
fn matmul_rank_one_product() {
    // Outer product u v^T where u is k×1 and v is 1×n gives a rank-1 matrix.
    // For Z/7 with u=[1,2,3], v=[4,5,6]: [[4,5,6],[8,10,12],[12,15,18]] mod 7 = [[4,5,6],[1,3,5],[5,1,4]].
    let p = 7;
    let u = MatZpn::from_fn(p, 1, 3, 1, |i, _| [1u64, 2, 3][i]);
    let v = MatZpn::from_fn(p, 1, 1, 3, |_, j| [4u64, 5, 6][j]);
    let r = u.matmul(&v);
    // row 0: 1*[4,5,6] = [4,5,6]
    // row 1: 2*[4,5,6] = [8,10,12] mod 7 = [1,3,5]
    // row 2: 3*[4,5,6] = [12,15,18] mod 7 = [5,1,4]
    assert_eq!(r.get(0, 0), 4);
    assert_eq!(r.get(0, 1), 5);
    assert_eq!(r.get(0, 2), 6);
    assert_eq!(r.get(1, 0), 1);
    assert_eq!(r.get(1, 1), 3);
    assert_eq!(r.get(1, 2), 5);
    assert_eq!(r.get(2, 0), 5);
    assert_eq!(r.get(2, 1), 1);
    assert_eq!(r.get(2, 2), 4);
}

#[test]
fn matmul_medium_size_stress() {
    let p = 13;
    let n = 1;
    let mut rng = ChaCha20Rng::seed_from_u64(500);
    let size = 20;
    let a = MatZpn::from_fn(p, n, size, size, |_, _| rng.random_range(0..p as u64));
    let b = MatZpn::from_fn(p, n, size, size, |_, _| rng.random_range(0..p as u64));
    assert_eq!(a.matmul(&b), reference_matmul(&a, &b));
}
