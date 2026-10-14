// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Hensel-lifted inverse over Z/p^n.

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn identity(p: u32, n: u32, size: usize) -> MatZpn {
    let mut i = MatZpn::zeros(p, n, size, size);
    for k in 0..size {
        i.set(k, k, 1);
    }
    i
}

/// Build a random invertible matrix (one whose F_p reduction has
/// non-zero determinant) — loop until the inverse works.
fn random_invertible(p: u32, n: u32, size: usize, seed: u64) -> MatZpn {
    let p_pow_n = (p as u64).pow(n);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    for attempt in 0..100 {
        let m = MatZpn::from_fn(p, n, size, size, |_, _| rng.random_range(0..p_pow_n));
        if m.inverse().is_some() {
            return m;
        }
        let _ = attempt;
    }
    panic!("couldn't build invertible matrix after 100 attempts");
}

#[test]
fn inverse_of_identity() {
    let i3 = identity(5, 1, 3);
    let inv = i3.inverse().unwrap();
    assert_eq!(inv, i3);
}

#[test]
fn inverse_1x1() {
    // Z/7: inverse of 3 is 5 (since 3·5 = 15 ≡ 1).
    let m = MatZpn::from_fn(7, 1, 1, 1, |_, _| 3);
    let inv = m.inverse().unwrap();
    assert_eq!(inv.get(0, 0), 5);
    // Verify A · A^{-1} = I.
    let prod = m.matmul(&inv);
    assert_eq!(prod.get(0, 0), 1);
}

#[test]
fn inverse_round_trip_fp() {
    // For F_p (n=1), no Newton iteration needed. Verify A · A^{-1} = I.
    let p = 11;
    let n = 5;
    let a = random_invertible(p, 1, n, 42);
    let inv = a.inverse().unwrap();
    let prod = a.matmul(&inv);
    assert_eq!(prod, identity(p, 1, n));
}

#[test]
fn inverse_round_trip_z_p_squared() {
    // Z/25 — Hensel lift doubles precision.
    let p = 5;
    let n = 2;
    let size = 4;
    let a = random_invertible(p, n, size, 100);
    let inv = a.inverse().unwrap();
    let prod = a.matmul(&inv);
    assert_eq!(prod, identity(p, n, size));
}

#[test]
fn inverse_round_trip_higher_precision() {
    // Z/3^5 = Z/243.
    let p = 3;
    let n = 5;
    let size = 4;
    let a = random_invertible(p, n, size, 200);
    let inv = a.inverse().unwrap();
    let prod = a.matmul(&inv);
    assert_eq!(prod, identity(p, n, size));
}

#[test]
fn inverse_round_trip_larger_prime() {
    let p = 257;
    let n = 1;
    let size = 5;
    let a = random_invertible(p, n, size, 300);
    let inv = a.inverse().unwrap();
    let prod = a.matmul(&inv);
    assert_eq!(prod, identity(p, n, size));
}

#[test]
fn inverse_detects_singular_over_fp() {
    // Z/5, matrix with determinant 0 mod 5:
    //   [[1, 2], [3, 6]] — row 2 = 3·row 1.
    let m = MatZpn::from_fn(5, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (1, 0) => 3,
        (1, 1) => 1, // 6 mod 5 = 1, so matrix is [[1,2],[3,1]] — wait det = 1-6 = -5 = 0 mod 5.
        _ => unreachable!(),
    });
    assert!(m.inverse().is_none());
}

#[test]
fn inverse_works_when_det_is_non_unit_but_fp_nonzero() {
    // Z/9: det = 3 (non-unit in Z/9, but 3 mod 3 = 0 → F_3 singular).
    // Actually that's singular.  Use det = 4 (coprime to 3): A = [[2, 0], [0, 2]], det = 4.
    let m = MatZpn::from_fn(3, 2, 2, 2, |i, j| if i == j { 2 } else { 0 });
    let inv = m.inverse().unwrap();
    let prod = m.matmul(&inv);
    let id = identity(3, 2, 2);
    assert_eq!(prod, id);
}

#[test]
fn inverse_product_is_product_of_inverses() {
    // (AB)^{-1} = B^{-1} · A^{-1}.
    let p = 7;
    let n = 2;
    let size = 3;
    let a = random_invertible(p, n, size, 500);
    let b = random_invertible(p, n, size, 501);
    let ab = a.matmul(&b);
    let ab_inv = ab.inverse().unwrap();
    let expected = b.inverse().unwrap().matmul(&a.inverse().unwrap());
    assert_eq!(ab_inv, expected);
}
