// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! MatZpn AVX2 u16 / u32 microkernels — cross-validation against the
//! scalar Barrett `MatZpn::matmul_scalar` path.
//!
//! The suite proves that the public runtime-dispatch entry and its explicit
//! AVX2 path remain identical to the scalar floor. Standalone consumers may
//! compile below Noether's x86-64-v3 artifact floor.

#![cfg(target_arch = "x86_64")]

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

// ─── u16 AVX2 ─────────────────────────────────────────────────────

#[test]
fn avx2_u16_identity_f5() {
    let i = MatZpn::identity(5, 1, 8);
    let a = rand_matzpn(5, 1, 8, 8, 1);
    let avx = a.matmul_avx2(&i).expect("Z/5 ≤ 2^15");
    let scalar = a.matmul_scalar(&i);
    assert!(mats_equal(&avx, &scalar));
}

#[test]
fn public_matmul_dispatch_matches_scalar_floor() {
    for &(p, exponent) in &[(5u32, 2u32), (65_537, 1)] {
        let a = rand_matzpn(p, exponent, 24, 31, 17 + u64::from(p));
        let b = rand_matzpn(p, exponent, 31, 19, 23 + u64::from(p));
        assert!(mats_equal(&a.matmul(&b), &a.matmul_scalar(&b)));
    }
}

#[test]
fn avx2_u16_zero() {
    let a = MatZpn::zeros(7, 2, 8, 8);
    let b = rand_matzpn(7, 2, 8, 8, 2);
    let avx = a.matmul_avx2(&b).expect("Z/49");
    assert!(avx.data.to_vec().iter().all(|&v| v == 0));
}

#[test]
fn avx2_u16_matches_scalar_square_z25() {
    for &sz in &[8usize, 16, 32, 64] {
        let a = rand_matzpn(5, 2, sz, sz, 100 + sz as u64);
        let b = rand_matzpn(5, 2, sz, sz, 200 + sz as u64);
        let avx = a.matmul_avx2(&b).expect("Z/25");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "sz={sz}");
    }
}

#[test]
fn avx2_u16_rectangular_and_tails() {
    for &(m, k, n) in &[
        (5usize, 10, 9),
        (13, 20, 17),
        (3, 8, 7),
        (1, 40, 1),
        (6, 1, 11),
    ] {
        let a = rand_matzpn(7, 2, m, k, 300 + m as u64);
        let b = rand_matzpn(7, 2, k, n, 400 + n as u64);
        let avx = a.matmul_avx2(&b).expect("Z/49");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn avx2_u16_k_larger_than_kc_slab() {
    // KC_U16 = 256 inside the driver; exercise ≥3 slabs + partial tail.
    for &(m, k, n) in &[(12usize, 600, 16), (8, 520, 8), (16, 256, 32)] {
        let a = rand_matzpn(7, 2, m, k, 4000 + m as u64);
        let b = rand_matzpn(7, 2, k, n, 4100 + n as u64);
        let avx = a.matmul_avx2(&b).expect("Z/49");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn avx2_u16_larger_primes() {
    for &(p, n) in &[(11u32, 1u32), (127, 1), (181, 2)] {
        // 181² = 32761 ≤ 2^15.
        let sz = 24;
        let a = rand_matzpn(p, n, sz, sz, 500 + p as u64);
        let b = rand_matzpn(p, n, sz, sz, 600 + p as u64);
        let avx = a.matmul_avx2(&b).expect("p^n ≤ 2^15");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "p={p} n={n}");
    }
}

#[test]
fn avx2_u16_returns_none_for_large_modulus() {
    let a = rand_matzpn(65521, 1, 4, 4, 700);
    let b = rand_matzpn(65521, 1, 4, 4, 701);
    assert!(a.matmul_avx2(&b).is_none());
}

// ─── u32 AVX2 ─────────────────────────────────────────────────────

#[test]
fn avx2_u32_identity_f65537() {
    let i = MatZpn::identity(65537, 1, 8);
    let a = rand_matzpn(65537, 1, 8, 8, 1001);
    let avx = a.matmul_avx2_u32(&i).expect("Z/65537 ≤ 2^31");
    let scalar = a.matmul_scalar(&i);
    assert!(mats_equal(&avx, &scalar));
}

#[test]
fn avx2_u32_zero() {
    let a = MatZpn::zeros(65537, 1, 8, 8);
    let b = rand_matzpn(65537, 1, 8, 8, 1002);
    let avx = a.matmul_avx2_u32(&b).expect("Z/65537");
    assert!(avx.data.to_vec().iter().all(|&v| v == 0));
}

#[test]
fn avx2_u32_matches_scalar_z65537() {
    for &sz in &[4usize, 8, 12, 16, 32] {
        let a = rand_matzpn(65537, 1, sz, sz, 1100 + sz as u64);
        let b = rand_matzpn(65537, 1, sz, sz, 1200 + sz as u64);
        let avx = a.matmul_avx2_u32(&b).expect("Z/65537 ≤ 2^31");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "sz={sz}");
    }
}

#[test]
fn avx2_u32_larger_primes() {
    for &p in &[1009u32, 65537, 1_000_003, 67_108_859] {
        let sz = 8;
        let a = rand_matzpn(p, 1, sz, sz, 2000 + p as u64);
        let b = rand_matzpn(p, 1, sz, sz, 3000 + p as u64);
        let avx = a.matmul_avx2_u32(&b).expect("p ≤ 2^31");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "p={p}");
    }
}

#[test]
fn avx2_u32_rectangular_and_tails() {
    for &(m, k, n) in &[(5usize, 10, 9), (11, 13, 15), (3, 5, 2), (1, 20, 1)] {
        let a = rand_matzpn(65537, 1, m, k, 4000 + m as u64);
        let b = rand_matzpn(65537, 1, k, n, 5000 + n as u64);
        let avx = a.matmul_avx2_u32(&b).expect("Z/65537");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn avx2_u32_k_larger_than_kc_slab() {
    // KC_U32 = 128; exercise ≥3 slabs plus a short final slab.
    for &(m, k, n) in &[(12usize, 400, 16), (8, 260, 8), (5, 128, 9)] {
        let a = rand_matzpn(65537, 1, m, k, 7000 + m as u64);
        let b = rand_matzpn(65537, 1, k, n, 7100 + n as u64);
        let avx = a.matmul_avx2_u32(&b).expect("Z/65537");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&avx, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn avx2_u32_associativity() {
    let sz = 8;
    let p = 65537u32;
    let a = rand_matzpn(p, 1, sz, sz, 6001);
    let b = rand_matzpn(p, 1, sz, sz, 6002);
    let c = rand_matzpn(p, 1, sz, sz, 6003);
    let ab = a.matmul_avx2_u32(&b).unwrap();
    let bc = b.matmul_avx2_u32(&c).unwrap();
    let lhs = ab.matmul_avx2_u32(&c).unwrap();
    let rhs = a.matmul_avx2_u32(&bc).unwrap();
    assert!(mats_equal(&lhs, &rhs));
}
