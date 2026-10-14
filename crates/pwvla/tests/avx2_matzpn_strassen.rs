// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strassen-Winograd recursion cross-validation against the tuned
//! AVX2 GEMM (`MatZpn::matmul_avx2` / `matmul_avx2_u32`) and the
//! scalar Barrett reference (`MatZpn::matmul_scalar`).
//!
//! The tests force deep recursion by passing a small explicit threshold
//! (typically 4) to `matmul_strassen_avx2_with_threshold`, so 16×16
//! and 32×32 inputs exercise multi-level Strassen + the 18-add
//! combine logic per level.

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

// ─── u16 Strassen ──────────────────────────────────────────────────

#[test]
fn strassen_u16_identity() {
    // Identity × random must equal random. Threshold 4 on an 8×8
    // forces one Strassen split (4×4 sub-blocks fall through to
    // matmul_u16_mod leaf).
    let i = MatZpn::identity(5, 2, 8);
    let a = rand_matzpn(5, 2, 8, 8, 10);
    let out = a.matmul_strassen_avx2_with_threshold(&i, 4).expect("Z/25");
    assert!(mats_equal(&out, &a));
}

#[test]
fn strassen_u16_zero() {
    let a = MatZpn::zeros(7, 2, 16, 16);
    let b = rand_matzpn(7, 2, 16, 16, 11);
    let out = a.matmul_strassen_avx2_with_threshold(&b, 4).expect("Z/49");
    assert!(out.data.to_vec().iter().all(|&v| v == 0));
}

#[test]
fn strassen_u16_matches_gemm_square() {
    // Threshold 4 on 16×16 → split to 8×8 → split to 4×4 leaves.
    // Two levels of recursion, 49 leaf matmuls total.
    for &sz in &[8usize, 16, 32, 64] {
        let a = rand_matzpn(5, 2, sz, sz, 100 + sz as u64);
        let b = rand_matzpn(5, 2, sz, sz, 200 + sz as u64);
        let strassen = a.matmul_strassen_avx2_with_threshold(&b, 4).expect("Z/25");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "sz={sz}");
    }
}

#[test]
fn strassen_u16_matches_gemm_rectangular() {
    // Rectangular (m ≠ k ≠ n) with all dims even + ≥ threshold
    // still recurses cleanly since Strassen doesn't require square.
    for &(m, k, n) in &[(8usize, 12, 16), (16, 8, 24), (32, 16, 8), (24, 24, 24)] {
        let a = rand_matzpn(7, 2, m, k, 300 + m as u64);
        let b = rand_matzpn(7, 2, k, n, 400 + n as u64);
        let strassen = a.matmul_strassen_avx2_with_threshold(&b, 4).expect("Z/49");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn strassen_u16_odd_dim_falls_through_to_gemm() {
    // Any odd dim at the outer call should hit the leaf immediately;
    // output still must equal scalar matmul bit-for-bit.
    for &(m, k, n) in &[(7usize, 8, 8), (8, 7, 8), (8, 8, 7), (15, 15, 15)] {
        let a = rand_matzpn(5, 2, m, k, 500 + m as u64);
        let b = rand_matzpn(5, 2, k, n, 600 + n as u64);
        let strassen = a.matmul_strassen_avx2_with_threshold(&b, 4).expect("Z/25");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn strassen_u16_threshold_above_size_matches_gemm() {
    // Threshold ≥ size → no recursion, pure gemm leaf.
    let a = rand_matzpn(5, 2, 32, 32, 700);
    let b = rand_matzpn(5, 2, 32, 32, 701);
    let strassen = a
        .matmul_strassen_avx2_with_threshold(&b, 1024)
        .expect("Z/25");
    let scalar = a.matmul_scalar(&b);
    assert!(mats_equal(&strassen, &scalar));
}

#[test]
fn strassen_u16_associativity() {
    // (AB)C == A(BC) under Strassen — both sides use the same
    // recursion, so this exercises the algebra rather than just
    // fresh cross-validation.
    let sz = 16;
    let a = rand_matzpn(5, 2, sz, sz, 800);
    let b = rand_matzpn(5, 2, sz, sz, 801);
    let c = rand_matzpn(5, 2, sz, sz, 802);
    let ab = a.matmul_strassen_avx2_with_threshold(&b, 4).unwrap();
    let bc = b.matmul_strassen_avx2_with_threshold(&c, 4).unwrap();
    let lhs = ab.matmul_strassen_avx2_with_threshold(&c, 4).unwrap();
    let rhs = a.matmul_strassen_avx2_with_threshold(&bc, 4).unwrap();
    assert!(mats_equal(&lhs, &rhs));
}

#[test]
fn strassen_u16_default_threshold_small_falls_through() {
    // Default threshold 512 on a 16×16 input → straight-through
    // gemm leaf; confirms the default-public API still returns
    // correct results on below-crossover inputs.
    let a = rand_matzpn(5, 2, 16, 16, 900);
    let b = rand_matzpn(5, 2, 16, 16, 901);
    let strassen = a.matmul_strassen_avx2(&b).expect("Z/25");
    let scalar = a.matmul_scalar(&b);
    assert!(mats_equal(&strassen, &scalar));
}

#[test]
fn strassen_u16_rejects_large_modulus() {
    let a = rand_matzpn(65521, 1, 8, 8, 1000);
    let b = rand_matzpn(65521, 1, 8, 8, 1001);
    assert!(a.matmul_strassen_avx2_with_threshold(&b, 4).is_none());
}

// ─── u32 Strassen ──────────────────────────────────────────────────

#[test]
fn strassen_u32_identity() {
    let i = MatZpn::identity(65537, 1, 8);
    let a = rand_matzpn(65537, 1, 8, 8, 2000);
    let out = a
        .matmul_strassen_avx2_u32_with_threshold(&i, 4)
        .expect("Z/65537");
    assert!(mats_equal(&out, &a));
}

#[test]
fn strassen_u32_zero() {
    let a = MatZpn::zeros(65537, 1, 16, 16);
    let b = rand_matzpn(65537, 1, 16, 16, 2001);
    let out = a
        .matmul_strassen_avx2_u32_with_threshold(&b, 4)
        .expect("Z/65537");
    assert!(out.data.to_vec().iter().all(|&v| v == 0));
}

#[test]
fn strassen_u32_matches_gemm_square() {
    for &sz in &[8usize, 16, 32] {
        let a = rand_matzpn(65537, 1, sz, sz, 2100 + sz as u64);
        let b = rand_matzpn(65537, 1, sz, sz, 2200 + sz as u64);
        let strassen = a
            .matmul_strassen_avx2_u32_with_threshold(&b, 4)
            .expect("Z/65537");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "sz={sz}");
    }
}

#[test]
fn strassen_u32_matches_gemm_rectangular() {
    for &(m, k, n) in &[(8usize, 12, 16), (16, 8, 8), (24, 24, 8)] {
        let a = rand_matzpn(65537, 1, m, k, 2300 + m as u64);
        let b = rand_matzpn(65537, 1, k, n, 2400 + n as u64);
        let strassen = a
            .matmul_strassen_avx2_u32_with_threshold(&b, 4)
            .expect("Z/65537");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "({m},{k},{n})");
    }
}

#[test]
fn strassen_u32_larger_primes() {
    for &p in &[1009u32, 65537, 1_000_003, 67_108_859] {
        let sz = 16;
        let a = rand_matzpn(p, 1, sz, sz, 2500 + p as u64);
        let b = rand_matzpn(p, 1, sz, sz, 2600 + p as u64);
        let strassen = a
            .matmul_strassen_avx2_u32_with_threshold(&b, 4)
            .expect("p ≤ 2^31");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&strassen, &scalar), "p={p}");
    }
}

#[test]
fn strassen_u32_associativity() {
    let sz = 16;
    let p = 65537u32;
    let a = rand_matzpn(p, 1, sz, sz, 3000);
    let b = rand_matzpn(p, 1, sz, sz, 3001);
    let c = rand_matzpn(p, 1, sz, sz, 3002);
    let ab = a.matmul_strassen_avx2_u32_with_threshold(&b, 4).unwrap();
    let bc = b.matmul_strassen_avx2_u32_with_threshold(&c, 4).unwrap();
    let lhs = ab.matmul_strassen_avx2_u32_with_threshold(&c, 4).unwrap();
    let rhs = a.matmul_strassen_avx2_u32_with_threshold(&bc, 4).unwrap();
    assert!(mats_equal(&lhs, &rhs));
}
