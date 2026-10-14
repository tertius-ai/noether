// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::axpy` and `MatZpn::dot` Level-1 cross-validation against
//! hand-computed oracles and a scalar-loop reference.

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_vec(p: u32, n: u32, len: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    (0..len).map(|_| rng.random_range(0..m)).collect()
}

fn axpy_reference(p: u32, n: u32, alpha: u64, x: &[u64], y: &[u64]) -> Vec<u64> {
    let modulus = (p as u64).pow(n) as u128;
    x.iter()
        .zip(y)
        .map(|(&xv, &yv)| ((alpha as u128 * xv as u128 + yv as u128) % modulus) as u64)
        .collect()
}

fn dot_reference(p: u32, n: u32, x: &[u64], y: &[u64]) -> u64 {
    let modulus = (p as u64).pow(n) as u128;
    let mut acc: u128 = 0;
    for (&xv, &yv) in x.iter().zip(y) {
        acc = (acc + xv as u128 * yv as u128) % modulus;
    }
    acc as u64
}

// ─── Hand-computed axpy ───────────────────────────────────────────

#[test]
fn axpy_hand_computed_z11() {
    // α=3, x=[1,2,3,4,5], y=[10, 10, 10, 10, 10] in Z/11.
    // 3·x = [3, 6, 9, 12, 15] ≡ [3, 6, 9, 1, 4] (mod 11).
    // 3·x + y = [13, 16, 19, 11, 14] ≡ [2, 5, 8, 0, 3] (mod 11).
    let x = vec![1u64, 2, 3, 4, 5];
    let y = vec![10u64; 5];
    let got = MatZpn::axpy(11, 1, 3, &x, &y);
    assert_eq!(got, vec![2u64, 5, 8, 0, 3]);
}

#[test]
fn axpy_alpha_zero_returns_y() {
    let y = vec![1u64, 2, 3];
    let x = vec![7u64, 8, 9];
    let got = MatZpn::axpy(11, 1, 0, &x, &y);
    assert_eq!(got, y);
}

#[test]
fn axpy_alpha_one_is_element_wise_add() {
    let x = vec![7u64, 8, 9];
    let y = vec![1u64, 2, 3];
    let got = MatZpn::axpy(11, 1, 1, &x, &y);
    assert_eq!(got, vec![8u64, 10, 1]); // 9+3 = 12 ≡ 1 mod 11
}

// ─── Hand-computed dot ────────────────────────────────────────────

#[test]
fn dot_hand_computed_z11() {
    // x=[1,2,3], y=[4,5,6]. Σ x[i]·y[i] = 4+10+18 = 32 ≡ 10 mod 11.
    let x = vec![1u64, 2, 3];
    let y = vec![4u64, 5, 6];
    let got = MatZpn::dot(11, 1, &x, &y);
    assert_eq!(got, 10);
}

#[test]
fn dot_of_zero_vector_is_zero() {
    let x = vec![0u64; 8];
    let y = vec![7u64, 8, 9, 10, 1, 2, 3, 4];
    assert_eq!(MatZpn::dot(11, 1, &x, &y), 0);
}

#[test]
fn dot_commutative() {
    let x = vec![3u64, 1, 4, 1, 5, 9, 2, 6];
    let y = vec![5u64, 3, 5, 8, 9, 7, 9, 3];
    let a = MatZpn::dot(11, 1, &x, &y);
    let b = MatZpn::dot(11, 1, &y, &x);
    assert_eq!(a, b);
}

// ─── Parametric cross-validation ──────────────────────────────────

fn axpy_sweep(p: u32, n: u32, len: usize, seed_base: u64) {
    let modulus = (p as u64).pow(n);
    let alphas = [0u64, 1, 2, modulus - 1];
    let x = rand_vec(p, n, len, seed_base);
    let y = rand_vec(p, n, len, seed_base.wrapping_add(1));
    for &alpha in &alphas {
        let got = MatZpn::axpy(p, n, alpha, &x, &y);
        let expected = axpy_reference(p, n, alpha, &x, &y);
        assert_eq!(got, expected, "axpy p={p} n={n} len={len} alpha={alpha}");
    }
}

fn dot_sweep(p: u32, n: u32, len: usize, seed_base: u64) {
    let x = rand_vec(p, n, len, seed_base);
    let y = rand_vec(p, n, len, seed_base.wrapping_add(1));
    let got = MatZpn::dot(p, n, &x, &y);
    let expected = dot_reference(p, n, &x, &y);
    assert_eq!(got, expected, "dot p={p} n={n} len={len}");
}

#[test]
fn axpy_matches_reference_across_moduli() {
    for &(p, n, len) in &[
        (5u32, 2u32, 16usize),
        (7, 2, 20),
        (65537, 1, 12),
        (1_000_003, 1, 24),
    ] {
        axpy_sweep(p, n, len, 100_000 + len as u64);
    }
}

#[test]
fn dot_matches_reference_across_moduli() {
    for &(p, n, len) in &[
        (5u32, 2u32, 16usize),
        (7, 2, 20),
        (65537, 1, 12),
        (1_000_003, 1, 24),
    ] {
        dot_sweep(p, n, len, 200_000 + len as u64);
    }
}

// ─── Algebraic identities ─────────────────────────────────────────

#[test]
fn axpy_linearity_in_alpha() {
    // axpy(α+β, x, 0) should equal axpy(α, x, axpy(β, x, 0)).
    let p = 11u32;
    let n = 1u32;
    let x = vec![3u64, 5, 7, 9];
    let zero = vec![0u64; 4];
    let alpha = 4u64;
    let beta = 6u64;
    let modulus = (p as u64).pow(n);
    let sum = (alpha + beta) % modulus;
    let a = MatZpn::axpy(p, n, sum, &x, &zero);
    let tmp = MatZpn::axpy(p, n, beta, &x, &zero);
    let b = MatZpn::axpy(p, n, alpha, &x, &tmp);
    assert_eq!(a, b);
}

#[test]
fn dot_distributes_over_axpy() {
    // dot(axpy(α, x, y), z) == α · dot(x, z) + dot(y, z)  (mod p^n)
    let p = 11u32;
    let n = 1u32;
    let x = vec![3u64, 1, 4, 1, 5];
    let y = vec![9u64, 2, 6, 5, 3];
    let z = vec![5u64, 8, 9, 7, 9];
    let alpha = 4u64;
    let lhs = MatZpn::dot(p, n, &MatZpn::axpy(p, n, alpha, &x, &y), &z);
    let dxz = MatZpn::dot(p, n, &x, &z);
    let dyz = MatZpn::dot(p, n, &y, &z);
    let modulus = (p as u64).pow(n);
    let rhs = (alpha * dxz + dyz) % modulus;
    assert_eq!(lhs, rhs);
}

// ─── Wiring into existing ops ─────────────────────────────────────

#[test]
fn dot_reproduces_gemv_1xn_slice() {
    // For an A with a single row, gemv(α=1, NoTrans, x, β=0, y=[0])
    // returns a 1-vector whose only entry equals dot(A_row, x).
    let p = 7u32;
    let n = 2u32;
    let len = 9;
    let row = rand_vec(p, n, len, 300_000);
    let x = rand_vec(p, n, len, 300_001);
    let a = MatZpn::from_fn(p, n, 1, len, |_, j| row[j]);
    let y_in = vec![0u64; 1];
    let gemv_out = a.gemv(1, pwvla::zpn::TransA::NoTrans, &x, 0, &y_in);
    let dot_out = MatZpn::dot(p, n, &row, &x);
    assert_eq!(gemv_out, vec![dot_out]);
}
