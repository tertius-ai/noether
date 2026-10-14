// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::gemv` cross-validation against the scalar oracle:
//! treat `x` as a single-column matrix, call `matmul` (optionally on
//! A's transpose), then scale by α and add β·y element-wise.

use pwvla::zpn::{MatZpn, TransA};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..m))
}

fn rand_vec(p: u32, n: u32, len: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    (0..len).map(|_| rng.random_range(0..m)).collect()
}

/// Oracle reference: wrap `x` as an `in_len × 1` matrix, matmul
/// against op(A), read back the single column, scale by α, add β·y.
fn gemv_reference(a: &MatZpn, alpha: u64, op: TransA, x: &[u64], beta: u64, y: &[u64]) -> Vec<u64> {
    let modulus = (a.p as u64).pow(a.n);
    let (out_len, in_len) = match op {
        TransA::NoTrans => (a.rows, a.cols),
        TransA::Trans => (a.cols, a.rows),
    };
    assert_eq!(x.len(), in_len);
    assert_eq!(y.len(), out_len);

    let a_eff = match op {
        TransA::NoTrans => MatZpn::from_fn(a.p, a.n, a.rows, a.cols, |i, j| a.get(i, j)),
        TransA::Trans => MatZpn::from_fn(a.p, a.n, a.cols, a.rows, |i, j| a.get(j, i)),
    };
    let x_col = MatZpn::from_fn(a.p, a.n, in_len, 1, |i, _| x[i]);
    let prod = a_eff.matmul(&x_col);

    (0..out_len)
        .map(|i| {
            let s = alpha as u128 * prod.get(i, 0) as u128;
            let b = beta as u128 * y[i] as u128;
            ((s + b) % modulus as u128) as u64
        })
        .collect()
}

// ─── Hand-computed 2×3 sanity ─────────────────────────────────────

#[test]
fn gemv_hand_computed_2x3_notrans_alpha_beta() {
    // A = [ [1, 2, 3], [4, 5, 6] ] over Z/11
    // x = [7, 8, 9], y_in = [10, 1], α = 2, β = 3.
    // op(A)·x = [1·7 + 2·8 + 3·9, 4·7 + 5·8 + 6·9] = [50, 122] ≡ [6, 1] (mod 11)
    // α·(A·x) + β·y = [12, 2] + [30, 3] = [42, 5] ≡ [9, 5] (mod 11)
    let a = MatZpn::from_fn(11, 1, 2, 3, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (0, 2) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (1, 2) => 6,
        _ => unreachable!(),
    });
    let x = vec![7u64, 8, 9];
    let y_in = vec![10u64, 1];
    let got = a.gemv(2, TransA::NoTrans, &x, 3, &y_in);
    assert_eq!(got, vec![9u64, 5]);
}

#[test]
fn gemv_hand_computed_trans_distinguishes_from_notrans() {
    // A = [ [1, 2, 3], [4, 5, 6] ] over Z/11.
    // Aᵀ·x where x has length 2 (the row-dim of A).
    // x = [2, 3].  Aᵀ = [[1, 4], [2, 5], [3, 6]].
    // Aᵀ·x = [1·2 + 4·3, 2·2 + 5·3, 3·2 + 6·3] = [14, 19, 24] ≡ [3, 8, 2] (mod 11)
    // α = 1, β = 0 → gemv output = Aᵀ·x.
    let a = MatZpn::from_fn(11, 1, 2, 3, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (0, 2) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (1, 2) => 6,
        _ => unreachable!(),
    });
    let x = vec![2u64, 3];
    let y_in = vec![0u64; 3];
    let got = a.gemv(1, TransA::Trans, &x, 0, &y_in);
    assert_eq!(got, vec![3u64, 8, 2]);
}

// ─── Cross-validation vs matmul oracle ─────────────────────────────

fn sweep_alpha_beta(p: u32, n: u32, rows: usize, cols: usize, seed_base: u64) {
    let modulus = (p as u64).pow(n);
    let alphas = [0u64, 1, 2, modulus - 1];
    let betas = [0u64, 1, 2, modulus - 1];
    let a = rand_matzpn(p, n, rows, cols, seed_base);
    for (ai, &alpha) in alphas.iter().enumerate() {
        for (bi, &beta) in betas.iter().enumerate() {
            for (oi, &op) in [TransA::NoTrans, TransA::Trans].iter().enumerate() {
                let (in_len, out_len) = match op {
                    TransA::NoTrans => (cols, rows),
                    TransA::Trans => (rows, cols),
                };
                let x = rand_vec(
                    p,
                    n,
                    in_len,
                    seed_base + 1000 + ai as u64 + bi as u64 * 10 + oi as u64 * 100,
                );
                let y_in = rand_vec(
                    p,
                    n,
                    out_len,
                    seed_base + 2000 + ai as u64 + bi as u64 * 10 + oi as u64 * 100,
                );
                let got = a.gemv(alpha, op, &x, beta, &y_in);
                let expected = gemv_reference(&a, alpha, op, &x, beta, &y_in);
                assert_eq!(
                    got, expected,
                    "gemv mismatch: p={p} n={n} rows={rows} cols={cols} \
                     alpha={alpha} beta={beta} op={op:?}"
                );
            }
        }
    }
}

#[test]
fn gemv_matches_reference_z25_square() {
    sweep_alpha_beta(5, 2, 8, 8, 10_000);
}

#[test]
fn gemv_matches_reference_z49_rectangular_tall() {
    sweep_alpha_beta(7, 2, 12, 5, 20_000);
}

#[test]
fn gemv_matches_reference_z49_rectangular_wide() {
    sweep_alpha_beta(7, 2, 5, 12, 30_000);
}

#[test]
fn gemv_matches_reference_z65537() {
    // u32-range modulus — u128 accumulator handles the products.
    sweep_alpha_beta(65537, 1, 6, 6, 40_000);
}

#[test]
fn gemv_matches_reference_z1_000_003_single_row() {
    sweep_alpha_beta(1_000_003, 1, 1, 10, 50_000);
}

#[test]
fn gemv_matches_reference_single_column() {
    sweep_alpha_beta(5, 2, 10, 1, 60_000);
}

// ─── Reduction to mat_vec + edge cases ─────────────────────────────

#[test]
fn gemv_alpha_1_beta_0_matches_mat_vec() {
    let a = rand_matzpn(5, 2, 9, 11, 70_000);
    let x = rand_vec(5, 2, 11, 70_001);
    let y_in = vec![0u64; 9];
    let got = a.gemv(1, TransA::NoTrans, &x, 0, &y_in);
    let expected = a.mat_vec(&x);
    assert_eq!(got, expected);
}

#[test]
fn gemv_alpha_0_collapses_to_beta_y() {
    let a = rand_matzpn(5, 2, 6, 8, 80_000);
    let x = rand_vec(5, 2, 8, 80_001);
    let y_in = rand_vec(5, 2, 6, 80_002);
    let beta = 3u64;
    let modulus = 25u64;
    let got = a.gemv(0, TransA::NoTrans, &x, beta, &y_in);
    let expected: Vec<u64> = y_in.iter().map(|&v| (beta * v) % modulus).collect();
    assert_eq!(got, expected);
}

#[test]
fn gemv_beta_0_drops_y_in() {
    let a = rand_matzpn(7, 2, 6, 6, 90_000);
    let x = rand_vec(7, 2, 6, 90_001);
    let y_garbage = vec![999u64; 6]; // not in [0, 49), but β=0 nukes it
    let y_zero = vec![0u64; 6];
    let g1 = a.gemv(1, TransA::NoTrans, &x, 0, &y_garbage);
    let g2 = a.gemv(1, TransA::NoTrans, &x, 0, &y_zero);
    assert_eq!(g1, g2);
}
