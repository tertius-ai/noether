// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::ger` cross-validation. Oracle: compute the outer product
//! `x · yᵀ` cell-by-cell, scale by α, add to `A`, reduce mod `p^n`.

use pwvla::zpn::MatZpn;
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

/// Oracle: build A + α·x·yᵀ cell by cell via u128 arithmetic, no
/// MatZpn APIs beyond `get`/`from_fn`.
fn ger_reference(a: &MatZpn, alpha: u64, x: &[u64], y: &[u64]) -> MatZpn {
    let modulus = (a.p as u64).pow(a.n);
    MatZpn::from_fn(a.p, a.n, a.rows, a.cols, |i, j| {
        let outer = (x[i] as u128 * y[j] as u128) % modulus as u128;
        let scaled = (alpha as u128 * outer) % modulus as u128;
        let sum = a.get(i, j) as u128 + scaled;
        (sum % modulus as u128) as u64
    })
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

// ─── Hand-computed 2×3 sanity ─────────────────────────────────────

#[test]
fn ger_hand_computed_2x3_z11() {
    // A = [[1, 2, 3], [4, 5, 6]], x = [7, 8], y = [9, 10, 11], α = 2.
    // x·yᵀ = [[7·9, 7·10, 7·11], [8·9, 8·10, 8·11]]
    //      = [[63, 70, 77], [72, 80, 88]]
    // 2·(x·yᵀ) = [[126, 140, 154], [144, 160, 176]]
    //          ≡ [[5, 8, 0], [1, 6, 0]] (mod 11)
    //            (126 = 11·11 + 5; 140 = 11·12 + 8; 154 = 11·14 = 154 → 0;
    //             144 = 11·13 + 1; 160 = 11·14 + 6; 176 = 11·16 = 0)
    // A + 2·(x·yᵀ) = [[1+5, 2+8, 3+0], [4+1, 5+6, 6+0]]
    //              = [[6, 10, 3], [5, 0, 6]] (mod 11; 11 → 0)
    let a = MatZpn::from_fn(11, 1, 2, 3, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (0, 2) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (1, 2) => 6,
        _ => unreachable!(),
    });
    let x = vec![7u64, 8];
    let y = vec![9u64, 10, 11];
    let got = a.ger(2, &x, &y);
    let expected = MatZpn::from_fn(11, 1, 2, 3, |i, j| match (i, j) {
        (0, 0) => 6,
        (0, 1) => 10,
        (0, 2) => 3,
        (1, 0) => 5,
        (1, 1) => 0,
        (1, 2) => 6,
        _ => unreachable!(),
    });
    assert!(mats_equal(&got, &expected));
}

// ─── Cross-validation vs manual outer product + add ────────────────

fn sweep(p: u32, n: u32, rows: usize, cols: usize, seed_base: u64) {
    let modulus = (p as u64).pow(n);
    let alphas = [0u64, 1, 2, modulus - 1];
    for (ai, &alpha) in alphas.iter().enumerate() {
        let a = rand_matzpn(p, n, rows, cols, seed_base + ai as u64);
        let x = rand_vec(p, n, rows, seed_base + 100 + ai as u64);
        let y = rand_vec(p, n, cols, seed_base + 200 + ai as u64);
        let got = a.ger(alpha, &x, &y);
        let expected = ger_reference(&a, alpha, &x, &y);
        assert!(
            mats_equal(&got, &expected),
            "ger mismatch: p={p} n={n} rows={rows} cols={cols} alpha={alpha}"
        );
    }
}

#[test]
fn ger_matches_reference_z25_square() {
    sweep(5, 2, 8, 8, 10_000);
}

#[test]
fn ger_matches_reference_z49_tall() {
    sweep(7, 2, 12, 5, 20_000);
}

#[test]
fn ger_matches_reference_z49_wide() {
    sweep(7, 2, 5, 12, 30_000);
}

#[test]
fn ger_matches_reference_z65537() {
    sweep(65537, 1, 6, 6, 40_000);
}

#[test]
fn ger_matches_reference_z1_000_003() {
    sweep(1_000_003, 1, 4, 7, 50_000);
}

// ─── Edge cases ───────────────────────────────────────────────────

#[test]
fn ger_alpha_zero_returns_a_unchanged() {
    let a = rand_matzpn(5, 2, 6, 8, 60_000);
    let x = rand_vec(5, 2, 6, 60_001);
    let y = rand_vec(5, 2, 8, 60_002);
    let got = a.ger(0, &x, &y);
    assert!(mats_equal(&got, &a));
}

#[test]
fn ger_zero_x_returns_a_unchanged() {
    let a = rand_matzpn(5, 2, 6, 8, 70_000);
    let x = vec![0u64; 6];
    let y = rand_vec(5, 2, 8, 70_001);
    let got = a.ger(1, &x, &y);
    assert!(mats_equal(&got, &a));
}

#[test]
fn ger_zero_a_yields_alpha_outer_product() {
    let p = 7u32;
    let n = 2u32;
    let rows = 5;
    let cols = 4;
    let alpha = 3u64;
    let zero_a = MatZpn::zeros(p, n, rows, cols);
    let x = rand_vec(p, n, rows, 80_000);
    let y = rand_vec(p, n, cols, 80_001);
    let got = zero_a.ger(alpha, &x, &y);
    let expected = ger_reference(&zero_a, alpha, &x, &y);
    assert!(mats_equal(&got, &expected));
}

#[test]
fn ger_rank_one_shape() {
    // For zero A, ger should produce exactly the outer product, which
    // has rank ≤ 1 — verifiable by checking any 2×2 minor has zero det.
    let p = 11u32;
    let n = 1u32;
    let rows = 4;
    let cols = 4;
    let zero_a = MatZpn::zeros(p, n, rows, cols);
    let x = rand_vec(p, n, rows, 90_000);
    let y = rand_vec(p, n, cols, 90_001);
    let out = zero_a.ger(1, &x, &y);
    let modulus = p as u64;
    for i0 in 0..rows {
        for i1 in (i0 + 1)..rows {
            for j0 in 0..cols {
                for j1 in (j0 + 1)..cols {
                    let a00 = out.get(i0, j0);
                    let a01 = out.get(i0, j1);
                    let a10 = out.get(i1, j0);
                    let a11 = out.get(i1, j1);
                    let det = ((a00 as u128 * a11 as u128) % modulus as u128 + modulus as u128
                        - (a01 as u128 * a10 as u128) % modulus as u128)
                        % modulus as u128;
                    assert_eq!(det, 0, "2×2 minor ({i0},{j0})-({i1},{j1}) has nonzero det");
                }
            }
        }
    }
}
