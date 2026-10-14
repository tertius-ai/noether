// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `trmv` / `trsv` cross-validated against the Level-3 oracles
//! (`trmm` / `trsm`) at nrhs = 1. Both Level-2 routines
//! are direct O(N²) specializations — these tests pin that they
//! produce the same outputs bit-for-bit.

use pwvla::zpn::{Diag, MatZpn, Side, TransA, Uplo};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..m))
}

fn rand_invertible_diag_a(p: u32, n: u32, dim: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let modulus = (p as u64).pow(n);
    MatZpn::from_fn(p, n, dim, dim, |i, j| {
        if i == j {
            loop {
                let v = rng.random_range(1..modulus);
                if v % p as u64 != 0 {
                    return v;
                }
            }
        } else {
            rng.random_range(0..modulus)
        }
    })
}

fn rand_vec(p: u32, n: u32, len: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    (0..len).map(|_| rng.random_range(0..m)).collect()
}

fn mat_col_to_vec(m: &MatZpn) -> Vec<u64> {
    (0..m.rows).map(|i| m.get(i, 0)).collect()
}

fn vec_to_col(v: &[u64], p: u32, n: u32) -> MatZpn {
    MatZpn::from_fn(p, n, v.len(), 1, |i, _| v[i])
}

// ─── Hand-computed 2×2 sanity ─────────────────────────────────────

#[test]
fn trmv_hand_computed_2x2_upper_notrans() {
    // A upper over Z/7:  [[2, 3], [*, 5]]   x = [1, 2]
    // y = A·x with A's lower entry ignored:
    //   y[0] = 2·1 + 3·2 = 8 ≡ 1
    //   y[1] = 5·2 = 10 ≡ 3
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 2,
        (0, 1) => 3,
        (1, 0) => 99, // garbage — ignored for Upper
        (1, 1) => 5,
        _ => unreachable!(),
    });
    let x = vec![1u64, 2];
    let y = a.trmv(Uplo::Upper, TransA::NoTrans, Diag::NonUnit, &x);
    assert_eq!(y, vec![1u64, 3]);
}

#[test]
fn trsv_hand_computed_2x2_upper_nonunit() {
    // Solve A·y = x with A upper non-unit: same A and x as trsm's
    // hand-computed test.
    //   y[1] = 2 / 5 ≡ 2·3 = 6 (mod 7; 5⁻¹=3)
    //   y[0] = (1 - 3·6) / 2 = -17/2 ≡ -17·4 = 2 (mod 7; 2⁻¹=4)
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 2,
        (0, 1) => 3,
        (1, 0) => 99,
        (1, 1) => 5,
        _ => unreachable!(),
    });
    let x = vec![1u64, 2];
    let y = a
        .trsv(Uplo::Upper, TransA::NoTrans, Diag::NonUnit, &x)
        .unwrap();
    assert_eq!(y, vec![2u64, 6]);
}

// ─── Cross-validation of trmv against trmm at nrhs=1 ──────────────

fn assert_trmv_matches_trmm(
    p: u32,
    n: u32,
    dim: usize,
    uplo: Uplo,
    op: TransA,
    diag: Diag,
    seed: u64,
) {
    let a = rand_matzpn(p, n, dim, dim, seed);
    let x = rand_vec(p, n, dim, seed.wrapping_add(1));

    let trmv_y = a.trmv(uplo, op, diag, &x);

    // For trmm, diag != NonUnit is supported via effective materialize;
    // we want the same output — but trmm doesn't take a Diag param. Use
    // effective-A from a hand-rolled helper so Diag::Unit is handled.
    // For Diag::NonUnit: trmm(alpha=1, side=L, uplo, op, A, x_col).
    // For Diag::Unit: temporarily overwrite A's diagonal to 1 before
    // calling trmm.
    let trmm_a = if matches!(diag, Diag::Unit) {
        MatZpn::from_fn(p, n, dim, dim, |i, j| if i == j { 1 } else { a.get(i, j) })
    } else {
        a.clone()
    };
    let x_col = vec_to_col(&x, p, n);
    let trmm_c = trmm_a.trmm(Side::Left, uplo, op, 1, &x_col);
    let trmm_y = mat_col_to_vec(&trmm_c);

    assert_eq!(
        trmv_y, trmm_y,
        "trmv != trmm[nrhs=1]: p={p} n={n} dim={dim} uplo={uplo:?} op={op:?} diag={diag:?}"
    );
}

#[test]
fn trmv_matches_trmm_all_8_combos_z25() {
    let mut seed = 10_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trmv_matches_trmm(5, 2, 9, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trmv_matches_trmm_z49_larger() {
    let mut seed = 20_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trmv_matches_trmm(7, 2, 12, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trmv_matches_trmm_z65537() {
    let mut seed = 30_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trmv_matches_trmm(65537, 1, 7, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

// ─── Cross-validation of trsv against trsm at nrhs=1 ──────────────

fn assert_trsv_matches_trsm(
    p: u32,
    n: u32,
    dim: usize,
    uplo: Uplo,
    op: TransA,
    diag: Diag,
    seed: u64,
) {
    let a = rand_invertible_diag_a(p, n, dim, seed);
    let x = rand_vec(p, n, dim, seed.wrapping_add(1));

    let trsv_y = a.trsv(uplo, op, diag, &x).expect("diag is coprime to p");

    let x_col = vec_to_col(&x, p, n);
    let trsm_x = a
        .trsm(Side::Left, uplo, op, diag, 1, &x_col)
        .expect("diag is coprime to p");
    let trsm_y = mat_col_to_vec(&trsm_x);

    assert_eq!(
        trsv_y, trsm_y,
        "trsv != trsm[nrhs=1]: p={p} n={n} dim={dim} uplo={uplo:?} op={op:?} diag={diag:?}"
    );
}

#[test]
fn trsv_matches_trsm_all_8_combos_z25() {
    let mut seed = 40_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trsv_matches_trsm(5, 2, 9, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trsv_matches_trsm_z49_larger() {
    let mut seed = 50_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trsv_matches_trsm(7, 2, 12, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trsv_matches_trsm_z65537() {
    let mut seed = 60_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                assert_trsv_matches_trsm(65537, 1, 7, uplo, op, diag, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

// ─── Round-trip: trmv(A, trsv(A, x)) == x ─────────────────────────

#[test]
fn trmv_of_trsv_is_identity() {
    // If we solve op(A)·y = x via trsv, then compute op(A)·y via
    // trmv, we should get x back — even for Unit diag.
    let p = 5u32;
    let n = 2u32;
    let dim = 8;
    let mut seed = 70_000u64;
    for &uplo in &[Uplo::Upper, Uplo::Lower] {
        for &op in &[TransA::NoTrans, TransA::Trans] {
            for &diag in &[Diag::Unit, Diag::NonUnit] {
                let a = rand_invertible_diag_a(p, n, dim, seed);
                let x = rand_vec(p, n, dim, seed.wrapping_add(1));
                let y = a.trsv(uplo, op, diag, &x).unwrap();
                let x_again = a.trmv(uplo, op, diag, &y);
                assert_eq!(x, x_again, "uplo={uplo:?} op={op:?} diag={diag:?}");
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trsv_rejects_non_unit_pivot() {
    // Diagonal entry in maximal ideal of Z/25.
    let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| match (i, j) {
        (0, 0) => 7,
        (1, 1) => 10, // 10 ≡ 0 mod 5 → non-unit
        (2, 2) => 3,
        (i, j) if i < j => 1,
        _ => 0,
    });
    let x = vec![1u64, 2, 3];
    assert!(a
        .trsv(Uplo::Upper, TransA::NoTrans, Diag::NonUnit, &x)
        .is_none());
}

#[test]
fn trsv_unit_diag_ignores_singular_stored_pivot() {
    let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| match (i, j) {
        (0, 0) => 7,
        (1, 1) => 10,
        (2, 2) => 3,
        (i, j) if i < j => 1,
        _ => 0,
    });
    let x = vec![1u64, 2, 3];
    let y = a
        .trsv(Uplo::Upper, TransA::NoTrans, Diag::Unit, &x)
        .unwrap();
    let x_round = a.trmv(Uplo::Upper, TransA::NoTrans, Diag::Unit, &y);
    assert_eq!(x_round, x);
}
