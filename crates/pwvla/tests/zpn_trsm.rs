// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::trsm` cross-validation. Two oracles:
//!
//! 1. **Round-trip via matmul**:
//!    `op(A) · X = alpha · B` (Left) or `X · op(A) = alpha · B`
//!    (Right), rebuilding the effective op(A) with the right
//!    diagonal handling and calling the scalar Barrett `matmul`.
//! 2. **Hand-computed 2×2** for a pinned sanity case.
//!
//! Covers all 16 (side × uplo × op × diag) combinations plus
//! pivot-singular rejection and alpha-scaling checks.

use pwvla::zpn::{Diag, MatZpn, Side, TransA, Uplo};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

/// Random triangular-compatible `A`: every diagonal entry is coprime
/// to `p` so trsm's pivots are all units in `Z/p^n`.
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

fn rand_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..m))
}

/// Build the effective dense op(A) after triangle masking, with unit
/// diagonal handling if requested. This is the reference op(A) that
/// trsm is implicitly solving against.
fn effective_a_trsm(a: &MatZpn, uplo: Uplo, op: TransA, diag: Diag) -> MatZpn {
    let dim = a.rows;
    MatZpn::from_fn(a.p, a.n, dim, dim, |i, j| {
        let (src_i, src_j) = match op {
            TransA::NoTrans => (i, j),
            TransA::Trans => (j, i),
        };
        let in_tri = match uplo {
            Uplo::Upper => src_i <= src_j,
            Uplo::Lower => src_i >= src_j,
        };
        if !in_tri {
            return 0;
        }
        if i == j && matches!(diag, Diag::Unit) {
            return 1;
        }
        a.get(src_i, src_j)
    })
}

fn scalar_scale(mat: &MatZpn, alpha: u64) -> MatZpn {
    let modulus = (mat.p as u64).pow(mat.n);
    MatZpn::from_fn(mat.p, mat.n, mat.rows, mat.cols, |i, j| {
        (mat.get(i, j) * alpha) % modulus
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

fn assert_trsm_round_trip(
    p: u32,
    n: u32,
    dim: usize,
    other: usize,
    alpha: u64,
    side: Side,
    uplo: Uplo,
    op: TransA,
    diag: Diag,
    seed: u64,
) {
    let a = rand_invertible_diag_a(p, n, dim, seed);
    let (b_rows, b_cols) = match side {
        Side::Left => (dim, other),
        Side::Right => (other, dim),
    };
    let b = rand_matzpn(p, n, b_rows, b_cols, seed.wrapping_add(1));

    let x = a
        .trsm(side, uplo, op, diag, alpha, &b)
        .expect("diagonal is coprime to p");

    let a_eff = effective_a_trsm(&a, uplo, op, diag);
    let check = match side {
        Side::Left => a_eff.matmul(&x),
        Side::Right => x.matmul(&a_eff),
    };
    let target = scalar_scale(&b, alpha);

    assert!(
        mats_equal(&check, &target),
        "trsm round-trip failed: p={p} n={n} dim={dim} other={other} alpha={alpha} \
         side={side:?} uplo={uplo:?} op={op:?} diag={diag:?}"
    );
}

// ─── Hand-computed 2×2 sanity ──────────────────────────────────────

#[test]
fn trsm_hand_computed_2x2_left_lower_unit_forward_sub() {
    // A (lower, unit diag) over Z/7:
    //   [ 1 * ]
    //   [ 3 1 ]   (* = ignored since diag=Unit)
    // B = [ 5 ; 4 ]^T (one column, two entries).
    // Solve A · X = B via forward substitution:
    //   X[0] = B[0] = 5
    //   X[1] = B[1] - 3·X[0] = 4 - 15 = -11 ≡ 3 (mod 7)
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 99,
        (0, 1) => 99, // garbage — ignored (Unit diag + Lower)
        (1, 0) => 3,
        (1, 1) => 99, // diag ignored; off-diag stored
        _ => unreachable!(),
    });
    let b = MatZpn::from_fn(7, 1, 2, 1, |i, _| match i {
        0 => 5,
        1 => 4,
        _ => unreachable!(),
    });
    let x = a
        .trsm(Side::Left, Uplo::Lower, TransA::NoTrans, Diag::Unit, 1, &b)
        .unwrap();
    assert_eq!(x.get(0, 0), 5);
    assert_eq!(x.get(1, 0), 3);
}

#[test]
fn trsm_hand_computed_2x2_left_upper_nonunit_back_sub() {
    // A (upper, non-unit) over Z/7:
    //   [ 2 3 ]
    //   [ * 5 ]   (* ignored since uplo=Upper)
    // B = [ 1 ; 2 ]^T
    // Back substitution:
    //   X[1] = B[1] / 5 = 2 / 5 ≡ 2 · 3 = 6 (mod 7; 5⁻¹=3 since 5·3=15≡1)
    //   X[0] = (B[0] - 3·X[1]) / 2 = (1 - 18) / 2 = -17 / 2 ≡ -17·4 = -68 ≡ -68+70 = 2 (mod 7)
    //         (2⁻¹ mod 7 = 4 since 2·4=8≡1)
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 2,
        (0, 1) => 3,
        (1, 0) => 99,
        (1, 1) => 5,
        _ => unreachable!(),
    });
    let b = MatZpn::from_fn(7, 1, 2, 1, |i, _| match i {
        0 => 1,
        1 => 2,
        _ => unreachable!(),
    });
    let x = a
        .trsm(
            Side::Left,
            Uplo::Upper,
            TransA::NoTrans,
            Diag::NonUnit,
            1,
            &b,
        )
        .unwrap();
    assert_eq!(x.get(1, 0), 6);
    assert_eq!(x.get(0, 0), 2);
}

// ─── Full combo cross-validation via matmul round-trip ─────────────

#[test]
fn trsm_all_16_combos_z25_square() {
    let p = 5u32;
    let n = 2u32;
    let dim = 6;
    let other = 6;
    let alpha = 3u64;
    let mut seed = 10_000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                for &diag in &[Diag::Unit, Diag::NonUnit] {
                    assert_trsm_round_trip(p, n, dim, other, alpha, side, uplo, op, diag, seed);
                    seed = seed.wrapping_add(1);
                }
            }
        }
    }
}

#[test]
fn trsm_all_16_combos_rectangular_b() {
    let p = 7u32;
    let n = 2u32;
    let dim = 8;
    let other = 5;
    let alpha = 11u64;
    let mut seed = 20_000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                for &diag in &[Diag::Unit, Diag::NonUnit] {
                    assert_trsm_round_trip(p, n, dim, other, alpha, side, uplo, op, diag, seed);
                    seed = seed.wrapping_add(1);
                }
            }
        }
    }
}

#[test]
fn trsm_all_16_combos_z65537_square() {
    // u32-range modulus, still well inside the u128 accumulator.
    let p = 65537u32;
    let n = 1u32;
    let dim = 5;
    let other = 5;
    let alpha = 42u64;
    let mut seed = 30_000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                for &diag in &[Diag::Unit, Diag::NonUnit] {
                    assert_trsm_round_trip(p, n, dim, other, alpha, side, uplo, op, diag, seed);
                    seed = seed.wrapping_add(1);
                }
            }
        }
    }
}

// ─── Pivot singularity + edge cases ────────────────────────────────

#[test]
fn trsm_rejects_non_unit_pivot() {
    // A in Z/25 with a diagonal entry in (5) — the maximal ideal —
    // is not a unit, so trsm with NonUnit diag must return None.
    let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| match (i, j) {
        (0, 0) => 7,          // unit
        (1, 1) => 10,         // 10 ≡ 0 mod 5 → in maximal ideal
        (2, 2) => 3,          // unit
        (i, j) if i < j => 1, // upper off-diag
        _ => 0,
    });
    let b = rand_matzpn(5, 2, 3, 4, 40_000);
    assert!(a
        .trsm(
            Side::Left,
            Uplo::Upper,
            TransA::NoTrans,
            Diag::NonUnit,
            1,
            &b
        )
        .is_none());
}

#[test]
fn trsm_unit_diag_ignores_singular_stored_pivot() {
    // Same matrix but with Diag::Unit — the stored zero on the
    // diagonal is masked to 1, so the solve succeeds even though
    // A's literal diagonal has a non-unit.
    let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| match (i, j) {
        (0, 0) => 7,
        (1, 1) => 10,
        (2, 2) => 3,
        (i, j) if i < j => 1,
        _ => 0,
    });
    let b = rand_matzpn(5, 2, 3, 4, 40_001);
    let x = a
        .trsm(Side::Left, Uplo::Upper, TransA::NoTrans, Diag::Unit, 1, &b)
        .unwrap();
    let a_eff = effective_a_trsm(&a, Uplo::Upper, TransA::NoTrans, Diag::Unit);
    let check = a_eff.matmul(&x);
    assert!(mats_equal(&check, &b));
}

#[test]
fn trsm_alpha_zero_yields_zero() {
    let a = rand_invertible_diag_a(5, 2, 6, 50_000);
    let b = rand_matzpn(5, 2, 6, 4, 50_001);
    let x = a
        .trsm(
            Side::Left,
            Uplo::Lower,
            TransA::NoTrans,
            Diag::NonUnit,
            0,
            &b,
        )
        .unwrap();
    for i in 0..x.rows {
        for j in 0..x.cols {
            assert_eq!(x.get(i, j), 0);
        }
    }
}

#[test]
fn trsm_identity_triangular_returns_alpha_b() {
    // Solving I · X = α·B gives X = α·B.
    let p = 7u32;
    let n = 2u32;
    let dim = 5;
    let alpha = 3u64;
    let i_mat = MatZpn::from_fn(p, n, dim, dim, |i, j| if i == j { 1 } else { 0 });
    let b = rand_matzpn(p, n, dim, 6, 60_000);
    let x = i_mat
        .trsm(
            Side::Left,
            Uplo::Upper,
            TransA::NoTrans,
            Diag::NonUnit,
            alpha,
            &b,
        )
        .unwrap();
    let expected = scalar_scale(&b, alpha);
    assert!(mats_equal(&x, &expected));
}
