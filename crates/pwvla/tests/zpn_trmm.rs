// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::trmm` cross-validation against the "materialize the
//! effective triangular op(A) + call scalar `matmul` + scalar-mul
//! alpha" oracle. Every combination of (side, uplo, op) is tested
//! on random matrices so that bit-for-bit equivalence with the
//! dense-gemm reference is pinned down end-to-end.

use pwvla::zpn::{MatZpn, Side, TransA, Uplo};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let m = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..m))
}

/// Reference: materialize the effective op(A) as a dense matrix
/// with out-of-triangle entries zeroed, then caller can gemm / mul
/// it against B.
fn effective_a(a: &MatZpn, uplo: Uplo, op: TransA) -> MatZpn {
    let dim = a.rows;
    MatZpn::from_fn(a.p, a.n, dim, dim, |i, j| {
        let (src_i, src_j) = match op {
            TransA::NoTrans => (i, j),
            TransA::Trans => (j, i),
        };
        let stored = match uplo {
            Uplo::Upper => src_i <= src_j,
            Uplo::Lower => src_i >= src_j,
        };
        if stored {
            a.get(src_i, src_j)
        } else {
            0
        }
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

fn assert_trmm_matches_reference(
    p: u32,
    n: u32,
    dim: usize,
    other: usize,
    alpha: u64,
    side: Side,
    uplo: Uplo,
    op: TransA,
    seed: u64,
) {
    let a = rand_matzpn(p, n, dim, dim, seed);
    let (b_rows, b_cols) = match side {
        Side::Left => (dim, other),
        Side::Right => (other, dim),
    };
    let b = rand_matzpn(p, n, b_rows, b_cols, seed.wrapping_add(1));

    let got = a.trmm(side, uplo, op, alpha, &b);

    let a_eff = effective_a(&a, uplo, op);
    let product = match side {
        Side::Left => a_eff.matmul(&b),
        Side::Right => b.matmul(&a_eff),
    };
    let expected = scalar_scale(&product, alpha);

    assert!(
        mats_equal(&got, &expected),
        "trmm mismatch: p={p} n={n} dim={dim} other={other} alpha={alpha} \
         side={side:?} uplo={uplo:?} op={op:?}"
    );
}

// ─── Hand-computed 2×2 sanity check ─────────────────────────────────

#[test]
fn trmm_hand_computed_2x2_left_upper_notrans() {
    // A upper 2×2 over Z/7:
    //   [ 2 3 ]          B = [ 1 4 ]
    //   [ * 5 ]              [ 2 6 ]
    // Only stored-triangle of A matters; unspecified entry ('*')
    // must not affect the output.
    //
    // C = A·B  with A's lower entry ignored (upper-triangular):
    //   C[0,0] = 2·1 + 3·2 = 8 ≡ 1  (mod 7)
    //   C[0,1] = 2·4 + 3·6 = 26 ≡ 5
    //   C[1,0] = 5·2 = 10 ≡ 3
    //   C[1,1] = 5·6 = 30 ≡ 2
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 2,
        (0, 1) => 3,
        (1, 0) => 99, // garbage — must be ignored since uplo=Upper
        (1, 1) => 5,
        _ => unreachable!(),
    });
    let b = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 4,
        (1, 0) => 2,
        (1, 1) => 6,
        _ => unreachable!(),
    });
    // "Garbage" is 99 mod 7 = 1, but get() returns the reduced form,
    // so A[1,0] reads as 1 — still, trmm must ignore it for Upper.
    let c = a.trmm(Side::Left, Uplo::Upper, TransA::NoTrans, 1, &b);
    assert_eq!(c.get(0, 0), 1);
    assert_eq!(c.get(0, 1), 5);
    assert_eq!(c.get(1, 0), 3);
    assert_eq!(c.get(1, 1), 2);
}

#[test]
fn trmm_hand_computed_2x2_alpha_scaling() {
    // Same A/B as above but with alpha = 2 over Z/7:
    //   C = 2 · A · B  where the unscaled C is [[1, 5], [3, 2]].
    // 2·[[1, 5], [3, 2]] = [[2, 10], [6, 4]] ≡ [[2, 3], [6, 4]] (mod 7).
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 2,
        (0, 1) => 3,
        (1, 0) => 0,
        (1, 1) => 5,
        _ => unreachable!(),
    });
    let b = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 4,
        (1, 0) => 2,
        (1, 1) => 6,
        _ => unreachable!(),
    });
    let c = a.trmm(Side::Left, Uplo::Upper, TransA::NoTrans, 2, &b);
    assert_eq!(c.get(0, 0), 2);
    assert_eq!(c.get(0, 1), 3);
    assert_eq!(c.get(1, 0), 6);
    assert_eq!(c.get(1, 1), 4);
}

// ─── Cross-validation against gemm on all 8 (side, uplo, op) combos ─

#[test]
fn trmm_all_combos_z25_square() {
    let p = 5u32;
    let n = 2u32;
    let dim = 7;
    let other = 7;
    let alpha = 3u64;
    let mut seed = 1000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                assert_trmm_matches_reference(p, n, dim, other, alpha, side, uplo, op, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trmm_all_combos_rectangular_b() {
    // Non-square B (Left: dim × other with dim ≠ other; Right: other × dim).
    let p = 7u32;
    let n = 2u32;
    let dim = 9;
    let other = 5;
    let alpha = 11u64;
    let mut seed = 2000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                assert_trmm_matches_reference(p, n, dim, other, alpha, side, uplo, op, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trmm_larger_prime_z65537_square() {
    // u32-range modulus; trmm's u128 accumulator handles this just fine.
    let p = 65537u32;
    let n = 1u32;
    let dim = 6;
    let other = 6;
    let alpha = 42u64;
    let mut seed = 3000u64;
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Upper, Uplo::Lower] {
            for &op in &[TransA::NoTrans, TransA::Trans] {
                assert_trmm_matches_reference(p, n, dim, other, alpha, side, uplo, op, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

#[test]
fn trmm_alpha_zero_yields_zero() {
    let a = rand_matzpn(5, 2, 8, 8, 4000);
    let b = rand_matzpn(5, 2, 8, 12, 4001);
    let c = a.trmm(Side::Left, Uplo::Upper, TransA::NoTrans, 0, &b);
    for i in 0..c.rows {
        for j in 0..c.cols {
            assert_eq!(c.get(i, j), 0);
        }
    }
}

#[test]
fn trmm_alpha_one_matches_unscaled_reference() {
    // Spot-check that alpha=1 gives exactly the materialize-then-gemm
    // product with no scaling step in the way.
    let p = 5u32;
    let n = 2u32;
    let dim = 10;
    let a = rand_matzpn(p, n, dim, dim, 5000);
    let b = rand_matzpn(p, n, dim, dim, 5001);
    let got = a.trmm(Side::Left, Uplo::Lower, TransA::Trans, 1, &b);
    let a_eff = effective_a(&a, Uplo::Lower, TransA::Trans);
    let expected = a_eff.matmul(&b);
    assert!(mats_equal(&got, &expected));
}

#[test]
fn trmm_identity_triangular_returns_b_scaled() {
    // I (upper triangular) times B equals B; trmm(L, U, N, alpha, B) = alpha·B.
    let p = 7u32;
    let n = 2u32;
    let dim = 8;
    let alpha = 3u64;
    let i_mat = MatZpn::from_fn(p, n, dim, dim, |i, j| if i == j { 1 } else { 0 });
    let b = rand_matzpn(p, n, dim, dim, 6000);
    let got = i_mat.trmm(Side::Left, Uplo::Upper, TransA::NoTrans, alpha, &b);
    let expected = scalar_scale(&b, alpha);
    assert!(mats_equal(&got, &expected));
}
