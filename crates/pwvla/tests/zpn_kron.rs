// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `MatZpn::kron` cross-validation: definitional check via indexed
//! `get(i, j)` comparison, plus algebraic identities (linearity,
//! mixed-product rule, identity preservation).

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

// ─── Hand-computed 2×2 ⊗ 2×2 sanity ───────────────────────────────

#[test]
fn kron_hand_computed_2x2_2x2_z7() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] over Z/7.
    // A ⊗ B is 4×4:
    //   [ 1·B | 2·B ]   [  5  6  10 12 ]   [ 5 6 3 5 ]
    //   [ 3·B | 4·B ] = [  7  8  14 16 ] ≡ [ 0 1 0 2 ]  (mod 7)
    //                   [ 15 18  20 24 ]   [ 1 4 6 3 ]
    //                   [ 21 24  28 32 ]   [ 0 3 0 4 ]
    let a = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 1,
        (0, 1) => 2,
        (1, 0) => 3,
        (1, 1) => 4,
        _ => unreachable!(),
    });
    let b = MatZpn::from_fn(7, 1, 2, 2, |i, j| match (i, j) {
        (0, 0) => 5,
        (0, 1) => 6,
        (1, 0) => 7,
        (1, 1) => 8,
        _ => unreachable!(),
    });
    let c = a.kron(&b);
    assert_eq!(c.rows, 4);
    assert_eq!(c.cols, 4);
    let expected = [[5u64, 6, 3, 5], [0, 1, 0, 2], [1, 4, 6, 3], [0, 3, 0, 4]];
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(c.get(i, j), expected[i][j], "cell ({i},{j})");
        }
    }
}

// ─── Definitional cross-validation ────────────────────────────────

/// Walk every cell of the output and compare against the definition
/// `C[i·p + k, j·q + l] = A[i, j] · B[k, l]`.
fn assert_kron_definition(a: &MatZpn, b: &MatZpn) {
    let c = a.kron(b);
    assert_eq!(c.rows, a.rows * b.rows);
    assert_eq!(c.cols, a.cols * b.cols);
    let modulus = (a.p as u64).pow(a.n) as u128;
    for i in 0..a.rows {
        for j in 0..a.cols {
            for k in 0..b.rows {
                for l in 0..b.cols {
                    let expected = ((a.get(i, j) as u128) * (b.get(k, l) as u128)) % modulus;
                    let got = c.get(i * b.rows + k, j * b.cols + l) as u128;
                    assert_eq!(got, expected, "({i},{j},{k},{l})");
                }
            }
        }
    }
}

#[test]
fn kron_matches_definition_z25() {
    assert_kron_definition(
        &rand_matzpn(5, 2, 3, 4, 10_000),
        &rand_matzpn(5, 2, 5, 2, 10_001),
    );
}

#[test]
fn kron_matches_definition_z49_with_rectangulars() {
    for &(ra, ca, rb, cb) in &[(1usize, 1, 4, 5), (4, 5, 1, 1), (2, 3, 3, 2), (5, 2, 2, 5)] {
        assert_kron_definition(
            &rand_matzpn(7, 2, ra, ca, 20_000 + ra as u64),
            &rand_matzpn(7, 2, rb, cb, 21_000 + cb as u64),
        );
    }
}

#[test]
fn kron_matches_definition_z65537() {
    assert_kron_definition(
        &rand_matzpn(65537, 1, 2, 3, 30_000),
        &rand_matzpn(65537, 1, 3, 2, 30_001),
    );
}

// ─── Algebraic identities ─────────────────────────────────────────

#[test]
fn kron_identity_preserves_shape_and_values() {
    // I_m ⊗ I_n = I_{mn}: m·n × m·n identity.
    let m = 3;
    let n = 4;
    let p = 7u32;
    let np = 1u32;
    let i_m = MatZpn::from_fn(p, np, m, m, |i, j| if i == j { 1 } else { 0 });
    let i_n = MatZpn::from_fn(p, np, n, n, |i, j| if i == j { 1 } else { 0 });
    let got = i_m.kron(&i_n);
    let expected = MatZpn::from_fn(p, np, m * n, m * n, |i, j| if i == j { 1 } else { 0 });
    assert!(mats_equal(&got, &expected));
}

#[test]
fn kron_mixed_product_rule() {
    // (A ⊗ B) · (C ⊗ D) = (A·C) ⊗ (B·D) — the fundamental identity
    // behind tensor-structured linear solves. Dimensions: A is (m,k),
    // C is (k,n); B is (p,r), D is (r,s).
    let p = 11u32;
    let np = 1u32;
    let a = rand_matzpn(p, np, 2, 3, 40_000);
    let c = rand_matzpn(p, np, 3, 2, 40_001);
    let b = rand_matzpn(p, np, 2, 4, 40_002);
    let d = rand_matzpn(p, np, 4, 3, 40_003);
    let lhs = a.kron(&b).matmul(&c.kron(&d));
    let rhs = a.matmul(&c).kron(&b.matmul(&d));
    assert!(mats_equal(&lhs, &rhs));
}

#[test]
fn kron_distributes_over_add() {
    // (A + A') ⊗ B = A ⊗ B + A' ⊗ B.
    let p = 11u32;
    let np = 1u32;
    let a1 = rand_matzpn(p, np, 3, 2, 50_000);
    let a2 = rand_matzpn(p, np, 3, 2, 50_001);
    let b = rand_matzpn(p, np, 4, 3, 50_002);
    let lhs = a1.add(&a2).kron(&b);
    let rhs = a1.kron(&b).add(&a2.kron(&b));
    assert!(mats_equal(&lhs, &rhs));
}

#[test]
fn kron_with_scalar_on_left() {
    // 1x1 matrix [α] kron B = α · B (up to reshape — 1x1⊗(p,q) is p×q).
    let p = 7u32;
    let np = 2u32;
    let alpha = 5u64;
    let one = MatZpn::from_fn(p, np, 1, 1, |_, _| alpha);
    let b = rand_matzpn(p, np, 4, 5, 60_000);
    let got = one.kron(&b);
    let modulus = 49u64;
    let expected = MatZpn::from_fn(p, np, 4, 5, |i, j| (alpha * b.get(i, j)) % modulus);
    assert!(mats_equal(&got, &expected));
}

#[test]
fn kron_zero_eats_everything() {
    let p = 5u32;
    let np = 2u32;
    let zero_a = MatZpn::zeros(p, np, 3, 4);
    let b = rand_matzpn(p, np, 2, 5, 70_000);
    let got = zero_a.kron(&b);
    assert_eq!(got.rows, 6);
    assert_eq!(got.cols, 20);
    for i in 0..got.rows {
        for j in 0..got.cols {
            assert_eq!(got.get(i, j), 0);
        }
    }
}
