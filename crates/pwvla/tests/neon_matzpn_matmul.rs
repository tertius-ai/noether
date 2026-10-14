// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! MatZpn NEON u16 microkernel — cross-validation against the scalar
//! Barrett `MatZpn::matmul` path.
//!
//! Cross-validation: the NEON kernel (`MatZpn::matmul_neon`) and the
//! scalar Barrett kernel (`MatZpn::matmul`) must produce identical
//! MatZpn outputs on every shape and modulus we exercise. The scalar
//! path is the canonical correctness oracle.
//!
//! Gated on `target_arch = "aarch64"` (the NEON path is only available
//! there). On x86_64 this file compiles to zero tests.

#![cfg(target_arch = "aarch64")]

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

// ─── identity / zero ─────────────────────────────────────────────

#[test]
fn matmul_neon_identity_z5() {
    let i = MatZpn::identity(5, 1, 16);
    let a = rand_matzpn(5, 1, 16, 16, 1);
    let neon = a.matmul_neon(&i).expect("modulus in range");
    let scalar = a.matmul_scalar(&i);
    assert!(mats_equal(&neon, &scalar));
}

#[test]
fn matmul_neon_zero_z7() {
    let a = MatZpn::zeros(7, 2, 16, 16);
    let b = rand_matzpn(7, 2, 16, 16, 2);
    let neon = a.matmul_neon(&b).expect("modulus in range");
    assert!(neon.data.to_vec().iter().all(|&v| v == 0));
}

// ─── cross-validate against scalar Barrett ──────────────────────

#[test]
fn matmul_neon_matches_scalar_square_z25() {
    // p=5, n=2 → modulus 25 ≤ 2^15.
    for &sz in &[8usize, 12, 16, 20, 32, 64] {
        let a = rand_matzpn(5, 2, sz, sz, 100 + sz as u64);
        let b = rand_matzpn(5, 2, sz, sz, 200 + sz as u64);
        let neon = a.matmul_neon(&b).expect("Z/25 ≤ 2^15");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&neon, &scalar), "mismatch at square sz={sz}");
    }
}

#[test]
fn matmul_neon_matches_scalar_rectangular_z49() {
    // p=7, n=2 → modulus 49. Shapes where none of m/k/n divide the tile.
    for &(m, k, n) in &[
        (4usize, 10, 8),
        (8, 15, 16),
        (12, 20, 24),
        (5, 13, 11),
        (17, 8, 9),
        (4, 4, 4),
    ] {
        let a = rand_matzpn(7, 2, m, k, 300 + m as u64 + k as u64 + n as u64);
        let b = rand_matzpn(7, 2, k, n, 400 + m as u64 + k as u64 + n as u64);
        let neon = a.matmul_neon(&b).expect("Z/49 ≤ 2^15");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&neon, &scalar), "mismatch at ({m}, {k}, {n})");
    }
}

#[test]
fn matmul_neon_matches_scalar_larger_prime() {
    // Exercise larger moduli (still ≤ 2^15).
    for &(p, n) in &[(11u32, 1u32), (13, 1), (31, 1), (127, 1), (181, 2)] {
        // 181² = 32761 ≤ 32768 ✓.
        let modulus = (p as u64).pow(n);
        assert!(modulus <= 32768, "test prepared wrong");
        let sz = 32;
        let a = rand_matzpn(p, n, sz, sz, 500 + p as u64 * 10 + n as u64);
        let b = rand_matzpn(p, n, sz, sz, 600 + p as u64 * 10 + n as u64);
        let neon = a.matmul_neon(&b).expect("modulus ≤ 2^15");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&neon, &scalar), "mismatch at p={p} n={n}");
    }
}

#[test]
fn matmul_neon_tail_rows_and_cols() {
    // Shapes where both m and n leave tails: m mod 4 ≠ 0 and n mod 8 ≠ 0.
    for &(m, k, n) in &[
        (5usize, 10, 9),
        (7, 13, 15),
        (3, 6, 7),
        (11, 3, 1),
        (1, 50, 3),
        (2, 1, 11),
    ] {
        let a = rand_matzpn(5, 2, m, k, 700 + m as u64);
        let b = rand_matzpn(5, 2, k, n, 800 + n as u64);
        let neon = a.matmul_neon(&b).expect("modulus in range");
        let scalar = a.matmul_scalar(&b);
        assert!(
            mats_equal(&neon, &scalar),
            "tail shape ({m},{k},{n}) mismatch"
        );
    }
}

#[test]
fn matmul_neon_empty_dims() {
    let a = MatZpn::zeros(5, 2, 0, 5);
    let b = MatZpn::zeros(5, 2, 5, 3);
    let neon = a.matmul_neon(&b).expect("modulus in range");
    assert_eq!(neon.rows, 0);
    assert_eq!(neon.cols, 3);
}

// ─── large-modulus dispatch ──────────────────────────────────────

#[test]
fn matmul_neon_returns_none_for_large_modulus() {
    // p=65521 (prime just below 2^16), n=1 → modulus > 2^15.
    let a = rand_matzpn(65521, 1, 4, 4, 900);
    let b = rand_matzpn(65521, 1, 4, 4, 901);
    assert!(a.matmul_neon(&b).is_none());
}

#[test]
fn matmul_neon_returns_none_for_p_squared_over_2_15() {
    // 181² = 32761 ≤ 2^15; 191² = 36481 > 2^15.
    let a = rand_matzpn(191, 2, 4, 4, 910);
    let b = rand_matzpn(191, 2, 4, 4, 911);
    assert!(a.matmul_neon(&b).is_none());
}

// ─── associativity via the NEON path ────────────────────────────

#[test]
fn matmul_neon_is_associative() {
    let sz = 12;
    let a = rand_matzpn(7, 1, sz, sz, 1000);
    let b = rand_matzpn(7, 1, sz, sz, 1001);
    let c = rand_matzpn(7, 1, sz, sz, 1002);
    let ab = a.matmul_neon(&b).unwrap();
    let bc = b.matmul_neon(&c).unwrap();
    let lhs = ab.matmul_neon(&c).unwrap();
    let rhs = a.matmul_neon(&bc).unwrap();
    assert!(mats_equal(&lhs, &rhs));
}

// ─── u32 NEON path (parallels the u16 tests above) ─────────────

#[test]
fn matmul_neon_u32_identity_small_prime() {
    // p=65537 (Fermat prime, fits 2^17), n=1 — modulus > 2^15 so u16
    // path rejects; u32 path handles it.
    let i = MatZpn::identity(65537, 1, 8);
    let a = rand_matzpn(65537, 1, 8, 8, 2001);
    let neon = a.matmul_neon_u32(&i).expect("Z/65537 ≤ 2^31");
    let scalar = a.matmul_scalar(&i);
    assert!(mats_equal(&neon, &scalar));
}

#[test]
fn matmul_neon_u32_zero_matrix() {
    let a = MatZpn::zeros(65537, 1, 8, 8);
    let b = rand_matzpn(65537, 1, 8, 8, 2002);
    let neon = a.matmul_neon_u32(&b).expect("modulus in range");
    assert!(neon.data.to_vec().iter().all(|&v| v == 0));
}

#[test]
fn matmul_neon_u32_matches_scalar_small_square_z65537() {
    // Modulus 65537 ≈ 2^16+1.
    for &sz in &[4usize, 8, 12, 16, 24] {
        let a = rand_matzpn(65537, 1, sz, sz, 2100 + sz as u64);
        let b = rand_matzpn(65537, 1, sz, sz, 2200 + sz as u64);
        let neon = a.matmul_neon_u32(&b).expect("Z/65537 ≤ 2^31");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&neon, &scalar), "mismatch at sz={sz}");
    }
}

#[test]
fn matmul_neon_u32_matches_scalar_larger_prime() {
    // Primes spanning the u32 sweet-spot range. MatZpn's default
    // guard_bits=2 caps slot_bits at 32, so the largest constructible
    // modulus has element_bits ≤ 30 — primes up to ~2^30 work.
    for &p in &[1009u32, 65537, 1_000_003, 67_108_859] {
        let sz = 8;
        let a = rand_matzpn(p, 1, sz, sz, 3000 + p as u64);
        let b = rand_matzpn(p, 1, sz, sz, 4000 + p as u64);
        let neon = a.matmul_neon_u32(&b).expect("p ≤ 2^31");
        let scalar = a.matmul_scalar(&b);
        assert!(mats_equal(&neon, &scalar), "mismatch at p={p}");
    }
}

#[test]
fn matmul_neon_u32_rectangular_and_tails() {
    // Shapes where both m and n leave 4×4 tile tails.
    for &(m, k, n) in &[
        (5usize, 10, 7),
        (9, 13, 11),
        (3, 5, 2),
        (15, 17, 19),
        (1, 10, 1),
    ] {
        let a = rand_matzpn(65537, 1, m, k, 5000 + m as u64 + n as u64);
        let b = rand_matzpn(65537, 1, k, n, 6000 + m as u64 + n as u64);
        let neon = a.matmul_neon_u32(&b).expect("modulus in range");
        let scalar = a.matmul_scalar(&b);
        assert!(
            mats_equal(&neon, &scalar),
            "tail shape ({m},{k},{n}) mismatch"
        );
    }
}

// Note: there is no "returns None for modulus > 2^31" test here —
// `PackedZpn` caps constructible moduli at element_bits ≤ 30 with its
// default guard_bits=2 (slot_bits ≤ 32 assertion), so the `None`
// branch in `matmul_neon_u32` is defense-in-depth for out-of-band
// callers rather than a reachable state via the normal MatZpn API.

#[test]
fn matmul_neon_u32_associativity() {
    let sz = 8;
    let p = 65537u32;
    let a = rand_matzpn(p, 1, sz, sz, 8001);
    let b = rand_matzpn(p, 1, sz, sz, 8002);
    let c = rand_matzpn(p, 1, sz, sz, 8003);
    let ab = a.matmul_neon_u32(&b).unwrap();
    let bc = b.matmul_neon_u32(&c).unwrap();
    let lhs = ab.matmul_neon_u32(&c).unwrap();
    let rhs = a.matmul_neon_u32(&bc).unwrap();
    assert!(mats_equal(&lhs, &rhs));
}
