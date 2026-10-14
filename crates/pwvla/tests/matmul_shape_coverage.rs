// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shape coverage for matmul kernels (NEON / AVX2 / scalar / GPU).
//!
//! Tests exercise the edge-tile path, empty dims, 1×N / N×1, below-tile,
//! non-power-of-2 dims, tile-boundary (MR, NR, MC, NC, KC ± 1), and
//! rectangular aspect ratios. Every result is cross-validated against
//! `scalar::matmul`, the portable reference.

use pwvla::mat::Mat;
use pwvla::scalar;

fn seeded_u16(rows: usize, cols: usize, seed: u64) -> Mat<u16> {
    let mut s = seed;
    let mut m = Mat::<u16>::zeros(rows, cols);
    for v in m.data.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (s >> 16) as u16;
    }
    m
}

fn cross_check(m: usize, k: usize, n: usize, seed: u64) {
    let a = seeded_u16(m, k, seed);
    let b = seeded_u16(k, n, seed.wrapping_add(1));
    let expected = scalar::matmul(&a, &b);
    assert_eq!(expected.rows, m);
    assert_eq!(expected.cols, n);

    #[cfg(target_arch = "aarch64")]
    {
        let got = pwvla::neon::matmul_u16(&a, &b);
        assert_eq!(got.rows, m, "neon: rows mismatch for {m}×{k}×{n}");
        assert_eq!(got.cols, n, "neon: cols mismatch for {m}×{k}×{n}");
        assert_eq!(got.data, expected.data, "neon disagrees at {m}×{k}×{n}");
    }

    #[cfg(target_arch = "x86_64")]
    {
        let got = pwvla::avx2::matmul_u16(&a, &b);
        assert_eq!(got.rows, m, "avx2: rows mismatch for {m}×{k}×{n}");
        assert_eq!(got.cols, n, "avx2: cols mismatch for {m}×{k}×{n}");
        assert_eq!(got.data, expected.data, "avx2 disagrees at {m}×{k}×{n}");
    }
}

#[test]
fn empty_dimensions_return_empty() {
    cross_check(0, 5, 3, 1);
    cross_check(4, 0, 3, 2);
    cross_check(4, 5, 0, 3);
    cross_check(0, 0, 0, 4);
}

#[test]
fn one_by_n_and_n_by_one() {
    // Row vector × matrix.
    cross_check(1, 7, 11, 10);
    cross_check(1, 1, 1, 11);
    cross_check(1, 64, 64, 12);
    // Matrix × column vector.
    cross_check(9, 7, 1, 13);
    cross_check(64, 64, 1, 14);
    // Inner product shape.
    cross_check(1, 128, 1, 15);
}

#[test]
fn below_register_tile() {
    // m < MR (4), n < NR (16, NEON) / 32 (AVX2).
    for m in 1..=3 {
        for n in 1..=5 {
            for k in 1..=5 {
                cross_check(m, k, n, (m * 1000 + k * 100 + n) as u64);
            }
        }
    }
}

#[test]
fn tile_boundary_plus_minus_one() {
    // MR=4 boundary on m.
    cross_check(3, 8, 16, 100);
    cross_check(4, 8, 16, 101);
    cross_check(5, 8, 16, 102);
    // NR=16 boundary on n (NEON).
    cross_check(8, 8, 15, 103);
    cross_check(8, 8, 16, 104);
    cross_check(8, 8, 17, 105);
    // Non-power-of-2 all around.
    cross_check(7, 11, 13, 106);
    cross_check(17, 19, 23, 107);
}

#[test]
fn cache_block_boundary() {
    // MC=128, KC=256, NC=512 in NEON. Cross the MC boundary on m.
    cross_check(127, 64, 64, 200);
    cross_check(128, 64, 64, 201);
    cross_check(129, 64, 64, 202);
    // Cross KC on k.
    cross_check(64, 255, 64, 203);
    cross_check(64, 256, 64, 204);
    cross_check(64, 257, 64, 205);
    // Cross NC on n.
    cross_check(64, 64, 511, 206);
    cross_check(64, 64, 512, 207);
    cross_check(64, 64, 513, 208);
}

#[test]
fn rectangular_aspect_ratios() {
    // Tall-and-skinny.
    cross_check(200, 50, 3, 300);
    // Short-and-wide.
    cross_check(3, 50, 200, 301);
    // Lopsided k.
    cross_check(32, 500, 32, 302);
    // Square at a mid-size.
    cross_check(97, 97, 97, 303);
}

#[test]
fn non_power_of_2_medium_square() {
    // Exercises both packed and edge-tile paths in the large-problem
    // regime (above the scalar-fallback threshold).
    cross_check(100, 100, 100, 400);
    cross_check(150, 150, 150, 401);
    cross_check(200, 200, 200, 402);
}
