// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Cross-backend equivalence: scalar == NEON == AVX2.

use pwvla::{scalar, Mat};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use rand::rngs::ChaCha20Rng;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use rand::{RngExt, SeedableRng};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn random_u16(rows: usize, cols: usize, seed: u64) -> Mat<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u16>::from_fn(rows, cols, |_, _| rng.random::<u16>())
}

#[test]
fn scalar_matches_manual_small() {
    // 2x2 worked example in Z/2^8.
    let a = Mat::<u8> {
        rows: 2,
        cols: 2,
        data: vec![1, 2, 3, 4],
    };
    let b = Mat::<u8> {
        rows: 2,
        cols: 2,
        data: vec![5, 6, 7, 8],
    };
    let c = scalar::matmul(&a, &b);
    // [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(c.data, vec![19, 22, 43, 50]);
}

#[test]
fn scalar_wraps_in_z2_8() {
    // 200 * 200 = 40000 ≡ 40000 mod 256 = 64.
    let a = Mat::<u8> {
        rows: 1,
        cols: 1,
        data: vec![200],
    };
    let b = Mat::<u8> {
        rows: 1,
        cols: 1,
        data: vec![200],
    };
    let c = scalar::matmul(&a, &b);
    assert_eq!(c.data, vec![64]);
}

#[test]
fn public_dispatch_matches_the_scalar_floor_for_every_cell_width() {
    macro_rules! check {
        ($cell:ty, $matmul:path) => {{
            let left = Mat::<$cell>::from_fn(19, 23, |row, column| {
                <$cell>::try_from((row * 31 + column * 17) % 251).unwrap()
            });
            let right = Mat::<$cell>::from_fn(23, 29, |row, column| {
                <$cell>::try_from((row * 13 + column * 37) % 241).unwrap()
            });
            assert_eq!($matmul(&left, &right), scalar::matmul(&left, &right));
        }};
    }

    check!(u8, pwvla::matmul_u8);
    check!(u16, pwvla::matmul_u16);
    check!(u32, pwvla::matmul_u32);
    check!(u64, pwvla::matmul_u64);
    check!(u128, pwvla::matmul_u128);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn strassen_matches_scalar_u16() {
    // Use a low crossover to actually exercise the recursion in tests.
    // Sizes chosen to hit multiple levels and the padding path.
    let xo = 64;
    for (m, k, n, seed) in [
        (128, 128, 128, 50), // 1 level, clean
        (256, 256, 256, 51), // 2 levels
        (257, 253, 259, 52), // all odd — exercises padding at every level
        (130, 192, 128, 53), // mixed parities
    ] {
        let a = random_u16(m, k, seed);
        let b = random_u16(k, n, seed + 100);
        let c_scalar = scalar::matmul(&a, &b);
        let c_strassen = pwvla::strassen::matmul_u16_with_crossover(&a, &b, xo);
        assert_eq!(c_scalar, c_strassen, "mismatch at {}x{}x{}", m, k, n);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_u16() {
    // Sizes chosen to stress: pure block-aligned, mid-size edges, and
    // sizes that cross MC=128 / KC=256 / NC=512 boundaries with tails.
    for (m, k, n, seed) in [
        (8, 8, 8, 1),
        (17, 23, 31, 2),
        (64, 64, 64, 3),
        (130, 257, 129, 4),
        (513, 512, 513, 5),
    ] {
        let a = random_u16(m, k, seed);
        let b = random_u16(k, n, seed + 100);
        let c_scalar = scalar::matmul(&a, &b);
        let c_neon = pwvla::neon::matmul_u16(&a, &b);
        assert_eq!(c_scalar, c_neon, "mismatch at {}x{}x{}", m, k, n);
    }
}

// FLINT cross-reference belongs in the consuming facade's dispatch tests;
// this crate differentially tests every optimized path against its scalar floor.

// The AVX2 mirror of `neon_matches_scalar_u16`. The safe entry retains runtime
// detection for standalone crate consumers compiled below Noether's v3 floor.
// The shapes cover aligned, mid-size, cache-block boundary, and tail cases.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_u16() {
    for (m, k, n, seed) in [
        (8, 8, 8, 1),
        (17, 23, 31, 2),
        (64, 64, 64, 3),
        (130, 257, 129, 4),
        (513, 512, 513, 5),
    ] {
        let a = random_u16(m, k, seed);
        let b = random_u16(k, n, seed + 100);
        let c_scalar = scalar::matmul(&a, &b);
        let c_avx2 = pwvla::avx2::matmul_u16(&a, &b);
        assert_eq!(c_scalar, c_avx2, "mismatch at {}x{}x{}", m, k, n);
    }
}
