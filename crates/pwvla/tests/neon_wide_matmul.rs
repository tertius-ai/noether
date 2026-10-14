// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! NEON matmul kernels for u8, u32, u64 — cross-validation against
//! the portable `scalar::matmul` reference across the same shape
//! matrix used by the u16 coverage suite.

#![cfg(target_arch = "aarch64")]

use pwvla::mat::Mat;
use pwvla::neon_wide::{matmul_u32, matmul_u64, matmul_u8};
use pwvla::scalar;

fn seeded_u8(rows: usize, cols: usize, seed: u64) -> Mat<u8> {
    let mut s = seed;
    let mut m = Mat::<u8>::zeros(rows, cols);
    for v in m.data.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (s >> 24) as u8;
    }
    m
}

fn seeded_u32(rows: usize, cols: usize, seed: u64) -> Mat<u32> {
    let mut s = seed;
    let mut m = Mat::<u32>::zeros(rows, cols);
    for v in m.data.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (s >> 16) as u32;
    }
    m
}

fn seeded_u64(rows: usize, cols: usize, seed: u64) -> Mat<u64> {
    let mut s = seed;
    let mut m = Mat::<u64>::zeros(rows, cols);
    for v in m.data.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = s;
    }
    m
}

#[test]
fn u32_cross_checks_scalar_across_shapes() {
    let cases = [
        (0, 5, 3),
        (4, 0, 3),
        (4, 5, 0),
        (1, 1, 1),
        (1, 7, 11),
        (9, 7, 1),
        (1, 128, 1),
        (3, 8, 16),
        (4, 8, 16),
        (5, 8, 16),
        (7, 11, 13),
        (17, 19, 23),
        (64, 64, 64),
        (100, 100, 100),
        (150, 150, 150),
    ];
    for (i, (m, k, n)) in cases.iter().enumerate() {
        let a = seeded_u32(*m, *k, (i * 1000) as u64);
        let b = seeded_u32(*k, *n, (i * 1000 + 1) as u64);
        let got = matmul_u32(&a, &b);
        let expected = scalar::matmul(&a, &b);
        assert_eq!(got.rows, expected.rows);
        assert_eq!(got.cols, expected.cols);
        assert_eq!(got.data, expected.data, "u32 disagrees at {m}×{k}×{n}");
    }
}

#[test]
fn u8_cross_checks_scalar_across_shapes() {
    let cases = [
        (0, 5, 3),
        (4, 0, 3),
        (4, 5, 0),
        (1, 1, 1),
        (1, 7, 11),
        (9, 7, 1),
        (1, 128, 1),
        (3, 8, 32),
        (4, 8, 32),
        (5, 8, 32),
        (7, 11, 33),
        (17, 19, 65),
        (64, 64, 64),
        (100, 100, 100),
        (150, 150, 150),
    ];
    for (i, (m, k, n)) in cases.iter().enumerate() {
        let a = seeded_u8(*m, *k, (i * 2000 + 7) as u64);
        let b = seeded_u8(*k, *n, (i * 2000 + 13) as u64);
        let got = matmul_u8(&a, &b);
        let expected = scalar::matmul(&a, &b);
        assert_eq!(got.rows, expected.rows);
        assert_eq!(got.cols, expected.cols);
        assert_eq!(got.data, expected.data, "u8 disagrees at {m}×{k}×{n}");
    }
}

#[test]
fn u64_delegates_to_scalar_and_agrees() {
    let cases = [(0, 5, 3), (1, 1, 1), (16, 16, 16), (32, 17, 11)];
    for (i, (m, k, n)) in cases.iter().enumerate() {
        let a = seeded_u64(*m, *k, (i * 3000 + 17) as u64);
        let b = seeded_u64(*k, *n, (i * 3000 + 23) as u64);
        let got = matmul_u64(&a, &b);
        let expected = scalar::matmul(&a, &b);
        assert_eq!(got.data, expected.data, "u64 disagrees at {m}×{k}×{n}");
    }
}
