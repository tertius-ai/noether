// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Streaming matmul tiling — `scalar::streaming_matmul_with` correctness.
//!
//! Tests the tiling / accumulate logic independently of any GPU by
//! injecting `scalar::matmul` as the per-chunk submatmul callable. The
//! same tiling is composed with the GPU matmul in `GpuContext::
//! streaming_matmul_u32`, so correctness here transfers to the GPU
//! path directly (modulo the GPU kernel, which has its own tests).
//!
//! Oracle: whole-matrix `scalar::matmul`. The tiled version must match
//! the whole-matrix result bit-for-bit in Z/2^BITS.

use pwvla::mat::Mat;
use pwvla::scalar;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn rand_mat_u32(rows: usize, cols: usize, seed: u64) -> Mat<u32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u32>::from_fn(rows, cols, |_, _| rng.random::<u32>())
}

fn rand_mat_u16(rows: usize, cols: usize, seed: u64) -> Mat<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u16>::from_fn(rows, cols, |_, _| rng.random::<u16>())
}

fn rand_mat_u8(rows: usize, cols: usize, seed: u64) -> Mat<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u8>::from_fn(rows, cols, |_, _| rng.random::<u8>())
}

// ─── matches oracle across shapes and block sizes ──────────────

#[test]
fn matches_scalar_square_u32_block_equals_dim() {
    // Block size equal to the full dim ⇒ single chunk, should match
    // `scalar::matmul` trivially.
    let a = rand_mat_u32(16, 16, 1);
    let b = rand_mat_u32(16, 16, 2);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 16, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_square_u32_block_1() {
    // Extreme block size 1 — exercises every edge path (bm=bn=bk=1),
    // and verifies single-element chunks compose correctly.
    let a = rand_mat_u32(5, 5, 3);
    let b = rand_mat_u32(5, 5, 4);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 1, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_multiple_block_sizes_square_u32() {
    // Fix the input, sweep over block sizes including non-multiples.
    let a = rand_mat_u32(32, 32, 5);
    let b = rand_mat_u32(32, 32, 6);
    let expected = scalar::matmul(&a, &b);
    for &bs in &[1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64] {
        let streamed = scalar::streaming_matmul_with(&a, &b, bs, |x, y| scalar::matmul(x, y));
        assert_eq!(streamed, expected, "mismatch at block_size={bs}");
    }
}

#[test]
fn matches_scalar_rectangular_u32() {
    // m, k, n all different; block size doesn't divide any of them.
    let a = rand_mat_u32(11, 17, 7);
    let b = rand_mat_u32(17, 13, 8);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 4, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_tall_skinny_u32() {
    // Tall × thin matmul: m >> n.
    let a = rand_mat_u32(64, 8, 9);
    let b = rand_mat_u32(8, 5, 10);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 3, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_short_wide_u32() {
    // Short × wide matmul: n >> m.
    let a = rand_mat_u32(3, 10, 11);
    let b = rand_mat_u32(10, 50, 12);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 7, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_single_row_single_col_u32() {
    // 1 × k · k × 1 — degenerates to an inner product; block size 4.
    let a = rand_mat_u32(1, 20, 13);
    let b = rand_mat_u32(20, 1, 14);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 4, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
    assert_eq!(streamed.rows, 1);
    assert_eq!(streamed.cols, 1);
}

// ─── consistent across widths ──────────────────────────────────

#[test]
fn matches_scalar_u16() {
    let a = rand_mat_u16(24, 17, 20);
    let b = rand_mat_u16(17, 13, 21);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 5, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_u8() {
    let a = rand_mat_u8(32, 40, 30);
    let b = rand_mat_u8(40, 24, 31);
    let expected = scalar::matmul(&a, &b);
    let streamed = scalar::streaming_matmul_with(&a, &b, 6, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
}

#[test]
fn matches_scalar_u64() {
    let a = rand_mat_u32(12, 15, 40).clone();
    // (just verify u64 generic path; build via from_fn)
    let mut rng = ChaCha20Rng::seed_from_u64(41);
    let a64 = Mat::<u64>::from_fn(12, 15, |_, _| rng.random::<u64>());
    let b64 = Mat::<u64>::from_fn(15, 9, |_, _| rng.random::<u64>());
    let expected = scalar::matmul(&a64, &b64);
    let streamed = scalar::streaming_matmul_with(&a64, &b64, 5, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, expected);
    let _ = a;
}

// ─── edge cases ────────────────────────────────────────────────

#[test]
fn zero_matrices_yield_zero() {
    let a = Mat::<u32>::zeros(8, 8);
    let b = rand_mat_u32(8, 8, 50);
    let streamed = scalar::streaming_matmul_with(&a, &b, 3, |x, y| scalar::matmul(x, y));
    for &v in &streamed.data {
        assert_eq!(v, 0);
    }
}

#[test]
fn identity_matmul_preserves_matrix() {
    let n = 10;
    let mut id = Mat::<u32>::zeros(n, n);
    for i in 0..n {
        id.data[i * n + i] = 1;
    }
    let a = rand_mat_u32(n, n, 60);
    let streamed = scalar::streaming_matmul_with(&id, &a, 3, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, a);
    let streamed = scalar::streaming_matmul_with(&a, &id, 3, |x, y| scalar::matmul(x, y));
    assert_eq!(streamed, a);
}

// ─── submatmul invocation counts ──────────────────────────────

#[test]
fn submatmul_invoked_correct_number_of_times() {
    // Count how many times submatmul is called — should equal
    // ⌈m/bs⌉ · ⌈n/bs⌉ · ⌈k/bs⌉.
    let a = rand_mat_u32(20, 12, 70);
    let b = rand_mat_u32(12, 15, 71);
    let bs = 5;
    let expected_count = 4 * 3 * 3; // ⌈20/5⌉ · ⌈15/5⌉ · ⌈12/5⌉
    let count = std::cell::Cell::new(0);
    let _streamed = scalar::streaming_matmul_with(&a, &b, bs, |x, y| {
        count.set(count.get() + 1);
        scalar::matmul(x, y)
    });
    assert_eq!(count.get(), expected_count);
}

#[test]
fn submatmul_receives_correct_tile_sizes() {
    // With m=20, k=12, n=15, bs=5: all chunks at bs except the trailing
    // k-chunk which has size 2 (since 12 = 5+5+2).
    let a = rand_mat_u32(20, 12, 80);
    let b = rand_mat_u32(12, 15, 81);
    let bs = 5;
    let seen_k_sizes = std::cell::RefCell::new(Vec::<usize>::new());
    let _ = scalar::streaming_matmul_with(&a, &b, bs, |x, y| {
        seen_k_sizes.borrow_mut().push(x.cols);
        assert_eq!(x.cols, y.rows, "inner-k mismatch");
        assert!(x.rows <= bs);
        assert!(y.cols <= bs);
        assert!(x.cols <= bs);
        scalar::matmul(x, y)
    });
    // Every (i, j) output-tile contributes the same sequence of k-sizes.
    let k_sizes = seen_k_sizes.borrow();
    let chunks_per_tile = 3; // 12 = 5+5+2
    let total_tiles = 4 * 3;
    assert_eq!(k_sizes.len(), total_tiles * chunks_per_tile);
    // First 3 entries for tile (0,0): 5, 5, 2.
    assert_eq!(&k_sizes[0..3], &[5, 5, 2]);
}
