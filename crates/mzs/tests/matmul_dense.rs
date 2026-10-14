// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse · dense F_2 matmul correctness (byte-buffer + bit-packed).
//!
//! Strategies:
//! - Hand-computed small cases that pin specific results bit-for-bit.
//! - Cross-validation of `matmul_dense` (byte) against
//!   `matmul_dense_packed` (bit-packed) through pack/unpack.
//! - Cross-validation of `matmul_dense` against sparse-`matmul` of the
//!   same data routed through `from_byte_buffer`. This is the closest
//!   thing to a "densify-via-m4rst" oracle that doesn't require the
//!   m4rst path-dep.
//! - Edge shapes and slack-bit handling for the packed variant.

use mzs::Mzs;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn random_mzs(rows: usize, cols: usize, density: f64, seed: u64) -> Mzs {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut m = Mzs::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if rng.random::<f64>() < density {
                m.set(i, j, 1);
            }
        }
    }
    m
}

fn random_byte_buffer(rows: usize, cols: usize, density: f64, seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| {
            if rng.random::<f64>() < density {
                1u8
            } else {
                0u8
            }
        })
        .collect()
}

/// Pack a row-major byte buffer into the LSB-first u64 layout used by
/// `matmul_dense_packed`.
fn pack(rows: usize, cols: usize, b: &[u8]) -> Vec<u64> {
    let stride = cols.div_ceil(64);
    let mut out = vec![0u64; rows * stride];
    if stride == 0 {
        return out;
    }
    for r in 0..rows {
        for c in 0..cols {
            if b[r * cols + c] & 1 == 1 {
                out[r * stride + c / 64] |= 1u64 << (c % 64);
            }
        }
    }
    out
}

/// Inverse of `pack`.
fn unpack(rows: usize, cols: usize, b: &[u64]) -> Vec<u8> {
    let stride = cols.div_ceil(64);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = ((b[r * stride + c / 64] >> (c % 64)) & 1) as u8;
        }
    }
    out
}

// ─── hand-computed ──────────────────────────────────────────────────

#[test]
fn matmul_dense_2x2_handcomputed() {
    // A = [[1, 1], [0, 1]], B = [[1, 0], [1, 1]] over F_2
    // A · B = [[0, 1], [1, 1]]
    let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    let b = [1u8, 0, 1, 1];
    assert_eq!(a.matmul_dense(2, 2, &b), vec![0, 1, 1, 1]);
}

#[test]
fn matmul_dense_packed_2x2_handcomputed() {
    let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    // B row 0 = [1, 0] → 0b01; B row 1 = [1, 1] → 0b11
    let b = [0b01u64, 0b11u64];
    let c = a.matmul_dense_packed(2, 2, &b);
    // C row 0 = [0, 1] → 0b10; C row 1 = [1, 1] → 0b11
    assert_eq!(c, vec![0b10u64, 0b11u64]);
}

#[test]
fn matmul_dense_high_bits_ignored() {
    // Bytes with garbage in high bits must still produce a clean {0, 1}
    // output. Same contract as from_byte_buffer.
    let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    let b = [0xFFu8, 0xFE, 0xFD, 0xFB]; // low bits: 1, 0, 1, 1
    assert_eq!(a.matmul_dense(2, 2, &b), vec![0, 1, 1, 1]);
}

#[test]
fn matmul_dense_packed_slack_bits_canonicalised() {
    // b_cols = 3 so stride = 1, tail = 3 bits used. The other 61 bits
    // of B's word are "slack" and the contract says they're ignored.
    // Output must come back with slack bits zeroed.
    let a = Mzs::from_triples(2, 2, &[(0, 0), (1, 1)]); // I_2
    let b = [0xFFFF_FFFF_FFFF_FFFFu64, 0xFFFF_FFFF_FFFF_FFFFu64];
    let c = a.matmul_dense_packed(2, 3, &b);
    // I · B (with slack masked) = both rows are 0b111 = 7.
    assert_eq!(c, vec![0b111u64, 0b111u64]);
}

// ─── algebraic invariants ───────────────────────────────────────────

#[test]
fn matmul_dense_identity_b_is_a() {
    // A · I = A. Byte form.
    let a = random_mzs(8, 7, 0.4, 42);
    let mut i_bytes = vec![0u8; 7 * 7];
    for k in 0..7 {
        i_bytes[k * 7 + k] = 1;
    }
    let out = a.matmul_dense(7, 7, &i_bytes);
    assert_eq!(out, a.to_byte_buffer());
}

#[test]
fn matmul_dense_zero_a_is_zero() {
    let a = Mzs::new(5, 4);
    let b = random_byte_buffer(4, 6, 0.5, 1);
    assert_eq!(a.matmul_dense(4, 6, &b), vec![0u8; 5 * 6]);
}

#[test]
fn matmul_dense_zero_b_is_zero() {
    let a = random_mzs(5, 4, 0.5, 2);
    let b = vec![0u8; 4 * 6];
    assert_eq!(a.matmul_dense(4, 6, &b), vec![0u8; 5 * 6]);
}

// ─── cross-validation against sparse matmul ─────────────────────────

fn dense_via_sparse(a: &Mzs, b_rows: usize, b_cols: usize, b: &[u8]) -> Vec<u8> {
    // The oracle: route B through Mzs::from_byte_buffer and run sparse
    // matmul, then densify.
    let b_mzs = Mzs::from_byte_buffer(b_rows, b_cols, b);
    a.matmul(&b_mzs).to_byte_buffer()
}

#[test]
fn matmul_dense_agrees_with_sparse_matmul_via_byte_buffer() {
    for (m, k, n, density, seed) in [
        (4, 7, 5, 0.3, 10),
        (8, 8, 8, 0.5, 11),
        (16, 1, 16, 0.5, 12),  // thin middle
        (1, 16, 16, 0.5, 13),  // thin top
        (16, 16, 1, 0.5, 14),  // thin right
        (32, 32, 17, 0.1, 15), // sparse
        (32, 32, 17, 0.9, 16), // dense
    ] {
        let a = random_mzs(m, k, density, seed);
        let b = random_byte_buffer(k, n, density, seed + 100);
        assert_eq!(
            a.matmul_dense(k, n, &b),
            dense_via_sparse(&a, k, n, &b),
            "matmul_dense diverged at shape ({m}, {k}, {n}) density {density}",
        );
    }
}

#[test]
fn matmul_dense_packed_agrees_with_byte_variant() {
    // Pack-then-multiply must agree with multiply-then-pack on the
    // byte path. Covers a range of (b_cols mod 64) so the stride
    // arithmetic and tail-mask are exercised.
    for (m, k, n, density, seed) in [
        (5, 6, 1, 0.5, 20),
        (5, 6, 63, 0.5, 21),
        (5, 6, 64, 0.5, 22),
        (5, 6, 65, 0.5, 23),
        (5, 6, 100, 0.5, 24),
        (8, 8, 200, 0.1, 25),
    ] {
        let a = random_mzs(m, k, density, seed);
        let b = random_byte_buffer(k, n, density, seed + 100);
        let dense_bytes = a.matmul_dense(k, n, &b);
        let dense_packed = a.matmul_dense_packed(k, n, &pack(k, n, &b));
        let packed_as_bytes = unpack(m, n, &dense_packed);
        assert_eq!(
            packed_as_bytes, dense_bytes,
            "packed ≠ byte path at shape ({m}, {k}, {n}) density {density}",
        );
    }
}

// ─── edge shapes ────────────────────────────────────────────────────

#[test]
fn matmul_dense_empty_shapes() {
    // 0×0 · 0×0
    let a = Mzs::new(0, 0);
    assert_eq!(a.matmul_dense(0, 0, &[]), Vec::<u8>::new());
    // 0×k · k×n
    let a = Mzs::new(0, 3);
    assert_eq!(a.matmul_dense(3, 5, &[1u8; 15]), Vec::<u8>::new(),);
    // m×k · k×0
    let a = random_mzs(4, 3, 0.5, 30);
    assert_eq!(a.matmul_dense(3, 0, &[]), Vec::<u8>::new());
    // m×0 · 0×n  — empty middle dim, output is zero.
    let a = Mzs::new(4, 0);
    assert_eq!(a.matmul_dense(0, 5, &[]), vec![0u8; 20]);
}

#[test]
fn matmul_dense_packed_empty_shapes() {
    let a = Mzs::new(0, 0);
    assert!(a.matmul_dense_packed(0, 0, &[]).is_empty());
    let a = Mzs::new(0, 3);
    assert!(a.matmul_dense_packed(3, 5, &[0u64, 0u64, 0u64]).is_empty());
    let a = Mzs::new(4, 0);
    assert_eq!(a.matmul_dense_packed(0, 5, &[]), vec![0u64; 4]);
}

// ─── shape-mismatch panics ──────────────────────────────────────────

#[test]
#[should_panic(expected = "matmul_dense: A.cols")]
fn matmul_dense_panics_on_inner_mismatch() {
    let a = Mzs::new(3, 4);
    a.matmul_dense(5, 6, &[0u8; 30]); // A.cols = 4, b_rows = 5
}

#[test]
#[should_panic(expected = "matmul_dense: b.len")]
fn matmul_dense_panics_on_b_len() {
    let a = Mzs::new(3, 4);
    a.matmul_dense(4, 5, &[0u8; 19]); // 19 != 4·5
}

#[test]
#[should_panic(expected = "matmul_dense_packed: A.cols")]
fn matmul_dense_packed_panics_on_inner_mismatch() {
    let a = Mzs::new(3, 4);
    a.matmul_dense_packed(5, 6, &[0u64; 5]);
}

#[test]
#[should_panic(expected = "matmul_dense_packed: b.len")]
fn matmul_dense_packed_panics_on_b_len() {
    let a = Mzs::new(3, 4);
    // b_cols = 100 → stride = 2 → expected length 4·2 = 8.
    a.matmul_dense_packed(4, 100, &[0u64; 7]);
}
