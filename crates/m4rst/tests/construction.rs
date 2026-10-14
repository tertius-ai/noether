// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for `Mzd` construction + bit IO.
//!
//! These tests do not depend on libm4ri; they verify intrinsic
//! properties of the bit-packed layout that must hold regardless of
//! the oracle. The libm4ri parity tests (`parity_libm4ri.rs`) add
//! bit-equality checks against the C oracle on top of these.

use m4rst::mzd::RADIX;
use m4rst::Mzd;

/// `Mzd::new` produces a matrix whose every bit reads 0.
#[test]
fn new_is_all_zero() {
    for &(r, c) in &[
        (0usize, 0usize),
        (1, 1),
        (1, 64),
        (1, 65),
        (7, 130),
        (64, 64),
        (65, 129),
    ] {
        let m = Mzd::new(r, c);
        assert_eq!(m.rows(), r);
        assert_eq!(m.cols(), c);
        for i in 0..r {
            for j in 0..c {
                assert_eq!(
                    m.get(i, j),
                    0,
                    "bit ({}, {}) of new({}, {}) was nonzero",
                    i,
                    j,
                    r,
                    c
                );
            }
        }
        // No excess bits set (M4RI invariant).
        for &w in m.as_words() {
            assert_eq!(w, 0);
        }
    }
}

/// `Mzd::identity(n)` has 1s exactly on the diagonal.
#[test]
fn identity_is_diagonal() {
    for &n in &[0usize, 1, 63, 64, 65, 128, 200] {
        let m = Mzd::identity(n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1 } else { 0 };
                assert_eq!(m.get(i, j), expected, "identity({})[{},{}] wrong", n, i, j);
            }
        }
    }
}

#[test]
fn identity_windows_match_materialized_submatrices() {
    let identity = Mzd::identity(193);
    for &(row_start, row_end, col_start, col_end) in &[
        (0, 0, 0, 0),
        (0, 17, 0, 65),
        (3, 70, 61, 129),
        (64, 193, 0, 63),
        (128, 193, 127, 193),
    ] {
        assert_eq!(
            Mzd::identity_window(row_start, row_end, col_start, col_end).as_words(),
            identity
                .submatrix(row_start, col_start, row_end, col_end)
                .as_words()
        );
    }
}

#[test]
fn strided_byte_windows_match_contiguous_conversion() {
    let rows = 7;
    let stride = 137;
    let col_start = 61;
    let cols = 65;
    let bytes: Vec<u8> = (0..rows * stride)
        .map(|index| ((index * 17 + index / 5) & 0xff) as u8)
        .collect();
    let contiguous: Vec<u8> = (0..rows)
        .flat_map(|row| bytes[row * stride + col_start..row * stride + col_start + cols].to_vec())
        .collect();
    assert_eq!(
        Mzd::from_strided_byte_buffer(rows, cols, stride, &bytes[col_start..]).as_words(),
        Mzd::from_byte_buffer(rows, cols, &contiguous).as_words()
    );
}

/// `get` / `set` round-trip on every bit of a small grid.
#[test]
fn set_get_round_trip() {
    let (rows, cols) = (7, 130);
    let mut m = Mzd::new(rows, cols);
    // Set a Diophantine pattern that exercises both low + high words.
    for i in 0..rows {
        for j in 0..cols {
            if (3 * i + 5 * j) % 7 == 0 {
                m.set(i, j, 1);
            }
        }
    }
    for i in 0..rows {
        for j in 0..cols {
            let want = if (3 * i + 5 * j) % 7 == 0 { 1 } else { 0 };
            assert_eq!(m.get(i, j), want);
        }
    }
}

/// `set` preserves the M4RI excess-bits-are-zero invariant in the high word.
#[test]
fn set_preserves_excess_zero() {
    let cols = 130; // 2 full words + 2 valid bits in word 2
    let mut m = Mzd::new(3, cols);
    for i in 0..3 {
        for j in 0..cols {
            m.set(i, j, 1);
        }
    }
    let valid_bits = cols % RADIX;
    let mask = (1u64 << valid_bits) - 1;
    for r in 0..3 {
        let row = m.row_words(r);
        assert_eq!(row[0], u64::MAX, "row {} word 0 should be all ones", r);
        assert_eq!(row[1], u64::MAX, "row {} word 1 should be all ones", r);
        assert_eq!(row[2], mask, "row {} word 2 should equal high_bitmask", r);
    }
}

/// `flip` toggles a bit twice → no-op.
#[test]
fn flip_is_involution() {
    let mut m = Mzd::random(5, 200, 0xDEAD_BEEF);
    let before = m.to_byte_buffer();
    m.flip(2, 150);
    m.flip(2, 150);
    assert_eq!(m.to_byte_buffer(), before);
}

/// `to_byte_buffer ∘ from_byte_buffer` is the identity on a row-major
/// `{0,1}`-byte buffer.
#[test]
fn byte_buffer_round_trip() {
    let rows = 5;
    let cols = 200;
    // Pattern: bit set iff (i*j + i + j) is even — easy to recompute by hand.
    let bytes: Vec<u8> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| ((i * j + i + j) & 1) as u8))
        .collect();
    let m = Mzd::from_byte_buffer(rows, cols, &bytes);
    assert_eq!(m.to_byte_buffer(), bytes);

    // Spot-check a few cells against the closed form.
    for &(i, j) in &[
        (0usize, 0usize),
        (1, 1),
        (3, 199),
        (4, 63),
        (4, 64),
        (4, 65),
    ] {
        let want = ((i * j + i + j) & 1) as u8;
        assert_eq!(m.get(i, j), want, "({}, {}) mismatch", i, j);
    }
}

/// `from_byte_buffer` ignores high bits of the input bytes (only `& 1` matters).
#[test]
fn from_byte_buffer_masks_high_bits() {
    // Both 0xAA (low bit 0) and 0xAB (low bit 1) should produce the same matrix
    // as 0x00 / 0x01.
    let bytes_a = vec![0u8, 1, 0, 1, 0, 1];
    let bytes_b = vec![0xAA, 0xAB, 0xFE, 0xFF, 0x10, 0x11];
    let a = Mzd::from_byte_buffer(2, 3, &bytes_a);
    let b = Mzd::from_byte_buffer(2, 3, &bytes_b);
    assert_eq!(a.as_words(), b.as_words());
}

/// `Mzd::random` is deterministic given the same seed, and `width=0`
/// edge cases don't panic.
#[test]
fn random_is_deterministic_and_handles_empty() {
    let m1 = Mzd::random(8, 128, 0xCAFE_F00D);
    let m2 = Mzd::random(8, 128, 0xCAFE_F00D);
    assert_eq!(m1.as_words(), m2.as_words());

    let empty_cols = Mzd::random(4, 0, 1);
    assert_eq!(empty_cols.rows(), 4);
    assert_eq!(empty_cols.cols(), 0);
    assert!(empty_cols.as_words().is_empty());

    let empty_rows = Mzd::random(0, 128, 1);
    assert_eq!(empty_rows.rows(), 0);
    assert!(empty_rows.as_words().is_empty());
}

/// `Mzd::random` density is ≈ 0.5 for a large matrix. With N = 8192 bits the
/// standard error is ~0.0055, so a ±0.02 envelope is comfortably within bounds
/// for any reasonable seed.
#[test]
fn random_density_close_to_half() {
    let m = Mzd::random(64, 128, 0x1234_5678_9ABC_DEF0);
    let mut ones = 0usize;
    for w in m.as_words() {
        ones += w.count_ones() as usize;
    }
    let total = 64 * 128;
    let density = ones as f64 / total as f64;
    assert!(
        (density - 0.5).abs() < 0.02,
        "random density was {} (expected ≈ 0.5)",
        density
    );
}

/// `high_bitmask()` reflects the actual valid-bits-in-high-word mask.
#[test]
fn high_bitmask_invariant() {
    use m4rst::mzd::RADIX;
    for &c in &[1usize, 32, 63, 64, 65, 128, 129, 256, 257] {
        let m = Mzd::new(4, c);
        let bitmask = m.high_bitmask();
        if c.is_multiple_of(RADIX) {
            assert_eq!(bitmask, u64::MAX, "cols={} should give full mask", c);
        } else {
            let expected = (1u64 << (c % RADIX)) - 1;
            assert_eq!(bitmask, expected, "cols={}", c);
        }
    }
}

/// `Mzd::from_matrix_read` materialises a view into a fresh owned
/// matrix bit-for-bit.
#[test]
fn from_matrix_read_round_trip() {
    let parent = Mzd::random(8, 200, 0xFEED);
    let view = parent.view(2, 64, 7, 192);
    let owned = Mzd::from_matrix_read(&view);
    assert_eq!(owned.rows(), 5);
    assert_eq!(owned.cols(), 128);
    for i in 0..owned.rows() {
        for j in 0..owned.cols() {
            assert_eq!(owned.get(i, j), parent.get(2 + i, 64 + j));
        }
    }
}

/// `extract_bits(row, col_start, k)` matches a bit-by-bit `get` oracle,
/// including the spans-two-words case.
#[test]
fn extract_bits_matches_get_oracle() {
    let m = Mzd::random(4, 200, 0xABCD);
    let ks: [usize; 6] = [1, 2, 7, 8, 31, 63];
    let starts: [usize; 8] = [0, 1, 7, 56, 60, 63, 64, 137];
    for &k in &ks {
        for &col_start in &starts {
            if col_start + k > m.cols() {
                continue;
            }
            for row in 0..m.rows() {
                let bits = m.extract_bits(row, col_start, k);
                let mut want: u64 = 0;
                for i in 0..k {
                    want |= (m.get(row, col_start + i) as u64) << i;
                }
                assert_eq!(
                    bits, want,
                    "extract_bits(row={}, col={}, k={}) gave {:#x}, expected {:#x}",
                    row, col_start, k, bits, want
                );
            }
        }
    }
}

#[test]
fn first_one_in_row_matches_bit_scan_at_word_boundaries() {
    let matrix = Mzd::random(19, 193, 0xF1757);
    for row in 0..matrix.rows() {
        for start in [0, 1, 63, 64, 65, 127, 128, 192, 193] {
            let expected = (start..matrix.cols()).find(|&col| matrix.get(row, col) != 0);
            assert_eq!(matrix.first_one_in_row(row, start), expected);
        }
    }
}

/// `submatrix ∘ paste_into` is the identity on a deterministic random matrix.
#[test]
fn submatrix_paste_round_trip() {
    let a = Mzd::random(20, 200, 0xFEED);
    // Word-aligned column offset (fast path).
    let s = a.submatrix(5, 64, 15, 192);
    assert_eq!(s.rows(), 10);
    assert_eq!(s.cols(), 128);
    for i in 0..s.rows() {
        for j in 0..s.cols() {
            assert_eq!(s.get(i, j), a.get(5 + i, 64 + j));
        }
    }
    // Misaligned column offset (slow path).
    let s2 = a.submatrix(2, 7, 18, 137);
    assert_eq!(s2.rows(), 16);
    assert_eq!(s2.cols(), 130);
    for i in 0..s2.rows() {
        for j in 0..s2.cols() {
            assert_eq!(s2.get(i, j), a.get(2 + i, 7 + j));
        }
    }
    // Paste back into a fresh canvas.
    let mut canvas = Mzd::new(20, 200);
    canvas.paste_into(5, 64, &s);
    for i in 0..s.rows() {
        for j in 0..s.cols() {
            assert_eq!(canvas.get(5 + i, 64 + j), a.get(5 + i, 64 + j));
        }
    }
    // Misaligned paste.
    let mut canvas2 = Mzd::new(20, 200);
    canvas2.paste_into(2, 7, &s2);
    for i in 0..s2.rows() {
        for j in 0..s2.cols() {
            assert_eq!(canvas2.get(2 + i, 7 + j), a.get(2 + i, 7 + j));
        }
    }
}

/// Arbitrary bit offsets use packed word stitching without changing the
/// cell-wise `submatrix`/`paste_into` contract or surrounding destination bits.
#[test]
fn unaligned_block_copy_matches_cell_oracle() {
    let source = Mzd::random(9, 257, 0xA11C_E5ED);
    for col_start in [1, 2, 7, 31, 63, 65, 100, 127, 129, 191] {
        for width in [1, 2, 31, 63, 64, 65, 100, 128] {
            let col_end = (col_start + width).min(source.cols());
            let block = source.submatrix(1, col_start, 8, col_end);
            for row in 0..block.rows() {
                for col in 0..block.cols() {
                    assert_eq!(
                        block.get(row, col),
                        source.get(row + 1, col_start + col),
                        "submatrix mismatch for start={col_start}, width={width}, cell=({row}, {col})"
                    );
                }
            }

            for destination_col in [1, 7, 31, 63, 65, 129] {
                if destination_col + block.cols() > 320 {
                    continue;
                }
                let mut actual = Mzd::random(11, 320, 0xB10C_C0F1);
                let mut expected = actual.clone();
                actual.paste_into(2, destination_col, &block);
                for row in 0..block.rows() {
                    for col in 0..block.cols() {
                        expected.set(2 + row, destination_col + col, block.get(row, col));
                    }
                }
                assert_eq!(
                    actual.as_words(),
                    expected.as_words(),
                    "paste mismatch for source start={col_start}, width={width}, destination={destination_col}"
                );
            }
        }
    }
}

/// Paste preserves bits outside the pasted rectangle.
#[test]
fn paste_preserves_outside() {
    let mut a = Mzd::random(16, 128, 0xBEEF_F00D);
    let snapshot = a.to_byte_buffer();
    let block = Mzd::random(4, 32, 0xC0DE);
    a.paste_into(8, 64, &block); // word-aligned
                                 // Bits outside the [8..12, 64..96) rectangle should be unchanged.
    for i in 0..16 {
        for j in 0..128 {
            if (8..12).contains(&i) && (64..96).contains(&j) {
                continue;
            }
            assert_eq!(
                a.get(i, j),
                snapshot[i * 128 + j],
                "paste damaged cell ({}, {})",
                i,
                j
            );
        }
    }
}

/// `Mzd::random` respects the high-bitmask invariant — excess bits stay zero.
#[test]
fn random_preserves_excess_zero() {
    let cols = 130;
    let m = Mzd::random(4, cols, 42);
    let valid_bits = cols % RADIX;
    let mask = (1u64 << valid_bits) - 1;
    for r in 0..m.rows() {
        let row = m.row_words(r);
        let high = row[row.len() - 1];
        assert_eq!(high & !mask, 0, "row {} has nonzero excess bits", r);
    }
}
