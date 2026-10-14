// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Tests for the zero-copy view types.
//!
//! Correctness is layered: (1) immutable views read the same cells as
//! Mzd::get does on the parent; (2) mutable views write back into the
//! parent; (3) `quad_split` produces 4 disjoint mut views that tile
//! the source; (4) modifications through quadrants don't corrupt
//! peers.

use m4rst::{MatrixRead, MatrixWrite, Mzd, MzdView};

#[test]
fn view_reads_match_parent() {
    let parent = Mzd::random(20, 256, 0xFEED);
    let v = parent.view(3, 64, 17, 192);
    assert_eq!(v.rows(), 14);
    assert_eq!(v.cols(), 128);
    for i in 0..v.rows() {
        for j in 0..v.cols() {
            assert_eq!(v.get(i, j), parent.get(3 + i, 64 + j));
        }
    }
}

#[test]
fn view_word_aligned_cols_partial_last_word() {
    // cols not multiple of 64: high_bitmask masks the tail.
    let parent = Mzd::random(8, 256, 0xCAFE);
    let v = parent.view(0, 64, 8, 200); // 136 cols → 3 words, last has 8 valid bits
    assert_eq!(v.cols(), 136);
    assert_eq!(v.width(), 3);
    assert_eq!(v.high_bitmask(), (1u64 << 8) - 1);
    for i in 0..v.rows() {
        for j in 0..v.cols() {
            assert_eq!(v.get(i, j), parent.get(i, 64 + j));
        }
    }
}

#[test]
#[should_panic(expected = "not word-aligned")]
fn view_panics_on_unaligned_col_offset() {
    let parent = Mzd::random(4, 256, 1);
    let _ = parent.view(0, 7, 4, 128); // c0 = 7, not multiple of 64
}

#[test]
fn view_full_matches_parent() {
    let parent = Mzd::random(10, 128, 42);
    let v = parent.full_view();
    assert_eq!(v.rows(), parent.rows());
    assert_eq!(v.cols(), parent.cols());
    for i in 0..v.rows() {
        for j in 0..v.cols() {
            assert_eq!(v.get(i, j), parent.get(i, j));
        }
    }
}

#[test]
fn mut_view_writes_back_to_parent() {
    let mut parent = Mzd::new(4, 128);
    {
        let mut v = parent.view_mut(0, 64, 4, 128);
        // Write a recognizable pattern in the right half.
        for i in 0..v.rows() {
            v.set(i, i, 1);
        }
    }
    // Right half (cols 64+i, i ∈ 0..4) should have 1s on the diagonal.
    for i in 0..4 {
        assert_eq!(
            parent.get(i, 64 + i),
            1,
            "diag bit ({}, {}) missing",
            i,
            64 + i
        );
    }
    // Left half untouched.
    for i in 0..4 {
        for j in 0..64 {
            assert_eq!(
                parent.get(i, j),
                0,
                "left-half cell ({}, {}) corrupted",
                i,
                j
            );
        }
    }
}

#[test]
fn quad_split_writes_are_disjoint() {
    // Build a 8 × 256 mut view, split into 4 quadrants of 4 × 128 each,
    // write a different pattern in each, then read back via the parent.
    let mut parent = Mzd::new(8, 256);
    {
        let v = parent.view_mut(0, 0, 8, 256);
        let (mut c11, _c12, mut c21, mut c22) = v.quad_split(4, 128);
        for i in 0..4 {
            for j in 0..128 {
                c11.set(i, j, 1); // C11 entirely 1s
            }
        }
        // C12 stays zero — verify quadrant boundaries.
        for i in 0..4 {
            for j in 0..128 {
                c21.set(i, j, ((i + j) & 1) as u8); // checkerboard
            }
        }
        for i in 0..4 {
            c22.set(i, i, 1); // diagonal
        }
    }
    // Verify C11 (rows 0..4, cols 0..128) all 1s.
    for i in 0..4 {
        for j in 0..128 {
            assert_eq!(parent.get(i, j), 1, "C11 ({}, {}) wrong", i, j);
        }
    }
    // Verify C12 (rows 0..4, cols 128..256) all 0s.
    for i in 0..4 {
        for j in 128..256 {
            assert_eq!(parent.get(i, j), 0, "C12 ({}, {}) wrong", i, j);
        }
    }
    // Verify C21 (rows 4..8, cols 0..128) is the checkerboard.
    for i in 0..4 {
        for j in 0..128 {
            let want = ((i + j) & 1) as u8;
            assert_eq!(parent.get(4 + i, j), want, "C21 ({}, {}) wrong", 4 + i, j);
        }
    }
    // Verify C22 (rows 4..8, cols 128..256) has diagonal 1s.
    for i in 0..4 {
        for j in 128..256 {
            let want = if i + 128 == j { 1 } else { 0 };
            assert_eq!(parent.get(4 + i, j), want, "C22 ({}, {}) wrong", 4 + i, j);
        }
    }
}

#[test]
fn view_extract_bits_matches_parent() {
    let parent = Mzd::random(5, 200, 0xABCDEF);
    let v = parent.view(1, 64, 4, 192);
    // Extract a slice via the view; compare to parent's extract_bits
    // at the equivalent (row, col) location.
    for k in &[1usize, 8, 31, 63] {
        for col_start in &[0usize, 1, 7, 56, 63] {
            if col_start + k > v.cols() {
                continue;
            }
            for row in 0..v.rows() {
                let a = v.extract_bits(row, *col_start, *k);
                let b = parent.extract_bits(1 + row, 64 + col_start, *k);
                assert_eq!(
                    a, b,
                    "extract_bits mismatch at ({}, {}, k={})",
                    row, col_start, k
                );
            }
        }
    }
}

#[test]
fn as_view_and_reborrow_compose() {
    // Sanity check: building a mut view, taking an immutable as_view,
    // and reading through it gives the same bits as the parent.
    let mut parent = Mzd::random(8, 128, 1);
    let snapshot = parent.to_byte_buffer();
    let mut v_mut = parent.view_mut(2, 0, 6, 128);
    {
        let v_ro: MzdView<'_> = v_mut.as_view();
        for i in 0..v_ro.rows() {
            for j in 0..v_ro.cols() {
                assert_eq!(v_ro.get(i, j), snapshot[(2 + i) * 128 + j]);
            }
        }
    }
    // Reborrow yields a fresh mut view with a shorter lifetime — what
    // Strassen passes to a recursive call without surrendering its own
    // ownership.
    {
        let mut reborrowed = v_mut.reborrow();
        reborrowed.set(0, 0, 1);
    }
    assert_eq!(parent.get(2, 0), 1, "reborrow write didn't propagate");
}
