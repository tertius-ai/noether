// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Elementary F_2 primitives — row/column swap and row XOR.
//!
//! The matrix-level XOR-into (`xor_into`) and cross-matrix
//! `xor_row_from` live on the [`MatrixWrite`](crate::view::MatrixWrite)
//! trait so they work uniformly over `Mzd`, `MzdView`, and
//! `MzdViewMut`. This file keeps the two row-ops that are intrinsically
//! within a single matrix (and so don't fit the cross-matrix trait
//! shape): `xor_row_self` and `swap_rows`.

use crate::mzd::Mzd;

/// XOR one equally sized packed word slice into another using the runtime
/// dispatched local kernel.
///
/// This is a low-level building block for other `m4rst` family crates whose
/// storage is still an ordinary packed `Mzd` once data reaches the local rank.
#[doc(hidden)]
#[inline]
pub fn xor_words(destination: &mut [u64], source: &[u64]) {
    assert_eq!(
        destination.len(),
        source.len(),
        "xor_words: packed slices must have equal length"
    );
    let _ = crate::simd::combine_n_into(destination, [source]);
}

impl Mzd {
    /// Copy one column from another equally tall packed matrix.
    ///
    /// This is a low-level building block for distributed callers whose
    /// source and destination columns live in different rank-local `Mzd`
    /// tiles. Only `destination_col` is modified.
    #[doc(hidden)]
    #[inline]
    pub fn copy_col_from(&mut self, destination_col: usize, source: &Self, source_col: usize) {
        assert_eq!(
            self.rows(),
            source.rows(),
            "copy_col_from: matrices must have equal row counts"
        );
        assert!(
            destination_col < self.cols(),
            "copy_col_from: destination column is out of bounds"
        );
        assert!(
            source_col < source.cols(),
            "copy_col_from: source column is out of bounds"
        );
        let destination_word = destination_col / 64;
        let destination_bit = destination_col % 64;
        let source_word = source_col / 64;
        let source_bit = source_col % 64;
        let destination_mask = 1_u64 << destination_bit;
        for row in 0..self.rows() {
            let value = (source.row_words(row)[source_word] >> source_bit) & 1;
            let word = &mut self.row_words_mut(row)[destination_word];
            *word = (*word & !destination_mask) | (value << destination_bit);
        }
    }

    /// XOR row `src_row` into row `dst_row` of `self`. Both within the
    /// same matrix. Standard elimination row-op; no-op if
    /// `src_row == dst_row` is forbidden (it would zero the row, never
    /// what callers want).
    #[inline]
    pub fn xor_row_self(&mut self, dst_row: usize, src_row: usize) {
        debug_assert!(dst_row < self.rows());
        debug_assert!(src_row < self.rows());
        debug_assert_ne!(dst_row, src_row, "xor_row_self: dst == src zeros the row");
        let w = self.width();
        let words = self.as_words_mut();
        let (lo, hi) = if dst_row < src_row {
            (dst_row, src_row)
        } else {
            (src_row, dst_row)
        };
        let (left, right) = words.split_at_mut(hi * w);
        let lo_slice = &mut left[lo * w..(lo + 1) * w];
        let hi_slice = &mut right[..w];
        let (dst_slice, src_slice) = if dst_row < src_row {
            (lo_slice, &*hi_slice)
        } else {
            (hi_slice, &*lo_slice)
        };
        xor_words(dst_slice, src_slice);
    }

    /// XOR row `src_row` into row `dst_row` starting at column
    /// `col_start` (inclusive), bounded by word index `word_end`
    /// (exclusive). Bits before `col_start` in `dst_row` are left
    /// untouched; bits in `dst_row` at and after column
    /// `word_end * 64` are also left untouched. Equivalent to
    /// libm4ri's `mzd_row_add_offset` with a temporary
    /// width-limiting window.
    ///
    /// Used by the M4R-block PLE kernel to clear column-prefixes
    /// of rows during the in-strip elimination phase, while
    /// preserving the lower-triangular `L` portion of the
    /// transformation matrix encoded in the bits below the pivot.
    #[inline]
    pub(crate) fn xor_row_offset_self(
        &mut self,
        dst_row: usize,
        src_row: usize,
        col_start: usize,
        word_end: usize,
    ) {
        debug_assert!(dst_row < self.rows());
        debug_assert!(src_row < self.rows());
        debug_assert_ne!(
            dst_row, src_row,
            "xor_row_offset_self: dst == src zeros the row"
        );
        let w = self.width();
        debug_assert!(word_end <= w);
        let wi = col_start / 64;
        if wi >= word_end {
            return;
        }
        let bit = col_start % 64;
        let head_mask: u64 = if bit == 0 {
            !0u64
        } else {
            !((1u64 << bit) - 1)
        };

        let words = self.as_words_mut();
        let (lo, hi) = if dst_row < src_row {
            (dst_row, src_row)
        } else {
            (src_row, dst_row)
        };
        let (left, right) = words.split_at_mut(hi * w);
        let lo_slice = &mut left[lo * w..(lo + 1) * w];
        let hi_slice = &mut right[..w];
        let (dst_slice, src_slice) = if dst_row < src_row {
            (lo_slice, &*hi_slice)
        } else {
            (hi_slice, &*lo_slice)
        };
        // First word: masked
        dst_slice[wi] ^= src_slice[wi] & head_mask;
        // Remaining words within [wi+1, word_end): full XOR
        xor_words(
            &mut dst_slice[wi + 1..word_end],
            &src_slice[wi + 1..word_end],
        );
    }

    /// Swap rows `i` and `j` in place. O(width) word swap, no
    /// allocation.
    #[inline]
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.rows());
        debug_assert!(j < self.rows());
        if i == j {
            return;
        }
        let w = self.width();
        let words = self.as_words_mut();
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let (left, right) = words.split_at_mut(hi * w);
        let a = &mut left[lo * w..(lo + 1) * w];
        let b = &mut right[..w];
        for k in 0..w {
            std::mem::swap(&mut a[k], &mut b[k]);
        }
    }

    /// Swap columns `i` and `j` in place.
    ///
    /// The operation walks each packed row once and exchanges the two bits
    /// without unpacking the matrix. It is useful to full-pivoting
    /// factorization in both local and distributed-memory callers.
    #[inline]
    pub fn swap_cols(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.cols());
        debug_assert!(j < self.cols());
        if i == j {
            return;
        }
        let i_word = i / 64;
        let i_bit = i % 64;
        let j_word = j / 64;
        let j_bit = j % 64;
        for row in 0..self.rows() {
            let words = self.row_words_mut(row);
            let different = ((words[i_word] >> i_bit) ^ (words[j_word] >> j_bit)) & 1;
            words[i_word] ^= different << i_bit;
            words[j_word] ^= different << j_bit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::xor_words;
    use crate::Mzd;

    #[test]
    fn xor_words_matches_scalar_across_vector_blocks_and_tails() {
        for len in [0, 1, 3, 4, 5, 17, 64, 259] {
            let mut actual: Vec<_> = (0..len)
                .map(|index| (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect();
            let source: Vec<_> = (0..len)
                .map(|index| (index as u64).rotate_left((index % 64) as u32))
                .collect();
            let expected: Vec<_> = actual
                .iter()
                .zip(&source)
                .map(|(destination, source)| destination ^ source)
                .collect();
            xor_words(&mut actual, &source);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn swap_cols_matches_bit_reference_across_words() {
        let original = Mzd::random(17, 193, 0xC011);
        for &(first, second) in &[(0, 0), (0, 63), (3, 64), (64, 192), (127, 128)] {
            let mut actual = original.clone();
            actual.swap_cols(first, second);
            for row in 0..original.rows() {
                for col in 0..original.cols() {
                    let source_col = if col == first {
                        second
                    } else if col == second {
                        first
                    } else {
                        col
                    };
                    assert_eq!(actual.get(row, col), original.get(row, source_col));
                }
            }
        }
    }

    #[test]
    fn copy_col_from_matches_bit_reference_and_preserves_other_columns() {
        let source = Mzd::random(17, 129, 0x50_0ACE);
        let original = Mzd::random(17, 193, 0xD357);
        for &(destination_col, source_col) in &[(0, 128), (63, 64), (64, 3), (192, 127)] {
            let mut actual = original.clone();
            actual.copy_col_from(destination_col, &source, source_col);
            for row in 0..actual.rows() {
                for col in 0..actual.cols() {
                    let expected = if col == destination_col {
                        source.get(row, source_col)
                    } else {
                        original.get(row, col)
                    };
                    assert_eq!(actual.get(row, col), expected);
                }
            }
        }
    }
}
