// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Matrix transpose over F_2.
//!
//! Tile-walks 64×64 bit-blocks via the internal `transpose64`
//! primitive (Warren's Hacker's Delight Ch. 7 5-pass shuffle —
//! lives at `src/transpose64.rs`, `pub(crate)`). For an `m × n`
//! matrix this is `ceil(m/64) · ceil(n/64)` block-transposes plus
//! boundary handling at the right and bottom edges, so the work
//! scales as `m · n / 64` word-ops rather than the bit-by-bit
//! kernel's `m · n` bit-ops.
//!
//! ## References
//!
//! * Warren, *Hacker's Delight* 2e, Ch. 7 § "Transposing a Bit Matrix"
//!   — the recursive bit-block transpose this is built on. The
//!   per-tile kernel lives in `src/transpose64.rs`.
//! * Knuth, *The Art of Computer Programming*, Vol. 4A,
//!   pre-fascicle 1B § 7.1.3.
//! * libm4ri's `mzd_transpose` in `mzd.c` (+ the tile-boundary
//!   handlers `_mzd_copy_transpose_lt64x64` / `_mzd_copy_transpose_64xlt64`
//!   at lines 471-700) is the engineering reference.

use crate::mzd::Mzd;
use crate::transpose64::transpose_64x64_inplace;

impl Mzd {
    /// Out-of-place transpose. Returns a new `cols × rows` matrix.
    ///
    /// Walks the input in 64×64 bit-tiles; each tile is transposed
    /// via the internal `transpose_64x64_inplace` primitive
    /// (Warren's Hacker's Delight Ch. 7 5-pass shuffle, 6 passes
    /// total). Boundary tiles at the right and bottom pad into a
    /// zeroed buffer and write back only the valid cells, so the
    /// kernel is general over any `m × n` (including degenerate
    /// 0-row / 0-col shapes).
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(8, 17, 1);
    /// assert_eq!(a.transpose().transpose().as_words(), a.as_words());
    /// ```
    pub fn transpose(&self) -> Mzd {
        let m = self.rows();
        let n = self.cols();
        let mut out = Mzd::new(n, m);
        if m == 0 || n == 0 {
            return out;
        }

        let dst_width = out.width();

        let n_tile_rows = m.div_ceil(64);
        let n_tile_cols = n.div_ceil(64);

        let mut buf = [0u64; 64];

        for tr in 0..n_tile_rows {
            let row_start = tr * 64;
            let r_count = std::cmp::min(64, m - row_start);

            for tc in 0..n_tile_cols {
                let col_start = tc * 64;
                let c_count = std::cmp::min(64, n - col_start);

                // Load `r_count` rows from src into buf[0..r_count].
                // Past `r_count` stays zero; past `c_count` within
                // each loaded word is zero by Mzd's
                // bits-past-valid-cols-are-zero invariant.
                buf.fill(0);
                for (i, slot) in buf.iter_mut().enumerate().take(r_count) {
                    *slot = self.row_words(row_start + i)[tc];
                }

                transpose_64x64_inplace(&mut buf);

                // After transpose, bit `i` of `buf[c]` is the
                // original bit at (i, c). Write `buf[c]` to word
                // `tr` of dst row `col_start + c` for c in
                // [0, c_count). Past `c_count`, no valid dst row.
                for (c, &word) in buf.iter().enumerate().take(c_count) {
                    out.row_words_mut(col_start + c)[tr] = word;
                }
            }
        }

        // Defence in depth: keep dst's high-bit invariant. Mask out
        // any bits in the last word of each row beyond `m` cols.
        // Required when m % 64 != 0 — buf[c] in that case may hold
        // src bits sourced from columns beyond `n`, which under the
        // invariant should be zero, but a writer bypassing the bit
        // accessor (e.g. test code mutating row_words_mut directly)
        // could plant non-zero bits there. This guards against
        // that path without slowing the hot loop.
        let m_tail = m % 64;
        if m_tail != 0 && dst_width != 0 {
            let mask = (1u64 << m_tail) - 1;
            let high_word = dst_width - 1;
            for r in 0..n {
                out.row_words_mut(r)[high_word] &= mask;
            }
        }

        out
    }
}
