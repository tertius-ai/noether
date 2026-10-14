// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bit-packed dense F_2 matrix — Rust port of M4RI's `mzd_t`.
//!
//! ## Layout
//!
//! Rows are stored consecutively in a flat `Vec<u64>` of length
//! `rows * width`, where `width = ceil(cols / 64)`. Row `r` starts at
//! word offset `r * width`. Within a row, column `c` lives in word
//! `c / 64` at bit position `c % 64`, **LSB-first** — i.e. bit 0 of a
//! word is the lowest-indexed column in that word's 64-column slice.
//! This matches `mzd_t`'s convention exactly (see `m4ri/misc.h`,
//! macros `__M4RI_GET_BIT` / `__M4RI_WRITE_BIT`).
//!
//! When `cols % 64 != 0`, the last word of each row carries "excess
//! bits" beyond column `cols - 1`. For a (non-windowed) matrix the
//! M4RI invariant is that those excess bits are zero, and we uphold
//! that invariant on every write. Callers can rely on it for fast
//! row-XOR loops that don't have to mask.
//!
//! Windowed (zero-copy submatrix) views are not yet implemented —
//! they land alongside the first kernel that needs them (M4RM /
//! Strassen). The current type owns its backing storage.

/// Number of F_2 elements packed into one `u64` word.
pub const RADIX: usize = 64;

/// Failure to size or reserve storage for an owned packed matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MzdAllocationError {
    /// The packed word count overflowed `usize`.
    SizeOverflow,
    /// The allocator rejected a reservation for the requested word count.
    AllocationFailed {
        /// Number of packed `u64` words requested.
        requested_words: usize,
    },
}

impl std::fmt::Display for MzdAllocationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeOverflow => write!(formatter, "packed matrix size overflows usize"),
            Self::AllocationFailed { requested_words } => write!(
                formatter,
                "allocator rejected {requested_words} packed matrix words"
            ),
        }
    }
}

impl std::error::Error for MzdAllocationError {}

/// Dense F_2 matrix, bit-packed `u64`-per-word, row-major.
///
/// Owns its backing storage. See module docs for the bit layout.
#[derive(Clone, Debug)]
pub struct Mzd {
    rows: usize,
    cols: usize,
    /// Number of u64 words per row (== `ceil(cols / 64)`, or `0` when `cols == 0`).
    width: usize,
    /// Mask for valid bits in the high word of each row.
    /// Equal to `u64::MAX` when `cols % 64 == 0` (and when `cols == 0`,
    /// `width == 0` so the mask is never consulted).
    high_bitmask: u64,
    data: Vec<u64>,
}

impl Mzd {
    /// Build an owned `Mzd` by copying from any
    /// [`MatrixRead`](crate::view::MatrixRead). For
    /// `&Mzd` this is `clone`; for `&MzdView` / `&MzdViewMut` it
    /// materialises the (possibly column-strided) view into a
    /// row-major-contiguous matrix. Used by the Strassen
    /// parallel-leaf path to feed view inputs to the rayon kernel
    /// (which requires contiguous backing for `par_chunks_mut`).
    pub fn from_matrix_read<R: crate::view::MatrixRead>(src: &R) -> Self {
        let mut out = Self::new(src.rows(), src.cols());
        let w = out.width;
        for r in 0..src.rows() {
            let s = src.row_words(r);
            let dst = &mut out.data[r * w..(r + 1) * w];
            dst.copy_from_slice(&s[..w]);
        }
        out
    }

    /// Construct an all-zero `rows × cols` matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::try_new(rows, cols).expect("Mzd::new: packed matrix allocation failed")
    }

    /// Fallibly construct an all-zero `rows × cols` matrix.
    pub fn try_new(rows: usize, cols: usize) -> Result<Self, MzdAllocationError> {
        let width = cols.div_ceil(RADIX);
        let high_bitmask = if cols.is_multiple_of(RADIX) {
            u64::MAX
        } else {
            (1u64 << (cols % RADIX)) - 1
        };
        let requested_words = rows
            .checked_mul(width)
            .ok_or(MzdAllocationError::SizeOverflow)?;
        let mut data = Vec::new();
        data.try_reserve_exact(requested_words)
            .map_err(|_| MzdAllocationError::AllocationFailed { requested_words })?;
        data.resize(requested_words, 0_u64);
        Ok(Self {
            rows,
            cols,
            width,
            high_bitmask,
            data,
        })
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of `u64` words per row.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Mask for valid bits in the high word of each row.
    #[inline]
    pub fn high_bitmask(&self) -> u64 {
        self.high_bitmask
    }

    /// Raw backing storage (length == `rows * width`).
    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.data
    }

    /// Mutable raw backing storage.
    #[inline]
    pub fn as_words_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Words backing row `r` (length == `width`).
    #[inline]
    pub fn row_words(&self, r: usize) -> &[u64] {
        let start = r * self.width;
        &self.data[start..start + self.width]
    }

    /// Mutable words backing row `r` (length == `width`).
    #[inline]
    pub fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        let start = r * self.width;
        &mut self.data[start..start + self.width]
    }

    /// Read bit at `(row, col)`. Returns 0 or 1.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        let word = self.data[row * self.width + col / RADIX];
        ((word >> (col % RADIX)) & 1) as u8
    }

    /// Write bit at `(row, col)`. `value` is masked to `value & 1`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: u8) {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        let idx = row * self.width + col / RADIX;
        let shift = col % RADIX;
        let v = (value as u64) & 1;
        let w = &mut self.data[idx];
        *w = (*w & !(1u64 << shift)) | (v << shift);
    }

    /// Flip bit at `(row, col)`.
    #[inline]
    pub fn flip(&mut self, row: usize, col: usize) {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        let idx = row * self.width + col / RADIX;
        let shift = col % RADIX;
        self.data[idx] ^= 1u64 << shift;
    }

    /// Return the first nonzero column in `row` at or after `col_start`.
    ///
    /// Scans packed words rather than testing individual bits. This is the
    /// local pivot-search primitive used by factorization code, including
    /// consumers that distribute ordinary `Mzd` row blocks across ranks.
    #[must_use]
    pub fn first_one_in_row(&self, row: usize, col_start: usize) -> Option<usize> {
        debug_assert!(row < self.rows);
        debug_assert!(col_start <= self.cols);
        if col_start == self.cols {
            return None;
        }

        let first_word = col_start / RADIX;
        let first_bit = col_start % RADIX;
        let last_word = self.width - 1;
        for word_index in first_word..self.width {
            let mut word = self.row_words(row)[word_index];
            if word_index == first_word && first_bit != 0 {
                word &= u64::MAX << first_bit;
            }
            if word_index == last_word {
                word &= self.high_bitmask;
            }
            if word != 0 {
                return Some(word_index * RADIX + word.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Copy out the `(r1 - r0) × (c1 - c0)` rectangle at `(r0, c0)` as a
    /// freshly-allocated owned `Mzd`. Half-open ranges (`r0 ≤ r < r1`,
    /// `c0 ≤ c < c1`).
    ///
    /// Used by Strassen's block split. Word-aligned columns use a word-stride
    /// copy and other offsets use bit extraction.
    pub fn submatrix(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> Mzd {
        self.try_submatrix(r0, c0, r1, c1)
            .expect("Mzd::submatrix: packed matrix allocation failed")
    }

    /// Fallible counterpart to [`Mzd::submatrix`].
    pub fn try_submatrix(
        &self,
        r0: usize,
        c0: usize,
        r1: usize,
        c1: usize,
    ) -> Result<Mzd, MzdAllocationError> {
        assert!(r0 <= r1 && r1 <= self.rows);
        assert!(c0 <= c1 && c1 <= self.cols);
        let rows = r1 - r0;
        let cols = c1 - c0;
        let mut out = Mzd::try_new(rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        // Fast path: source column offset is word-aligned, so we copy
        // whole words. The common Strassen case (N a multiple of 128)
        // hits this path at every recursion level.
        if c0.is_multiple_of(RADIX) {
            let w_out = out.width;
            let src_word_off = c0 / RADIX;
            let cols_in_last = cols % RADIX;
            for i in 0..rows {
                let src_start = (r0 + i) * self.width + src_word_off;
                let dst_start = i * w_out;
                for w in 0..w_out {
                    out.data[dst_start + w] = self.data[src_start + w];
                }
                if cols_in_last != 0 {
                    let mask = (1u64 << cols_in_last) - 1;
                    out.data[dst_start + w_out - 1] &= mask;
                }
            }
            return Ok(out);
        }
        // General packed path: stitch adjacent source words when the requested
        // rectangle begins between word boundaries.
        let src_word_off = c0 / RADIX;
        let bit_off = c0 % RADIX;
        for i in 0..rows {
            let src_start = (r0 + i) * self.width + src_word_off;
            let dst_start = i * out.width;
            for w in 0..out.width {
                let mut value = self.data[src_start + w] >> bit_off;
                if src_word_off + w + 1 < self.width {
                    value |= self.data[src_start + w + 1] << (RADIX - bit_off);
                }
                out.data[dst_start + w] = value;
            }
            let last = dst_start + out.width - 1;
            out.data[last] &= out.high_bitmask;
        }
        Ok(out)
    }

    /// Write `src` into `self` at top-left `(r0, c0)`. The block must
    /// fit. Inverse of [`Mzd::submatrix`].
    pub fn paste_into(&mut self, r0: usize, c0: usize, src: &Mzd) {
        assert!(r0 + src.rows <= self.rows);
        assert!(c0 + src.cols <= self.cols);
        if src.rows == 0 || src.cols == 0 {
            return;
        }
        if c0.is_multiple_of(RADIX) {
            // Word-aligned destination — paste whole words, but be
            // careful about the destination's last word (which carries
            // bits beyond src's columns and must be preserved).
            let dst_word_off = c0 / RADIX;
            let cols_in_last = src.cols % RADIX;
            for i in 0..src.rows {
                let src_start = i * src.width;
                let dst_start = (r0 + i) * self.width + dst_word_off;
                let full_words = if cols_in_last == 0 {
                    src.width
                } else {
                    src.width - 1
                };
                for w in 0..full_words {
                    self.data[dst_start + w] = src.data[src_start + w];
                }
                if cols_in_last != 0 {
                    let mask = (1u64 << cols_in_last) - 1;
                    let target_word = &mut self.data[dst_start + src.width - 1];
                    *target_word =
                        (*target_word & !mask) | (src.data[src_start + src.width - 1] & mask);
                }
            }
            return;
        }
        // General packed path. Walk destination words and merge only the
        // covered bit interval, preserving the prefix/suffix outside `src`.
        let paste_end = c0 + src.cols;
        let first_word = c0 / RADIX;
        let last_word = (paste_end - 1) / RADIX;
        for i in 0..src.rows {
            let dst_row = (r0 + i) * self.width;
            for destination_word in first_word..=last_word {
                let word_start = destination_word * RADIX;
                let overlap_start = word_start.max(c0);
                let overlap_end = (word_start + RADIX).min(paste_end);
                let count = overlap_end - overlap_start;
                let destination_shift = overlap_start - word_start;
                let source_col = overlap_start - c0;
                let bits = src.extract_bits(i, source_col, count);
                let low_mask = if count == RADIX {
                    u64::MAX
                } else {
                    (1_u64 << count) - 1
                };
                let mask = low_mask << destination_shift;
                let destination = &mut self.data[dst_row + destination_word];
                *destination = (*destination & !mask) | ((bits << destination_shift) & mask);
            }
        }
    }

    /// Extract `k` consecutive bits at `(row, col_start)` into a `u64`,
    /// LSB-first to match the stored bit order: bit 0 of the result is
    /// `self.get(row, col_start)`, bit 1 is `self.get(row, col_start+1)`,
    /// and so on. Requires `1 ≤ k ≤ 63` and the range to be in bounds.
    ///
    /// Used by M4RM (table index) and the M4R elimination kernels.
    #[inline]
    pub fn extract_bits(&self, row: usize, col_start: usize, k: usize) -> u64 {
        debug_assert!((1..=RADIX).contains(&k));
        debug_assert!(row < self.rows);
        debug_assert!(col_start + k <= self.cols);
        let row_start = row * self.width;
        let w_idx = col_start / RADIX;
        let bit_off = col_start % RADIX;
        // mask = (1 << k) - 1 with k=64 underflow guarded.
        let mask = if k == RADIX {
            u64::MAX
        } else {
            (1u64 << k) - 1
        };
        let low = self.data[row_start + w_idx] >> bit_off;
        if bit_off + k <= RADIX {
            // Fully inside one word. For bit_off=0, k=64, the shift `>> 0`
            // is a no-op and we return word & u64::MAX = whole word.
            low & mask
        } else {
            // Spans two words: stitch in the high half from word w_idx + 1.
            let high = self.data[row_start + w_idx + 1] << (RADIX - bit_off);
            (low | high) & mask
        }
    }

    /// `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::identity_window(0, n, 0, n)
    }

    /// Construct a rectangular window of an identity matrix.
    ///
    /// The half-open row and column ranges use coordinates in the same
    /// conceptual square matrix. This is a low-level building block for
    /// callers that store an identity matrix as independent local tiles.
    #[doc(hidden)]
    pub fn identity_window(
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        Self::try_identity_window(row_start, row_end, col_start, col_end)
            .expect("Mzd::identity_window: packed matrix allocation failed")
    }

    /// Fallible counterpart to [`Mzd::identity_window`].
    #[doc(hidden)]
    pub fn try_identity_window(
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Self, MzdAllocationError> {
        assert!(
            row_start <= row_end,
            "Mzd::identity_window: invalid row range"
        );
        assert!(
            col_start <= col_end,
            "Mzd::identity_window: invalid column range"
        );
        let mut matrix = Self::try_new(row_end - row_start, col_end - col_start)?;
        for diagonal in row_start.max(col_start)..row_end.min(col_end) {
            matrix.set(diagonal - row_start, diagonal - col_start, 1);
        }
        Ok(matrix)
    }

    /// Construct from a row-major byte buffer holding one F_2 element per
    /// byte. The value of each cell is taken from the byte's **low bit**
    /// (`byte & 1`), matching the engine's `M4rstStorage::from_bytes`
    /// convention. Higher bits in each byte are ignored.
    ///
    /// `bytes.len()` must equal `rows * cols`.
    pub fn from_byte_buffer(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            rows.checked_mul(cols)
                .expect("Mzd::from_byte_buffer: rows * cols overflows usize"),
            "Mzd::from_byte_buffer: bytes.len() must equal rows * cols"
        );
        Self::from_strided_byte_buffer(rows, cols, cols, bytes)
    }

    /// Construct from a row-major byte window with an explicit row stride.
    ///
    /// Each cell is read from the low bit of its byte. `bytes` may contain
    /// data after the requested window, but must cover `rows` rows of `cols`
    /// bytes whose starts are `row_stride` bytes apart.
    #[doc(hidden)]
    pub fn from_strided_byte_buffer(
        rows: usize,
        cols: usize,
        row_stride: usize,
        bytes: &[u8],
    ) -> Self {
        Self::try_from_strided_byte_buffer(rows, cols, row_stride, bytes)
            .expect("Mzd::from_strided_byte_buffer: packed matrix allocation failed")
    }

    /// Fallible counterpart to [`Mzd::from_strided_byte_buffer`].
    #[doc(hidden)]
    pub fn try_from_strided_byte_buffer(
        rows: usize,
        cols: usize,
        row_stride: usize,
        bytes: &[u8],
    ) -> Result<Self, MzdAllocationError> {
        assert!(
            rows == 0 || row_stride >= cols,
            "Mzd::from_strided_byte_buffer: row stride is shorter than a row"
        );
        let required = if rows == 0 {
            0
        } else {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|prefix| prefix.checked_add(cols))
                .expect("Mzd::from_strided_byte_buffer: byte window size overflows usize")
        };
        assert!(
            bytes.len() >= required,
            "Mzd::from_strided_byte_buffer: byte buffer is too short"
        );
        let mut m = Self::try_new(rows, cols)?;
        if m.width == 0 || rows == 0 {
            return Ok(m);
        }
        for r in 0..rows {
            let row_start = r * row_stride;
            let row_bytes = &bytes[row_start..row_start + cols];
            let row = m.row_words_mut(r);
            // Pack 64 bytes at a time into one u64, LSB-first.
            let full_words = cols / RADIX;
            for w in 0..full_words {
                let chunk = &row_bytes[w * RADIX..(w + 1) * RADIX];
                let mut acc: u64 = 0;
                for (i, &b) in chunk.iter().enumerate() {
                    acc |= ((b as u64) & 1) << i;
                }
                row[w] = acc;
            }
            let tail = cols % RADIX;
            if tail != 0 {
                let chunk = &row_bytes[full_words * RADIX..];
                let mut acc: u64 = 0;
                for (i, &b) in chunk.iter().enumerate() {
                    acc |= ((b as u64) & 1) << i;
                }
                // Tail bits already fit inside high_bitmask by construction.
                row[full_words] = acc;
            }
        }
        Ok(m)
    }

    /// Inverse of [`Mzd::from_byte_buffer`]. Produces a row-major
    /// `rows * cols` byte buffer where each byte is 0 or 1.
    pub fn to_byte_buffer(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.cols];
        if self.width == 0 || self.rows == 0 {
            return out;
        }
        for r in 0..self.rows {
            let row = self.row_words(r);
            let row_bytes = &mut out[r * self.cols..(r + 1) * self.cols];
            let full_words = self.cols / RADIX;
            for w in 0..full_words {
                let word = row[w];
                let chunk = &mut row_bytes[w * RADIX..(w + 1) * RADIX];
                for (i, b) in chunk.iter_mut().enumerate() {
                    *b = ((word >> i) & 1) as u8;
                }
            }
            let tail = self.cols % RADIX;
            if tail != 0 {
                let word = row[full_words];
                let chunk = &mut row_bytes[full_words * RADIX..];
                for (i, b) in chunk.iter_mut().enumerate() {
                    *b = ((word >> i) & 1) as u8;
                }
            }
        }
        out
    }

    /// Pseudorandom `rows × cols` matrix. Reproducible from `seed`.
    ///
    /// Uses an index-addressable SplitMix64 stream. It is not cryptographic,
    /// but is deterministic for a given seed so test failures replay exactly
    /// and distributed tiles can be generated without materializing or
    /// advancing through the preceding words.
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        Self::try_random(rows, cols, seed).expect("Mzd::random: packed matrix allocation failed")
    }

    /// Fallible counterpart to [`Mzd::random`].
    pub fn try_random(rows: usize, cols: usize, seed: u64) -> Result<Self, MzdAllocationError> {
        Self::try_random_word_aligned_submatrix(rows, cols, 0, rows, 0, cols, seed)
    }

    /// Generate a word-aligned submatrix of the deterministic random matrix
    /// described by `global_rows`, `global_cols`, and `seed`.
    ///
    /// This produces exactly the same value as
    /// `Mzd::random(global_rows, global_cols, seed).submatrix(...)` without
    /// materializing the full matrix. `col_start` must be word-aligned; the
    /// other bounds may be arbitrary. It is primarily a low-level hook for
    /// distributed storage, where each rank constructs only its owned tiles.
    #[doc(hidden)]
    pub fn random_word_aligned_submatrix(
        global_rows: usize,
        global_cols: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        seed: u64,
    ) -> Self {
        Self::try_random_word_aligned_submatrix(
            global_rows,
            global_cols,
            row_start,
            row_end,
            col_start,
            col_end,
            seed,
        )
        .expect("Mzd::random_word_aligned_submatrix: packed matrix allocation failed")
    }

    /// Fallible counterpart to [`Mzd::random_word_aligned_submatrix`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_random_word_aligned_submatrix(
        global_rows: usize,
        global_cols: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        seed: u64,
    ) -> Result<Self, MzdAllocationError> {
        assert!(row_start <= row_end && row_end <= global_rows);
        assert!(col_start <= col_end && col_end <= global_cols);
        assert!(
            col_start.is_multiple_of(RADIX),
            "Mzd::random_word_aligned_submatrix: col_start must be word-aligned"
        );

        let mut matrix = Self::try_new(row_end - row_start, col_end - col_start)?;
        if matrix.width == 0 || matrix.rows == 0 {
            return Ok(matrix);
        }

        let global_width = global_cols.div_ceil(RADIX);
        let col_word_start = col_start / RADIX;
        let word_offset = row_start
            .checked_mul(global_width)
            .and_then(|offset| offset.checked_add(col_word_start))
            .ok_or(MzdAllocationError::SizeOverflow)?;
        let mut row_word_offset = word_offset;
        let width = matrix.width;
        let high_bitmask = matrix.high_bitmask;

        for local_row in 0..matrix.rows {
            for (local_word, word) in matrix.row_words_mut(local_row).iter_mut().enumerate() {
                let global_word = row_word_offset
                    .checked_add(local_word)
                    .ok_or(MzdAllocationError::SizeOverflow)?;
                *word = splitmix64_word(seed, global_word);
            }
            matrix.row_words_mut(local_row)[width - 1] &= high_bitmask;
            if local_row + 1 < matrix.rows {
                row_word_offset = row_word_offset
                    .checked_add(global_width)
                    .ok_or(MzdAllocationError::SizeOverflow)?;
            }
        }
        Ok(matrix)
    }
}

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64_word(seed: u64, word_offset: usize) -> u64 {
    let counter = u64::try_from(word_offset)
        .expect("Mzd::random_word_aligned_submatrix: word offset exceeds u64")
        .checked_add(1)
        .expect("Mzd::random_word_aligned_submatrix: word offset exceeds u64");
    let mut value = seed.wrapping_add(counter.wrapping_mul(SPLITMIX64_GAMMA));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{Mzd, MzdAllocationError};

    #[test]
    fn fallible_construction_reports_size_overflow() {
        assert!(matches!(
            Mzd::try_new(usize::MAX, 65),
            Err(MzdAllocationError::SizeOverflow)
        ));
        assert_eq!(Mzd::try_new(usize::MAX, 0).unwrap().as_words(), &[]);
    }

    #[test]
    fn random_word_aligned_submatrices_match_materialized_matrix() {
        for &(rows, cols, seed) in &[
            (0, 0, 0),
            (1, 1, 1),
            (7, 193, 0),
            (11, 257, 0xDEAD_BEEF),
            (129, 1089, 0x1729),
        ] {
            let full = Mzd::random(rows, cols, seed);
            let mut bounds = vec![(0, rows, 0, cols)];
            if rows > 2 && cols > 128 {
                bounds.extend([
                    (1, rows - 1, 0, 64),
                    (0, rows.min(3), 64, cols.min(193)),
                    (rows / 2, rows, 128, cols),
                ]);
            }
            for (row_start, row_end, col_start, col_end) in bounds {
                let actual = Mzd::random_word_aligned_submatrix(
                    rows, cols, row_start, row_end, col_start, col_end, seed,
                );
                let expected = full.submatrix(row_start, col_start, row_end, col_end);
                assert_eq!(actual.as_words(), expected.as_words());
            }
        }
    }

    #[test]
    fn random_stream_avoids_large_square_rank_collapse() {
        let matrix = Mzd::random(2048, 2048, 5929);
        assert_eq!(matrix.ple_rank(), 2048);
    }
}
