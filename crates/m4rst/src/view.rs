// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Zero-copy submatrix views — the substrate Strassen actually wins on.
//!
//! ## Why this exists
//!
//! AGENTS.md core invariant 5: *"Submatrix windows are zero-copy.
//! Strassen and M4RM both rely on this."* Without zero-copy views,
//! the 2×2-block split inside Strassen pays 8 × O(N²/4) copies per
//! recursion level, which is enough to make Strassen *lose* against
//! a vectorised M4RM at every realistic size. With views the splits
//! are just pointer arithmetic and Strassen's 7-vs-8 multiplications
//! saving actually shows up.
//!
//! ## The trait pair
//!
//! [`MatrixRead`] is the bag of methods every kernel needs from its
//! inputs: shape + word-stride row access. [`MatrixWrite`] extends
//! it with mutable row access for kernels that write a destination.
//!
//! Both [`Mzd`] and the view types in this module implement these
//! traits, so a single generic kernel covers owned matrices, borrowed
//! windows, and the Strassen 2×2 quadrants alike.
//!
//! ## Column-offset alignment
//!
//! Views are restricted to **word-aligned column offsets** (`c0 % 64
//! == 0`). This keeps row access a flat slice into the parent
//! storage — no bit shifting per word read — which is the whole
//! point. Strassen's 2×2 split lands on word-aligned boundaries iff
//! every dim is a multiple of 128; the Strassen dispatcher therefore
//! gates on that. Non-aligned cases fall through to M4RM via
//! `Mzd::matmul`'s top-level dispatch, where the alignment doesn't
//! matter.

use crate::mzd::{Mzd, RADIX};

// ---- read trait + view --------------------------------------------------

/// Read-only access to an F_2 bit-packed matrix. Implemented by
/// [`Mzd`], [`MzdView`], and [`MzdViewMut`].
pub trait MatrixRead {
    /// Number of rows.
    fn rows(&self) -> usize;
    /// Number of columns.
    fn cols(&self) -> usize;
    /// Number of `u64` words per row (== `ceil(cols / 64)`).
    fn width(&self) -> usize;
    /// Mask for the valid bits in the high word of each row.
    fn high_bitmask(&self) -> u64;
    /// Words backing row `r` (length == `width()`).
    fn row_words(&self, r: usize) -> &[u64];

    /// Read bit `(r, c)`. Default impl reads via `row_words`; no need
    /// to override.
    #[inline]
    fn get(&self, r: usize, c: usize) -> u8 {
        ((self.row_words(r)[c / RADIX] >> (c % RADIX)) & 1) as u8
    }

    /// Extract `k ≤ 64` consecutive bits at `(r, c)` as a `u64`,
    /// LSB-first. Mirrors [`Mzd::extract_bits`] but works through
    /// the trait. `k = 64` is needed by the NTABLES=8 M4RM kernel
    /// (kk = NTABLES · k = 64 bit-strip read).
    #[inline]
    fn extract_bits(&self, r: usize, col_start: usize, k: usize) -> u64 {
        debug_assert!((1..=RADIX).contains(&k));
        debug_assert!(col_start + k <= self.cols());
        let words = self.row_words(r);
        let w_idx = col_start / RADIX;
        let bit_off = col_start % RADIX;
        let mask = if k == RADIX {
            u64::MAX
        } else {
            (1u64 << k) - 1
        };
        let low = words[w_idx] >> bit_off;
        if bit_off + k <= RADIX {
            low & mask
        } else {
            let high = words[w_idx + 1] << (RADIX - bit_off);
            (low | high) & mask
        }
    }
}

/// Mutable access. Adds row-words mutation on top of [`MatrixRead`].
pub trait MatrixWrite: MatrixRead {
    /// Mutable words backing row `r`.
    fn row_words_mut(&mut self, r: usize) -> &mut [u64];

    /// Write bit `(r, c)` to `value & 1`.
    #[inline]
    fn set(&mut self, r: usize, c: usize, value: u8) {
        let shift = c % RADIX;
        let idx = c / RADIX;
        let v = (value as u64) & 1;
        let w = &mut self.row_words_mut(r)[idx];
        *w = (*w & !(1u64 << shift)) | (v << shift);
    }

    /// Overwrite `self` with `src` cell-for-cell. Like `xor_into` but
    /// assigning rather than accumulating. Used by Strassen to set
    /// up the input-sum scratches without first zeroing them.
    fn assign<R: MatrixRead>(&mut self, src: &R) {
        assert_eq!(self.rows(), src.rows(), "assign: row mismatch");
        assert_eq!(self.cols(), src.cols(), "assign: col mismatch");
        let w = self.width();
        debug_assert_eq!(w, src.width());
        for r in 0..self.rows() {
            let s = src.row_words(r);
            let dst = self.row_words_mut(r);
            dst[..w].copy_from_slice(&s[..w]);
        }
    }

    /// Zero every cell. For Strassen this fills the per-iteration M
    /// scratch between recursive calls.
    fn fill_zero(&mut self) {
        for r in 0..self.rows() {
            for word in self.row_words_mut(r).iter_mut() {
                *word = 0;
            }
        }
    }

    /// In-place element-wise XOR: `self ^= src`. Panics on shape
    /// mismatch. The generic over `MatrixRead` lets this take an
    /// `&Mzd`, an `&MzdView`, or an `&MzdViewMut` interchangeably.
    fn xor_into<R: MatrixRead>(&mut self, src: &R) {
        assert_eq!(self.rows(), src.rows(), "xor_into: row mismatch");
        assert_eq!(self.cols(), src.cols(), "xor_into: col mismatch");
        let w = self.width();
        debug_assert_eq!(w, src.width());
        for r in 0..self.rows() {
            let s = src.row_words(r);
            let dst = self.row_words_mut(r);
            crate::xor::xor_words(&mut dst[..w], &s[..w]);
        }
    }

    /// XOR a single row of `src` into row `dst_row` of `self`. Both
    /// must have equal column count (same word width). Inner loop of
    /// matmul + elimination.
    #[inline]
    fn xor_row_from<R: MatrixRead>(&mut self, dst_row: usize, src: &R, src_row: usize) {
        debug_assert_eq!(self.width(), src.width());
        debug_assert!(dst_row < self.rows());
        debug_assert!(src_row < src.rows());
        let s = src.row_words(src_row);
        let dst = self.row_words_mut(dst_row);
        let w = dst.len();
        crate::xor::xor_words(dst, &s[..w]);
    }
}

// ---- Mzd implements both ------------------------------------------------

impl MatrixRead for Mzd {
    #[inline]
    fn rows(&self) -> usize {
        Mzd::rows(self)
    }
    #[inline]
    fn cols(&self) -> usize {
        Mzd::cols(self)
    }
    #[inline]
    fn width(&self) -> usize {
        Mzd::width(self)
    }
    #[inline]
    fn high_bitmask(&self) -> u64 {
        Mzd::high_bitmask(self)
    }
    #[inline]
    fn row_words(&self, r: usize) -> &[u64] {
        Mzd::row_words(self, r)
    }
}

impl MatrixWrite for Mzd {
    #[inline]
    fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        Mzd::row_words_mut(self, r)
    }
}

// ---- MzdView (immutable borrow) -----------------------------------------

/// Immutable submatrix view: shape + (row_offset, col_word_offset,
/// parent_rowstride) tuple that aliases a parent `Mzd`'s storage.
#[derive(Clone, Copy)]
pub struct MzdView<'a> {
    rows: usize,
    cols: usize,
    width: usize,
    high_bitmask: u64,
    /// Words from one row of the parent to the next.
    parent_rowstride: usize,
    /// Index of the first row of the view in the parent.
    row_offset: usize,
    /// Index of the first u64 word in each row of the view in the
    /// parent's row layout. Always word-aligned (we panic otherwise
    /// in the constructor).
    col_word_offset: usize,
    data: &'a [u64],
}

impl<'a> MzdView<'a> {
    /// Construct a view over the parent's rows `[r0, r1)` and cols
    /// `[c0, c1)`. `c0` must be a multiple of 64.
    pub fn new(parent: &'a Mzd, r0: usize, c0: usize, r1: usize, c1: usize) -> Self {
        assert!(r0 <= r1 && r1 <= parent.rows());
        assert!(c0 <= c1 && c1 <= parent.cols());
        assert!(
            c0.is_multiple_of(RADIX),
            "MzdView: column offset {} is not word-aligned (must be multiple of {})",
            c0,
            RADIX
        );
        let rows = r1 - r0;
        let cols = c1 - c0;
        let width = cols.div_ceil(RADIX);
        let high_bitmask = if cols.is_multiple_of(RADIX) {
            u64::MAX
        } else {
            (1u64 << (cols % RADIX)) - 1
        };
        Self {
            rows,
            cols,
            width,
            high_bitmask,
            parent_rowstride: parent.width(),
            row_offset: r0,
            col_word_offset: c0 / RADIX,
            data: parent.as_words(),
        }
    }
}

impl<'a> MzdView<'a> {
    /// Re-slice this view into a sub-view of the same parent storage.
    /// `c0` must be a multiple of 64. Coordinates are relative to
    /// `self` (not the original parent).
    pub fn view(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> MzdView<'a> {
        assert!(r0 <= r1 && r1 <= self.rows);
        assert!(c0 <= c1 && c1 <= self.cols);
        assert!(
            c0.is_multiple_of(RADIX),
            "MzdView::view: c0 not word-aligned"
        );
        let rows = r1 - r0;
        let cols = c1 - c0;
        let width = cols.div_ceil(RADIX);
        let high_bitmask = if cols.is_multiple_of(RADIX) {
            u64::MAX
        } else {
            (1u64 << (cols % RADIX)) - 1
        };
        MzdView {
            rows,
            cols,
            width,
            high_bitmask,
            parent_rowstride: self.parent_rowstride,
            row_offset: self.row_offset + r0,
            col_word_offset: self.col_word_offset + c0 / RADIX,
            data: self.data,
        }
    }
}

impl<'a> MatrixRead for MzdView<'a> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn high_bitmask(&self) -> u64 {
        self.high_bitmask
    }
    #[inline]
    fn row_words(&self, r: usize) -> &[u64] {
        debug_assert!(r < self.rows);
        let start = (self.row_offset + r) * self.parent_rowstride + self.col_word_offset;
        &self.data[start..start + self.width]
    }
}

// ---- MzdViewMut (mutable borrow + disjoint split) -----------------------

/// Mutable submatrix view. The single-view case is borrow-checked the
/// normal way; the **4-quadrant disjoint split** ([`MzdViewMut::quad_split`])
/// uses `unsafe` internally to hand out four mutable sub-views of the
/// same underlying buffer.
///
/// We store a raw pointer + lifetime marker so the quadrant split can
/// build four disjoint views without each borrowing from `self`. Any
/// access goes through `row_words[_mut]` which validates the row
/// index in debug mode.
pub struct MzdViewMut<'a> {
    rows: usize,
    cols: usize,
    width: usize,
    high_bitmask: u64,
    parent_rowstride: usize,
    row_offset: usize,
    col_word_offset: usize,
    /// Raw pointer into the parent's backing storage, valid for the
    /// view's window. The `'a` lifetime ties us back to the parent's
    /// borrow.
    data: *mut u64,
    /// Length of `data` in u64 words (i.e., the parent buffer's
    /// length); used for debug-mode bounds checks.
    data_len: usize,
    _marker: std::marker::PhantomData<&'a mut [u64]>,
}

// Safety: `MzdViewMut`'s raw pointer is only ever dereferenced through
// disjoint row × col-word rectangles. The `quad_split` and `row_split`
// methods are the only ways to create multiple `MzdViewMut`s into the
// same parent, and both produce non-overlapping sub-rectangles by
// construction. Send is safe for the same reason any &mut into a
// disjoint slice is. We don't implement Sync — there's no API that
// shares a single `&MzdViewMut` across threads.
unsafe impl<'a> Send for MzdViewMut<'a> {}

impl<'a> MzdViewMut<'a> {
    /// Build a top-level view over the parent.
    pub fn new(parent: &'a mut Mzd, r0: usize, c0: usize, r1: usize, c1: usize) -> Self {
        assert!(r0 <= r1 && r1 <= parent.rows());
        assert!(c0 <= c1 && c1 <= parent.cols());
        assert!(
            c0.is_multiple_of(RADIX),
            "MzdViewMut: column offset {} not word-aligned",
            c0
        );
        let rows = r1 - r0;
        let cols = c1 - c0;
        let width = cols.div_ceil(RADIX);
        let high_bitmask = if cols.is_multiple_of(RADIX) {
            u64::MAX
        } else {
            (1u64 << (cols % RADIX)) - 1
        };
        let parent_rowstride = parent.width();
        let data_len = parent.as_words().len();
        let data = parent.as_words_mut().as_mut_ptr();
        Self {
            rows,
            cols,
            width,
            high_bitmask,
            parent_rowstride,
            row_offset: r0,
            col_word_offset: c0 / RADIX,
            data,
            data_len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Split into four equal-shape quadrants `(c11, c12, c21, c22)`
    /// at `(m2, n2)`. The four returned mut views tile `self` exactly
    /// — disjoint by construction.
    ///
    /// `n2` must be a multiple of 64.
    pub fn quad_split(
        self,
        m2: usize,
        n2: usize,
    ) -> (
        MzdViewMut<'a>,
        MzdViewMut<'a>,
        MzdViewMut<'a>,
        MzdViewMut<'a>,
    ) {
        assert!(m2 <= self.rows);
        assert!(n2 <= self.cols);
        assert!(
            n2.is_multiple_of(RADIX),
            "quad_split: n2 must be word-aligned"
        );

        let make_quadrant = |r0: usize, c0: usize, r1: usize, c1: usize| -> MzdViewMut<'a> {
            let rows = r1 - r0;
            let cols = c1 - c0;
            let width = cols.div_ceil(RADIX);
            let high_bitmask = if cols.is_multiple_of(RADIX) {
                u64::MAX
            } else {
                (1u64 << (cols % RADIX)) - 1
            };
            MzdViewMut {
                rows,
                cols,
                width,
                high_bitmask,
                parent_rowstride: self.parent_rowstride,
                row_offset: self.row_offset + r0,
                col_word_offset: self.col_word_offset + c0 / RADIX,
                // Safety: the four sub-rectangles tile `self` exactly
                // and we move `self` (consume it) so no other handle
                // to this region survives. Each sub-view holds rows
                // `row_offset + r0 .. row_offset + r1` and word
                // columns `col_word_offset + c0/64 ..
                // col_word_offset + c1/64`, which is disjoint for
                // distinct quadrants.
                data: self.data,
                data_len: self.data_len,
                _marker: std::marker::PhantomData,
            }
        };

        let c11 = make_quadrant(0, 0, m2, n2);
        let c12 = make_quadrant(0, n2, m2, self.cols);
        let c21 = make_quadrant(m2, 0, self.rows, n2);
        let c22 = make_quadrant(m2, n2, self.rows, self.cols);
        (c11, c12, c21, c22)
    }

    /// Split into row-chunks of size `chunk_rows` (last chunk may be
    /// smaller). The returned chunks tile `self` exactly and are
    /// disjoint by row range — safe to send to different threads.
    /// Used by the parallel NTABLES kernel that operates on views.
    ///
    /// Consumes `self`. Use `reborrow().row_split(...)` if you need
    /// to keep the outer view alive.
    pub fn row_split(self, chunk_rows: usize) -> Vec<MzdViewMut<'a>> {
        assert!(chunk_rows > 0, "row_split: chunk_rows must be > 0");
        let m = self.rows;
        let mut chunks = Vec::with_capacity(m.div_ceil(chunk_rows.max(1)));
        let mut start = 0;
        while start < m {
            let end = (start + chunk_rows).min(m);
            chunks.push(MzdViewMut {
                rows: end - start,
                cols: self.cols,
                width: self.width,
                high_bitmask: self.high_bitmask,
                parent_rowstride: self.parent_rowstride,
                row_offset: self.row_offset + start,
                col_word_offset: self.col_word_offset,
                // Safety: each chunk owns a disjoint row range
                // [self.row_offset + start, self.row_offset + end).
                // `quad_split`'s same-buffer rationale applies.
                data: self.data,
                data_len: self.data_len,
                _marker: std::marker::PhantomData,
            });
            start = end;
        }
        chunks
    }

    /// Construct an immutable view borrowing from this mutable one.
    pub fn as_view(&self) -> MzdView<'_> {
        // Safety: we hand out a *shared* (`&'_`) reference into the
        // backing buffer; the caller's `&self` borrow on us prevents
        // any conflicting writes for the lifetime of the returned
        // view.
        let slice = unsafe { std::slice::from_raw_parts(self.data, self.data_len) };
        MzdView {
            rows: self.rows,
            cols: self.cols,
            width: self.width,
            high_bitmask: self.high_bitmask,
            parent_rowstride: self.parent_rowstride,
            row_offset: self.row_offset,
            col_word_offset: self.col_word_offset,
            data: slice,
        }
    }

    /// Reborrow as a fresh mutable view with a shorter lifetime.
    /// Used by Strassen to pass quadrant views down a recursive call
    /// without giving up the outer ownership.
    pub fn reborrow(&mut self) -> MzdViewMut<'_> {
        MzdViewMut {
            rows: self.rows,
            cols: self.cols,
            width: self.width,
            high_bitmask: self.high_bitmask,
            parent_rowstride: self.parent_rowstride,
            row_offset: self.row_offset,
            col_word_offset: self.col_word_offset,
            data: self.data,
            data_len: self.data_len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> MatrixRead for MzdViewMut<'a> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn high_bitmask(&self) -> u64 {
        self.high_bitmask
    }
    #[inline]
    fn row_words(&self, r: usize) -> &[u64] {
        debug_assert!(r < self.rows);
        let start = (self.row_offset + r) * self.parent_rowstride + self.col_word_offset;
        debug_assert!(start + self.width <= self.data_len);
        // Safety: `start + width ≤ data_len` (debug-checked) and the
        // `'a` marker ties us back to the parent's borrow.
        unsafe { std::slice::from_raw_parts(self.data.add(start), self.width) }
    }
}

impl<'a> MatrixWrite for MzdViewMut<'a> {
    #[inline]
    fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        debug_assert!(r < self.rows);
        let start = (self.row_offset + r) * self.parent_rowstride + self.col_word_offset;
        debug_assert!(start + self.width <= self.data_len);
        // Safety: per `quad_split`'s contract the four quadrant views
        // touch disjoint rows × col-word rectangles, so simultaneously-
        // live `&mut` slices into the same underlying buffer never
        // overlap.
        unsafe { std::slice::from_raw_parts_mut(self.data.add(start), self.width) }
    }
}

// ---- Mzd helpers that build views ---------------------------------------

impl Mzd {
    /// Zero-copy immutable view over `[r0, r1) × [c0, c1)`. `c0` must
    /// be a multiple of 64.
    pub fn view(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> MzdView<'_> {
        MzdView::new(self, r0, c0, r1, c1)
    }

    /// Zero-copy mutable view. `c0` must be a multiple of 64.
    pub fn view_mut(&mut self, r0: usize, c0: usize, r1: usize, c1: usize) -> MzdViewMut<'_> {
        MzdViewMut::new(self, r0, c0, r1, c1)
    }

    /// Immutable view over the entire matrix.
    pub fn full_view(&self) -> MzdView<'_> {
        let r = self.rows();
        let c = self.cols();
        MzdView::new(self, 0, 0, r, c)
    }

    /// Mutable view over the entire matrix.
    pub fn full_view_mut(&mut self) -> MzdViewMut<'_> {
        let r = self.rows();
        let c = self.cols();
        MzdViewMut::new(self, 0, 0, r, c)
    }
}
