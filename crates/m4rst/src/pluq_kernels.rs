// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! M4R-block PLE engine kernels.
//!
//! This module hosts the in-place strip, table, row-application, and driver
//! kernels used by the PLE implementation:
//!
//! * [`ple_submatrix`] — the per-strip pivot-find and column-clear kernel;
//!   mirrors libm4ri's
//!   `_mzd_ple_submatrix` (`ple_russian.c:119-188`). Identifies up
//!   to `k` pivots inside the kk-column strip starting at
//!   `(start_row, start_col)` and clears the lower-triangular
//!   portion lazily.
//! * [`ple_a10`] — apply the strip's transformation
//!   matrix to the left of the strip (the `A10` panel); mirrors
//!   `_mzd_ple_a10` (`ple_russian.c:325-345`). Cleans up the bits
//!   above the diagonal of the L panel.
//! * [`PleTable`] + [`make_table_ple`] — the Gray-code table
//!   builder for full-rank strips. `pivots[j] == j` in that branch, so the
//!   table can be
//!   built in natural-binary order without the M / E / B / ord
//!   permutation arrays. This is also what
//!   [`combine_n_into`][`crate::matmul::combine::combine_n_into`]
//!   already does inside the M4RM matmul kernel — the only PLE
//!   operation shared with matmul.
//!
//! ## References
//!
//! * Albrecht, Pernet, *Efficient Decomposition of Dense Matrices
//!   over GF(2)*, [arXiv:1006.1744](https://arxiv.org/abs/1006.1744),
//!   2010 — the PLE decomposition this engine ports.
//! * libm4ri's `m4rst/ple_russian.c` — engineering reference.

use crate::mzd::Mzd;

/// Maximum supported strip width `kk = NTABLES · k`. Libm4rst's
/// `_mzd_ple_russian` asserts `kk ≤ m4rst_radix = 64`, so we cap at
/// 64 — `extract_bits` cannot return more than a single word
/// anyway. This bound is used to size the `os` / `bm` scratch
/// buffers in [`ple_submatrix`].
pub(crate) const MAX_PLE_KK: usize = 64;

/// In-place PLE pivot search and elimination over a kk-column
/// strip of `A` starting at `(start_row, start_col)`. Mirrors
/// libm4ri's `_mzd_ple_submatrix` (`ple_russian.c:119-188`).
///
/// Walks columns `start_col..start_col+k`. For each column,
/// searches rows `[start_row + rank, stop_row)` for a row with a
/// nonzero bit in the strip prefix; when found, clears the bits
/// at *earlier* strip pivots lazily (preserving the L
/// transformation matrix encoded below the pivot) and, if a 1
/// remains at the current column, takes it as the next pivot.
///
/// Row operations during pivot search are bounded by word index
/// `splitblock_word` (exclusive). The caller passes
/// `splitblock_word ≥ ceil((start_col + k) / 64) + 1` (plus a
/// cache-line-friendly bump) to limit the in-strip elimination
/// width; the remaining tail is dispatched by the apply kernel.
///
/// Returns `(rank, done_row)` where `rank` is the number of
/// pivots identified (≤ k) and `done_row` is the largest row
/// index processed by the lazy column-clear. The caller uses
/// `done_row` to know which rows still need the post-strip clear.
///
/// `p` and `q` are the in-place row / column permutation arrays
/// (libm4ri's `mzp_t::values`). `pivots[i]` records the
/// in-strip column index where pivot `i` was found (range
/// `0..k`). `done[i]` tracks the last row processed for pivot
/// `i` during the lazy clear, used to avoid double-XORing rows
/// across iterations.
///
/// # Panics
///
/// * `k > MAX_PLE_KK`
/// * `start_row + k > stop_row` or `stop_row > a.rows()`
/// * `start_col + k > a.cols()`
/// * `splitblock_word > a.width()`
/// * `pivots.len() < k` or `done.len() < k`
/// * `p.len() < a.rows()` or `q.len() < a.cols()`
#[allow(clippy::too_many_arguments)] // mirrors libm4ri's signature exactly
pub(crate) fn ple_submatrix(
    a: &mut Mzd,
    start_row: usize,
    stop_row: usize,
    start_col: usize,
    k: usize,
    p: &mut [usize],
    q: &mut [usize],
    pivots: &mut [usize],
    done: &mut [usize],
    splitblock_word: usize,
) -> (usize, usize) {
    assert!(k <= MAX_PLE_KK, "ple_submatrix: k={k} > MAX_PLE_KK");
    assert!(stop_row <= a.rows());
    assert!(start_row + k <= stop_row || k == 0);
    assert!(start_col + k <= a.cols() || k == 0);
    assert!(splitblock_word <= a.width());
    assert!(pivots.len() >= k);
    assert!(done.len() >= k);
    assert!(p.len() >= a.rows());
    assert!(q.len() >= a.cols());

    // Per-strip-position word/bit caches: os[c] is the word index
    // of column `start_col + c`, bm[c] is the bit mask within that
    // word. Used by the inner clear loop to peek at row bits
    // without going through extract_bits's general path.
    let mut os = [0usize; MAX_PLE_KK];
    let mut bm = [0u64; MAX_PLE_KK];

    let mut rank = 0usize;

    for curr_pos in 0..k {
        let col = start_col + curr_pos;
        os[curr_pos] = col / 64;
        bm[curr_pos] = 1u64 << (col % 64);

        // Search rows [start_row + rank, stop_row) for a pivot.
        let mut found: Option<usize> = None;
        let search_start = start_row + rank;
        for i in search_start..stop_row {
            // Quick rejection: are any of bits start_col..=start_col+curr_pos
            // of row i nonzero? If all zero, no pivot here yet, move on.
            let bits = a.extract_bits(i, start_col, curr_pos + 1);
            if bits == 0 {
                continue;
            }
            // Clear earlier pivots in this row, lazily. The L-portion
            // (bit at and before the pivot column) is preserved so the
            // transformation matrix can be reconstructed.
            for l in 0..rank {
                if done[l] < i {
                    let pivot_l_word = os[pivots[l]];
                    let pivot_l_mask = bm[pivots[l]];
                    let has_bit = (a.row_words(i)[pivot_l_word] & pivot_l_mask) != 0;
                    if has_bit {
                        a.xor_row_offset_self(
                            i,
                            start_row + l,
                            start_col + pivots[l] + 1,
                            splitblock_word,
                        );
                    }
                    done[l] = i;
                }
            }
            // After clearing earlier pivots, is the bit at the current
            // column position now 1? If so, take it as the pivot.
            if (a.row_words(i)[os[curr_pos]] & bm[curr_pos]) != 0 {
                found = Some(i);
                break;
            }
        }

        if let Some(i) = found {
            p[start_row + rank] = i;
            a.swap_rows(i, start_row + rank);
            q[start_row + rank] = col;
            pivots[rank] = curr_pos;
            done[rank] = i;
            rank += 1;
        }
    }

    // done_row = max of done[0..rank] if all pivots found; else
    // we processed every row down to nrows-1.
    let done_row = if rank < k {
        a.rows().saturating_sub(1)
    } else {
        *done[..rank].iter().max().unwrap_or(&0)
    };

    // Final clear: for each pivot `c2`, walk rows
    // (done[c2], done_row] and clear the column-`pivots[c2]` bit
    // if set. This finishes the rows that were not yet processed
    // during the lazy clear inside the pivot search.
    for c2 in 0..rank {
        let pc = start_col + pivots[c2];
        // libm4ri bails on c2 once pivots[c2] hits the last column
        // (no bits past it to XOR). Mirror that early-exit.
        if pc + 1 >= a.cols() {
            continue;
        }
        if done[c2] >= done_row {
            continue;
        }
        let word = os[pivots[c2]];
        let mask = bm[pivots[c2]];
        for r2 in (done[c2] + 1)..=done_row {
            if (a.row_words(r2)[word] & mask) != 0 {
                a.xor_row_offset_self(r2, start_row + c2, pc + 1, splitblock_word);
            }
        }
    }

    (rank, done_row)
}

/// Apply the strip's transformation matrix `L` to the `A10` panel
/// (the rectangle of `A` to the LEFT of the strip — column word
/// indices `[0, start_col / 64)`). Mirrors libm4ri's
/// `_mzd_ple_a10` (`ple_russian.c:325-345`).
///
/// The PLE inner kernel does its row swaps and XORs only over
/// columns `[start_col, start_col + kk)`; the bits left of
/// `start_col` (the `A10` block) are left untouched by
/// [`ple_submatrix`]. This function walks through the
/// `done[c2]+1 .. done_row+1` rows for each pivot `c2` and XORs
/// `row[start_row + c2]` into them on those LEFT-of-strip
/// columns, completing the elimination.
///
/// `pivots`, `done`, and `done_row` come from the [`ple_submatrix`]
/// return values for the same strip.
pub(crate) fn ple_a10(
    a: &mut Mzd,
    start_row: usize,
    start_col: usize,
    done_row: usize,
    rank: usize,
    pivots: &[usize],
    done: &[usize],
) {
    debug_assert!(rank <= pivots.len());
    debug_assert!(rank <= done.len());

    let pre_word_end = start_col / 64;
    if pre_word_end == 0 {
        // No columns left of the strip; nothing to apply.
        return;
    }

    let mut os = [0usize; MAX_PLE_KK];
    let mut bm = [0u64; MAX_PLE_KK];
    for c in 0..rank {
        let col = start_col + pivots[c];
        os[c] = col / 64;
        bm[c] = 1u64 << (col % 64);
    }

    // Same lazy-vs-final structure as ple_submatrix but on the
    // A10 columns. We only need to apply against the bits below
    // each pivot's `done[c2]` and up through `done_row`.
    for c2 in 0..rank {
        if done[c2] >= done_row {
            continue;
        }
        let word = os[c2];
        let mask = bm[c2];
        for r2 in (done[c2] + 1)..=done_row {
            if (a.row_words(r2)[word] & mask) != 0 {
                a.xor_row_offset_self(r2, start_row + c2, 0, pre_word_end);
            }
        }
    }
}

/// Gray-code lookup table for the M4R-block PLE apply step.
///
/// Stores `2^knar` linear combinations of pivot rows
/// `A[r..r+knar]`. Entry `p` (natural binary indexing) is the
/// XOR of `A[r + j]` for each bit `j` set in `p`. The lookup is
/// direct: `row^=T[v]` for the value `v` extracted from the
/// target row's `knar`-bit slice at the strip's start column.
///
/// Layout: a flat `Vec<u64>` of `2^knar × row_stride` words, with
/// entry `p` occupying `entries[p * row_stride .. (p+1) * row_stride]`.
/// Allocated once per strip and reused across all NTABLES strips by
/// [`ple_russian`].
///
/// ## Full-rank construction
///
/// libm4ri's `mzd_make_table_ple` carries two code paths:
/// `!fullrank` uses `m4rst_spread_bits` + `m4rst_codebook` to handle
/// strips where `pivots[j] != j`; `fullrank` skips that machinery. This table
/// stores the natural-binary-indexed XOR combinations for that path.
pub(crate) struct PleTable {
    /// Flat buffer of size `2^knar × row_stride`. Entry `p` lives at
    /// `entries[p * row_stride .. (p+1) * row_stride]`.
    entries: Vec<u64>,
    /// Words per table entry. Equals the source matrix's width.
    row_stride: usize,
    /// Strip-width: `kk = NTABLES · k`. Table holds `2^knar` rows.
    knar: usize,
    /// `e[strip_bits] = p` where `T[p]`'s strip-z bit pattern equals
    /// `strip_bits`. Lets [`ple_apply`] recover the linear-combination
    /// coefficient from a row's strip-bit pattern without modifying
    /// the pivot rows in `A`. Mirrors libm4ri's `ple_table_t::E`.
    e: Vec<usize>,
}

impl PleTable {
    /// Allocate a zero-initialised table sized for a strip of
    /// width `knar` operating on rows of `row_stride` words.
    /// The buffer is `(2^knar) × row_stride` u64s; the table is
    /// reset between strips via [`PleTable::reset`].
    ///
    /// # Panics
    ///
    /// * `knar > MAX_PLE_KK`
    pub(crate) fn new(knar: usize, row_stride: usize) -> Self {
        assert!(
            knar <= MAX_PLE_KK,
            "PleTable::new: knar={knar} > MAX_PLE_KK"
        );
        let n_entries = 1usize << knar;
        let entries = vec![0u64; n_entries * row_stride];
        let e = vec![0usize; n_entries];
        Self {
            entries,
            row_stride,
            knar,
            e,
        }
    }

    /// Number of entries in the table (`2^knar`).
    pub(crate) fn n_entries(&self) -> usize {
        1usize << self.knar
    }

    /// Get table row `p` as a slice (length `row_stride`).
    pub(crate) fn row(&self, p: usize) -> &[u64] {
        let start = p * self.row_stride;
        &self.entries[start..start + self.row_stride]
    }

    /// Re-zero entry 0 (the all-zero combination). The other
    /// entries are overwritten by [`make_table_ple`] in
    /// natural-binary order, so they don't need pre-zeroing.
    fn reset(&mut self) {
        for w in &mut self.entries[..self.row_stride] {
            *w = 0;
        }
    }
}

/// Build the Gray-code XOR-combination table for a PLE strip.
/// Mirrors libm4ri's `mzd_make_table_ple` (`ple_russian.c:191-295`)
/// restricted to the fullrank branch.
///
/// For each `p` in `1..2^knar`:
///
/// ```text
/// table[p] = table[p ^ lowbit(p)] ^ A[r + log2(lowbit(p))]
/// ```
///
/// which is the standard natural-binary Gray-code-table
/// construction (one row-XOR per entry, O(2^knar) total).
/// The same recurrence drives
/// [`combine_n_into`][`crate::matmul::combine::combine_n_into`]'s
/// table build (`src/matmul/m4rm.rs:740-751`).
///
/// `block_word` is the word index inside each row that holds the
/// strip; `shift` is the bit offset within `block_word` where this
/// table's k-bit strip pattern lives (= `z * k_per` for the z-th
/// table of the NTABLES-wide super-strip).
///
/// `pivot_strip_mask[j]` is AND'd against the pivot row at offset
/// `r + j` when computing the strip word of table entries,
/// providing a virtual L-encoding clear without modifying `A`.
/// For pivot row at absolute strip-relative position `m = shift + j`,
/// pass `mask = !((1 << m) - 1)` (keep bits ≥ m, clear bits < m).
/// Pass `[!0; knar]` for no masking. Mirrors libm4ri's
/// `_mzd_ple_to_e` (`ple_russian.c:364-377`).
///
/// Populates `table.e[strip_bits] = p` so [`ple_apply`] can recover
/// the linear-combination coefficient from a row's strip-bit
/// pattern. The strip-bit pattern is the `knar`-bit slice of
/// `T[p].entries[block_word]` starting at `shift`.
///
/// # Panics
///
/// * `r + table.knar > a.rows()`
/// * `a.width() != table.row_stride`
/// * `block_word >= table.row_stride`
/// * `pivot_strip_mask.len() < table.knar`
pub(crate) fn make_table_ple(
    a: &Mzd,
    r: usize,
    table: &mut PleTable,
    block_word: usize,
    shift: usize,
    pivot_strip_mask: &[u64],
) {
    assert!(r + table.knar <= a.rows());
    assert_eq!(
        a.width(),
        table.row_stride,
        "make_table_ple: a.width() must equal table.row_stride"
    );
    assert!(block_word < table.row_stride);
    assert!(pivot_strip_mask.len() >= table.knar);
    table.reset();
    let w = table.row_stride;
    let n_entries = table.n_entries();
    let knar_mask: u64 = if table.knar == 64 {
        !0u64
    } else {
        (1u64 << table.knar) - 1
    };
    table.e[0] = 0;
    for p in 1..n_entries {
        let lowbit_idx = p.trailing_zeros() as usize;
        let prev_p = p ^ (1usize << lowbit_idx);
        let src_row = a.row_words(r + lowbit_idx);
        let mask = pivot_strip_mask[lowbit_idx];
        // table[p] = table[prev_p] XOR (src_row, with strip word
        // masked by pivot_strip_mask[lowbit_idx]).
        let (left, right) = table.entries.split_at_mut(p * w);
        let dst = &mut right[..w];
        let prev = &left[prev_p * w..(prev_p + 1) * w];
        for i in 0..w {
            let src_word = if i == block_word {
                src_row[i] & mask
            } else {
                src_row[i]
            };
            dst[i] = prev[i] ^ src_word;
        }
        // Populate E[] inverse-permutation entry.
        let strip_bits = ((dst[block_word] >> shift) & knar_mask) as usize;
        table.e[strip_bits] = p;
    }
}

/// Row-apply engine for the M4R-block PLE step. Mirrors libm4ri's
/// `_mzd_process_rows_ple_<N>` (template-expanded from
/// `ple_russian_template.h`) and `_mzd_ple_a11_<N>` collapsed into
/// a single kernel — in the fullrank-only setting the two
/// libm4ri kernels reduce to the same operation (M, E, and B
/// permutation arrays all become identity / zero).
///
/// For each row `r` in `[start_row, stop_row)`:
///
/// 1. Read `NT · k_per` bits of `a` at column `writecol` (one
///    `extract_bits` call).
/// 2. Split into `NT` sub-indices of `k_per` bits each.
/// 3. Look up `tables[z].row(idx_z)` for each `z`.
/// 4. Fuse all `NT` table rows + the destination row into one
///    XOR pass via
///    [`combine_n_into`][`crate::matmul::combine::combine_n_into`],
///    starting at word `writecol / 64` of `a`.
///
/// The XOR starts at `block = writecol / 64`, leaving words
/// `[0, block)` of the destination row untouched. This matches
/// libm4ri's `_mzd_combine(m, t, wide)` with `m = mzd_row + block`
/// — necessary because the L-encoding bits at columns `< writecol`
/// in the target row are the work of earlier strips and must not
/// be disturbed. The table's `[0, block)` words are also untouched
/// during construction (`make_table_ple` over the full row leaves them at
/// their initial zero value).
///
/// The kernel reuses
/// [`combine_n_into`][`crate::matmul::combine::combine_n_into`]
/// directly.
///
/// `xor_start_word` is the destination-row word index at which the
/// combine starts. Two distinct call sites in
/// [`ple_russian`] use this differently:
///
/// * For rows below the strip (`[done_row+1, nrows)`) — libm4ri's
///   `_mzd_process_rows_ple` — pass `xor_start_word = writecol / 64`.
///   The combine clears the strip word and propagates the U-tail.
/// * For rows in-between (`[curr_row+knar, done_row+1)`) — libm4ri's
///   `_mzd_ple_a11` — pass `xor_start_word = splitblock_word`. The
///   strip word has already been processed by `ple_submatrix`'s
///   lazy clear up to `splitblock_word`; the combine only needs to
///   finish words past that point.
///
/// # Panics
///
/// * `NT * k_per > 64` (`extract_bits` cannot return more than a word).
/// * `stop_row > a.rows()`
/// * any `tables[z].knar != k_per`
/// * any `tables[z].row_stride != a.width()`
pub(crate) fn ple_apply<const NT: usize>(
    a: &mut Mzd,
    start_row: usize,
    stop_row: usize,
    writecol: usize,
    k_per: usize,
    tables: [&PleTable; NT],
    xor_start_word: usize,
) {
    assert!(start_row <= stop_row);
    assert!(stop_row <= a.rows());
    let kk = NT * k_per;
    assert!(
        kk <= 64,
        "ple_apply: NT * k_per ={kk} > 64; extract_bits cannot return >1 word"
    );
    for tbl in tables.iter() {
        debug_assert_eq!(tbl.knar, k_per);
        debug_assert_eq!(tbl.row_stride, a.width());
    }
    if start_row == stop_row {
        return;
    }
    let width = a.width();
    if xor_start_word >= width {
        return;
    }
    let mask: u64 = if k_per == 64 {
        !0u64
    } else {
        (1u64 << k_per) - 1
    };

    // Strip word index inside each table entry — set by
    // make_table_ple to writecol/64.
    let strip_word = writecol / 64;

    // bits-ripple correction with E[] permutation lookup. Mirrors
    // libm4ri's `bits ^= B[N - z][x[N - z]]` at
    // `ple_russian_template.h:71-107`, with `x[z] = E[z][bits_at_z]`.
    //
    // The natural-binary lookup `idx = (bits >> sh) & mask` would
    // give the wrong table entry when pivot rows have strip-bit
    // contamination from L-encoding (their bits at strip[i<m]) plus
    // U-portion (their bits at strip[i>m]). E[bits_at_z] returns
    // the table index whose strip pattern at strip-z equals
    // bits_at_z — the canonical-basis coefficient subset that
    // produces those strip bits. After all NT ripple iterations
    // `bits` is 0, so the combine clears the strip word.
    for r in start_row..stop_row {
        let mut bits = a.extract_bits(r, writecol, kk);
        let mut indices = [0usize; NT];
        for (z, idx_slot) in indices.iter_mut().enumerate().take(NT) {
            let bits_at_z = ((bits >> (z * k_per)) & mask) as usize;
            let idx = tables[z].e[bits_at_z];
            *idx_slot = idx;
            bits ^= tables[z].row(idx)[strip_word];
        }
        let strip_mask = if kk == 64 {
            u64::MAX
        } else {
            (1_u64 << kk) - 1
        };
        debug_assert_eq!(
            bits & strip_mask,
            0,
            "bits-ripple with E[] should clear the super-strip bits after NT iterations"
        );
        let mut t_rows: [&[u64]; NT] = [&[][..]; NT];
        for (z, slot) in t_rows.iter_mut().enumerate().take(NT) {
            *slot = &tables[z].row(indices[z])[xor_start_word..];
        }
        let dst_words = &mut a.row_words_mut(r)[xor_start_word..];
        crate::matmul::combine::combine_n_into::<NT>(dst_words, t_rows);
    }
}

/// Default per-table strip width and NTABLES for the PLE driver.
/// Mirrors libm4ri's choice (kk = NTABLES · k ≤ 64) plus m4rst's
/// matmul NT=8 selection. Yields kk = 64 at k = 8.
const PLE_NTABLES: usize = 8;
const PLE_K_DEFAULT: usize = 8;

/// Top-level M4R-block PLE driver.
///
/// Operates in place on `a`, identifying pivots strip-by-strip via
/// [`ple_submatrix`], applying the L transformation to the A10 panel
/// via [`ple_a10`], and clearing pivot columns in the rows below
/// the strip via [`ple_apply`] (fullrank fast path) or a
/// per-pivot sequential XOR fallback (rank-deficient strip path).
///
/// On return:
/// * `a` holds the PLE-form in-place (U above-and-including the
///   pivot diagonal, L bits below the pivot column of each row).
/// * `p[i]` for `i in 0..rank` records the original row index of
///   the i-th pivot (the row that was swapped to position `i`).
/// * `q[i]` for `i in 0..rank` records the column index where the
///   i-th pivot landed.
/// * Returns rank.
///
/// Mirrors libm4ri's `_mzd_ple_russian` (`ple_russian.c:380-623`) with two
/// simplifications:
///   1. Fullrank table apply only; rank-deficient strips fall back
///      to per-pivot sequential XOR rather than the spread-bits
///      codebook.
///   2. Reads the pivot rows directly from `a` rather than
///      materialising a separate `U` matrix (libm4ri's
///      `_mzd_ple_to_e`).
///
/// # Panics
///
/// * `p.len() < a.rows()`
/// * `q.len() < a.cols()`
pub(crate) fn ple_russian(a: &mut Mzd, p: &mut [usize], q: &mut [usize]) -> usize {
    let nrows = a.rows();
    let ncols = a.cols();
    assert!(p.len() >= nrows);
    assert!(q.len() >= ncols);

    // Initialise permutations as identity. p[i] = i, q[i] = i.
    for (i, slot) in p.iter_mut().enumerate().take(nrows) {
        *slot = i;
    }
    for (i, slot) in q.iter_mut().enumerate().take(ncols) {
        *slot = i;
    }

    if nrows == 0 || ncols == 0 {
        return 0;
    }

    // Per-strip scratch — see ple_submatrix's MAX_PLE_KK bound.
    let mut pivots = [0usize; MAX_PLE_KK];
    let mut done = [0usize; MAX_PLE_KK];

    // Persistent tables (one per NTABLES), allocated once and reused
    // across strips.
    let width = a.width();
    let mut tables: Vec<PleTable> = (0..PLE_NTABLES)
        .map(|_| PleTable::new(PLE_K_DEFAULT, width))
        .collect();

    let mut curr_row = 0usize;
    let mut curr_col = 0usize;
    let mut rank = 0usize;

    while curr_col < ncols && curr_row < nrows {
        let k = PLE_K_DEFAULT;
        let kk = (k * PLE_NTABLES)
            .min(ncols - curr_col)
            .min(nrows - curr_row);

        // Splitblock = exclusive word upper bound for in-strip ops.
        // **Forced to `width`** so `ple_submatrix`'s lazy + final
        // clear fully process in-between rows. Both apply paths
        // (fast and fallback) then only need to handle the
        // below-strip rows. Using the full width preserves a simple invariant
        // across both apply paths.
        let splitblock_word = width;
        let _ = kk; // splitblock formula `min(max(by_kk, by_cache), width)` not used

        let (knar, done_row) = ple_submatrix(
            a,
            curr_row,
            nrows,
            curr_col,
            kk,
            p,
            q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );

        // Apply the strip's L transformation to the A10 panel
        // (columns left of curr_col).
        ple_a10(a, curr_row, curr_col, done_row, knar, &pivots, &done);

        if knar == 0 {
            // Rank-deficient strip: no pivots found. Search for the
            // next single pivot in the remaining sub-matrix; if
            // found, apply it and advance one row + one column.
            let (found, pi, pj) = find_first_pivot(a, curr_row, curr_col);
            if !found {
                break;
            }
            p[curr_row] = pi;
            q[curr_row] = pj;
            a.swap_rows(curr_row, pi);
            // Clear column pj in rows below curr_row.
            if pj + 1 < ncols {
                let pj_word = pj / 64;
                let pj_mask = 1u64 << (pj % 64);
                for l in (curr_row + 1)..nrows {
                    if (a.row_words(l)[pj_word] & pj_mask) != 0 {
                        a.xor_row_offset_self(l, curr_row, pj + 1, width);
                    }
                }
            }
            curr_col = pj + 1;
            curr_row += 1;
            rank += 1;
            continue;
        }

        // Fullrank or partial-rank strip with knar > 0 pivots. Take
        // the M4R-block table-apply branch when the strip is both
        // full (kk == k * NTABLES) and fullrank (knar == kk).
        //
        // The M4R-block fast path. With splitblock_word=width above,
        // ple_submatrix's lazy + final clear fully processes
        // in-between rows; ple_a11's xor_start_word = splitblock_word
        // = width then short-circuits to a no-op. Only
        // process_rows_ple (below-strip rows) does meaningful work.
        if knar == kk && kk == k * PLE_NTABLES {
            // M4R-block fast path. Three steps:
            //
            //   1. Build NTABLES Gray-code tables from the strip's
            //      pivot rows. Use the virtual-L-clear mask so each
            //      table sees pivot rows with strip[0..m) = 0
            //      (canonical-form), without modifying A.
            //   2. Apply tables to rows [curr_row+knar, done_row+1)
            //      from word splitblock_word onward — these rows
            //      were processed by ple_submatrix's lazy clear up
            //      to splitblock_word but need apply for words past
            //      that (libm4ri's _mzd_ple_a11).
            //   3. Apply tables to rows [done_row+1, nrows) from
            //      word curr_col/64 onward — these rows weren't
            //      touched by ple_submatrix (libm4ri's
            //      _mzd_process_rows_ple).
            debug_assert_eq!(
                curr_col % 64,
                0,
                "ple_russian table-apply assumes curr_col is 64-aligned"
            );
            debug_assert_eq!(kk, 64, "ple_russian table-apply assumes kk == 64");
            let block_word = curr_col / 64;
            // Per-row strip-clear mask for virtual L-clearing. For
            // pivot row at strip-relative position m = z*k + j,
            // mask = !((1 << m) - 1) keeps bits ≥ m, clears bits < m.
            let mut strip_masks = [!0u64; MAX_PLE_KK];
            for (m, slot) in strip_masks.iter_mut().enumerate().take(knar) {
                *slot = !((1u64 << m) - 1);
            }
            // Build NTABLES tables.
            for (z, tbl) in tables.iter_mut().enumerate().take(PLE_NTABLES) {
                let mask_slice = &strip_masks[z * k..(z + 1) * k];
                make_table_ple(a, curr_row + z * k, tbl, block_word, z * k, mask_slice);
            }
            let table_refs: [&PleTable; PLE_NTABLES] = std::array::from_fn(|i| &tables[i]);
            // (2) ple_a11: in-between rows [curr_row+knar, done_row+1).
            let a11_start = curr_row + knar;
            let a11_end = done_row + 1;
            if a11_start < a11_end {
                ple_apply::<PLE_NTABLES>(
                    a,
                    a11_start,
                    a11_end,
                    curr_col,
                    k,
                    table_refs,
                    splitblock_word,
                );
            }
            // (3) process_rows_ple: below-strip rows [done_row+1, nrows).
            let below_start = done_row + 1;
            if below_start < nrows {
                ple_apply::<PLE_NTABLES>(
                    a,
                    below_start,
                    nrows,
                    curr_col,
                    k,
                    table_refs,
                    block_word,
                );
            }
        } else {
            // Sequential per-pivot XOR fallback for below-strip rows.
            // The in-between rows (curr_row+knar, done_row+1) are
            // fully processed by ple_submatrix's lazy + final clear
            // because we force `splitblock_word = width` for the
            // sequential path (see the per-strip splitblock above).
            for i in (done_row + 1)..nrows {
                for (piv, &pivot_pos) in pivots.iter().enumerate().take(knar) {
                    let pcol = curr_col + pivot_pos;
                    let pcol_word = pcol / 64;
                    let pcol_mask = 1u64 << (pcol % 64);
                    if (a.row_words(i)[pcol_word] & pcol_mask) != 0 {
                        a.xor_row_offset_self(i, curr_row + piv, pcol + 1, width);
                    }
                }
            }
        }

        curr_col += kk;
        curr_row += knar;
        rank += knar;
    }

    rank
}

/// Find the lexicographically-first (column-then-row) nonzero in
/// `a[start_row.., start_col..]`. Returns `(found, row, col)`.
/// Used by [`ple_russian`] when a strip is rank-deficient and we
/// need to advance past it to the next pivot.
fn find_first_pivot(a: &Mzd, start_row: usize, start_col: usize) -> (bool, usize, usize) {
    let mut pivot: Option<(usize, usize)> = None;
    for row in start_row..a.rows() {
        if let Some(col) = a.first_one_in_row(row, start_col) {
            if pivot.is_none_or(|(best_row, best_col)| (col, row) < (best_col, best_row)) {
                pivot = Some((row, col));
            }
        }
    }
    pivot.map_or((false, 0, 0), |(row, col)| (true, row, col))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity-pivot input: A is the n×n identity. ple_submatrix
    /// should find n pivots without doing any column clears, since
    /// each diagonal-1 is already isolated.
    #[test]
    fn ple_submatrix_identity() {
        let n = 16usize;
        let mut a = Mzd::identity(n);
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let mut pivots = vec![0usize; n];
        let mut done = vec![0usize; n];
        let splitblock_word = a.width();
        let (rank, _done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            n,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, n, "identity should have full rank");
        for (i, &piv) in pivots.iter().enumerate().take(n) {
            assert_eq!(piv, i, "pivot {i} should be at strip col {i}");
        }
        // A should still be the identity after the elimination.
        assert_eq!(a.as_words(), Mzd::identity(n).as_words());
    }

    /// Lower-triangular full-rank input: each diagonal 1 is the
    /// pivot for its column. The L-portion bits below the diagonal
    /// are preserved by PLE (encoding the transformation matrix),
    /// and the U-portion above the diagonal is zero by
    /// construction.
    ///
    /// Net effect on this input: A is unchanged. This tests that
    /// PLE preserves the L-encoding rather than zeroing it out
    /// (a common porting bug — Gauss-Jordan zeros below pivots;
    /// PLE keeps them).
    #[test]
    fn ple_submatrix_lower_triangular_full_rank() {
        let n = 8usize;
        let mut a = Mzd::new(n, n);
        for i in 0..n {
            for j in 0..=i {
                a.set(i, j, 1);
            }
        }
        let original = a.clone();
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let mut pivots = vec![0usize; n];
        let mut done = vec![0usize; n];
        let splitblock_word = a.width();
        let (rank, done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            n,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, n, "lower-triangular all-ones is full-rank");
        assert_eq!(done_row, n - 1);
        assert_eq!(
            a.as_words(),
            original.as_words(),
            "PLE preserves L on a lower-triangular input"
        );
        // Pivots track the strip-column index for each rank;
        // diagonal pivots in this case.
        for (i, &piv) in pivots.iter().enumerate().take(n) {
            assert_eq!(piv, i);
        }
    }

    /// Mixed input that drives the inner-clear path nontrivially.
    /// A has two rows that share their first nonzero column —
    /// the second row's contributions to the first column must
    /// be XORed away during the column-1 pivot search.
    ///
    /// A =
    ///   1 1 0 0
    ///   1 0 1 0
    ///   0 1 1 1
    ///   0 0 0 1
    ///
    /// After PLE:
    /// * Col 0 pivot lands on row 0.
    /// * Col 1 pivot search reaches row 1 (bits [1, 0, …]); the
    ///   lazy-clear XORs row 0 into row 1 starting at col 1
    ///   (so row 1 becomes [1, 1, 1, 0]) and the bit at col 1
    ///   is now 1 — pivot found.
    /// * Col 1 done[0] is now 1; col 2 pivot search visits row 2
    ///   (bits [0, 1, 1, 1]); clears col 0 (no bit) and col 1
    ///   (XOR row 1 starting at col 2: row 2 becomes
    ///   [0, 1, 0, 1]); col 2 bit is now 0 — no pivot for col 2.
    ///   Continues to row 3 ([0, 0, 0, 1]); col 0 and col 1
    ///   bits are 0; col 2 bit is 0 — no pivot. Strip exits with
    ///   rank < k.
    /// * Net rank should be 3 (the matrix is rank-3 since
    ///   rows 0+1+2 sum to row 3).
    #[test]
    fn ple_submatrix_inner_clear_path() {
        let n = 4usize;
        let mut a = Mzd::new(n, n);
        a.set(0, 0, 1);
        a.set(0, 1, 1);
        a.set(1, 0, 1);
        a.set(1, 2, 1);
        a.set(2, 1, 1);
        a.set(2, 2, 1);
        a.set(2, 3, 1);
        a.set(3, 3, 1);
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let mut pivots = vec![0usize; n];
        let mut done = vec![0usize; n];
        let splitblock_word = a.width();
        let (rank, _done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            n,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, 3, "matrix has rank 3");
        // Spot-check the L-portion preservation: row 1 col 0 stays 1.
        assert_eq!(a.get(1, 0), 1, "L-portion bit (1, 0) preserved");
    }

    /// Rank-deficient: the all-zero matrix has zero pivots and
    /// returns rank = 0.
    #[test]
    fn ple_submatrix_all_zero() {
        let n = 8usize;
        let mut a = Mzd::new(n, n);
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let mut pivots = vec![0usize; n];
        let mut done = vec![0usize; n];
        let splitblock_word = a.width();
        let (rank, _done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            n,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, 0);
        assert_eq!(a.as_words(), Mzd::new(n, n).as_words());
    }

    /// Row swap: the pivot for column 0 is in row 5 (not 0).
    /// After ple_submatrix the strip-pivots should land on row 0,
    /// and `p[0]` should record the original row 5.
    #[test]
    fn ple_submatrix_records_row_swap() {
        let n = 8usize;
        // A: row 5 = [1, 0, ...], row 0 = [0, 1, 0, ...], etc.
        // (an identity with rows 0 and 5 swapped).
        let mut a = Mzd::new(n, n);
        for i in 0..n {
            let src = if i == 0 {
                5
            } else if i == 5 {
                0
            } else {
                i
            };
            a.set(i, src, 1);
        }
        let mut p = (0..n).collect::<Vec<_>>();
        let mut q = (0..n).collect::<Vec<_>>();
        let mut pivots = vec![0usize; n];
        let mut done = vec![0usize; n];
        let splitblock_word = a.width();
        let (rank, _done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            n,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, n);
        // p[0] should record the row we picked the first pivot
        // from: that's the row originally at index 5 (which had
        // its 1 in column 0).
        assert_eq!(p[0], 5, "P[0] should record the row-5 pivot");
    }

    /// Strip-narrower-than-matrix: only the first 4 columns get
    /// pivoted; the rest of A (columns 4..n) is untouched by the
    /// strip's elimination.
    #[test]
    fn ple_submatrix_narrow_strip() {
        let n = 8usize;
        let mut a = Mzd::identity(n);
        let mut p = (0..n).collect::<Vec<_>>();
        let mut q = (0..n).collect::<Vec<_>>();
        let mut pivots = vec![0usize; 4];
        let mut done = vec![0usize; 4];
        let splitblock_word = a.width();
        let (rank, _done_row) = ple_submatrix(
            &mut a,
            0,
            n,
            0,
            4,
            &mut p,
            &mut q,
            &mut pivots,
            &mut done,
            splitblock_word,
        );
        assert_eq!(rank, 4);
        // First 4 diagonal pivots; rest of A unchanged (identity
        // already had no off-diagonals).
        assert_eq!(a.as_words(), Mzd::identity(n).as_words());
    }

    /// Direct check of the Gray-code table: T[p] equals the XOR
    /// of the rows of A indicated by the bits of p (natural
    /// binary). Tests every entry of a small 16-row × 256-col
    /// table built from a known random input.
    #[test]
    fn make_table_ple_natural_binary_combinations() {
        let cols = 256usize;
        let knar = 4usize;
        let a = Mzd::random(knar, cols, 0x000B_00B5);
        let mut table = PleTable::new(knar, a.width());
        make_table_ple(&a, 0, &mut table, 0, 0, &[!0u64; MAX_PLE_KK]);

        let w = table.row_stride;
        // T[0] is zero by construction.
        assert!(
            table.row(0).iter().all(|&x| x == 0),
            "T[0] should be all-zero"
        );
        // For each p in [1, 2^knar), reconstruct the expected
        // XOR-sum cell-by-cell and compare to T[p].
        let n_entries = table.n_entries();
        let mut expected = vec![0u64; w];
        for p in 1..n_entries {
            for word in expected.iter_mut().take(w) {
                *word = 0;
            }
            for j in 0..knar {
                if (p >> j) & 1 == 1 {
                    let row_j = a.row_words(j);
                    for i in 0..w {
                        expected[i] ^= row_j[i];
                    }
                }
            }
            assert_eq!(
                table.row(p),
                expected.as_slice(),
                "T[{p}] disagrees with hand-XOR of rows {p:b}"
            );
        }
    }

    /// Reset between strip invocations: a fresh `make_table_ple`
    /// call must reproduce the same table regardless of any prior
    /// state in the buffer.
    #[test]
    fn make_table_ple_is_idempotent_under_reuse() {
        let cols = 130usize;
        let knar = 3usize;
        let a = Mzd::random(knar, cols, 0xCAFE_BAB6);
        let mut table = PleTable::new(knar, a.width());
        make_table_ple(&a, 0, &mut table, 0, 0, &[!0u64; MAX_PLE_KK]);
        let first: Vec<u64> = table.entries.clone();
        // Re-call with no other change.
        make_table_ple(&a, 0, &mut table, 0, 0, &[!0u64; MAX_PLE_KK]);
        assert_eq!(
            table.entries, first,
            "second make_table_ple should produce identical output"
        );
    }

    /// Single-table ple_apply: with NT=1 and one pivot row of A,
    /// the apply step should zero the pivot bit and propagate the
    /// U-portion XOR into every target row that had that bit set.
    /// Exercises the basic read-bits → lookup → XOR loop.
    #[test]
    fn ple_apply_single_table_zeroes_pivot_bit() {
        // A: 5 rows × 64 cols. Row 0 is the pivot row [1, 1, 1, 0, ..., 0].
        // Rows 1..5 each have bit 0 set (= the pivot column) and a
        // varying tail pattern.
        let nrows = 5usize;
        let cols = 64usize;
        let mut a = Mzd::new(nrows, cols);
        // Pivot row 0: bits at cols 0, 1, 2.
        a.set(0, 0, 1);
        a.set(0, 1, 1);
        a.set(0, 2, 1);
        // Rows 1..5 with bit 0 set + their own non-pivot tail.
        for r in 1..nrows {
            a.set(r, 0, 1);
            a.set(r, 10 + r, 1); // a distinct bit per row
        }

        // Build a 1-table with knar=1 from row 0 of A.
        let mut t = PleTable::new(1, a.width());
        make_table_ple(&a, 0, &mut t, 0, 0, &[!0u64; MAX_PLE_KK]);

        // Sanity: T[0] = 0, T[1] = row 0 of A.
        assert!(t.row(0).iter().all(|&w| w == 0));
        assert_eq!(t.row(1), a.row_words(0));

        // Apply to rows 1..5 at writecol=0. Each row's bit-0 is 1,
        // so idx = 1, T[1] (= row 0) is XORed in: clears bit 0,
        // and adds bits at 1, 2.
        ple_apply::<1>(&mut a, 1, nrows, 0, 1, [&t], 0);

        // After: rows 1..5 have bit-0 cleared, bits 1 and 2
        // flipped (added), and bit (10+r) untouched.
        for r in 1..nrows {
            assert_eq!(a.get(r, 0), 0, "row {r} bit 0 should be cleared");
            assert_eq!(a.get(r, 1), 1, "row {r} bit 1 should be XOR-in from T[1]");
            assert_eq!(a.get(r, 2), 1, "row {r} bit 2 should be XOR-in from T[1]");
            assert_eq!(a.get(r, 10 + r), 1, "row {r} distinct bit untouched");
        }
        // Pivot row 0 is outside [start_row, stop_row), so unchanged.
        assert_eq!(a.get(0, 0), 1);
        assert_eq!(a.get(0, 1), 1);
        assert_eq!(a.get(0, 2), 1);
    }

    /// Two-table ple_apply (NT=2): kk = 2 × 2 = 4 pivot bits in
    /// the super-strip; super-strip splits as strip-0 = bits [0, 2),
    /// strip-1 = bits [2, 4). Drive a 4-row pivot block and a
    /// distinct target row pattern.
    #[test]
    fn ple_apply_two_tables_clears_super_strip() {
        let knar = 2usize;
        let cols = 128usize;
        let mut a = Mzd::new(8, cols);
        // 4 pivot rows: row j has bit at col j for j=0..4, plus a
        // distinct U-tail bit for each.
        for j in 0..4 {
            a.set(j, j, 1);
            a.set(j, 64 + j, 1); // U tail
        }
        // Target row 4: [bit 0, bit 2, bit 64, bit 66] — pivot
        // bits 0 and 2 set, in strip-0 and strip-1 respectively.
        a.set(4, 0, 1);
        a.set(4, 2, 1);
        a.set(4, 64, 1);
        a.set(4, 66, 1);

        // Build tables — strip 0 from rows [0, 2), strip 1 from rows [2, 4).
        let mut t0 = PleTable::new(knar, a.width());
        let mut t1 = PleTable::new(knar, a.width());
        make_table_ple(&a, 0, &mut t0, 0, 0, &[!0u64; MAX_PLE_KK]);
        make_table_ple(&a, 2, &mut t1, 0, knar, &[!0u64; MAX_PLE_KK]);

        // Apply at writecol=0. kk = 4, NT = 2, k_per = 2.
        ple_apply::<2>(&mut a, 4, 5, 0, knar, [&t0, &t1], 0);

        // After: row 4's bits [0..4) at col 0..3 should be zero.
        for c in 0..4 {
            assert_eq!(a.get(4, c), 0, "row 4 col {c} should be cleared");
        }
        // Row 4's U-tail bits: 64 ^ 64 = 0 (from T[1] of strip 0);
        // 66 ^ 66 = 0 (from T[1] of strip 1). All cleared.
        for c in 64..68 {
            assert_eq!(a.get(4, c), 0, "row 4 col {c} U-tail should be cleared");
        }
    }

    /// ple_russian rank matches naive at the smaller multi-super-strip
    /// sizes (N=384 = 6 super-strips at kk=64). Confirms the
    /// M4R-block fast path's `_mzd_ple_a11` and
    /// `_mzd_process_rows_ple` semantics work across multiple super-strips.
    #[test]
    fn ple_russian_rank_matches_naive_multi_super_strip_small() {
        for &n in &[384usize] {
            for &seed in &[0xC0DEu64, 0xBEEF, 0xFEEDF00D, 0xDEADBEEF] {
                let original = Mzd::random(n, n, seed);
                let mut work = original.clone();
                let mut p = vec![0usize; n];
                let mut q = vec![0usize; n];
                let r_ple = ple_russian(&mut work, &mut p, &mut q);
                let r_naive = original.rank();
                assert_eq!(
                    r_ple, r_naive,
                    "ple_russian rank disagrees with naive at N={n} seed={seed:#x}"
                );
            }
        }
    }

    /// 16-seed sweep at multi-super-strip sizes N ∈ {640, 1024,
    /// 2048}. `splitblock_word=width` lets `ple_submatrix` handle in-between
    /// rows fully; all cases must agree with naive rank.
    #[test]
    fn ple_russian_rank_matches_naive_at_large_n() {
        let mut failures = 0usize;
        let mut total = 0usize;
        for &n in &[640usize, 1024, 2048] {
            for seed_bump in 0..16u64 {
                let seed = 0xC0DE_BABE_u64.wrapping_add(seed_bump.wrapping_mul(0x9E37_79B9));
                let original = Mzd::random(n, n, seed);
                let mut work = original.clone();
                let mut p = vec![0usize; n];
                let mut q = vec![0usize; n];
                let r_ple = ple_russian(&mut work, &mut p, &mut q);
                let r_naive = original.rank();
                total += 1;
                if r_ple != r_naive {
                    failures += 1;
                    eprintln!("FAIL N={n} seed={seed:#x}: r_ple={r_ple} r_naive={r_naive}");
                }
            }
        }
        assert_eq!(
            failures, 0,
            "M4R-block fast path failed on {failures}/{total} cases"
        );
    }

    /// ple_russian rank matches naive Gauss-Jordan rank on a
    /// random N=512 input — the dispatch crossover where the
    /// M4R-block path takes over (min(m, n) > 256). Sanity-checks
    /// the whole driver: pivot search, table build, fullrank
    /// apply, no rank-deficient detours.
    #[test]
    fn ple_russian_rank_matches_naive_at_n_512() {
        for &seed in &[0xCAFE_F00Du64, 0xDEAD_BEEFu64, 0x1234_5678u64] {
            let n = 512usize;
            let original = Mzd::random(n, n, seed);
            let mut work = original.clone();
            let mut p = vec![0usize; n];
            let mut q = vec![0usize; n];
            let r_ple = ple_russian(&mut work, &mut p, &mut q);
            // Compare against the naive rank.
            let r_naive = original.rank();
            assert_eq!(
                r_ple, r_naive,
                "ple_russian rank disagrees with Mzd::rank at seed={seed:#x}"
            );
        }
    }

    /// Same property at smaller N to characterise the bug surface.
    #[test]
    fn ple_russian_rank_matches_naive_smaller() {
        for &seed in &[0xDEAD_BEEFu64, 0xDEAD_BEEF_DEAD_BEEFu64] {
            for &n in &[64usize, 65, 128, 129, 192, 256, 257, 384] {
                let original = Mzd::random(n, n, seed);
                let mut work = original.clone();
                let mut p = vec![0usize; n];
                let mut q = vec![0usize; n];
                let r_ple = ple_russian(&mut work, &mut p, &mut q);
                let r_naive = original.rank();
                assert_eq!(
                    r_ple, r_naive,
                    "ple_russian rank disagrees at N={n} seed={seed:#x}"
                );
            }
        }
    }

    /// Rank-deficient input: a 16×16 matrix of rank 8 (two copies of
    /// the same 8 rows). Exercises the find_first_pivot rank-deficient
    /// path inside ple_russian.
    #[test]
    fn ple_russian_handles_rank_deficient() {
        let n = 16usize;
        let mut a = Mzd::new(n, n);
        for r in 0..8 {
            for c in 0..n {
                if ((r + c) ^ 0xA5) & 1 == 1 {
                    a.set(r, c, 1);
                    a.set(r + 8, c, 1); // duplicate the row
                }
            }
        }
        let original_rank = a.rank();
        let mut work = a.clone();
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let r_ple = ple_russian(&mut work, &mut p, &mut q);
        assert_eq!(r_ple, original_rank);
        assert!(r_ple < n, "rank-deficient input should have rank < n");
    }

    /// Identity input: full rank, every pivot lands at the
    /// diagonal column, every q[i] = i.
    #[test]
    fn ple_russian_identity_input() {
        let n = 64usize;
        let mut a = Mzd::identity(n);
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let r = ple_russian(&mut a, &mut p, &mut q);
        assert_eq!(r, n);
        // After identity input, no row swaps occurred — p stays identity.
        for (i, &pi) in p.iter().enumerate().take(n) {
            assert_eq!(pi, i, "identity input shouldn't induce row swaps");
        }
        // The work matrix is still the identity (PLE leaves
        // already-isolated pivots alone).
        assert_eq!(a.as_words(), Mzd::identity(n).as_words());
    }

    /// All-zero input: rank 0, ple_russian returns immediately
    /// via find_first_pivot reporting no pivot.
    #[test]
    fn ple_russian_all_zero() {
        let n = 32usize;
        let mut a = Mzd::new(n, n);
        let mut p = vec![0usize; n];
        let mut q = vec![0usize; n];
        let r = ple_russian(&mut a, &mut p, &mut q);
        assert_eq!(r, 0);
    }

    /// Sanity check on ple_a10: when start_col == 0 (no A10
    /// region), the function is a no-op.
    #[test]
    fn ple_a10_no_op_when_start_col_zero() {
        let n = 8usize;
        let a = Mzd::random(n, n, 0xA10A);
        let mut work = a.clone();
        let pivots = vec![0usize, 1, 2, 3];
        let done = vec![0usize, 1, 2, 3];
        ple_a10(&mut work, 0, 0, 3, 4, &pivots, &done);
        assert_eq!(work.as_words(), a.as_words(), "start_col=0 should be no-op");
    }
}
