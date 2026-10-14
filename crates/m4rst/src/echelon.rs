// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Row echelon / reduced row echelon over F_2.
//!
//! Standard Gauss-Jordan: walk columns left-to-right, find a pivot
//! row, swap it up, and clear the pivot column in all other rows by
//! XOR. The `full` flag toggles between RREF (`full=true`, clear
//! above and below) and plain REF (`full=false`, only clear below).
//!
//! The unblocked implementation is checked against libm4ri's
//! `mzd_echelonize` in the feature-gated parity suite.
//!
//! ## References
//!
//! * Bard, *Accelerating Cryptanalysis with the Method of Four
//!   Russians*, IACR ePrint 2006/251 — the original Gray-code-table
//!   blocked elimination over F_2.
//! * Albrecht, Bard, Hart, *Algorithm 898: Efficient Multiplication
//!   of Dense Matrices over GF(2)*, ACM TOMS 37(1):9, 2010 — the
//!   M4R-block elimination engineering used in libm4ri's
//!   `echelonform.c` (`mzd_echelonize`, `mzd_echelonize_m4rst`).
//! * Albrecht, Pernet, *Efficient Decomposition of Dense Matrices
//!   over GF(2)*, [arXiv:1006.1744](https://arxiv.org/abs/1006.1744),
//!   2010 — the PLE-to-RREF reduction that backs
//!   `mzd_echelonize_pluq`, the asymptotically-best path.
//!
//! This module is the unblocked Gauss-Jordan correctness floor.

use crate::{mzd::Mzd, xor::xor_words};

/// Build a packed row-space basis for one word-aligned column panel.
///
/// This low-level operation lets distributed-memory callers keep MPI
/// ownership outside `m4rst` while delegating rank-local packed elimination to
/// the local kernel crate. Every returned row starts at `panel_word` and
/// retains the full suffix needed to update trailing columns.
#[doc(hidden)]
pub fn packed_word_basis<'a>(
    matrices: impl IntoIterator<Item = &'a Mzd>,
    panel_word: usize,
    panel_width: usize,
) -> Vec<Vec<u64>> {
    assert!(panel_width <= crate::mzd::RADIX);
    if panel_width == 0 {
        return Vec::new();
    }
    let mut basis: Vec<Option<Vec<u64>>> = vec![None; panel_width];
    for matrix in matrices {
        assert!(panel_word < matrix.width());
        for row in 0..matrix.rows() {
            let mut candidate = matrix.row_words(row)[panel_word..].to_vec();
            for (pivot, pivot_row) in basis.iter_mut().enumerate() {
                if candidate[0] & (1_u64 << pivot) == 0 {
                    continue;
                }
                if let Some(words) = pivot_row {
                    xor_words(&mut candidate, words);
                } else {
                    *pivot_row = Some(candidate);
                    break;
                }
            }
        }
    }
    basis.into_iter().flatten().collect()
}

/// Apply a flattened basis returned by [`packed_word_basis`] to one matrix.
///
/// `payload` contains consecutive full-suffix rows beginning at
/// `panel_word`. The routine clears each represented pivot from every local
/// row using the runtime-dispatched packed XOR primitive.
#[doc(hidden)]
pub fn apply_packed_word_basis(matrix: &mut Mzd, panel_word: usize, payload: &[u64]) {
    let rows = 0..matrix.rows();
    apply_packed_word_basis_to_rows(matrix, rows, panel_word, payload);
}

/// Apply a flattened packed basis to a selected range of local rows.
///
/// This is the bounded-row counterpart to [`apply_packed_word_basis`]. It is
/// used by distributed callers during a reverse panel sweep, where later
/// pivots must update only the rows preceding their panel.
#[doc(hidden)]
pub fn apply_packed_word_basis_to_rows(
    matrix: &mut Mzd,
    rows: std::ops::Range<usize>,
    panel_word: usize,
    payload: &[u64],
) {
    assert!(rows.start <= rows.end && rows.end <= matrix.rows());
    assert!(panel_word < matrix.width());
    let suffix_words = matrix.width() - panel_word;
    assert!(payload.len().is_multiple_of(suffix_words));
    for row in rows {
        let destination = &mut matrix.row_words_mut(row)[panel_word..];
        for pivot_row in payload.chunks_exact(suffix_words) {
            let pivot = pivot_row[0].trailing_zeros();
            debug_assert!(pivot < crate::mzd::RADIX as u32);
            if destination[0] & (1_u64 << pivot) != 0 {
                xor_words(destination, pivot_row);
            }
        }
    }
}

/// Build a packed row-space basis and carry a row-aligned companion through
/// the same elimination.
///
/// The coefficient matrices determine pivots in the word-aligned panel. Each
/// companion row receives exactly the XOR operations applied to its matching
/// coefficient row. This supports augmented distributed elimination without
/// concatenating the full coefficient and companion matrices.
#[doc(hidden)]
pub fn packed_word_basis_with_companion<'a>(
    matrices: impl IntoIterator<Item = (&'a Mzd, &'a Mzd)>,
    panel_word: usize,
    panel_width: usize,
) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    assert!(panel_width <= crate::mzd::RADIX);
    if panel_width == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut basis: Vec<Option<(Vec<u64>, Vec<u64>)>> = vec![None; panel_width];
    for (coefficients, companion) in matrices {
        assert_eq!(coefficients.rows(), companion.rows());
        assert!(panel_word < coefficients.width());
        for row in 0..coefficients.rows() {
            let mut coefficient_candidate = coefficients.row_words(row)[panel_word..].to_vec();
            let mut companion_candidate = companion.row_words(row).to_vec();
            for (pivot, pivot_row) in basis.iter_mut().enumerate() {
                if coefficient_candidate[0] & (1_u64 << pivot) == 0 {
                    continue;
                }
                if let Some((pivot_coefficients, pivot_companion)) = pivot_row {
                    xor_words(&mut coefficient_candidate, pivot_coefficients);
                    xor_words(&mut companion_candidate, pivot_companion);
                } else {
                    *pivot_row = Some((coefficient_candidate, companion_candidate));
                    break;
                }
            }
        }
    }
    basis.into_iter().flatten().unzip()
}

/// Apply paired packed coefficient and companion basis payloads to selected
/// row-aligned matrices.
#[doc(hidden)]
pub fn apply_packed_word_basis_with_companion_to_rows(
    coefficients: &mut Mzd,
    companion: &mut Mzd,
    rows: std::ops::Range<usize>,
    panel_word: usize,
    coefficient_payload: &[u64],
    companion_payload: &[u64],
) {
    assert_eq!(coefficients.rows(), companion.rows());
    assert!(rows.start <= rows.end && rows.end <= coefficients.rows());
    assert!(panel_word < coefficients.width());
    let coefficient_suffix_words = coefficients.width() - panel_word;
    assert!(coefficient_payload
        .len()
        .is_multiple_of(coefficient_suffix_words));
    let basis_count = coefficient_payload.len() / coefficient_suffix_words;
    let companion_words = companion.width();
    assert_eq!(companion_payload.len(), basis_count * companion_words);

    for row in rows {
        let coefficient_destination = &mut coefficients.row_words_mut(row)[panel_word..];
        let companion_destination = companion.row_words_mut(row);
        for basis_index in 0..basis_count {
            let coefficient_start = basis_index * coefficient_suffix_words;
            let coefficient_row = &coefficient_payload
                [coefficient_start..coefficient_start + coefficient_suffix_words];
            let pivot = coefficient_row[0].trailing_zeros();
            debug_assert!(pivot < crate::mzd::RADIX as u32);
            if coefficient_destination[0] & (1_u64 << pivot) != 0 {
                xor_words(coefficient_destination, coefficient_row);
                let companion_start = basis_index * companion_words;
                xor_words(
                    companion_destination,
                    &companion_payload[companion_start..companion_start + companion_words],
                );
            }
        }
    }
}

/// Reduce one ordered packed-word basis to RREF while carrying a companion.
///
/// Rows must have distinct, increasing pivots in `panel_word`; this is the
/// representation returned by [`packed_word_basis_with_companion`] after
/// ordering by pivot. The returned values are absolute coefficient columns.
#[doc(hidden)]
pub fn rref_packed_word_basis_with_companion(
    coefficients: &mut Mzd,
    companion: &mut Mzd,
    panel_word: usize,
) -> Vec<usize> {
    assert_eq!(coefficients.rows(), companion.rows());
    if coefficients.rows() == 0 {
        return Vec::new();
    }
    assert!(panel_word < coefficients.width());
    let panel_start = panel_word * crate::mzd::RADIX;
    let panel_width = (coefficients.cols() - panel_start).min(crate::mzd::RADIX);
    let mut pivots = Vec::with_capacity(coefficients.rows());
    for row in 0..coefficients.rows() {
        let panel = coefficients.row_words(row)[panel_word];
        assert_ne!(panel, 0);
        let pivot = panel.trailing_zeros() as usize;
        assert!(pivot < panel_width);
        assert!(pivots.last().is_none_or(|previous| *previous < pivot));
        pivots.push(pivot);
    }

    for pivot_index in (0..coefficients.rows()).rev() {
        let pivot = pivots[pivot_index];
        let pivot_coefficients = coefficients.row_words(pivot_index)[panel_word..].to_vec();
        let pivot_companion = companion.row_words(pivot_index).to_vec();
        for row in 0..pivot_index {
            if coefficients.row_words(row)[panel_word] & (1_u64 << pivot) != 0 {
                xor_words(
                    &mut coefficients.row_words_mut(row)[panel_word..],
                    &pivot_coefficients,
                );
                xor_words(companion.row_words_mut(row), &pivot_companion);
            }
        }
    }
    pivots
        .into_iter()
        .map(|pivot| panel_start + pivot)
        .collect()
}

/// Find the first row in `rows` whose `pivot_col` bit is set.
///
/// Distributed elimination uses this rank-local scan before MPI chooses the
/// globally earliest candidate. Keeping the packed lookup here avoids a
/// second implementation of local pivot search in the MPI crate.
#[doc(hidden)]
pub fn first_pivot_row(
    matrix: &Mzd,
    rows: std::ops::Range<usize>,
    pivot_col: usize,
) -> Option<usize> {
    first_pivot_row_where(matrix, rows, pivot_col, |_| true)
}

/// Find the first admissible row whose `pivot_col` bit is set.
///
/// The predicate receives a local row index. This variant supports exact
/// augmented elimination, where previously chosen equation rows remain in
/// place and must be excluded from subsequent local pivot searches.
#[doc(hidden)]
pub fn first_pivot_row_where(
    matrix: &Mzd,
    rows: std::ops::Range<usize>,
    pivot_col: usize,
    mut admissible: impl FnMut(usize) -> bool,
) -> Option<usize> {
    assert!(rows.start <= rows.end && rows.end <= matrix.rows());
    assert!(pivot_col < matrix.cols());
    rows.into_iter()
        .find(|&row| admissible(row) && matrix.get(row, pivot_col) != 0)
}

/// Apply one packed pivot row to selected local target rows.
///
/// `pivot` must contain exactly one row with its `pivot_col` bit set and must
/// have zero words before the word containing that pivot. The callback records
/// each eliminated row, allowing witness-producing callers to update their
/// lower factor without duplicating the packed bit test or XOR loop.
#[doc(hidden)]
pub fn apply_packed_pivot_to_rows(
    matrix: &mut Mzd,
    rows: std::ops::Range<usize>,
    pivot_col: usize,
    pivot: &Mzd,
    mut on_eliminated: impl FnMut(usize),
) {
    assert!(rows.start <= rows.end && rows.end <= matrix.rows());
    assert_eq!(pivot.rows(), 1);
    assert_eq!(pivot.cols(), matrix.cols());
    assert!(pivot_col < matrix.cols());
    assert_ne!(pivot.get(0, pivot_col), 0);
    let first_word = pivot_col / crate::mzd::RADIX;
    debug_assert!(pivot.row_words(0)[..first_word]
        .iter()
        .all(|word| *word == 0));
    for row in rows {
        if matrix.get(row, pivot_col) != 0 {
            xor_words(
                &mut matrix.row_words_mut(row)[first_word..],
                &pivot.row_words(0)[first_word..],
            );
            on_eliminated(row);
        }
    }
}

impl Mzd {
    /// Reduce `self` to row echelon form in place; returns the rank.
    ///
    /// If `full == true`, produces reduced row echelon form
    /// (RREF, à la Gauss-Jordan): every pivot column has exactly one
    /// 1, in the pivot row. If `false`, only sub-pivot entries are
    /// cleared (REF — upper triangle stays).
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let mut a = Mzd::identity(8);
    /// assert_eq!(a.echelonize(true), 8);     // I is full-rank
    /// assert_eq!(a.as_words(), Mzd::identity(8).as_words()); // RREF of I = I
    /// ```
    pub fn echelonize(&mut self, full: bool) -> usize {
        if full && self.rows() >= 64 && self.cols() >= 64 {
            return self.echelonize_m4r();
        }
        self.echelonize_classical(full)
    }

    /// Classical row-at-a-time correctness floor used for small matrices and
    /// as the differential oracle for the blocked path.
    fn echelonize_classical(&mut self, full: bool) -> usize {
        let m = self.rows();
        let n = self.cols();
        let mut pivot_row = 0usize;
        for col in 0..n {
            if pivot_row >= m {
                break;
            }
            // Find a row at or below pivot_row with a 1 in this column.
            let mut r = pivot_row;
            while r < m && self.get(r, col) == 0 {
                r += 1;
            }
            if r == m {
                continue;
            }
            if r != pivot_row {
                self.swap_rows(pivot_row, r);
            }
            // Clear column `col` in other rows. If `full`, clear everywhere
            // else; otherwise only below the pivot.
            let start = if full { 0 } else { pivot_row + 1 };
            for i in start..m {
                if i != pivot_row && self.get(i, col) == 1 {
                    self.xor_row_self(i, pivot_row);
                }
            }
            pivot_row += 1;
        }
        pivot_row
    }

    /// Blocked Method-of-Four-Russians Gauss--Jordan elimination.
    ///
    /// Each panel first constructs at most eight pivot rows whose panel
    /// columns form an identity matrix.  Two 16-row tables encode the low and
    /// high halves of their linear combinations, then clear the whole panel
    /// from every other row in one fused packed-row pass.  Unlike the PLE
    /// engine this retains the original column order, so the result is the
    /// canonical RREF expected by the public API.
    fn echelonize_m4r(&mut self) -> usize {
        const PANEL_RANK: usize = 8;
        const TABLE_RANK: usize = PANEL_RANK / 2;

        let rows = self.rows();
        let cols = self.cols();
        let width = self.width();
        let mut pivot_row = 0usize;
        let mut next_col = 0usize;

        while pivot_row < rows && next_col < cols {
            let block_start = pivot_row;
            let mut pivot_cols = [0usize; PANEL_RANK];
            let mut block_rank = 0usize;

            while block_rank < PANEL_RANK && pivot_row < rows && next_col < cols {
                let col = next_col;
                next_col += 1;
                let mut found = None;
                for row in pivot_row..rows {
                    for (basis_index, &basis_col) in pivot_cols.iter().enumerate().take(block_rank)
                    {
                        if self.get(row, basis_col) != 0 {
                            self.xor_row_self(row, block_start + basis_index);
                        }
                    }
                    if self.get(row, col) != 0 {
                        found = Some(row);
                        break;
                    }
                }
                let Some(found_row) = found else {
                    continue;
                };
                self.swap_rows(pivot_row, found_row);
                for basis_index in 0..block_rank {
                    if self.get(block_start + basis_index, col) != 0 {
                        self.xor_row_self(block_start + basis_index, pivot_row);
                    }
                }
                pivot_cols[block_rank] = col;
                block_rank += 1;
                pivot_row += 1;
            }

            if block_rank == 0 {
                continue;
            }

            let first_word = pivot_cols[0] / 64;
            let suffix_width = width - first_word;
            let low_rank = block_rank.min(TABLE_RANK);
            let high_rank = block_rank - low_rank;
            let low_count = 1usize << low_rank;
            let high_count = 1usize << high_rank;
            let mut low_table = vec![0u64; low_count * suffix_width];
            let mut high_table = vec![0u64; high_count * suffix_width];
            for (basis_offset, table) in [(0usize, &mut low_table), (low_rank, &mut high_table)] {
                let table_rank = if basis_offset == 0 {
                    low_rank
                } else {
                    high_rank
                };
                for combination in 1..1usize << table_rank {
                    let basis_index = combination.trailing_zeros() as usize;
                    let previous = combination & (combination - 1);
                    let destination_start = combination * suffix_width;
                    let previous_start = previous * suffix_width;
                    let basis = self.row_words(block_start + basis_offset + basis_index);
                    for word in 0..suffix_width {
                        table[destination_start + word] =
                            table[previous_start + word] ^ basis[first_word + word];
                    }
                }
            }

            let consecutive = pivot_cols[..block_rank]
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 1);
            for row in 0..rows {
                if (block_start..block_start + block_rank).contains(&row) {
                    continue;
                }
                let combination = if consecutive {
                    self.extract_bits(row, pivot_cols[0], block_rank) as usize
                } else {
                    pivot_cols.iter().enumerate().take(block_rank).fold(
                        0usize,
                        |combination, (basis_index, &col)| {
                            combination | (usize::from(self.get(row, col)) << basis_index)
                        },
                    )
                };
                if combination == 0 {
                    continue;
                }
                let low = combination & (low_count - 1);
                let high = combination >> low_rank;
                let low_start = low * suffix_width;
                let high_start = high * suffix_width;
                let destination = &mut self.row_words_mut(row)[first_word..];
                for word in 0..suffix_width {
                    destination[word] ^=
                        low_table[low_start + word] ^ high_table[high_start + word];
                }
            }
        }
        pivot_row
    }

    /// Rank of `self`. Computes via a clone + RREF echelonize, so
    /// `self` is left unmodified. O(m · n · k) where k = min(m, n).
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// assert_eq!(Mzd::identity(64).rank(), 64);
    /// assert_eq!(Mzd::new(10, 20).rank(), 0);  // zero matrix
    /// ```
    pub fn rank(&self) -> usize {
        let mut copy = self.clone();
        copy.echelonize(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_packed_pivot_to_rows, apply_packed_word_basis, apply_packed_word_basis_to_rows,
        apply_packed_word_basis_with_companion_to_rows, first_pivot_row, first_pivot_row_where,
        packed_word_basis, packed_word_basis_with_companion, rref_packed_word_basis_with_companion,
    };
    use crate::{mzd::RADIX, xor::xor_words, Mzd};

    #[test]
    fn blocked_rref_matches_classical_across_neighboring_shapes() {
        for &(rows, cols) in &[(64, 64), (96, 129), (128, 192), (192, 192), (193, 257)] {
            for seed in [1, 0x5929, 0xDEAD_BEEF] {
                let source = Mzd::random(rows, cols, seed);
                let mut expected = source.clone();
                let expected_rank = expected.echelonize_classical(true);
                let mut actual = source;
                let actual_rank = actual.echelonize_m4r();
                assert_eq!(
                    actual_rank, expected_rank,
                    "shape {rows}x{cols}, seed {seed}"
                );
                assert_eq!(
                    actual.as_words(),
                    expected.as_words(),
                    "shape {rows}x{cols}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn blocked_rref_handles_rank_deficient_and_zero_neighbors() {
        let mut source = Mzd::random(96, 160, 0x1729);
        for row in 48..source.rows() {
            let source_row = row % 48;
            let values = source.row_words(source_row).to_vec();
            source.row_words_mut(row).copy_from_slice(&values);
        }
        let mut expected = source.clone();
        let expected_rank = expected.echelonize_classical(true);
        let mut actual = source;
        assert_eq!(actual.echelonize_m4r(), expected_rank);
        assert_eq!(actual.as_words(), expected.as_words());

        for (rows, cols) in [(0, 0), (0, 128), (128, 0)] {
            let mut matrix = Mzd::new(rows, cols);
            assert_eq!(matrix.echelonize_m4r(), 0);
            assert_eq!(matrix.rows(), rows);
            assert_eq!(matrix.cols(), cols);
        }
    }

    #[test]
    fn packed_word_basis_spans_multiple_local_blocks_and_clears_its_pivots() {
        let source = Mzd::random(97, 130, 0x1729);
        let first = source.submatrix(0, 0, 31, source.cols());
        let second = source.submatrix(31, 0, 72, source.cols());
        let third = source.submatrix(72, 0, source.rows(), source.cols());
        let mut blocks = [first, second, third];
        let basis = packed_word_basis(blocks.iter(), 0, RADIX);
        let projected = source.submatrix(0, 0, source.rows(), RADIX);
        assert_eq!(basis.len(), projected.rank());

        let payload: Vec<_> = basis.iter().flatten().copied().collect();
        let pivots: Vec<_> = basis
            .iter()
            .map(|row| row[0].trailing_zeros() as usize)
            .collect();
        for block in &mut blocks {
            apply_packed_word_basis(block, 0, &payload);
            for row in 0..block.rows() {
                assert!(pivots.iter().all(|&col| block.get(row, col) == 0));
            }
        }
    }

    #[test]
    fn packed_word_basis_can_update_only_a_row_prefix() {
        let source = Mzd::random(19, 130, 0x2026);
        let basis = packed_word_basis([&source], 1, RADIX);
        let payload: Vec<_> = basis.iter().flatten().copied().collect();
        let mut selected = source.clone();
        apply_packed_word_basis_to_rows(&mut selected, 0..7, 1, &payload);

        let mut all = source.clone();
        apply_packed_word_basis(&mut all, 1, &payload);
        for row in 0..7 {
            assert_eq!(selected.row_words(row), all.row_words(row));
        }
        for row in 7..source.rows() {
            assert_eq!(selected.row_words(row), source.row_words(row));
        }
    }

    #[test]
    fn packed_word_basis_carries_companion_through_rref() {
        let suffix = Mzd::random(19, 66, 0xA11CE);
        let mut coefficients = Mzd::new(19, 130);
        coefficients.paste_into(0, RADIX, &suffix);
        let companion = Mzd::random(19, 11, 0xB0B);
        let coefficient_first = coefficients.submatrix(0, 0, 7, coefficients.cols());
        let coefficient_second =
            coefficients.submatrix(7, 0, coefficients.rows(), coefficients.cols());
        let companion_first = companion.submatrix(0, 0, 7, companion.cols());
        let companion_second = companion.submatrix(7, 0, companion.rows(), companion.cols());
        let (coefficient_basis, companion_basis) = packed_word_basis_with_companion(
            [
                (&coefficient_first, &companion_first),
                (&coefficient_second, &companion_second),
            ],
            1,
            RADIX,
        );
        assert_eq!(coefficient_basis.len(), companion_basis.len());

        let mut basis_coefficients = Mzd::new(coefficient_basis.len(), coefficients.cols());
        let mut basis_companion = Mzd::new(companion_basis.len(), companion.cols());
        for (row, words) in coefficient_basis.iter().enumerate() {
            basis_coefficients.row_words_mut(row)[1..].copy_from_slice(words);
        }
        for (row, words) in companion_basis.iter().enumerate() {
            basis_companion.row_words_mut(row).copy_from_slice(words);
        }

        let mut augmented = Mzd::new(
            coefficient_basis.len(),
            coefficients.cols() + companion.cols(),
        );
        augmented.paste_into(0, 0, &basis_coefficients);
        augmented.paste_into(0, coefficients.cols(), &basis_companion);
        assert_eq!(augmented.echelonize(true), coefficient_basis.len());

        let pivots =
            rref_packed_word_basis_with_companion(&mut basis_coefficients, &mut basis_companion, 1);
        assert_eq!(
            basis_coefficients.as_words(),
            augmented
                .submatrix(0, 0, augmented.rows(), coefficients.cols())
                .as_words()
        );
        assert_eq!(
            basis_companion.as_words(),
            augmented
                .submatrix(0, coefficients.cols(), augmented.rows(), augmented.cols())
                .as_words()
        );

        let coefficient_payload: Vec<_> = coefficient_basis.into_iter().flatten().collect();
        let companion_payload: Vec<_> = companion_basis.into_iter().flatten().collect();
        let mut residual_coefficients = coefficients;
        let mut residual_companion = companion;
        let residual_rows = residual_coefficients.rows();
        apply_packed_word_basis_with_companion_to_rows(
            &mut residual_coefficients,
            &mut residual_companion,
            0..residual_rows,
            1,
            &coefficient_payload,
            &companion_payload,
        );
        for row in 0..residual_coefficients.rows() {
            assert!(pivots
                .iter()
                .all(|pivot| residual_coefficients.get(row, *pivot) == 0));
        }
    }

    #[test]
    fn packed_pivot_helpers_scan_apply_and_record_selected_rows() {
        let mut matrix = Mzd::random(13, 130, 0x51ced);
        let mut pivot = Mzd::new(1, matrix.cols());
        pivot.set(0, 70, 1);
        pivot.set(0, 93, 1);
        pivot.set(0, 129, 1);

        matrix.set(2, 70, 1);
        matrix.set(3, 70, 0);
        matrix.set(4, 70, 1);
        assert_eq!(first_pivot_row(&matrix, 3..11, 70), Some(4));
        assert_eq!(
            first_pivot_row_where(&matrix, 3..11, 70, |row| row > 4),
            (5..11).find(|&row| matrix.get(row, 70) != 0)
        );

        let mut expected = matrix.clone();
        let mut expected_rows = Vec::new();
        for row in 3..11 {
            if expected.get(row, 70) != 0 {
                xor_words(
                    &mut expected.row_words_mut(row)[1..],
                    &pivot.row_words(0)[1..],
                );
                expected_rows.push(row);
            }
        }
        let mut actual_rows = Vec::new();
        apply_packed_pivot_to_rows(&mut matrix, 3..11, 70, &pivot, |row| {
            actual_rows.push(row);
        });
        assert_eq!(actual_rows, expected_rows);
        assert_eq!(matrix.as_words(), expected.as_words());
    }
}
