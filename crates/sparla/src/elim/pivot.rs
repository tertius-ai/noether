// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Pivot-row selection for sparse Gaussian elimination.
//!
//! The first heuristic — "min-nnz active row that has a nonzero in
//! the target column" — is a degenerate Markowitz: it minimises the
//! cost of the broadcast row op that immediately follows, ignoring
//! the column count. Cheap to compute, deterministic, and a strong
//! baseline. It deliberately avoids the column-counting cost of the full
//! Markowitz product `(rnnz-1)(cnnz-1)`.

/// Pick the row to use as pivot for `col`, from among the active
/// rows that have a nonzero at `col`.
///
/// `rows[r]` is the sparse representation of row `r` as a slice of
/// `(col, value)` pairs, sorted by column. `active[r] == true` means
/// row `r` has not yet been used as a pivot and is eligible.
///
/// Returns `None` if no active row has a nonzero in `col` (i.e. the
/// column is not a pivot column).
///
/// Determinism: among rows tied on minimum nnz, the smallest row
/// index wins (the iteration order is `0..rows`, and the comparison
/// is strict `<`, so the first minimum holds).
pub(crate) fn select_pivot_row(
    rows: &[Vec<(usize, u32)>],
    active: &[bool],
    col: usize,
) -> Option<usize> {
    debug_assert_eq!(rows.len(), active.len());
    let mut best: Option<(usize, usize)> = None; // (nnz, row_index)
    for (r, row) in rows.iter().enumerate() {
        if !active[r] {
            continue;
        }
        if row.binary_search_by_key(&col, |e| e.0).is_ok() {
            let nnz = row.len();
            match best {
                None => best = Some((nnz, r)),
                Some((bn, _)) if nnz < bn => best = Some((nnz, r)),
                _ => {}
            }
        }
    }
    best.map(|(_, r)| r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_when_column_is_empty() {
        let rows = vec![vec![(0, 1u32)], vec![(1, 1u32)]];
        let active = vec![true, true];
        assert_eq!(select_pivot_row(&rows, &active, 2), None);
    }

    #[test]
    fn picks_minimum_nnz_row() {
        // row 0: nnz 3, has col 1
        // row 1: nnz 1, has col 1
        // row 2: nnz 2, has col 1
        // expect row 1.
        let rows = vec![
            vec![(0, 1u32), (1, 2u32), (2, 3u32)],
            vec![(1, 5u32)],
            vec![(0, 1u32), (1, 4u32)],
        ];
        let active = vec![true, true, true];
        assert_eq!(select_pivot_row(&rows, &active, 1), Some(1));
    }

    #[test]
    fn skips_inactive_rows() {
        // row 0 has col 1 but is inactive; pick row 2.
        let rows = vec![vec![(1, 7u32)], vec![(0, 3u32)], vec![(1, 2u32), (3, 5u32)]];
        let active = vec![false, true, true];
        assert_eq!(select_pivot_row(&rows, &active, 1), Some(2));
    }

    #[test]
    fn ties_break_to_smallest_row_index() {
        // rows 1 and 3 both have nnz 1 in col 0; expect row 1.
        let rows = vec![
            vec![(2, 1u32), (3, 2u32)],
            vec![(0, 1u32)],
            vec![(2, 4u32)],
            vec![(0, 5u32)],
        ];
        let active = vec![true, true, true, true];
        assert_eq!(select_pivot_row(&rows, &active, 0), Some(1));
    }

    #[test]
    fn skips_rows_missing_the_target_column() {
        // Only row 2 has col 5.
        let rows = vec![
            vec![(0, 1u32), (1, 1u32)],
            vec![(3, 1u32), (4, 1u32)],
            vec![(2, 1u32), (5, 1u32)],
        ];
        let active = vec![true, true, true];
        assert_eq!(select_pivot_row(&rows, &active, 5), Some(2));
    }
}
