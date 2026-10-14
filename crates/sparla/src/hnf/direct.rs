// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Direct sparse Hermite Normal Form over `Z`.
//!
//! ## Algorithm
//!
//! Column-by-column over `0..ncols`:
//!
//! 1. Collect active rows that have a nonzero in this column ("the
//!    candidates").
//! 2. Euclidean-reduce candidates pairwise: repeatedly sort by
//!    `|value at col c|`, take the two smallest, and apply
//!    `row[o] -= floor(o_val / p_val) · row[p]`. When the second
//!    row's `col c` entry hits zero, drop it; otherwise the
//!    remainder is strictly smaller than the previous pivot, so the
//!    process terminates (Euclid).
//! 3. The single surviving candidate is the pivot row for `col c`;
//!    its `col c` entry is the GCD of all original candidates' `col c`
//!    entries, signed.
//!
//! After all columns are processed the result is in row echelon form
//! over `Z` but not yet canonical: pivots may be negative, and
//! entries above a pivot may be outside `[0, pivot)`. [`hnf`] runs
//! the canonical post-pass; the pre-canonical [`column_reduce_z`]
//! stays `pub(crate)` because the SNF kernel reuses it as-is —
//! dense SNF doesn't care about above-pivot reduction.

use std::cmp::Ordering;
use std::mem;

use crate::elim::reduce_z::{row_axpy, row_get, row_negate};
use crate::storage::{CsrZ, SparseRingStorage};
use crate::Integer;

/// Pre-canonical column reduction over `Z`. The internal state the
/// SNF kernel reuses.
pub(crate) struct ColumnReductionZ {
    /// Working rows in the order they came in. Pivot rows live at
    /// the indices given by `pivot_rows`; non-pivot rows are empty
    /// (column reduction zeroed them).
    pub working: Vec<Vec<(usize, Integer)>>,
    /// Pivot columns in ascending order. Length equals the rank.
    pub pivot_cols: Vec<usize>,
    /// `pivot_rows[i]` is the index into `working` of the row that
    /// became the pivot for `pivot_cols[i]`.
    pub pivot_rows: Vec<usize>,
}

/// Run the column-by-column Euclidean reduction without the
/// canonical-HNF post-pass. `pub(crate)` because SNF reuses this
/// directly.
pub(crate) fn column_reduce_z(matrix: &CsrZ) -> ColumnReductionZ {
    let nrows = matrix.rows();
    let ncols = matrix.cols();

    let mut working: Vec<Vec<(usize, Integer)>> = (0..nrows)
        .map(|r| matrix.iter_row(r).map(|(c, v)| (c, v.clone())).collect())
        .collect();

    let mut active = vec![true; nrows];
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut pivot_rows: Vec<usize> = Vec::new();

    for c in 0..ncols {
        let mut candidates: Vec<usize> = (0..nrows)
            .filter(|&r| active[r] && row_get(&working[r], c).is_some())
            .collect();
        if candidates.is_empty() {
            continue;
        }

        while candidates.len() > 1 {
            // Sort by |value at col c| ascending.
            candidates.sort_by(|&a, &b| {
                let av = row_get(&working[a], c).unwrap().abs();
                let bv = row_get(&working[b], c).unwrap().abs();
                av.cmp(&bv)
            });
            let p_idx = candidates[0];
            let o_idx = candidates[1];

            // q = floor(mat[o][c] / mat[p][c]). Floor-division gives
            // the Python `divmod` semantics that the reference uses.
            let q: Integer = {
                let ov = row_get(&working[o_idx], c).unwrap();
                let pv = row_get(&working[p_idx], c).unwrap();
                ov.div_floor(pv)
            };

            if q.cmp0() != Ordering::Equal {
                // working[o_idx] -= q * working[p_idx].
                // Take pivot row out so we can borrow it while
                // mutating the other row; put it back after.
                let src = mem::take(&mut working[p_idx]);
                row_axpy(&mut working[o_idx], &src, &q);
                working[p_idx] = src;
            }

            if row_get(&working[o_idx], c).is_none() {
                // Remainder was exactly zero — this row is no
                // longer a candidate. Don't change `active`; the
                // row may still have nonzeros in later columns.
                candidates.remove(1);
            }
            // Else: a smaller remainder lives at o_idx; the next
            // iteration's sort will swap roles.
        }

        let p = candidates[0];
        active[p] = false;
        pivot_cols.push(c);
        pivot_rows.push(p);
    }

    ColumnReductionZ {
        working,
        pivot_cols,
        pivot_rows,
    }
}

/// Exact `Z`-rank of `matrix` — the number of pivot columns
/// produced by [`column_reduce_z`].
///
/// Same shape as [`hnf`] but skips the canonical post-pass: no
/// sign-normalisation, no above-pivot reduction. Cheaper when only
/// the rank is needed.
pub fn rank(matrix: &CsrZ) -> usize {
    column_reduce_z(matrix).pivot_cols.len()
}

/// Canonical Hermite Normal Form over `Z`.
///
/// Output rows are in canonical order: pivot rows first, sorted by
/// pivot column (ascending), then any all-zero rows.
///
/// Properties of the result:
/// - Upper-triangular: row `i`'s leading nonzero is at column
///   `pivot_cols[i]`, which is strictly greater than row `i-1`'s
///   pivot column.
/// - Each pivot is positive.
/// - Each above-pivot entry `(i, pivot_cols[k])` with `i < k`
///   satisfies `0 ≤ entry < pivot_cols[k]`'s pivot value.
pub fn hnf(matrix: &CsrZ) -> CsrZ {
    let nrows = matrix.rows();
    let ncols = matrix.cols();

    let ColumnReductionZ {
        mut working,
        pivot_cols,
        pivot_rows,
    } = column_reduce_z(matrix);

    // Canonical post-pass: walk pivots in ascending pivot-col order,
    // sign-normalising and reducing entries above each new pivot in
    // every earlier pivot row.
    for (i, &p_idx) in pivot_rows.iter().enumerate() {
        let pc = pivot_cols[i];

        // Pivot positive.
        if row_get(&working[p_idx], pc).unwrap().cmp0() == Ordering::Less {
            row_negate(&mut working[p_idx]);
        }

        // Reduce earlier pivot rows' entry at column pc into
        // [0, pivot_val).
        let pivot_val: Integer = row_get(&working[p_idx], pc).unwrap().clone();
        let pivot_row = mem::take(&mut working[p_idx]);
        for &prev_p_idx in &pivot_rows[..i] {
            let coeff_opt = row_get(&working[prev_p_idx], pc).cloned();
            if let Some(coeff) = coeff_opt {
                let q: Integer = coeff.div_floor(&pivot_val);
                if q.cmp0() != Ordering::Equal {
                    row_axpy(&mut working[prev_p_idx], &pivot_row, &q);
                }
            }
        }
        working[p_idx] = pivot_row;
    }

    // Emit in canonical row order: pivot rows in ascending pivot
    // column, then all-zero rows.
    let pivot_set: std::collections::HashSet<usize> = pivot_rows.iter().copied().collect();
    let mut out_rows: Vec<Vec<(usize, Integer)>> = Vec::with_capacity(nrows);
    for &p in &pivot_rows {
        out_rows.push(mem::take(&mut working[p]));
    }
    for (r, row) in working.iter_mut().enumerate() {
        if !pivot_set.contains(&r) {
            debug_assert!(
                row.is_empty(),
                "non-pivot row {} should be empty after column reduction",
                r
            );
            out_rows.push(mem::take(row));
        }
    }

    CsrZ::from_sorted_rows(nrows, ncols, out_rows)
}

/// Same data as [`ColumnReductionZ`] plus the unimodular `m × m`
/// transform matrix `U` such that `U · A = working` (in canonical
/// row order — see [`hnf_with_transforms`] for the final emit).
pub(crate) struct ColumnReductionZWithTransforms {
    /// Working representation of `U · A`.
    pub working: Vec<Vec<(usize, Integer)>>,
    /// Pivot columns of the input, ascending. Length = rank.
    pub pivot_cols: Vec<usize>,
    /// `pivot_rows[i]` is the index into `working` of the row that
    /// became the pivot for `pivot_cols[i]`.
    pub pivot_rows: Vec<usize>,
    /// The accumulated transform matrix `U`. Starts as identity and
    /// mirrors every row op applied to `working`. Sparse rows, same
    /// canonical invariant as `working`.
    pub transforms: Vec<Vec<(usize, Integer)>>,
}

/// Column reduction over `Z` with the unimodular transform tracked.
///
/// Same algorithm as [`column_reduce_z`], but every row op applied
/// to the working matrix is mirrored on a parallel `U` that starts
/// as the `m × m` identity. After this returns, `U · A = working`.
pub(crate) fn column_reduce_z_with_transforms(matrix: &CsrZ) -> ColumnReductionZWithTransforms {
    let nrows = matrix.rows();
    let ncols = matrix.cols();

    let mut working: Vec<Vec<(usize, Integer)>> = (0..nrows)
        .map(|r| matrix.iter_row(r).map(|(c, v)| (c, v.clone())).collect())
        .collect();
    let mut transforms: Vec<Vec<(usize, Integer)>> =
        (0..nrows).map(|r| vec![(r, Integer::from(1))]).collect();

    let mut active = vec![true; nrows];
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut pivot_rows: Vec<usize> = Vec::new();

    for c in 0..ncols {
        let mut candidates: Vec<usize> = (0..nrows)
            .filter(|&r| active[r] && row_get(&working[r], c).is_some())
            .collect();
        if candidates.is_empty() {
            continue;
        }

        while candidates.len() > 1 {
            candidates.sort_by(|&a, &b| {
                let av = row_get(&working[a], c).unwrap().abs();
                let bv = row_get(&working[b], c).unwrap().abs();
                av.cmp(&bv)
            });
            let p_idx = candidates[0];
            let o_idx = candidates[1];

            let q: Integer = {
                let ov = row_get(&working[o_idx], c).unwrap();
                let pv = row_get(&working[p_idx], c).unwrap();
                ov.div_floor(pv)
            };

            if q.cmp0() != Ordering::Equal {
                // working[o] -= q · working[p], mirrored on transforms.
                let src = mem::take(&mut working[p_idx]);
                row_axpy(&mut working[o_idx], &src, &q);
                working[p_idx] = src;

                let tsrc = mem::take(&mut transforms[p_idx]);
                row_axpy(&mut transforms[o_idx], &tsrc, &q);
                transforms[p_idx] = tsrc;
            }

            if row_get(&working[o_idx], c).is_none() {
                candidates.remove(1);
            }
        }

        let p = candidates[0];
        active[p] = false;
        pivot_cols.push(c);
        pivot_rows.push(p);
    }

    ColumnReductionZWithTransforms {
        working,
        pivot_cols,
        pivot_rows,
        transforms,
    }
}

/// Canonical Hermite Normal Form over `Z` paired with the
/// unimodular transform: `(H, U)` such that `U · A = H` and `U` is
/// invertible over `Z` (`det(U) ∈ {±1}`).
///
/// `H` is in canonical row order (pivot rows first, sorted by pivot
/// column, then any all-zero rows) with positive pivots and
/// above-pivot entries in `[0, pivot_value)` — same shape as
/// [`hnf`].
///
/// The transform `U` is `m × m`, sparse storage. It records every
/// row operation applied during column reduction and the canonical
/// post-pass, so `U · A == H` holds bit-for-bit and can be
/// verified via [`crate::matmul::matmul_z`].
pub fn hnf_with_transforms(matrix: &CsrZ) -> (CsrZ, CsrZ) {
    let nrows = matrix.rows();
    let ncols = matrix.cols();

    let ColumnReductionZWithTransforms {
        mut working,
        pivot_cols,
        pivot_rows,
        mut transforms,
    } = column_reduce_z_with_transforms(matrix);

    // Canonical post-pass: mirror sign-normalisation and above-pivot
    // reduction on `transforms`.
    for (i, &p_idx) in pivot_rows.iter().enumerate() {
        let pc = pivot_cols[i];

        if row_get(&working[p_idx], pc).unwrap().cmp0() == Ordering::Less {
            row_negate(&mut working[p_idx]);
            row_negate(&mut transforms[p_idx]);
        }

        let pivot_val: Integer = row_get(&working[p_idx], pc).unwrap().clone();
        let pivot_row = mem::take(&mut working[p_idx]);
        let pivot_trans = mem::take(&mut transforms[p_idx]);
        for &prev_p_idx in &pivot_rows[..i] {
            let coeff_opt = row_get(&working[prev_p_idx], pc).cloned();
            if let Some(coeff) = coeff_opt {
                let q: Integer = coeff.div_floor(&pivot_val);
                if q.cmp0() != Ordering::Equal {
                    row_axpy(&mut working[prev_p_idx], &pivot_row, &q);
                    row_axpy(&mut transforms[prev_p_idx], &pivot_trans, &q);
                }
            }
        }
        working[p_idx] = pivot_row;
        transforms[p_idx] = pivot_trans;
    }

    // Emit in canonical row order: pivot rows first, then non-pivot
    // rows of `working` (those are all zero by column-reduction
    // invariant; their corresponding `transforms` rows are the
    // unimodular "witness" that the original row of A was a
    // Z-linear combination of pivot rows and hence reduced to 0).
    let pivot_set: std::collections::HashSet<usize> = pivot_rows.iter().copied().collect();
    let mut h_rows: Vec<Vec<(usize, Integer)>> = Vec::with_capacity(nrows);
    let mut u_rows: Vec<Vec<(usize, Integer)>> = Vec::with_capacity(nrows);
    for &p in &pivot_rows {
        h_rows.push(mem::take(&mut working[p]));
        u_rows.push(mem::take(&mut transforms[p]));
    }
    for (r, (w, t)) in working.iter_mut().zip(transforms.iter_mut()).enumerate() {
        if !pivot_set.contains(&r) {
            debug_assert!(w.is_empty());
            h_rows.push(mem::take(w));
            u_rows.push(mem::take(t));
        }
    }

    let h = CsrZ::from_sorted_rows(nrows, ncols, h_rows);
    let u = CsrZ::from_sorted_rows(nrows, nrows, u_rows);
    (h, u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    fn csr_from_dense(dense: &[Vec<i64>]) -> CsrZ {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
        for (r, row) in dense.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0 {
                    triples.push((r, c, z(v)));
                }
            }
        }
        CsrZ::from_coo(rows, cols, &triples)
    }

    fn dense_of(m: &CsrZ) -> Vec<Vec<Integer>> {
        m.to_dense()
    }

    #[test]
    fn empty_matrix_hnf_is_empty() {
        let h = hnf(&CsrZ::zeros(0, 0));
        assert_eq!(h.rows(), 0);
        assert_eq!(h.cols(), 0);
        assert_eq!(h.nnz(), 0);
    }

    #[test]
    fn zero_matrix_hnf_is_zero() {
        let h = hnf(&CsrZ::zeros(3, 4));
        assert_eq!(h.nnz(), 0);
        assert_eq!(h.rows(), 3);
        assert_eq!(h.cols(), 4);
    }

    #[test]
    fn identity_is_its_own_hnf() {
        let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        let h = hnf(&a);
        assert_eq!(
            dense_of(&h),
            vec![
                vec![z(1), z(0), z(0)],
                vec![z(0), z(1), z(0)],
                vec![z(0), z(0), z(1)]
            ]
        );
    }

    #[test]
    fn negative_pivot_gets_sign_normalised() {
        // Single row with negative leading entry. HNF must flip
        // signs so the pivot is positive.
        let a = csr_from_dense(&[vec![-3, 6]]);
        let h = hnf(&a);
        assert_eq!(dense_of(&h), vec![vec![z(3), z(-6)]]);
        // Hmm: after sign flip the entry at col 1 is -6, which is
        // not in [0, 3). But there's no pivot at col 1 to reduce
        // it against. So canonical HNF just sign-flips the pivot
        // row; that's correct.
    }

    #[test]
    fn two_rows_with_common_factor() {
        // [[2, 4], [6, 8]] — GCD of column 0 is 2.
        // HNF: row 0 = [2, 4], row 1 = [0, 4·6/2 - 8] worked by hand:
        //   q = 6/2 = 3; row 1 -= 3 * row 0 = [6 - 6, 8 - 12] = [0, -4].
        //   Process col 1: pivot val from row 1 = -4 → flip → 4.
        //   Reduce above: row 0[col 1] = 4. q = floor(4/4) = 1.
        //     row 0 -= 1 * row 1 = [2, 4 - 4] = [2, 0].
        // → [[2, 0], [0, 4]].
        let a = csr_from_dense(&[vec![2, 4], vec![6, 8]]);
        let h = hnf(&a);
        assert_eq!(dense_of(&h), vec![vec![z(2), z(0)], vec![z(0), z(4)]]);
    }
}
