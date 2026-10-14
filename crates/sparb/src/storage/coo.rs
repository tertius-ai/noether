// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! COO ingestion helper for [`CsrArb`].
//!
//! The user-facing input format is `(row, col, value)` triples —
//! easy to produce from any upstream pipeline (FEM element-loop
//! emits, a pyo3 glue layer, a test fixture). The `from_coo_*`
//! constructors funnel through [`build_csr`] for the sort /
//! dedupe-sum / drop-exact-zero / bounds-check dance, so the CSR
//! invariant is enforced in exactly one place.
//!
//! ## CSR invariant produced
//!
//! - `indptr.len() == rows + 1`, `indptr[0] == 0`,
//!   `indptr[rows] == data.len() == indices.len()`.
//! - Within each row, `indices[indptr[r]..indptr[r+1]]` is strictly
//!   increasing and every entry lies in `[0, cols)`.
//! - Every entry of `data` is a non-(exact-)zero ball.
//!
//! ## Duplicate handling
//!
//! Duplicate `(row, col)` triples are **summed** in ball arithmetic
//! at the matrix's working precision (scipy convention). If two
//! triples at the same `(row, col)` sum to the *exact* zero ball,
//! no entry is stored — but a sum that merely *contains* zero with
//! a nonzero radius **is** stored, because dropping it would discard
//! a rigorous enclosure.
//!
//! [`CsrArb`]: super::CsrArb

/// A COO triple: `(row, col, value)`.
pub type Triple<T> = (usize, usize, T);

/// Canonicalise a list of COO triples into the three CSR arrays.
///
/// `add_assign(acc, v)` must implement `acc += v` for the target
/// ring (ball addition at the working precision, for `Arb`).
/// `is_zero(acc)` must report exactly the values that should be
/// dropped from storage — for `Arb`, the *exact* zero ball only.
///
/// # Panics
///
/// Panics if any triple's `row >= rows` or `col >= cols`.
pub(crate) fn build_csr<T>(
    rows: usize,
    cols: usize,
    mut indexed: Vec<Triple<T>>,
    add_assign: impl Fn(&mut T, T),
    is_zero: impl Fn(&T) -> bool,
) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    for &(r, c, _) in indexed.iter() {
        assert!(
            r < rows,
            "sparb: COO row index {r} out of bounds (rows = {rows})"
        );
        assert!(
            c < cols,
            "sparb: COO col index {c} out of bounds (cols = {cols})"
        );
    }

    indexed.sort_by_key(|&(r, c, _)| (r, c));

    let mut indptr = vec![0usize; rows + 1];
    let mut indices: Vec<usize> = Vec::with_capacity(indexed.len());
    let mut data: Vec<T> = Vec::with_capacity(indexed.len());

    let mut iter = indexed.into_iter();
    let Some((mut cur_r, mut cur_c, mut acc)) = iter.next() else {
        return (indptr, indices, data);
    };

    for (r, c, v) in iter {
        if r == cur_r && c == cur_c {
            add_assign(&mut acc, v);
        } else {
            if !is_zero(&acc) {
                indices.push(cur_c);
                data.push(acc);
                indptr[cur_r + 1] += 1;
            }
            cur_r = r;
            cur_c = c;
            acc = v;
        }
    }
    if !is_zero(&acc) {
        indices.push(cur_c);
        data.push(acc);
        indptr[cur_r + 1] += 1;
    }

    // Per-row counts → cumulative offsets.
    for r in 0..rows {
        indptr[r + 1] += indptr[r];
    }
    (indptr, indices, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use i64 as a stand-in ring — `build_csr` is ring-generic
    // and the `Arb` instantiation is exercised in `csr_arb.rs`.

    #[test]
    fn empty_input_yields_zero_csr() {
        let (indptr, indices, data) = build_csr::<i64>(3, 4, vec![], |a, b| *a += b, |x| *x == 0);
        assert_eq!(indptr, vec![0; 4]);
        assert!(indices.is_empty());
        assert!(data.is_empty());
    }

    #[test]
    fn sorts_and_packs_unique_entries() {
        let (indptr, indices, data) = build_csr::<i64>(
            2,
            3,
            vec![(1, 2, 7), (0, 1, 3), (1, 0, 5)],
            |a, b| *a += b,
            |x| *x == 0,
        );
        assert_eq!(indptr, vec![0, 1, 3]);
        assert_eq!(indices, vec![1, 0, 2]);
        assert_eq!(data, vec![3, 5, 7]);
    }

    #[test]
    fn sums_duplicates_at_same_cell() {
        let (indptr, indices, data) = build_csr::<i64>(
            1,
            1,
            vec![(0, 0, 2), (0, 0, 3), (0, 0, 5)],
            |a, b| *a += b,
            |x| *x == 0,
        );
        assert_eq!(indptr, vec![0, 1]);
        assert_eq!(indices, vec![0]);
        assert_eq!(data, vec![10]);
    }

    #[test]
    fn duplicates_summing_to_zero_are_dropped() {
        let (indptr, indices, data) = build_csr::<i64>(
            1,
            2,
            vec![(0, 0, 5), (0, 1, 4), (0, 0, -5)],
            |a, b| *a += b,
            |x| *x == 0,
        );
        assert_eq!(indptr, vec![0, 1]);
        assert_eq!(indices, vec![1]);
        assert_eq!(data, vec![4]);
    }

    #[test]
    #[should_panic(expected = "row index 5 out of bounds")]
    fn out_of_bounds_row_panics() {
        let _ = build_csr::<i64>(2, 2, vec![(5, 0, 1)], |a, b| *a += b, |x| *x == 0);
    }

    #[test]
    #[should_panic(expected = "col index 9 out of bounds")]
    fn out_of_bounds_col_panics() {
        let _ = build_csr::<i64>(2, 2, vec![(0, 9, 1)], |a, b| *a += b, |x| *x == 0);
    }
}
