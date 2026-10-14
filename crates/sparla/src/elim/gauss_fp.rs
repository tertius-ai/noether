// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse Gaussian elimination over `F_p` to reduced row echelon
//! form.
//!
//! Operates on a working representation `Vec<Vec<(col, val)>>` where
//! each inner vector is sorted by column and contains no zero
//! entries. Convert to/from [`CsrFp`] at the boundaries.
//!
//! The algorithm is the standard column-by-column sweep: for each
//! column, pick an active pivot row via [`select_pivot_row`],
//! normalise it to a leading 1, and eliminate the column in every
//! other row (including already-pivoted rows, so the output is full
//! RREF, not just REF). After all pivots are found, rows are
//! reordered so pivot rows come first sorted by pivot column, then
//! the all-zero rows.

use std::mem;

use crate::storage::{CsrFp, SparseRingStorage};

use super::pivot::select_pivot_row;

/// Result of reducing a sparse `F_p` matrix to RREF.
#[derive(Clone, Debug)]
pub struct RrefFp {
    /// The matrix in reduced row-echelon form: pivot rows first
    /// (sorted by pivot column) with each pivot a leading `1` and
    /// every other entry in its column zero; then any all-zero rows.
    pub matrix: CsrFp,
    /// Number of pivot columns — equivalently, the rank of the
    /// input matrix over `F_p`.
    pub rank: usize,
    /// Pivot columns of the input, in ascending order. Length
    /// `rank`. The pivot in column `pivot_cols[i]` lives at row `i`
    /// of [`Self::matrix`].
    pub pivot_cols: Vec<usize>,
}

/// Reduce a sparse `F_p` matrix to RREF via column-by-column sparse
/// Gaussian elimination.
///
/// `O(rows · cols · log row_nnz)` for the column-scan structure
/// alone, plus the elimination work. Fill-in is governed by the
/// pivot heuristic in [`select_pivot_row`].
pub fn rref(matrix: &CsrFp) -> RrefFp {
    let prime = matrix.modulus() as u32;
    let nrows = matrix.rows();
    let ncols = matrix.cols();

    let mut working: Vec<Vec<(usize, u32)>> =
        (0..nrows).map(|r| matrix.iter_row(r).collect()).collect();

    let mut active = vec![true; nrows];
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut pivot_rows: Vec<usize> = Vec::new();
    // Reused across every axpy so the merge doesn't allocate a fresh
    // result Vec per call (millions of calls on a large RREF).
    let mut scratch: Vec<(usize, u32)> = Vec::new();

    for c in 0..ncols {
        let p = match select_pivot_row(&working, &active, c) {
            None => continue,
            Some(p) => p,
        };

        // Normalise pivot row so the entry at col `c` is 1.
        let off = working[p]
            .binary_search_by_key(&c, |e| e.0)
            .expect("pivot row must have a nonzero at the pivot column");
        let val = working[p][off].1;
        if val != 1 {
            let inv = inv_mod_p(val, prime);
            scale_row(&mut working[p], inv, prime);
        }

        // Take the pivot row out so we can borrow it immutably while
        // mutating every other row. Put it back when the broadcast
        // completes.
        let pivot_row = mem::take(&mut working[p]);
        for (r, row) in working.iter_mut().enumerate() {
            if r == p {
                continue;
            }
            let off_r = match row.binary_search_by_key(&c, |e| e.0) {
                Ok(off) => off,
                Err(_) => continue,
            };
            let coeff = row[off_r].1;
            axpy_mod_p(row, &pivot_row, coeff, prime, &mut scratch);
        }
        working[p] = pivot_row;

        active[p] = false;
        pivot_cols.push(c);
        pivot_rows.push(p);
    }

    // Re-emit rows in canonical RREF order: pivot rows first
    // (already in ascending pivot-col order because we processed
    // columns left-to-right), then zero rows.
    let rank = pivot_cols.len();
    let mut out_rows: Vec<Vec<(usize, u32)>> = Vec::with_capacity(nrows);
    for &p in &pivot_rows {
        out_rows.push(mem::take(&mut working[p]));
    }
    for r in 0..nrows {
        if active[r] {
            debug_assert!(
                working[r].is_empty(),
                "non-pivot row should be empty after full RREF"
            );
            out_rows.push(mem::take(&mut working[r]));
        }
    }

    let result = CsrFp::from_sorted_rows(nrows, ncols, prime, out_rows);
    RrefFp {
        matrix: result,
        rank,
        pivot_cols,
    }
}

/// Multiply every entry of `row` by `c` mod `p`.
///
/// Caller's contract: `c != 0`, so no entry goes to zero and the
/// sparsity pattern is preserved.
fn scale_row(row: &mut [(usize, u32)], c: u32, prime: u32) {
    debug_assert!(c != 0, "scale_row called with zero scalar");
    let p = prime as u64;
    let cu = c as u64;
    for entry in row.iter_mut() {
        entry.1 = ((entry.1 as u64 * cu) % p) as u32;
    }
}

/// `target -= coeff * src` over `F_p`, preserving the sorted-columns /
/// no-zero invariant on `target`. Textbook `O(|target| + |src|)` merge
/// over the two sorted rows.
///
/// The merged row is built into `scratch` and swapped into `target`, so
/// the caller's reused buffer absorbs the allocation: in steady state
/// this is allocation-free, which is the dominant saving across the
/// millions of axpy calls in a large RREF.
fn axpy_mod_p(
    target: &mut Vec<(usize, u32)>,
    src: &[(usize, u32)],
    coeff: u32,
    prime: u32,
    scratch: &mut Vec<(usize, u32)>,
) {
    if coeff == 0 {
        return;
    }
    let p = prime as u64;
    let cu = coeff as u64;
    scratch.clear();
    scratch.reserve(target.len() + src.len());

    let t = target.as_slice();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < t.len() && j < src.len() {
        let (tc, tv) = t[i];
        let (sc, sv) = src[j];
        if tc < sc {
            scratch.push((tc, tv));
            i += 1;
        } else if tc > sc {
            // Fill-in: insert -coeff·sv in column sc.
            let prod = (cu * sv as u64) % p;
            if prod != 0 {
                scratch.push((sc, (p - prod) as u32));
            }
            j += 1;
        } else {
            // tv - coeff·sv (mod p); both operands are in [0, p), so the
            // difference is in [0, 2p) — one conditional subtraction, no
            // second `%`.
            let prod = (cu * sv as u64) % p;
            let diff = tv as u64 + p - prod;
            let new_v = if diff >= p { diff - p } else { diff };
            if new_v != 0 {
                scratch.push((tc, new_v as u32));
            }
            i += 1;
            j += 1;
        }
    }
    scratch.extend_from_slice(&t[i..]);
    while j < src.len() {
        let (sc, sv) = src[j];
        let prod = (cu * sv as u64) % p;
        if prod != 0 {
            scratch.push((sc, (p - prod) as u32));
        }
        j += 1;
    }
    mem::swap(target, scratch);
}

/// Multiplicative inverse of `a` in `F_p` via Fermat's little
/// theorem: `a^(p-2) mod p`.
///
/// Caller's contract: `a != 0` and `p` is prime.
pub(crate) fn inv_mod_p(a: u32, prime: u32) -> u32 {
    debug_assert!(a != 0, "inv_mod_p of zero");
    debug_assert!(prime >= 2);
    pow_mod_p(a, prime - 2, prime)
}

/// `base^exp mod prime`, square-and-multiply.
pub(crate) fn pow_mod_p(base: u32, exp: u32, prime: u32) -> u32 {
    let p = prime as u64;
    let mut result: u64 = 1;
    let mut b = (base as u64) % p;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % p;
        }
        b = (b * b) % p;
        e >>= 1;
    }
    result as u32
}

/// Determinant of a square sparse `F_p` matrix.
///
/// Runs the same column-by-column sweep as [`rref`] but stops short
/// of normalising pivot rows — the pivot products contribute
/// directly to the determinant. Permutation sign comes from the
/// inversion parity of the pivot-row sequence.
///
/// Returns `0` if the matrix is singular (some column has no
/// available pivot in the still-active rows).
///
/// # Panics
///
/// Panics if `matrix` is not square.
pub fn det(matrix: &CsrFp) -> u32 {
    assert_eq!(
        matrix.rows(),
        matrix.cols(),
        "det requires a square matrix; got {} × {}",
        matrix.rows(),
        matrix.cols()
    );
    let n = matrix.rows();
    if n == 0 {
        return 1; // empty product
    }
    let prime = matrix.modulus() as u32;
    let p = prime as u64;

    let mut working: Vec<Vec<(usize, u32)>> =
        (0..n).map(|r| matrix.iter_row(r).collect()).collect();

    let mut active = vec![true; n];
    let mut pivot_rows: Vec<usize> = Vec::with_capacity(n);
    let mut product: u64 = 1;
    let mut scratch: Vec<(usize, u32)> = Vec::new();

    for c in 0..n {
        let pivot = match select_pivot_row(&working, &active, c) {
            None => return 0, // singular
            Some(p) => p,
        };
        let off = working[pivot]
            .binary_search_by_key(&c, |e| e.0)
            .expect("pivot row must hold the pivot column");
        let pivot_val = working[pivot][off].1;
        product = (product * pivot_val as u64) % p;

        pivot_rows.push(pivot);
        active[pivot] = false;

        // Eliminate column `c` in every still-active row so that
        // future pivot values reflect the post-elimination state —
        // which is what makes the running product equal `det(A)`.
        let inv = inv_mod_p(pivot_val, prime);
        let pivot_row = mem::take(&mut working[pivot]);
        for (r, row) in working.iter_mut().enumerate() {
            if !active[r] {
                continue;
            }
            let off_r = match row.binary_search_by_key(&c, |e| e.0) {
                Ok(off) => off,
                Err(_) => continue,
            };
            let row_val = row[off_r].1;
            let coeff = ((row_val as u64 * inv as u64) % p) as u32;
            axpy_mod_p(row, &pivot_row, coeff, prime, &mut scratch);
        }
        working[pivot] = pivot_row;
    }

    let inv_count = permutation_inversions(&pivot_rows);
    if inv_count.is_multiple_of(2) {
        product as u32
    } else if product == 0 {
        0
    } else {
        (p - product) as u32
    }
}

/// Number of inversions in the sequence — pairs `(i, j)` with
/// `i < j` and `perm[i] > perm[j]`. Used by [`det`] to recover the
/// sign of the pivot-row permutation; an even count means the
/// permutation is even (sign `+1`), odd means odd (sign `−1`).
///
/// `O(n²)`; determinant computation uses it only on the pivot permutation.
fn permutation_inversions(perm: &[usize]) -> usize {
    let mut count = 0;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                count += 1;
            }
        }
    }
    count
}

/// Basis for the null space `{x ∈ F_p^n : A · x = 0}` as columns of
/// the returned `n × nullity` sparse matrix.
///
/// Derived from [`rref`]: for each free column `f` (index `j` in
/// the output), the kernel basis vector has `1` at row `f` and
/// `-RREF[i][f]` at each pivot row's pivot column. Sparse-in
/// sparse-out per AGENTS.md invariant 4 — each column has at most
/// `1 + rank` nonzeros, far less than `n` at typical PH-scale.
pub fn kernel(matrix: &CsrFp) -> CsrFp {
    let prime = matrix.modulus() as u32;
    let n = matrix.cols();
    let RrefFp {
        matrix: rref_mat,
        pivot_cols,
        ..
    } = rref(matrix);
    let p = prime as u64;

    let free_cols: Vec<usize> = (0..n)
        .filter(|c| pivot_cols.binary_search(c).is_err())
        .collect();
    let nullity = free_cols.len();

    let mut k_rows: Vec<Vec<(usize, u32)>> = (0..n).map(|_| Vec::new()).collect();
    for (j, &f) in free_cols.iter().enumerate() {
        // The "1" in the free-column position of the j-th basis
        // vector.
        k_rows[f].push((j, 1));
        // The "-RREF[i][f]" entries in each pivot column.
        for (i, &pc) in pivot_cols.iter().enumerate() {
            if let Some(val) = rref_mat.get(i, f) {
                debug_assert!(val != 0);
                let neg = (p - val as u64) % p;
                if neg != 0 {
                    k_rows[pc].push((j, neg as u32));
                }
            }
        }
    }
    // Each pivot row's vec is built in increasing-j order (outer
    // loop is over j); free-col rows have a single push. The
    // sorted-columns CSR invariant holds.
    CsrFp::from_sorted_rows(n, nullity, prime, k_rows)
}

/// Basis for the column space `{A · x : x ∈ F_p^n}` as columns of
/// the returned `m × rank` sparse matrix.
///
/// Derived from [`rref`]: the basis is simply the pivot columns of
/// the *original* `A` (not of the RREF) — those columns are
/// linearly independent and span the same subspace as all of `A`'s
/// columns.
pub fn image(matrix: &CsrFp) -> CsrFp {
    let prime = matrix.modulus() as u32;
    let m = matrix.rows();
    let pivot_cols = rref(matrix).pivot_cols;
    let rank = pivot_cols.len();

    let mut img_rows: Vec<Vec<(usize, u32)>> = (0..m).map(|_| Vec::new()).collect();
    for (r, row) in img_rows.iter_mut().enumerate() {
        for (c, v) in matrix.iter_row(r) {
            if let Ok(j) = pivot_cols.binary_search(&c) {
                row.push((j, v));
            }
        }
    }
    CsrFp::from_sorted_rows(m, rank, prime, img_rows)
}

/// Solve `A · x = b` over `F_p`, returning one particular solution
/// as a sparse `n × 1` matrix, or `None` if the system is
/// inconsistent.
///
/// Algorithm: augment `[A | b]`, compute only forward echelon form,
/// check the remaining rows for inconsistency, and back-substitute with free
/// variables set to zero. Unlike a full RREF solve, pivot rows are not
/// broadcast into already-resolved rows.
///
/// All solutions are then `x + kernel(A)`. Pair with [`kernel`] to
/// enumerate the affine solution space.
///
/// # Panics
///
/// Panics if `b.rows() != matrix.rows()` or `b.cols() != 1`.
pub fn solve(matrix: &CsrFp, b: &CsrFp) -> Option<CsrFp> {
    assert_eq!(
        b.rows(),
        matrix.rows(),
        "b must have same row count as A: {} vs {}",
        b.rows(),
        matrix.rows()
    );
    assert_eq!(b.cols(), 1, "b must be a column vector (m × 1)");
    let m = matrix.rows();
    let n = matrix.cols();
    let prime = matrix.modulus() as u32;

    // Augmented matrix [A | b], built as triples for from_coo.
    // (Using from_coo rather than from_sorted_rows because b's
    // column n entry needs to interleave with A's row entries; the
    // canonicalisation in build_csr handles ordering.)
    let mut aug_triples: Vec<(usize, usize, i64)> = Vec::new();
    for r in 0..m {
        for (c, v) in matrix.iter_row(r) {
            aug_triples.push((r, c, v as i64));
        }
        if let Some(v) = b.get(r, 0) {
            aug_triples.push((r, n, v as i64));
        }
    }
    let aug = CsrFp::from_coo(m, n + 1, &aug_triples, prime as u64);
    let mut working = (0..m)
        .map(|row| aug.iter_row(row).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut active = vec![true; m];
    let mut pivot_columns = Vec::new();
    let mut pivot_rows = Vec::new();
    let mut scratch = Vec::new();
    for column in 0..n {
        let Some(pivot) = select_pivot_row(&working, &active, column) else {
            continue;
        };
        let offset = working[pivot]
            .binary_search_by_key(&column, |entry| entry.0)
            .expect("a selected solve pivot contains its column");
        let value = working[pivot][offset].1;
        if value != 1 {
            let inverse = inv_mod_p(value, prime);
            scale_row(&mut working[pivot], inverse, prime);
        }
        let pivot_row = mem::take(&mut working[pivot]);
        for (row_index, row) in working.iter_mut().enumerate() {
            if row_index == pivot || !active[row_index] {
                continue;
            }
            let Ok(offset) = row.binary_search_by_key(&column, |entry| entry.0) else {
                continue;
            };
            let coefficient = row[offset].1;
            axpy_mod_p(row, &pivot_row, coefficient, prime, &mut scratch);
        }
        working[pivot] = pivot_row;
        active[pivot] = false;
        pivot_columns.push(column);
        pivot_rows.push(pivot);
    }

    for (row, is_active) in working.iter().zip(&active) {
        if *is_active
            && row
                .iter()
                .any(|(column, value)| *column == n && *value != 0)
        {
            return None;
        }
    }

    let modulus = u64::from(prime);
    let mut solution = vec![0_u32; n];
    for (&pivot_column, &pivot_row) in pivot_columns.iter().zip(&pivot_rows).rev() {
        let row = &working[pivot_row];
        let mut value = row
            .binary_search_by_key(&n, |entry| entry.0)
            .ok()
            .map_or(0, |offset| row[offset].1);
        for &(column, coefficient) in row {
            if column <= pivot_column || column >= n || solution[column] == 0 {
                continue;
            }
            let product = (u64::from(coefficient) * u64::from(solution[column])) % modulus;
            let difference = u64::from(value) + modulus - product;
            value = if difference >= modulus {
                (difference - modulus) as u32
            } else {
                difference as u32
            };
        }
        solution[pivot_column] = value;
    }
    let x_rows = solution
        .into_iter()
        .map(|value| {
            if value == 0 {
                Vec::new()
            } else {
                vec![(0, value)]
            }
        })
        .collect();
    Some(CsrFp::from_sorted_rows(n, 1, prime, x_rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 7;

    #[test]
    fn inv_mod_p_is_left_inverse_for_all_units() {
        for p in [2u32, 3, 5, 7, 11, 13, 101, 65537] {
            for a in 1..p {
                let inv = inv_mod_p(a, p);
                assert_eq!(
                    (a as u64 * inv as u64) % p as u64,
                    1,
                    "inv({}, {}) = {} but {} * {} mod {} != 1",
                    a,
                    p,
                    inv,
                    a,
                    inv,
                    p
                );
            }
        }
    }

    #[test]
    fn pow_mod_p_basic_values() {
        assert_eq!(pow_mod_p(2, 0, 7), 1);
        assert_eq!(pow_mod_p(2, 3, 7), 1); // 8 mod 7
        assert_eq!(pow_mod_p(3, 5, 7), 5); // 243 mod 7
    }

    #[test]
    fn scale_row_multiplies_in_place() {
        let mut row = vec![(0, 2u32), (3, 5u32)];
        scale_row(&mut row, 3, P);
        // 2*3 = 6; 5*3 = 15 ≡ 1 (mod 7)
        assert_eq!(row, vec![(0, 6), (3, 1)]);
    }

    #[test]
    fn axpy_subtracts_into_disjoint_columns() {
        let mut t = vec![(0, 1u32), (2, 3u32)];
        let s = vec![(1, 5u32), (3, 6u32)];
        let mut scratch = Vec::new();
        // t -= 1 * s, so t gains -5 ≡ 2 in col 1, -6 ≡ 1 in col 3.
        axpy_mod_p(&mut t, &s, 1, P, &mut scratch);
        assert_eq!(t, vec![(0, 1), (1, 2), (2, 3), (3, 1)]);
    }

    #[test]
    fn axpy_combines_overlapping_columns() {
        let mut t = vec![(0, 4u32), (1, 5u32)];
        let s = vec![(0, 1u32), (1, 2u32)];
        let mut scratch = Vec::new();
        // t -= 2 * s; col 0: 4 - 2 = 2; col 1: 5 - 4 = 1.
        axpy_mod_p(&mut t, &s, 2, P, &mut scratch);
        assert_eq!(t, vec![(0, 2), (1, 1)]);
    }

    #[test]
    fn axpy_drops_cells_that_zero_out() {
        let mut t = vec![(0, 3u32), (1, 4u32)];
        let s = vec![(0, 3u32), (1, 4u32)];
        let mut scratch = Vec::new();
        // t -= 1 * s; every column zeros.
        axpy_mod_p(&mut t, &s, 1, P, &mut scratch);
        assert!(t.is_empty());
    }

    #[test]
    fn axpy_with_zero_coefficient_is_noop() {
        let mut t = vec![(0, 3u32), (5, 6u32)];
        let original = t.clone();
        let mut scratch = Vec::new();
        axpy_mod_p(&mut t, &[(0, 1u32), (5, 1u32)], 0, P, &mut scratch);
        assert_eq!(t, original);
    }

    #[test]
    fn axpy_extends_when_target_shorter() {
        let mut t = vec![(0, 1u32)];
        let s = vec![(0, 1u32), (1, 1u32), (2, 1u32)];
        let mut scratch = Vec::new();
        // t -= 1 * s: col 0 becomes 0 (dropped), cols 1,2 become 6,6.
        axpy_mod_p(&mut t, &s, 1, P, &mut scratch);
        assert_eq!(t, vec![(1, 6), (2, 6)]);
    }

    #[test]
    fn axpy_long_target_short_src_fill_in() {
        // A long target with a short pivot whose columns fall in the
        // interior gaps (some present, some fill-in), with leading and
        // trailing runs. Cross-checked against a naive merge.
        let mut t = vec![(0, 1u32), (1, 2), (2, 3), (5, 4), (6, 5), (9, 6)];
        // src touches col 2 (present), col 4 (fill-in), col 6 (present).
        let s = vec![(2, 1u32), (4, 3), (6, 2)];
        let coeff = 2u32;
        let expected = naive_axpy(&t, &s, coeff, P);
        let mut scratch = Vec::new();
        axpy_mod_p(&mut t, &s, coeff, P, &mut scratch);
        assert_eq!(t, expected);
    }

    #[test]
    fn axpy_matches_naive_on_random_rows() {
        // Differential test of the run-copy merge against a textbook
        // O(|t|+|s|) merge across an LCG-driven sweep of shapes, gaps,
        // overlaps, and primes.
        let mut state: u64 = 0x00C0_FFEE_1234_5678;
        let mut next = |m: u64| -> u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut scratch = Vec::new();
        for &p in &[2u32, 3, 7, 101, 65537] {
            for _ in 0..400 {
                let ncols = 1 + next(40) as usize;
                let build = |n: &mut dyn FnMut(u64) -> u64| -> Vec<(usize, u32)> {
                    let mut cols: Vec<usize> = (0..ncols).filter(|_| n(2) == 0).collect();
                    cols.sort_unstable();
                    cols.dedup();
                    cols.into_iter()
                        .map(|c| (c, 1 + n((p - 1) as u64) as u32))
                        .collect()
                };
                let mut t = build(&mut next);
                let s = build(&mut next);
                let coeff = next(p as u64) as u32;
                let expected = naive_axpy(&t, &s, coeff, p);
                axpy_mod_p(&mut t, &s, coeff, p, &mut scratch);
                assert_eq!(t, expected, "p={} coeff={}", p, coeff);
            }
        }
    }

    #[test]
    fn axpy_reuses_scratch_across_calls() {
        // The scratch buffer ping-pongs with `target` via mem::swap, so
        // a second call must still produce the right answer (regression
        // guard for the swap-based buffer reuse).
        let mut scratch = Vec::new();
        let mut t = vec![(0, 4u32), (1, 5u32)];
        axpy_mod_p(&mut t, &[(0, 1u32), (1, 2u32)], 2, P, &mut scratch);
        assert_eq!(t, vec![(0, 2), (1, 1)]);
        // Reuse the same scratch on a differently shaped op.
        let mut t2 = vec![(2, 3u32)];
        axpy_mod_p(&mut t2, &[(0, 1u32), (2, 3u32)], 1, P, &mut scratch);
        // col 0: -1 ≡ 6; col 2: 3 - 3 = 0 (dropped).
        assert_eq!(t2, vec![(0, 6)]);
    }

    /// Textbook `target -= coeff·src` merge — the oracle for the
    /// run-copy [`axpy_mod_p`].
    fn naive_axpy(
        target: &[(usize, u32)],
        src: &[(usize, u32)],
        coeff: u32,
        prime: u32,
    ) -> Vec<(usize, u32)> {
        let p = prime as u64;
        let cu = coeff as u64;
        let mut acc: std::collections::BTreeMap<usize, u64> =
            target.iter().map(|&(c, v)| (c, v as u64)).collect();
        if coeff != 0 {
            for &(sc, sv) in src {
                let prod = (cu * sv as u64) % p;
                let e = acc.entry(sc).or_insert(0);
                *e = (*e + p - prod) % p;
            }
        }
        acc.into_iter()
            .filter(|&(_, v)| v != 0)
            .map(|(c, v)| (c, v as u32))
            .collect()
    }

    // -- det --------------------------------------------------------

    fn csr_from_dense(dense: &[Vec<i64>], prime: u64) -> CsrFp {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut triples = Vec::new();
        for (r, row) in dense.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v.rem_euclid(prime as i64) != 0 {
                    triples.push((r, c, v));
                }
            }
        }
        CsrFp::from_coo(rows, cols, &triples, prime)
    }

    #[test]
    fn det_of_empty_matrix_is_one() {
        assert_eq!(det(&CsrFp::zeros(0, 0, P as u64)), 1);
    }

    #[test]
    fn det_of_zero_matrix_is_zero() {
        assert_eq!(det(&CsrFp::zeros(3, 3, P as u64)), 0);
    }

    #[test]
    fn det_of_identity_is_one() {
        let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], P as u64);
        assert_eq!(det(&a), 1);
    }

    #[test]
    fn det_of_singular_is_zero() {
        // Row 1 = 2 * row 0.
        let a = csr_from_dense(&[vec![1, 2, 3], vec![2, 4, 6], vec![0, 1, 0]], P as u64);
        assert_eq!(det(&a), 0);
    }

    #[test]
    fn det_two_by_two() {
        // [[2, 3], [4, 5]]: det = 10 - 12 = -2 ≡ 5 (mod 7).
        let a = csr_from_dense(&[vec![2, 3], vec![4, 5]], P as u64);
        assert_eq!(det(&a), 5);
    }

    #[test]
    fn det_swap_matrix_is_minus_one() {
        // [[0, 1], [1, 0]]: det = -1 ≡ 6 (mod 7).
        let a = csr_from_dense(&[vec![0, 1], vec![1, 0]], P as u64);
        assert_eq!(det(&a), 6);
    }

    #[test]
    fn det_three_cycle_is_one() {
        // Permutation (0 → 2 → 1 → 0): even, det = +1.
        let a = csr_from_dense(&[vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]], P as u64);
        assert_eq!(det(&a), 1);
    }

    #[test]
    fn det_full_reversal_3x3_is_minus_one() {
        // Reverse-identity n=3 → sign = -1.
        let a = csr_from_dense(&[vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]], P as u64);
        assert_eq!(det(&a), 6); // -1 mod 7
    }

    #[test]
    fn det_one_by_one() {
        let a = csr_from_dense(&[vec![5]], P as u64);
        assert_eq!(det(&a), 5);
    }

    #[test]
    #[should_panic(expected = "det requires a square matrix")]
    fn det_rejects_rectangular() {
        let a = csr_from_dense(&[vec![1, 2, 3], vec![4, 5, 6]], P as u64);
        let _ = det(&a);
    }

    // -- permutation_inversions -------------------------------------

    #[test]
    fn permutation_inversions_identity() {
        assert_eq!(permutation_inversions(&[0, 1, 2, 3]), 0);
    }

    #[test]
    fn permutation_inversions_single_swap() {
        assert_eq!(permutation_inversions(&[1, 0]), 1);
        assert_eq!(permutation_inversions(&[0, 2, 1, 3]), 1);
    }

    #[test]
    fn permutation_inversions_full_reverse() {
        // n=4 reversed: pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) all inverted.
        assert_eq!(permutation_inversions(&[3, 2, 1, 0]), 6);
    }

    #[test]
    fn permutation_inversions_empty_and_singleton() {
        assert_eq!(permutation_inversions(&[]), 0);
        assert_eq!(permutation_inversions(&[42]), 0);
    }
}
