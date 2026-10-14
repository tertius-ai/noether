// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrix over a small prime field in compressed-sparse-row
//! layout.
//!
//! The modulus satisfies `p ≤ 2³¹`, so every reduced value fits in `u32`.
//! Arithmetic widens to `u64` to hold
//! intermediates; the storage stays compact.
//!
//! `p` is taken on trust as prime — checking primality is non-trivial
//! and orthogonal to the storage layer.

use crate::elim::gauss_fp;
use crate::elim::{rref, RrefFp};

use super::coo::{build_csr, Triple};
use super::csr_z::CsrZ;
use super::SparseRingStorage;

/// Largest prime modulus supported by [`CsrFp`]. With `p ≤ 2³¹`, reduced
/// values fit in `u32` and any pairwise product fits in `u64`.
pub const MAX_PRIME: u64 = 1u64 << 31;

/// Sparse `rows × cols` matrix over `F_p` (small prime) in CSR layout.
///
/// Values are stored as `u32` in `[0, p)`; arithmetic widens to `u64`
/// internally. The modulus is carried with the matrix so the type
/// system can't accidentally mix matrices over different fields.
#[derive(Clone, Debug)]
pub struct CsrFp {
    rows: usize,
    cols: usize,
    prime: u32,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<u32>,
}

impl CsrFp {
    /// The all-zero `rows × cols` matrix over `F_p`.
    ///
    /// # Panics
    ///
    /// Panics if `prime < 2` or `prime > 2³¹`.
    pub fn zeros(rows: usize, cols: usize, prime: u64) -> Self {
        let prime = check_prime(prime);
        Self {
            rows,
            cols,
            prime,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Build from a slice of `(row, col, value)` triples.
    ///
    /// Triple values are accepted as `i64` (so callers can pass
    /// signed entries — boundary matrices commonly have `±1`) and
    /// reduced modulo `prime` before storage. Duplicate `(row, col)`
    /// triples are summed mod `p`; cells that reduce to `0` are not
    /// stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::storage::{CsrFp, SparseRingStorage};
    ///
    /// let a = CsrFp::from_coo(
    ///     2,
    ///     3,
    ///     &[(0, 0, 3), (0, 1, -1), (1, 2, 5)],
    ///     7,
    /// );
    /// assert_eq!(a.nnz(), 3);
    /// // -1 reduces to 6 mod 7.
    /// assert_eq!(a.get(0, 1), Some(6));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `prime < 2`, `prime > 2³¹`, or any triple's row /
    /// col is out of bounds.
    pub fn from_coo(rows: usize, cols: usize, triples: &[Triple<i64>], prime: u64) -> Self {
        let prime = check_prime(prime);
        let p_u32 = prime;
        let p_u64 = prime as u64;
        let reduced: Vec<Triple<u32>> = triples
            .iter()
            .map(|&(r, c, v)| (r, c, reduce_i64(v, p_u64)))
            .collect();
        let (indptr, indices, data) = build_csr(
            rows,
            cols,
            reduced,
            move |acc, v| {
                let s = (*acc as u64 + v as u64) % p_u64;
                *acc = s as u32;
            },
            |v| *v == 0,
        );
        Self {
            rows,
            cols,
            prime: p_u32,
            indptr,
            indices,
            data,
        }
    }

    /// Build directly from a list of already-sorted, already-reduced
    /// rows. Skips the COO sort / dedupe / mod-reduce pipeline.
    ///
    /// Caller's contract (checked under `debug_assertions` only):
    /// `sorted_rows.len() == rows`; within each inner vec, column
    /// indices are strictly increasing and lie in `[0, cols)`; every
    /// value is nonzero and lies in `[0, prime)`. Used by
    /// [`crate::elim::rref`] to skip a re-canonicalisation when
    /// emitting the RREF matrix.
    pub(crate) fn from_sorted_rows(
        rows: usize,
        cols: usize,
        prime: u32,
        sorted_rows: Vec<Vec<(usize, u32)>>,
    ) -> Self {
        debug_assert_eq!(sorted_rows.len(), rows);
        let nnz_estimate: usize = sorted_rows.iter().map(|r| r.len()).sum();
        let mut indptr = vec![0usize; rows + 1];
        let mut indices: Vec<usize> = Vec::with_capacity(nnz_estimate);
        let mut data: Vec<u32> = Vec::with_capacity(nnz_estimate);
        for (r, row) in sorted_rows.into_iter().enumerate() {
            let mut last_col: Option<usize> = None;
            for (c, v) in row {
                debug_assert!(c < cols, "col index {} out of bounds (cols={})", c, cols);
                debug_assert!(v != 0, "zero value passed to from_sorted_rows");
                debug_assert!(v < prime, "value {} not reduced mod prime {}", v, prime);
                debug_assert!(
                    last_col.is_none_or(|lc| lc < c),
                    "column indices not strictly increasing within row {}",
                    r
                );
                last_col = Some(c);
                indices.push(c);
                data.push(v);
            }
            indptr[r + 1] = indices.len();
        }
        Self {
            rows,
            cols,
            prime,
            indptr,
            indices,
            data,
        }
    }

    /// Prime modulus of the field.
    pub fn modulus(&self) -> u64 {
        self.prime as u64
    }

    /// Reduce to row echelon form via sparse Gaussian elimination.
    ///
    /// Returns the canonical RREF matrix (pivot rows on top, sorted
    /// by pivot column, then zero rows), the rank, and the pivot
    /// columns. See [`RrefFp`] for the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::storage::CsrFp;
    ///
    /// // A 2 × 3 matrix of rank 2.
    /// let a = CsrFp::from_coo(
    ///     2,
    ///     3,
    ///     &[(0, 0, 1), (0, 1, 2), (1, 2, 1)],
    ///     7,
    /// );
    /// let result = a.rref();
    /// assert_eq!(result.rank, 2);
    /// assert_eq!(result.pivot_cols, vec![0, 2]);
    /// ```
    pub fn rref(&self) -> RrefFp {
        rref(self)
    }

    /// Rank of the matrix over `F_p`. Computed via [`Self::rref`].
    pub fn rank(&self) -> usize {
        self.rref().rank
    }

    /// Sparse · sparse `F_p` matrix multiplication. See
    /// [`crate::matmul::matmul_fp`] for the algorithm and the
    /// validated perf shape against FLINT-dense.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::storage::{CsrFp, SparseRingStorage};
    ///
    /// // A · kernel(A) = 0 — a foundational invariant tied to matmul.
    /// let a = CsrFp::from_coo(
    ///     2,
    ///     3,
    ///     &[(0, 0, 1), (0, 1, 2), (1, 2, 1)],
    ///     7,
    /// );
    /// let k = a.kernel();
    /// let prod = a.matmul(&k);
    /// assert_eq!(prod.nnz(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on dimension mismatch or modulus mismatch.
    pub fn matmul(&self, other: &CsrFp) -> CsrFp {
        crate::matmul::matmul_fp(self, other)
    }

    /// Transpose: swap rows and columns. The result is `cols × rows`
    /// and stores entry `(c, r, v)` for every stored entry `(r, c, v)`
    /// of `self`. Modulus is preserved.
    ///
    /// `O(rows + nnz)` — one pass populates per-column buckets,
    /// already sorted by `(row, col)` because `iter_row` walks
    /// rows in `0..rows` order.
    pub fn transpose(&self) -> CsrFp {
        let m = self.rows();
        let n = self.cols();
        let mut tr_rows: Vec<Vec<(usize, u32)>> = (0..n).map(|_| Vec::new()).collect();
        for r in 0..m {
            for (c, v) in self.iter_row(r) {
                tr_rows[c].push((r, v));
            }
        }
        CsrFp::from_sorted_rows(n, m, self.prime, tr_rows)
    }

    /// Determinant of a square matrix over `F_p`, in `[0, prime)`.
    ///
    /// Returns `0` for singular inputs. See [`gauss_fp::det`] for
    /// the algorithm. The empty `0 × 0` matrix returns `1` (empty
    /// product).
    ///
    /// # Panics
    ///
    /// Panics if `self` is not square.
    pub fn det(&self) -> u32 {
        gauss_fp::det(self)
    }

    /// Basis for the null space `{x ∈ F_p^n : A · x = 0}` as columns
    /// of an `n × nullity` sparse matrix. See [`gauss_fp::kernel`].
    pub fn kernel(&self) -> CsrFp {
        gauss_fp::kernel(self)
    }

    /// Basis for the column space `{A · x}` as columns of an
    /// `m × rank` sparse matrix. See [`gauss_fp::image`].
    pub fn image(&self) -> CsrFp {
        gauss_fp::image(self)
    }

    /// Solve `A · x = b` over `F_p`. Returns one particular solution
    /// as an `n × 1` sparse vector, or `None` if `b` is not in the
    /// column space of `self`. Combine with [`Self::kernel`] to
    /// enumerate all solutions: `x + kernel(A)`. See
    /// [`gauss_fp::solve`].
    ///
    /// # Panics
    ///
    /// Panics if `b.rows() != self.rows()` or `b.cols() != 1`.
    pub fn solve(&self, b: &CsrFp) -> Option<CsrFp> {
        gauss_fp::solve(self, b)
    }

    /// Reduce a [`CsrZ`] to a sparse `F_p` matrix entry-wise.
    ///
    /// Each Z entry is reduced via Euclidean remainder (so negative
    /// values land in `[0, prime)`), and entries that reduce to `0`
    /// are dropped. Cheaper than going through COO: entries arrive
    /// already sorted by `(row, col)`, so the canonical CSR shape
    /// is built directly via [`Self::from_sorted_rows`].
    ///
    /// # Panics
    ///
    /// Panics if `prime < 2` or `prime > 2³¹`.
    pub fn from_z(matrix: &CsrZ, prime: u64) -> Self {
        let prime_u32 = check_prime(prime);
        let nrows = matrix.rows();
        let ncols = matrix.cols();
        let mut sorted_rows: Vec<Vec<(usize, u32)>> = (0..nrows).map(|_| Vec::new()).collect();
        for (r, row) in sorted_rows.iter_mut().enumerate().take(nrows) {
            for (c, v) in matrix.iter_row(r) {
                let val = v.mod_u(prime_u32);
                if val != 0 {
                    row.push((c, val));
                }
            }
        }
        Self::from_sorted_rows(nrows, ncols, prime_u32, sorted_rows)
    }

    /// Look up entry `(row, col)`. Returns `None` for a structural
    /// zero. The returned `u32` lies in `[0, prime)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows` or `col >= cols`.
    pub fn get(&self, row: usize, col: usize) -> Option<u32> {
        assert!(row < self.rows, "row index {} out of bounds", row);
        assert!(col < self.cols, "col index {} out of bounds", col);
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .binary_search(&col)
            .ok()
            .map(|off| self.data[s + off])
    }

    /// Iterate the stored entries of `row` in column-ascending order.
    /// Yields `(col, value)` pairs with `value` in `[0, prime)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows`.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, u32)> + '_ {
        assert!(row < self.rows, "row index {} out of bounds", row);
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .iter()
            .copied()
            .zip(self.data[s..e].iter().copied())
    }

    /// Convert to a sorted-by-`(row, col)` COO triple list. Values
    /// are returned as `u32` in `[0, prime)`.
    pub fn to_coo(&self) -> Vec<Triple<u32>> {
        let mut out = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                out.push((r, c, v));
            }
        }
        out
    }

    /// Densify into a row-major `Vec<Vec<u32>>`. Useful for parity
    /// tests against dense `nmod_mat` references.
    pub fn to_dense(&self) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = (0..self.rows).map(|_| vec![0u32; self.cols]).collect();
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in self.iter_row(r) {
                row[c] = v;
            }
        }
        out
    }

    /// `true` if every stored entry lies on the main diagonal.
    pub fn is_diagonal(&self) -> bool {
        for r in 0..self.rows {
            for (c, _) in self.iter_row(r) {
                if r != c {
                    return false;
                }
            }
        }
        true
    }

    /// `true` if the matrix equals its transpose (over `F_p`).
    /// Non-square matrices return `false`.
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                if self.get(c, r) != Some(v) {
                    return false;
                }
            }
        }
        true
    }

    /// Sum of the diagonal entries, reduced mod `p`.
    pub fn trace(&self) -> u32 {
        let p = self.prime as u64;
        let mut acc: u64 = 0;
        let n = self.rows.min(self.cols);
        for i in 0..n {
            if let Some(v) = self.get(i, i) {
                acc = (acc + v as u64) % p;
            }
        }
        acc as u32
    }

    /// Element-wise sum in canonical CSR storage.
    pub fn add(&self, other: &Self) -> Self {
        self.linear_combine(other, false)
    }

    /// Element-wise difference in canonical CSR storage.
    pub fn sub(&self, other: &Self) -> Self {
        self.linear_combine(other, true)
    }

    fn linear_combine(&self, other: &Self, subtract: bool) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        assert_eq!(self.prime, other.prime);
        let modulus = self.prime as u64;
        let mut indptr = Vec::with_capacity(self.rows + 1);
        let mut indices = Vec::with_capacity(self.nnz() + other.nnz());
        let mut data = Vec::with_capacity(self.nnz() + other.nnz());
        indptr.push(0);
        for row in 0..self.rows {
            let (mut left, left_end) = (self.indptr[row], self.indptr[row + 1]);
            let (mut right, right_end) = (other.indptr[row], other.indptr[row + 1]);
            while left < left_end || right < right_end {
                match (
                    self.indices.get(left).filter(|_| left < left_end),
                    other.indices.get(right).filter(|_| right < right_end),
                ) {
                    (Some(&left_col), Some(&right_col)) if left_col == right_col => {
                        let left_value = self.data[left] as u64;
                        let right_value = other.data[right] as u64;
                        let value = if subtract {
                            (left_value + modulus - right_value) % modulus
                        } else {
                            (left_value + right_value) % modulus
                        } as u32;
                        if value != 0 {
                            indices.push(left_col);
                            data.push(value);
                        }
                        left += 1;
                        right += 1;
                    }
                    (Some(&left_col), Some(&right_col)) if left_col < right_col => {
                        indices.push(left_col);
                        data.push(self.data[left]);
                        left += 1;
                    }
                    (Some(&left_col), None) => {
                        indices.push(left_col);
                        data.push(self.data[left]);
                        left += 1;
                    }
                    (_, Some(&right_col)) => {
                        let value = if subtract {
                            self.prime - other.data[right]
                        } else {
                            other.data[right]
                        };
                        indices.push(right_col);
                        data.push(value);
                        right += 1;
                    }
                    _ => unreachable!("at least one CSR row cursor remains"),
                }
            }
            indptr.push(indices.len());
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            prime: self.prime,
            indptr,
            indices,
            data,
        }
    }

    /// Additive inverse in canonical CSR storage.
    pub fn neg(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            prime: self.prime,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|&value| self.prime - value).collect(),
        }
    }

    /// Multiply every stored coefficient by a reduced field scalar.
    pub fn scale(&self, scalar: u64) -> Self {
        let scalar = scalar % self.prime as u64;
        if scalar == 0 {
            return Self::zeros(self.rows, self.cols, self.prime as u64);
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            prime: self.prime,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self
                .data
                .iter()
                .map(|&value| (u64::from(value) * scalar % self.prime as u64) as u32)
                .collect(),
        }
    }
}

impl SparseRingStorage for CsrFp {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn nnz(&self) -> usize {
        self.data.len()
    }
}

#[inline]
fn check_prime(prime: u64) -> u32 {
    assert!(prime >= 2, "sparla: prime must be >= 2 (got {})", prime);
    assert!(
        prime <= MAX_PRIME,
        "sparla: prime must be <= 2^31 ({}); got {}",
        MAX_PRIME,
        prime
    );
    prime as u32
}

/// Reduce an `i64` value into `[0, p)`. `rem_euclid` keeps the result
/// non-negative for negative inputs without branching on sign.
#[inline]
fn reduce_i64(v: i64, p: u64) -> u32 {
    v.rem_euclid(p as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 7;

    fn t(r: usize, c: usize, v: i64) -> Triple<i64> {
        (r, c, v)
    }

    #[test]
    fn zeros_has_no_entries() {
        let a = CsrFp::zeros(2, 3, P);
        assert_eq!(a.rows(), 2);
        assert_eq!(a.cols(), 3);
        assert_eq!(a.nnz(), 0);
        assert_eq!(a.modulus(), P);
        assert!(a.is_zero());
    }

    #[test]
    fn from_coo_reduces_mod_p() {
        let a = CsrFp::from_coo(
            1,
            4,
            &[t(0, 0, 10), t(0, 1, -1), t(0, 2, 14), t(0, 3, -8)],
            P,
        );
        // 10 % 7 = 3; -1 mod 7 = 6; 14 % 7 = 0 (dropped); -8 mod 7 = 6.
        assert_eq!(a.get(0, 0), Some(3));
        assert_eq!(a.get(0, 1), Some(6));
        assert_eq!(a.get(0, 2), None);
        assert_eq!(a.get(0, 3), Some(6));
        assert_eq!(a.nnz(), 3);
    }

    #[test]
    fn duplicate_entries_sum_mod_p() {
        let a = CsrFp::from_coo(1, 2, &[t(0, 0, 4), t(0, 0, 5), t(0, 1, 1)], P);
        // 4 + 5 = 9 ≡ 2 (mod 7).
        assert_eq!(a.get(0, 0), Some(2));
        assert_eq!(a.get(0, 1), Some(1));
    }

    #[test]
    fn duplicates_summing_to_p_are_dropped() {
        let a = CsrFp::from_coo(1, 2, &[t(0, 0, 3), t(0, 0, 4), t(0, 1, 6)], P);
        // 3 + 4 = 7 ≡ 0; drop.
        assert_eq!(a.get(0, 0), None);
        assert_eq!(a.get(0, 1), Some(6));
        assert_eq!(a.nnz(), 1);
    }

    #[test]
    fn iter_row_yields_sorted_entries() {
        let a = CsrFp::from_coo(2, 3, &[t(0, 2, 5), t(0, 0, 1), t(0, 1, 3)], P);
        let row0: Vec<(usize, u32)> = a.iter_row(0).collect();
        assert_eq!(row0, vec![(0, 1), (1, 3), (2, 5)]);
        let row1: Vec<(usize, u32)> = a.iter_row(1).collect();
        assert!(row1.is_empty());
    }

    #[test]
    fn to_dense_and_to_coo() {
        let a = CsrFp::from_coo(2, 2, &[t(0, 0, 1), t(0, 1, 2), t(1, 1, -3)], P);
        assert_eq!(a.to_dense(), vec![vec![1, 2], vec![0, 4]]); // -3 mod 7 = 4
        let mut coo = a.to_coo();
        coo.sort();
        assert_eq!(coo, vec![(0, 0, 1), (0, 1, 2), (1, 1, 4)]);
    }

    #[test]
    fn predicates() {
        let diag = CsrFp::from_coo(3, 3, &[t(0, 0, 1), t(1, 1, 2), t(2, 2, 3)], P);
        assert!(diag.is_diagonal());
        assert!(diag.is_symmetric());
        assert_eq!(diag.trace(), 6);

        let asym = CsrFp::from_coo(2, 2, &[t(0, 1, 1), t(1, 0, 2)], P);
        assert!(!asym.is_symmetric());

        let sym = CsrFp::from_coo(2, 2, &[t(0, 1, 3), t(1, 0, 3), t(0, 0, 4)], P);
        assert!(sym.is_symmetric());

        let rect = CsrFp::from_coo(2, 4, &[t(0, 0, 5), t(1, 1, 6), t(1, 3, 4)], P);
        assert!(!rect.is_square());
        assert!(!rect.is_symmetric());
        assert_eq!(rect.trace(), (5 + 6) % 7);
    }

    #[test]
    fn density_matches_nnz_over_total() {
        let a = CsrFp::from_coo(2, 5, &[t(0, 0, 1), t(1, 4, 2)], P);
        assert_eq!(a.density(), 2.0 / 10.0);
    }

    #[test]
    #[should_panic(expected = "prime must be >= 2")]
    fn rejects_prime_below_two() {
        let _ = CsrFp::zeros(1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "prime must be <= 2^31")]
    fn rejects_prime_above_2_to_the_31() {
        let _ = CsrFp::zeros(1, 1, (1u64 << 31) + 1);
    }

    #[test]
    fn accepts_prime_at_upper_bound() {
        // The largest 31-bit prime is 2^31 - 1 = 2147483647 (a Mersenne prime).
        let a = CsrFp::from_coo(1, 1, &[t(0, 0, 5)], (1u64 << 31) - 1);
        assert_eq!(a.modulus(), (1u64 << 31) - 1);
    }

    #[test]
    fn negative_value_reduction_uses_euclidean_remainder() {
        let a = CsrFp::from_coo(1, 3, &[t(0, 0, -1), t(0, 1, -7), t(0, 2, -100)], P);
        assert_eq!(a.get(0, 0), Some(6));
        assert_eq!(a.get(0, 1), None); // -7 mod 7 = 0
        assert_eq!(a.get(0, 2), Some(5)); // -100 mod 7 = -100 + 15·7 = 5
    }

    #[test]
    fn transpose_swaps_indices() {
        let a = CsrFp::from_coo(2, 3, &[t(0, 0, 1), t(0, 2, 2), t(1, 1, 3)], P);
        let at = a.transpose();
        assert_eq!(at.rows(), 3);
        assert_eq!(at.cols(), 2);
        assert_eq!(at.modulus(), P);
        assert_eq!(at.get(0, 0), Some(1));
        assert_eq!(at.get(1, 1), Some(3));
        assert_eq!(at.get(2, 0), Some(2));
        // Round-trip.
        let att = at.transpose();
        assert_eq!(att.to_dense(), a.to_dense());
    }

    #[test]
    fn transpose_zero_matrix() {
        let z = CsrFp::zeros(2, 5, P).transpose();
        assert_eq!(z.rows(), 5);
        assert_eq!(z.cols(), 2);
        assert_eq!(z.nnz(), 0);
        assert_eq!(z.modulus(), P);
    }
}
