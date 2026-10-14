// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrix over the integers in compressed-sparse-row layout.
//!
//! Entries are FLINT-backed [`crate::Integer`] values. The
//! storage invariant — strictly increasing column indices per row, no
//! explicit zeros — is enforced by every constructor and assumed by
//! every accessor.

use crate::crt::{crt_combine, symmetric_lift};
use crate::hnf;
use crate::prime_set::{hadamard_bound, select_primes_for_bound};
use crate::snf;
use crate::Integer;

use super::coo::{build_csr, Triple};
use super::csr_fp::CsrFp;
use super::SparseRingStorage;

/// Sparse `rows × cols` matrix over `Z` in CSR layout.
#[derive(Clone, Debug)]
pub struct CsrZ {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<Integer>,
}

impl CsrZ {
    /// The all-zero `rows × cols` matrix (no stored entries).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Build from a slice of `(row, col, value)` triples.
    ///
    /// Duplicate `(row, col)` triples are summed; entries that end up
    /// at zero are not stored. Triples are cloned into owned `Integer`
    /// values (use [`CsrZ::from_coo_owned`] to move).
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::Integer;
    /// use sparla::storage::{CsrZ, SparseRingStorage};
    ///
    /// let a = CsrZ::from_coo(
    ///     2,
    ///     2,
    ///     &[
    ///         (0, 0, Integer::from(1)),
    ///         (0, 1, Integer::from(2)),
    ///         (1, 1, Integer::from(3)),
    ///     ],
    /// );
    /// assert_eq!(a.rows(), 2);
    /// assert_eq!(a.nnz(), 3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any triple's row is `>= rows` or col is `>= cols`.
    pub fn from_coo(rows: usize, cols: usize, triples: &[Triple<Integer>]) -> Self {
        let owned: Vec<Triple<Integer>> = triples
            .iter()
            .map(|(r, c, v)| (*r, *c, v.clone()))
            .collect();
        Self::from_coo_owned(rows, cols, owned)
    }

    /// Move-based variant of [`CsrZ::from_coo`] that consumes the
    /// triple list and avoids cloning each `Integer`.
    pub fn from_coo_owned(rows: usize, cols: usize, triples: Vec<Triple<Integer>>) -> Self {
        let (indptr, indices, data) = build_csr(
            rows,
            cols,
            triples,
            |acc, v| *acc += v,
            |v| v.cmp0() == std::cmp::Ordering::Equal,
        );
        Self {
            rows,
            cols,
            indptr,
            indices,
            data,
        }
    }

    /// Build directly from a list of already-sorted, already-canonical
    /// rows. Skips the COO sort / dedupe pipeline.
    ///
    /// Caller's contract (checked under `debug_assertions` only):
    /// `sorted_rows.len() == rows`; within each inner vec column
    /// indices are strictly increasing and lie in `[0, cols)`; every
    /// value is nonzero. Used by [`crate::hnf::hnf`] to skip a
    /// re-canonicalisation when emitting the HNF matrix.
    pub(crate) fn from_sorted_rows(
        rows: usize,
        cols: usize,
        sorted_rows: Vec<Vec<(usize, Integer)>>,
    ) -> Self {
        debug_assert_eq!(sorted_rows.len(), rows);
        let nnz_estimate: usize = sorted_rows.iter().map(|r| r.len()).sum();
        let mut indptr = vec![0usize; rows + 1];
        let mut indices: Vec<usize> = Vec::with_capacity(nnz_estimate);
        let mut data: Vec<Integer> = Vec::with_capacity(nnz_estimate);
        for (r, row) in sorted_rows.into_iter().enumerate() {
            let mut last_col: Option<usize> = None;
            for (c, v) in row {
                debug_assert!(c < cols, "col index {} out of bounds (cols={})", c, cols);
                debug_assert!(
                    v.cmp0() != std::cmp::Ordering::Equal,
                    "zero value passed to from_sorted_rows"
                );
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
            indptr,
            indices,
            data,
        }
    }

    /// Canonical Hermite Normal Form over `Z`.
    ///
    /// Output rows are in canonical order: pivot rows first (sorted
    /// by pivot column), then any all-zero rows. Pivots are positive
    /// and each above-pivot entry lies in `[0, pivot_value)`. See
    /// [`hnf::hnf`] for the algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::Integer;
    /// use sparla::storage::CsrZ;
    ///
    /// // [[2, 4], [6, 8]] reduces to [[2, 0], [0, 4]].
    /// let a = CsrZ::from_coo(2, 2, &[
    ///     (0, 0, Integer::from(2)), (0, 1, Integer::from(4)),
    ///     (1, 0, Integer::from(6)), (1, 1, Integer::from(8)),
    /// ]);
    /// let h = a.hnf();
    /// assert_eq!(h.get(0, 0), Some(&Integer::from(2)));
    /// assert_eq!(h.get(0, 1), None);  // above-pivot entry reduced to 0
    /// assert_eq!(h.get(1, 1), Some(&Integer::from(4)));
    /// ```
    pub fn hnf(&self) -> CsrZ {
        hnf::hnf(self)
    }

    /// Hermite Normal Form paired with the unimodular transform:
    /// returns `(H, U)` such that `U · A = H` and `U` is invertible
    /// over `Z`. See [`hnf::hnf_with_transforms`].
    pub fn hnf_with_transforms(&self) -> (CsrZ, CsrZ) {
        hnf::hnf_with_transforms(self)
    }

    /// Z-basis of the column space `image(A) ⊆ Z^m` as columns of
    /// an `m × rank` sparse matrix. See [`crate::solve::image`].
    pub fn image(&self) -> CsrZ {
        crate::solve::image(self)
    }

    /// Z-basis of the kernel `{x ∈ Z^n : A · x = 0}` as columns of
    /// an `n × nullity` sparse matrix. See [`crate::solve::kernel`].
    pub fn kernel(&self) -> CsrZ {
        crate::solve::kernel(self)
    }

    /// Solve `A · x = b` over `Z`. Returns one particular solution
    /// as an `n × 1` sparse vector, or `None` if no integer
    /// solution exists. See [`crate::solve::solve_left`].
    ///
    /// # Panics
    ///
    /// Panics if `b.rows() != self.rows()` or `b.cols() != 1`.
    pub fn solve_left(&self, b: &CsrZ) -> Option<CsrZ> {
        crate::solve::solve_left(self, b)
    }

    /// Smith Normal Form diagonal of the matrix, padded with zeros to
    /// length `min(rows, cols)`.
    ///
    /// The returned vector satisfies `d_0 | d_1 | ... | d_{r-1}` and
    /// `d_i > 0` for `i < rank`; trailing entries are zero. See
    /// [`snf::snf`] for the pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::Integer;
    /// use sparla::storage::CsrZ;
    ///
    /// // diag(2, 3): coprime, so gcd 1 and lcm 6 → SNF = (1, 6).
    /// let a = CsrZ::from_coo(2, 2, &[
    ///     (0, 0, Integer::from(2)),
    ///     (1, 1, Integer::from(3)),
    /// ]);
    /// assert_eq!(a.snf(), vec![Integer::from(1), Integer::from(6)]);
    /// ```
    pub fn snf(&self) -> Vec<Integer> {
        snf::snf(self)
    }

    /// Exact `Z`-rank of the matrix. See [`hnf::rank`].
    pub fn rank(&self) -> usize {
        hnf::rank(self)
    }

    /// Sparse · sparse `Z` matrix multiplication. See
    /// [`crate::matmul::matmul_z`] for the algorithm and the
    /// validated perf shape against FLINT-dense.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::Integer;
    /// use sparla::storage::CsrZ;
    ///
    /// // [[1, 2], [3, 4]] · [[5, 6], [7, 8]] = [[19, 22], [43, 50]].
    /// let a = CsrZ::from_coo(2, 2, &[
    ///     (0, 0, Integer::from(1)), (0, 1, Integer::from(2)),
    ///     (1, 0, Integer::from(3)), (1, 1, Integer::from(4)),
    /// ]);
    /// let b = CsrZ::from_coo(2, 2, &[
    ///     (0, 0, Integer::from(5)), (0, 1, Integer::from(6)),
    ///     (1, 0, Integer::from(7)), (1, 1, Integer::from(8)),
    /// ]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.get(0, 0), Some(&Integer::from(19)));
    /// assert_eq!(c.get(1, 1), Some(&Integer::from(50)));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self.cols() != other.rows()`.
    pub fn matmul(&self, other: &CsrZ) -> CsrZ {
        crate::matmul::matmul_z(self, other)
    }

    /// Transpose: swap rows and columns. The result is `cols × rows`
    /// and stores entry `(c, r)` for every stored entry `(r, c)` of
    /// `self`.
    ///
    /// `O(rows + nnz)` — one pass over `self` populates per-column
    /// buckets, which are already sorted by row of `self` (=
    /// column of the transpose) because `iter_row` yields rows in
    /// `0..rows` order.
    pub fn transpose(&self) -> CsrZ {
        let m = self.rows();
        let n = self.cols();
        let mut tr_rows: Vec<Vec<(usize, Integer)>> = (0..n).map(|_| Vec::new()).collect();
        for r in 0..m {
            for (c, v) in self.iter_row(r) {
                tr_rows[c].push((r, v.clone()));
            }
        }
        CsrZ::from_sorted_rows(n, m, tr_rows)
    }

    /// Elementary divisors of the matrix: the nonzero invariant
    /// factors `d_0 | d_1 | ... | d_{r-1}` where `r = rank`.
    ///
    /// Derived from [`Self::snf`] by stripping the trailing zero padding:
    /// zero matrix → `[]`, identity n×n → `[1; n]`, and
    /// `diag(2, 6, 0)` → `[2, 6]`.
    pub fn elementary_divisors(&self) -> Vec<Integer> {
        let mut diag = self.snf();
        diag.retain(|d| d.cmp0() != std::cmp::Ordering::Equal);
        diag
    }

    /// Determinant of a square matrix over `Z`, via multi-prime CRT.
    ///
    /// The pipeline:
    ///
    /// 1. Compute the Hadamard upper bound `D ≥ |det(A)|` via
    ///    [`hadamard_bound`].
    /// 2. Pick distinct primes (largest 31-bit first, descending)
    ///    whose product strictly exceeds `2·D` via
    ///    [`select_primes_for_bound`].
    /// 3. For each prime `p`, reduce `A` mod `p` via
    ///    [`CsrFp::from_z`] and compute `det(A mod p)` with
    ///    [`crate::elim::gauss_fp::det`].
    /// 4. [`crt_combine`] folds the per-prime residues into a single
    ///    value mod the prime product; [`symmetric_lift`] maps it
    ///    into `(-prod/2, prod/2]`, recovering the signed integer
    ///    determinant.
    ///
    /// `O(k · per_prime_det_cost)` where `k` is the number of primes
    /// (typically a handful for matrices that fit in working
    /// memory). Each per-prime kernel runs in `u64` arithmetic — no
    /// GMP cost in the inner loop, in contrast to the direct
    /// integer kernel which suffers value swell.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparla::Integer;
    /// use sparla::storage::CsrZ;
    ///
    /// // det [[1, 2], [3, 4]] = 1·4 - 2·3 = -2.
    /// let a = CsrZ::from_coo(2, 2, &[
    ///     (0, 0, Integer::from(1)), (0, 1, Integer::from(2)),
    ///     (1, 0, Integer::from(3)), (1, 1, Integer::from(4)),
    /// ]);
    /// assert_eq!(a.det(), Integer::from(-2));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self` is not square.
    pub fn det(&self) -> Integer {
        assert_eq!(
            self.rows(),
            self.cols(),
            "det requires a square matrix; got {} × {}",
            self.rows(),
            self.cols()
        );
        let n = self.rows();
        if n == 0 {
            return Integer::from(1);
        }

        let hadamard = hadamard_bound(self);
        let bound: Integer = hadamard * 2u32;
        let primes = select_primes_for_bound(&bound);

        let pairs: Vec<(Integer, Integer)> = primes
            .into_iter()
            .map(|p| {
                let det_fp = CsrFp::from_z(self, p as u64).det();
                (Integer::from(det_fp), Integer::from(p))
            })
            .collect();

        let (value, modulus) = crt_combine(&pairs);
        symmetric_lift(&value, &modulus)
    }

    /// Look up entry `(row, col)`.
    ///
    /// Returns a borrowed reference when the cell is explicitly
    /// stored, otherwise `None` (representing a structural zero).
    ///
    /// Cost is `O(log nnz_in_row)` via binary search on the sorted
    /// `indices` slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows` or `col >= cols`.
    pub fn get(&self, row: usize, col: usize) -> Option<&Integer> {
        assert!(row < self.rows, "row index {} out of bounds", row);
        assert!(col < self.cols, "col index {} out of bounds", col);
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .binary_search(&col)
            .ok()
            .map(|off| &self.data[s + off])
    }

    /// Iterate the stored entries of `row` in column-ascending order.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows`.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, &Integer)> + '_ {
        assert!(row < self.rows, "row index {} out of bounds", row);
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .iter()
            .copied()
            .zip(self.data[s..e].iter())
    }

    /// Convert to a sorted-by-`(row, col)` COO triple list, cloning
    /// each `Integer`.
    pub fn to_coo(&self) -> Vec<Triple<Integer>> {
        let mut out = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                out.push((r, c, v.clone()));
            }
        }
        out
    }

    /// Densify into a row-major `Vec<Vec<Integer>>`. Useful for
    /// parity tests against dense reference implementations.
    pub fn to_dense(&self) -> Vec<Vec<Integer>> {
        let mut out: Vec<Vec<Integer>> = (0..self.rows)
            .map(|_| vec![Integer::new(); self.cols])
            .collect();
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in self.iter_row(r) {
                row[c].assign(v);
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

    /// `true` if the matrix equals its transpose. Requires a square
    /// matrix; returns `false` for non-square shapes.
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                match self.get(c, r) {
                    Some(w) if w == v => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Sum of the diagonal entries — `Σ A[i, i]`.
    pub fn trace(&self) -> Integer {
        let mut acc = Integer::new();
        let n = self.rows.min(self.cols);
        for i in 0..n {
            if let Some(v) = self.get(i, i) {
                acc += v;
            }
        }
        acc
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
                        let mut value = self.data[left].clone();
                        if subtract {
                            value -= &other.data[right];
                        } else {
                            value += &other.data[right];
                        }
                        if value.cmp0() != std::cmp::Ordering::Equal {
                            indices.push(left_col);
                            data.push(value);
                        }
                        left += 1;
                        right += 1;
                    }
                    (Some(&left_col), Some(&right_col)) if left_col < right_col => {
                        indices.push(left_col);
                        data.push(self.data[left].clone());
                        left += 1;
                    }
                    (Some(&left_col), None) => {
                        indices.push(left_col);
                        data.push(self.data[left].clone());
                        left += 1;
                    }
                    (_, Some(&right_col)) => {
                        indices.push(right_col);
                        data.push(if subtract {
                            -&other.data[right]
                        } else {
                            other.data[right].clone()
                        });
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
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|value| -value).collect(),
        }
    }

    /// Multiply every stored coefficient by an integer scalar.
    pub fn scale(&self, scalar: &Integer) -> Self {
        if scalar.cmp0() == std::cmp::Ordering::Equal {
            return Self::zeros(self.rows, self.cols);
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|value| value * scalar).collect(),
        }
    }
}

impl SparseRingStorage for CsrZ {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn t(r: usize, c: usize, v: i64) -> Triple<Integer> {
        (r, c, Integer::from(v))
    }

    #[test]
    fn zeros_has_no_entries() {
        let a = CsrZ::zeros(3, 4);
        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 4);
        assert_eq!(a.nnz(), 0);
        assert!(a.is_zero());
        assert_eq!(a.density(), 0.0);
        assert!(!a.is_square());
    }

    #[test]
    fn from_coo_orders_indices_within_row() {
        let a = CsrZ::from_coo(2, 3, &[t(1, 2, 7), t(0, 2, 4), t(0, 0, 1), t(1, 0, -3)]);
        assert_eq!(a.nnz(), 4);
        assert_eq!(a.indptr, vec![0, 2, 4]);
        assert_eq!(a.indices, vec![0, 2, 0, 2]);
        // get() round-trips every stored entry.
        assert_eq!(a.get(0, 0), Some(&Integer::from(1)));
        assert_eq!(a.get(0, 2), Some(&Integer::from(4)));
        assert_eq!(a.get(1, 0), Some(&Integer::from(-3)));
        assert_eq!(a.get(1, 2), Some(&Integer::from(7)));
        // Structural zero.
        assert_eq!(a.get(0, 1), None);
        assert_eq!(a.get(1, 1), None);
    }

    #[test]
    fn duplicate_entries_are_summed() {
        let a = CsrZ::from_coo(1, 2, &[t(0, 0, 10), t(0, 0, 5), t(0, 1, 1)]);
        assert_eq!(a.get(0, 0), Some(&Integer::from(15)));
        assert_eq!(a.get(0, 1), Some(&Integer::from(1)));
        assert_eq!(a.nnz(), 2);
    }

    #[test]
    fn duplicates_cancelling_to_zero_are_dropped() {
        let a = CsrZ::from_coo(1, 2, &[t(0, 0, 4), t(0, 0, -4), t(0, 1, 9)]);
        assert_eq!(a.nnz(), 1);
        assert_eq!(a.get(0, 0), None);
        assert_eq!(a.get(0, 1), Some(&Integer::from(9)));
    }

    #[test]
    fn from_coo_owned_consumes_without_clone() {
        let triples = vec![t(0, 0, 3), t(0, 1, 4)];
        let a = CsrZ::from_coo_owned(1, 2, triples);
        assert_eq!(a.nnz(), 2);
    }

    #[test]
    fn iter_row_yields_sorted_entries() {
        let a = CsrZ::from_coo(2, 3, &[t(0, 2, 5), t(0, 0, 1), t(0, 1, 3)]);
        let row0: Vec<_> = a.iter_row(0).map(|(c, v)| (c, v.clone())).collect();
        assert_eq!(
            row0,
            vec![
                (0, Integer::from(1)),
                (1, Integer::from(3)),
                (2, Integer::from(5))
            ]
        );
        let row1: Vec<_> = a.iter_row(1).collect();
        assert!(row1.is_empty());
    }

    #[test]
    fn to_dense_matches_inputs() {
        let a = CsrZ::from_coo(2, 2, &[t(0, 0, 1), t(0, 1, 2), t(1, 1, -3)]);
        let d = a.to_dense();
        assert_eq!(d[0][0], Integer::from(1));
        assert_eq!(d[0][1], Integer::from(2));
        assert_eq!(d[1][0], Integer::from(0));
        assert_eq!(d[1][1], Integer::from(-3));
    }

    #[test]
    fn to_coo_round_trips_through_from_coo() {
        let original = CsrZ::from_coo(3, 3, &[t(0, 1, 2), t(2, 0, 7), t(1, 2, -1)]);
        let triples = original.to_coo();
        let rebuilt = CsrZ::from_coo(3, 3, &triples);
        assert_eq!(rebuilt.indptr, original.indptr);
        assert_eq!(rebuilt.indices, original.indices);
        assert_eq!(rebuilt.data, original.data);
    }

    #[test]
    fn density_is_nnz_over_total_cells() {
        let a = CsrZ::from_coo(2, 4, &[t(0, 0, 1), t(0, 1, 1)]);
        assert_eq!(a.density(), 2.0 / 8.0);
        let empty = CsrZ::zeros(0, 5);
        assert_eq!(empty.density(), 0.0);
    }

    #[test]
    fn is_diagonal_predicate() {
        let diag = CsrZ::from_coo(3, 3, &[t(0, 0, 1), t(1, 1, 2), t(2, 2, 3)]);
        assert!(diag.is_diagonal());
        let off = CsrZ::from_coo(3, 3, &[t(0, 0, 1), t(1, 2, 5)]);
        assert!(!off.is_diagonal());
        // The zero matrix is vacuously diagonal.
        assert!(CsrZ::zeros(2, 2).is_diagonal());
    }

    #[test]
    fn is_symmetric_predicate() {
        let sym = CsrZ::from_coo(
            3,
            3,
            &[t(0, 1, 5), t(1, 0, 5), t(0, 2, 7), t(2, 0, 7), t(1, 1, 9)],
        );
        assert!(sym.is_symmetric());
        let asym = CsrZ::from_coo(2, 2, &[t(0, 1, 1), t(1, 0, 2)]);
        assert!(!asym.is_symmetric());
        // Non-square is never symmetric (no consistent transpose shape).
        let wide = CsrZ::from_coo(1, 2, &[t(0, 1, 3)]);
        assert!(!wide.is_symmetric());
    }

    #[test]
    fn trace_sums_diagonal() {
        let a = CsrZ::from_coo(3, 3, &[t(0, 0, 2), t(1, 1, 5), t(2, 2, -1), t(0, 1, 99)]);
        assert_eq!(a.trace(), Integer::from(6));
        // Rectangular: trace runs to min(rows, cols).
        let wide = CsrZ::from_coo(2, 4, &[t(0, 0, 3), t(1, 1, 4), t(1, 3, 100)]);
        assert_eq!(wide.trace(), Integer::from(7));
    }

    #[test]
    fn transpose_swaps_indices() {
        // [[1, 0, 2], [0, 3, 0]] → [[1, 0], [0, 3], [2, 0]].
        let a = CsrZ::from_coo(2, 3, &[t(0, 0, 1), t(0, 2, 2), t(1, 1, 3)]);
        let at = a.transpose();
        assert_eq!(at.rows(), 3);
        assert_eq!(at.cols(), 2);
        assert_eq!(at.get(0, 0), Some(&Integer::from(1)));
        assert_eq!(at.get(1, 1), Some(&Integer::from(3)));
        assert_eq!(at.get(2, 0), Some(&Integer::from(2)));
        // Round-trip.
        let att = at.transpose();
        assert_eq!(att.to_dense(), a.to_dense());
    }

    #[test]
    fn transpose_empty_and_zero() {
        assert_eq!(CsrZ::zeros(0, 0).transpose().rows(), 0);
        let z3x4 = CsrZ::zeros(3, 4).transpose();
        assert_eq!(z3x4.rows(), 4);
        assert_eq!(z3x4.cols(), 3);
        assert_eq!(z3x4.nnz(), 0);
    }

    #[test]
    fn transpose_preserves_nnz_and_negative_values() {
        let a = CsrZ::from_coo(2, 2, &[t(0, 0, -5), t(0, 1, 7), t(1, 0, -11)]);
        let at = a.transpose();
        assert_eq!(at.nnz(), 3);
        assert_eq!(at.get(0, 0), Some(&Integer::from(-5)));
        assert_eq!(at.get(0, 1), Some(&Integer::from(-11)));
        assert_eq!(at.get(1, 0), Some(&Integer::from(7)));
    }

    #[test]
    fn elementary_divisors_of_zero_matrix_is_empty() {
        assert!(CsrZ::zeros(3, 3).elementary_divisors().is_empty());
    }

    #[test]
    fn elementary_divisors_of_identity() {
        let i3 = CsrZ::from_coo(3, 3, &[t(0, 0, 1), t(1, 1, 1), t(2, 2, 1)]);
        assert_eq!(
            i3.elementary_divisors(),
            vec![Integer::from(1), Integer::from(1), Integer::from(1)]
        );
    }

    #[test]
    fn elementary_divisors_strips_zero_padding() {
        // diag(2, 6, 0) → SNF = (2, 6, 0). Elementary divisors drop the 0.
        let a = CsrZ::from_coo(3, 3, &[t(0, 0, 2), t(1, 1, 6)]);
        assert_eq!(
            a.elementary_divisors(),
            vec![Integer::from(2), Integer::from(6)]
        );
    }

    #[test]
    fn elementary_divisors_count_matches_rank() {
        let a = CsrZ::from_coo(3, 3, &[t(0, 0, 1), t(0, 1, 2), t(1, 0, 2), t(1, 1, 4)]);
        let r = a.rank();
        let ed = a.elementary_divisors();
        assert_eq!(ed.len(), r);
    }
}
