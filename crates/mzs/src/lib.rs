// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Sparse linear algebra over **F_2**.
//!
//! This is the sparse counterpart to `m4rst`'s dense, bit-packed matrices.
//! It supports exact multiplication, transpose, elimination, rank, solve, and
//! nullspace.
//!
//! ## Storage
//!
//! Each row is a sorted `Vec<u32>` of the column indices of the `1`s
//! in that row. No value array — F_2 means every stored entry is `1`.
//!
//! ## Headline op
//!
//! [`Mzs::matmul`] — sparse · sparse F_2 matmul. For each row `i` of
//! `A`, accumulate XORs of `B`'s rows indexed by `A.row_indices(i)`,
//! emit the set bits in sorted order as `C.row_indices(i)`. Cost
//! `O(rows · (nnz(A_row) · avg_nnz(B_row)))` plus a per-row sort.

/// Sparse F_2 matrix, indexed-row storage.
///
/// See the crate-level docs for the storage layout.
///
/// # Examples
///
/// ```
/// use mzs::Mzs;
/// let a = Mzs::from_triples(2, 2, &[(0, 0), (1, 1)]);  // I_2
/// let b = Mzs::from_triples(2, 2, &[(0, 1), (1, 0)]);  // swap
/// let c = a.matmul(&b);                                 // I·swap = swap
/// assert_eq!(c, b);
/// ```
#[derive(Debug, Clone)]
pub struct Mzs {
    rows: usize,
    cols: usize,
    /// One sorted `Vec<u32>` per row, holding the column indices of
    /// the `1`s in that row.
    row_indices: Vec<Vec<u32>>,
}

impl Mzs {
    /// Construct an empty `rows × cols` F_2 matrix. All cells start
    /// at zero.
    ///
    /// Panics if `cols > u32::MAX` (column indices are stored as
    /// `u32` for compactness).
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let m = Mzs::new(3, 5);
    /// assert_eq!(m.rows(), 3);
    /// assert_eq!(m.cols(), 5);
    /// assert_eq!(m.nnz(), 0);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(
            cols <= u32::MAX as usize,
            "mzs: cols ({cols}) exceeds u32::MAX"
        );
        Self {
            rows,
            cols,
            row_indices: vec![Vec::new(); rows],
        }
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

    /// Total number of stored nonzeros across all rows.
    pub fn nnz(&self) -> usize {
        self.row_indices.iter().map(|r| r.len()).sum()
    }

    /// Density (`nnz / (rows · cols)`). Returns `0.0` for a matrix
    /// with no cells.
    pub fn density(&self) -> f64 {
        let total = self.rows.saturating_mul(self.cols);
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Read cell `(i, j)`. Returns `0` or `1`. Bounds-checked in
    /// debug builds.
    pub fn get(&self, i: usize, j: usize) -> u8 {
        debug_assert!(i < self.rows && j < self.cols);
        if self.row_indices[i].binary_search(&(j as u32)).is_ok() {
            1
        } else {
            0
        }
    }

    /// Write `value & 1` to cell `(i, j)`. Bounds-checked in debug
    /// builds.
    pub fn set(&mut self, i: usize, j: usize, value: u8) {
        debug_assert!(i < self.rows && j < self.cols);
        let j32 = j as u32;
        let row = &mut self.row_indices[i];
        match row.binary_search(&j32) {
            Ok(idx) => {
                if value & 1 == 0 {
                    row.remove(idx);
                }
            }
            Err(idx) => {
                if value & 1 == 1 {
                    row.insert(idx, j32);
                }
            }
        }
    }

    /// Toggle cell `(i, j)` — flip its value (`0 → 1` or `1 → 0`).
    /// Equivalent to `set(i, j, get(i, j) ^ 1)`.
    pub fn flip(&mut self, i: usize, j: usize) {
        let v = self.get(i, j);
        self.set(i, j, v ^ 1);
    }

    /// Number of nonzero entries in row `r`.
    pub fn row_nnz(&self, r: usize) -> usize {
        self.row_indices[r].len()
    }

    /// Borrow the sorted column-index slice of row `r`.
    pub fn row_indices(&self, r: usize) -> &[u32] {
        &self.row_indices[r]
    }

    /// Build from a list of `(row, col)` 1-positions. Duplicate
    /// entries are XORed together — each repetition toggles the
    /// cell, so even-count entries cancel.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let m = Mzs::from_triples(2, 2, &[(0, 0), (0, 0), (1, 1)]);
    /// // (0,0) cancels itself; only (1,1) survives.
    /// assert_eq!(m.get(0, 0), 0);
    /// assert_eq!(m.get(1, 1), 1);
    /// assert_eq!(m.nnz(), 1);
    /// ```
    pub fn from_triples(rows: usize, cols: usize, triples: &[(usize, usize)]) -> Self {
        let mut m = Self::new(rows, cols);
        for &(i, j) in triples {
            m.flip(i, j);
        }
        m
    }

    /// Out-of-place transpose. Returns a new `cols × rows` matrix.
    ///
    /// Cost: `O(rows + nnz)`. Iterates set cells row-major and
    /// appends to the transposed row; since rows are processed in
    /// order, the transposed lists are sorted by construction.
    pub fn transpose(&self) -> Self {
        let mut out = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for &j in &self.row_indices[i] {
                out.row_indices[j as usize].push(i as u32);
            }
        }
        out
    }

    /// Element-wise sum over `F_2`, equivalently row-wise symmetric
    /// difference of the stored column-index sets.
    ///
    /// # Panics
    ///
    /// On shape mismatch.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "add: matrix shape mismatch"
        );
        let mut out = Self::new(self.rows, self.cols);
        for ((dst, left), right) in out
            .row_indices
            .iter_mut()
            .zip(&self.row_indices)
            .zip(&other.row_indices)
        {
            dst.reserve(left.len() + right.len());
            let (mut i, mut j) = (0, 0);
            while i < left.len() && j < right.len() {
                match left[i].cmp(&right[j]) {
                    std::cmp::Ordering::Less => {
                        dst.push(left[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        dst.push(right[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        i += 1;
                        j += 1;
                    }
                }
            }
            dst.extend_from_slice(&left[i..]);
            dst.extend_from_slice(&right[j..]);
        }
        out
    }

    /// Multiply by a scalar in `F_2`.
    pub fn scalar_mul(&self, scalar: u8) -> Self {
        if scalar & 1 == 0 {
            Self::new(self.rows, self.cols)
        } else {
            self.clone()
        }
    }

    /// Densify to a row-major `Vec<u8>` of length `rows · cols`,
    /// one byte per cell (`0` or `1`). Matches `m4rst::Mzd`'s byte-
    /// buffer convention exactly, so the round-trip
    /// `m4rst::Mzd::from_byte_buffer(rows, cols, &mzs.to_byte_buffer())`
    /// is the bridge to dense F_2 LA without either crate having to
    /// path-dep the other.
    ///
    /// Cost: `O(rows · cols)` time and space.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let m = Mzs::from_triples(2, 3, &[(0, 0), (1, 2)]);
    /// assert_eq!(m.to_byte_buffer(), vec![1, 0, 0, 0, 0, 1]);
    /// ```
    pub fn to_byte_buffer(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.cols];
        for i in 0..self.rows {
            let base = i * self.cols;
            for &j in &self.row_indices[i] {
                out[base + j as usize] = 1;
            }
        }
        out
    }

    /// Build from a row-major `&[u8]` of length `rows · cols`, one
    /// byte per cell. Each byte's low bit (`byte & 1`) is the F_2
    /// value; high bits are ignored. Inverse of
    /// [`to_byte_buffer`](Self::to_byte_buffer).
    ///
    /// # Panics
    ///
    /// If `bytes.len() != rows · cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// // Round-trip through a byte buffer is the identity.
    /// let a = Mzs::from_triples(3, 4, &[(0, 0), (0, 2), (1, 3), (2, 1)]);
    /// let bytes = a.to_byte_buffer();
    /// let b = Mzs::from_byte_buffer(3, 4, &bytes);
    /// assert_eq!(a, b);
    /// ```
    pub fn from_byte_buffer(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        assert_eq!(
            bytes.len(),
            rows * cols,
            "from_byte_buffer: bytes.len() ({}) != rows · cols ({})",
            bytes.len(),
            rows * cols,
        );
        let mut m = Self::new(rows, cols);
        for i in 0..rows {
            let base = i * cols;
            let row = &mut m.row_indices[i];
            for j in 0..cols {
                if bytes[base + j] & 1 == 1 {
                    row.push(j as u32);
                }
            }
        }
        m
    }

    /// Sparse · sparse F_2 matrix multiplication. Returns `C = A · B`.
    ///
    /// Algorithm: for each row `i` of `A`, walk the nonzero columns
    /// `k` of row `i`; for each such `k`, XOR row `k` of `B` into the
    /// `C[i]` accumulator. Emit the set bits in sorted order.
    ///
    /// Cost: `O(sum_i nnz(A.row(i)) · avg_nnz(B.row(k)))` toggles, plus
    /// `O(touched · log touched)` per row for the sort. Beats dense
    /// matmul when the density of `A · B` is below ~1.5%.
    ///
    /// # Panics
    ///
    /// On shape mismatch: `self.cols() == other.rows()` required.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// // [[1 1], [0 1]] · [[1 0], [1 1]] = [[0 1], [1 1]] over F_2.
    /// let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    /// let b = Mzs::from_triples(2, 2, &[(0, 0), (1, 0), (1, 1)]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.get(0, 0), 0);
    /// assert_eq!(c.get(0, 1), 1);
    /// assert_eq!(c.get(1, 0), 1);
    /// assert_eq!(c.get(1, 1), 1);
    /// ```
    pub fn matmul(&self, other: &Mzs) -> Mzs {
        assert_eq!(
            self.cols, other.rows,
            "matmul: A.cols ({}) != B.rows ({})",
            self.cols, other.rows,
        );
        let m = self.rows;
        let n = other.cols;
        let mut out = Mzs::new(m, n);
        if n == 0 || m == 0 {
            return out;
        }
        // Per-row dense bool accumulator. `touched` tracks which slots
        // we've poked at so the per-row reset is O(touched), not O(n).
        let mut acc = vec![false; n];
        let mut touched: Vec<u32> = Vec::new();
        for i in 0..m {
            touched.clear();
            for &k in &self.row_indices[i] {
                for &j in &other.row_indices[k as usize] {
                    acc[j as usize] ^= true;
                    touched.push(j);
                }
            }
            // Dedupe and sort so the row emits sorted unique indices,
            // and so we don't try to read/clear the same acc-slot twice.
            touched.sort_unstable();
            touched.dedup();
            let row = &mut out.row_indices[i];
            row.reserve(touched.len());
            for &j in &touched {
                if acc[j as usize] {
                    row.push(j);
                }
                // Clear for the next row.
                acc[j as usize] = false;
            }
        }
        out
    }

    /// Parallel sparse · sparse F_2 matmul. Same algorithm as
    /// [`matmul`](Self::matmul), but parallelizes over output rows
    /// via rayon — each thread handles a subset of rows with its own
    /// dense-bool accumulator and touched-list scratch.
    ///
    /// Bit-identical to [`matmul`](Self::matmul); cross-validated in
    /// `tests/proptest.rs` and `tests/matmul.rs`.
    ///
    /// Gated on `--features parallel`. Not the default: at small `N`
    /// or very low density, the per-row thread-pool dispatch cost
    /// exceeds the parallelism gain. Callers who know their workload
    /// is large enough to benefit should opt in explicitly.
    ///
    /// # Panics
    ///
    /// On shape mismatch: `self.cols() == other.rows()` required.
    #[cfg(feature = "parallel")]
    pub fn matmul_parallel(&self, other: &Mzs) -> Mzs {
        use rayon::prelude::*;
        assert_eq!(
            self.cols, other.rows,
            "matmul_parallel: A.cols ({}) != B.rows ({})",
            self.cols, other.rows,
        );
        let n = other.cols;
        let mut out = Mzs::new(self.rows, n);
        if self.rows == 0 || n == 0 {
            return out;
        }
        // Per-output-row work. Each thread allocates its own accumulator and
        // touched scratch; rayon's work-stealing distributes the rows.
        out.row_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, row_out)| {
                let mut acc = vec![false; n];
                let mut touched: Vec<u32> = Vec::new();
                for &k in &self.row_indices[i] {
                    for &j in &other.row_indices[k as usize] {
                        acc[j as usize] ^= true;
                        touched.push(j);
                    }
                }
                touched.sort_unstable();
                touched.dedup();
                row_out.reserve(touched.len());
                for &j in &touched {
                    if acc[j as usize] {
                        row_out.push(j);
                    }
                }
            });
        out
    }

    /// Sparse · dense F_2 matmul against a bit-packed dense `B`.
    ///
    /// `B` is row-major bit-packed: each row is
    /// `stride = ⌈b_cols / 64⌉` `u64` words, with column `j` of row
    /// `r` carried by bit `j % 64` of `b[r * stride + j / 64]`
    /// (LSB-first within each word — same layout convention as
    /// `m4rst::Mzd`). Bits past column `b_cols - 1` in the last
    /// word of each row are ignored on input; the caller is not
    /// required to zero them.
    ///
    /// Output is the same packed layout: `Vec<u64>` of length
    /// `self.rows * stride`, with the post-`b_cols` slack bits
    /// masked to zero so the result is canonically clean.
    ///
    /// Cost: `O(self.rows · avg_nnz(A_row) · stride)` u64 XORs.
    /// Each inner XOR processes 64 columns of `B` at once — this is
    /// the headline sparse·dense fast path.
    ///
    /// # Panics
    ///
    /// If `self.cols() != b_rows`, or `b.len() != b_rows * stride`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// // A = I_2, so A · B = B.
    /// let a = Mzs::from_triples(2, 2, &[(0, 0), (1, 1)]);
    /// // B has 2 rows, 2 cols, stride = 1. Bit j of word 0 is column j.
    /// let b = [0b01u64, 0b10u64];
    /// let c = a.matmul_dense_packed(2, 2, &b);
    /// assert_eq!(c, vec![0b01u64, 0b10u64]);
    /// ```
    pub fn matmul_dense_packed(&self, b_rows: usize, b_cols: usize, b: &[u64]) -> Vec<u64> {
        assert_eq!(
            self.cols, b_rows,
            "matmul_dense_packed: A.cols ({}) != B.rows ({})",
            self.cols, b_rows,
        );
        let stride = b_cols.div_ceil(64);
        assert_eq!(
            b.len(),
            b_rows * stride,
            "matmul_dense_packed: b.len ({}) != b_rows · stride ({} · {})",
            b.len(),
            b_rows,
            stride,
        );
        let mut out = vec![0u64; self.rows * stride];
        if stride == 0 || self.rows == 0 {
            return out;
        }
        for i in 0..self.rows {
            let o_base = i * stride;
            for &k in &self.row_indices[i] {
                let b_base = (k as usize) * stride;
                for w in 0..stride {
                    out[o_base + w] ^= b[b_base + w];
                }
            }
        }
        // Canonicalise slack bits in the last word: input bits past
        // column `b_cols - 1` are ignored per the contract, so their
        // XOR contribution gets cleaned out here.
        let tail_bits = b_cols % 64;
        if tail_bits != 0 {
            let mask = (1u64 << tail_bits) - 1;
            let last = stride - 1;
            for i in 0..self.rows {
                out[i * stride + last] &= mask;
            }
        }
        out
    }

    /// Sparse · dense F_2 matmul against a byte-buffer dense `B`.
    ///
    /// `B` is a row-major `&[u8]` of length `b_rows · b_cols`, one
    /// byte per cell — same convention as
    /// [`from_byte_buffer`](Self::from_byte_buffer) and
    /// [`to_byte_buffer`](Self::to_byte_buffer). Each byte's low bit
    /// (`byte & 1`) carries the F_2 value; high bits are ignored.
    ///
    /// Output is the same layout: `Vec<u8>` of length
    /// `self.rows · b_cols`, with each byte in `{0, 1}`.
    ///
    /// Convenience wrapper. Cost is
    /// `O(self.rows · avg_nnz(A_row) · b_cols)` byte XORs — for
    /// large `b_cols` or repeated calls against the same `B`,
    /// prefer [`matmul_dense_packed`](Self::matmul_dense_packed),
    /// which gets ~64× the inner-loop throughput on the same work.
    ///
    /// # Panics
    ///
    /// If `self.cols() != b_rows`, or `b.len() != b_rows * b_cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// // A = [[1, 1], [0, 1]], B = [[1, 0], [1, 1]] over F_2.
    /// // A · B = [[0, 1], [1, 1]].
    /// let a = Mzs::from_triples(2, 2, &[(0, 0), (0, 1), (1, 1)]);
    /// let b = [1u8, 0, 1, 1];
    /// assert_eq!(a.matmul_dense(2, 2, &b), vec![0, 1, 1, 1]);
    /// ```
    pub fn matmul_dense(&self, b_rows: usize, b_cols: usize, b: &[u8]) -> Vec<u8> {
        assert_eq!(
            self.cols, b_rows,
            "matmul_dense: A.cols ({}) != B.rows ({})",
            self.cols, b_rows,
        );
        assert_eq!(
            b.len(),
            b_rows * b_cols,
            "matmul_dense: b.len ({}) != b_rows · b_cols ({} · {})",
            b.len(),
            b_rows,
            b_cols,
        );
        let mut out = vec![0u8; self.rows * b_cols];
        if b_cols == 0 || self.rows == 0 {
            return out;
        }
        for i in 0..self.rows {
            let (o_start, o_end) = (i * b_cols, (i + 1) * b_cols);
            let o_row = &mut out[o_start..o_end];
            for &k in &self.row_indices[i] {
                let b_base = (k as usize) * b_cols;
                let b_row = &b[b_base..b_base + b_cols];
                for (o, bv) in o_row.iter_mut().zip(b_row.iter()) {
                    *o ^= *bv;
                }
            }
        }
        // XOR is bitwise; bit 0 of the accumulator equals XOR-of-bit-0s
        // of the inputs regardless of high-bit garbage, so masking once
        // at the end produces the canonical {0, 1} output.
        for x in out.iter_mut() {
            *x &= 1;
        }
        out
    }
}

impl PartialEq for Mzs {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.row_indices == other.row_indices
    }
}

impl Eq for Mzs {}

// ─── Gaussian elimination ───────────────────────────────────────────
//
// rref as the foundation. rank / nullspace / solve_left are wrappers.
// All elimination is via `xor_row_self` — symmetric-differencing two
// sorted column-index lists in O(nnz_dst + nnz_src). Over F_2 there's
// no separate "scale pivot to 1" step; the pivot is already 1 if it
// exists.

impl Mzs {
    /// Swap rows `i` and `j` in place. `O(1)` — just a `Vec::swap` on
    /// the row-storage list.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.rows && j < self.rows);
        if i != j {
            self.row_indices.swap(i, j);
        }
    }

    /// XOR row `src` into row `dst` of `self`. Symmetric-differences
    /// the two sorted column-index lists in `O(nnz(dst) + nnz(src))`.
    /// Private helper for the Gaussian path.
    fn xor_row_self(&mut self, dst: usize, src: usize) {
        debug_assert!(dst < self.rows && src < self.rows);
        debug_assert_ne!(dst, src, "xor_row_self: dst == src zeros the row");
        let (lo, hi) = if dst < src { (dst, src) } else { (src, dst) };
        let (left, right) = self.row_indices.split_at_mut(hi);
        let lo_row = &left[lo];
        let hi_row = &right[0];
        let (src_row, dst_row_ref) = if dst < src {
            (hi_row, lo_row)
        } else {
            (lo_row, hi_row)
        };
        let mut merged = Vec::with_capacity(dst_row_ref.len() + src_row.len());
        let (mut i, mut j) = (0, 0);
        while i < dst_row_ref.len() && j < src_row.len() {
            match dst_row_ref[i].cmp(&src_row[j]) {
                std::cmp::Ordering::Less => {
                    merged.push(dst_row_ref[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(src_row[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        merged.extend_from_slice(&dst_row_ref[i..]);
        merged.extend_from_slice(&src_row[j..]);
        self.row_indices[dst] = merged;
    }

    /// Reduced row echelon form over `F_2`. Returns `(rref, rank)`.
    ///
    /// Gauss-Jordan with column-by-column pivoting: find the first
    /// row at or below the current pivot row with a `1` in the
    /// current column; if found, swap it up and XOR-eliminate that
    /// column from every other row; if not, the column is non-pivot
    /// and we advance.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let i = Mzs::from_triples(3, 3, &[(0, 0), (1, 1), (2, 2)]);
    /// let (rref, rank) = i.rref();
    /// assert_eq!(rank, 3);
    /// assert_eq!(rref, i);
    /// ```
    pub fn rref(&self) -> (Mzs, usize) {
        let mut work = self.clone();
        let m = work.rows;
        let n = work.cols;
        let mut rank = 0usize;
        let mut col = 0usize;
        while rank < m && col < n {
            let mut pivot_row: Option<usize> = None;
            for r in rank..m {
                if work.get(r, col) == 1 {
                    pivot_row = Some(r);
                    break;
                }
            }
            let pi = match pivot_row {
                Some(r) => r,
                None => {
                    col += 1;
                    continue;
                }
            };
            if pi != rank {
                work.swap_rows(pi, rank);
            }
            for i in 0..m {
                if i == rank {
                    continue;
                }
                if work.get(i, col) == 1 {
                    work.xor_row_self(i, rank);
                }
            }
            rank += 1;
            col += 1;
        }
        (work, rank)
    }

    /// Rank of `self` over `F_2`. Computed via [`rref`](Self::rref);
    /// `self` is unmodified.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let i = Mzs::from_triples(4, 4, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
    /// assert_eq!(i.rank(), 4);
    /// let z = Mzs::new(3, 5);
    /// assert_eq!(z.rank(), 0);
    /// ```
    pub fn rank(&self) -> usize {
        let (_, r) = self.rref();
        r
    }

    /// Compute a basis for the right nullspace: returns an
    /// `n × (n - rank)` matrix whose columns span
    /// `ker(self) = { v ∈ F_2^n : self · v = 0 }`.
    ///
    /// If `self` has full column rank, returns an `n × 0` matrix.
    /// Rank-nullity: `self.rank() + self.nullspace().cols() == self.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzs::Mzs;
    /// let a = Mzs::from_triples(2, 3, &[(0, 0), (0, 2), (1, 1), (1, 2)]);
    /// let k = a.nullspace();
    /// let ak = a.matmul(&k);
    /// assert_eq!(ak.nnz(), 0);
    /// assert_eq!(a.rank() + k.cols(), a.cols());
    /// ```
    pub fn nullspace(&self) -> Mzs {
        let n = self.cols;
        let (rref, rank) = self.rref();
        let mut pivot_col: Vec<usize> = Vec::with_capacity(rank);
        for i in 0..rank {
            // The leftmost 1 in row i is the pivot column.
            let row = rref.row_indices(i);
            pivot_col.push(row[0] as usize);
        }
        let mut is_pivot = vec![false; n];
        for &pc in &pivot_col {
            is_pivot[pc] = true;
        }
        let nullity = n - rank;
        let mut basis = Mzs::new(n, nullity);
        let mut bc = 0usize;
        for k in 0..n {
            if is_pivot[k] {
                continue;
            }
            // basis vector for free variable k:
            // - v[k] = 1
            // - for each pivot row i with rref[i, k] = 1, v[pivot_col[i]] = 1
            basis.set(k, bc, 1);
            for (i, &pc) in pivot_col.iter().enumerate() {
                if rref.get(i, k) == 1 {
                    basis.set(pc, bc, 1);
                }
            }
            bc += 1;
        }
        basis
    }

    /// Solve `A · X = B` over `F_2`. Returns `Some(X)` if consistent,
    /// `None` otherwise.
    ///
    /// `A` is `m × n`, `B` is `m × n_rhs`, `X` is `n × n_rhs`. The
    /// algorithm augments `[A | B]`, reduces to RREF, and reads `X`
    /// off the pivot rows. Inconsistency manifests as a pivot column
    /// landing in the `B`-part — that's the row `0 = nonzero`.
    ///
    /// When `A` is rank-deficient and the system is solvable, returns
    /// the natural solution with free variables set to zero; combine
    /// with [`nullspace`](Self::nullspace) to enumerate the affine
    /// solution set.
    pub fn solve_left(a: &Mzs, b: &Mzs) -> Option<Mzs> {
        assert_eq!(
            a.rows(),
            b.rows(),
            "solve_left: A.rows ({}) != B.rows ({})",
            a.rows(),
            b.rows(),
        );
        let m = a.rows;
        let n = a.cols;
        let n_rhs = b.cols;
        let mut aug = Mzs::new(m, n + n_rhs);
        for i in 0..m {
            for &j in a.row_indices(i) {
                aug.set(i, j as usize, 1);
            }
            for &j in b.row_indices(i) {
                aug.set(i, n + j as usize, 1);
            }
        }
        let (rref, rank) = aug.rref();
        let mut pivot_col: Vec<usize> = Vec::with_capacity(rank);
        for i in 0..rank {
            let pc = rref.row_indices(i)[0] as usize;
            if pc >= n {
                return None;
            }
            pivot_col.push(pc);
        }
        let mut x = Mzs::new(n, n_rhs);
        for (i, &pc) in pivot_col.iter().enumerate() {
            for c in 0..n_rhs {
                if rref.get(i, n + c) == 1 {
                    x.set(pc, c, 1);
                }
            }
        }
        Some(x)
    }
}
