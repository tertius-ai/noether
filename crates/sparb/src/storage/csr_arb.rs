// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrix over real balls in compressed-sparse-row layout.
//!
//! Entries are [`Arb`] balls — each a rigorous enclosure of a real
//! number. The storage invariant — strictly increasing column
//! indices per row, no *exact*-zero entries — is enforced by every
//! constructor and assumed by every accessor. A ball that merely
//! contains zero (nonzero radius) is a real entry and is kept.
//!
//! The matrix carries a single working precision (`prec`, in bits)
//! shared by every operation that produces new balls — COO
//! dedupe-sums and SpMV accumulation.

use crate::ball::Arb;

use super::coo::{build_csr, Triple};
use super::SparseRingStorage;

/// Sparse `rows × cols` matrix over real balls (`Arb`) in CSR
/// layout, at a fixed working precision.
#[derive(Clone, Debug)]
pub struct CsrArb {
    rows: usize,
    cols: usize,
    prec: i64,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<Arb>,
}

impl CsrArb {
    /// The all-zero `rows × cols` matrix at `prec` bits (no stored
    /// entries).
    ///
    /// # Panics
    ///
    /// Panics if `prec < 2` (FLINT's minimum working precision).
    pub fn zeros(rows: usize, cols: usize, prec: i64) -> Self {
        assert!(prec >= 2, "CsrArb: prec must be >= 2 bits; got {prec}");
        Self {
            rows,
            cols,
            prec,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Build from `(row, col, midpoint, radius)` `f64` quadruples —
    /// the pragmatic ingest path (a pyo3 glue layer, a FEM
    /// element-loop, a test fixture all speak `f64` pairs).
    ///
    /// # Dedupe semantics
    ///
    /// Duplicate `(row, col)` entries are summed in ball arithmetic
    /// **at `prec`**: each pair-sum uses FLINT `arb_add` with the
    /// matrix's working precision, so the resulting entry is the
    /// rounded ball sum, *not* an exact accumulator. This matters
    /// when the same `(row, col, mid, rad)` list is ingested at
    /// different precs: the *entries themselves* depend on `prec`
    /// when there are duplicates, even if the input list is fixed.
    /// Tests that sweep over `prec` should either dedupe their COO
    /// data first or build a matrix once at the highest prec and
    /// downstream-truncate.
    ///
    /// # Storage invariant
    ///
    /// An entry whose summed ball is the *exact* zero ball (`{0}`,
    /// radius 0) is not stored — that's a structural zero. A ball
    /// that merely contains zero (`mid = 0, rad > 0`) IS stored;
    /// dropping it would lose its rigorous enclosure.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparb::CsrArb;
    /// let a = CsrArb::from_coo_f64(
    ///     2, 2, 64,
    ///     &[(0, 0, 3.0, 0.0), (1, 1, 5.0, 0.01)],
    /// );
    /// assert_eq!(a.get(0, 0).unwrap().midpoint(), 3.0);
    /// assert!(a.get(0, 1).is_none()); // structural zero
    /// assert!(a.get(1, 1).unwrap().radius(64) >= 0.01);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `prec < 2`, if any `row >= rows` / `col >= cols`,
    /// or if any radius is negative or non-finite.
    pub fn from_coo_f64(
        rows: usize,
        cols: usize,
        prec: i64,
        triples: &[(usize, usize, f64, f64)],
    ) -> Self {
        assert!(prec >= 2, "CsrArb: prec must be >= 2 bits; got {prec}");
        let balls: Vec<Triple<Arb>> = triples
            .iter()
            .map(|&(r, c, mid, rad)| (r, c, Arb::from_mid_rad(mid, rad)))
            .collect();
        Self::from_coo_balls(rows, cols, prec, balls)
    }

    /// Build from owned `(row, col, Arb)` triples — the move-based
    /// ingest that avoids cloning each ball. Duplicate `(row, col)`
    /// entries are summed at `prec`.
    ///
    /// # Panics
    ///
    /// Panics if `prec < 2` or if any `row >= rows` / `col >= cols`.
    pub fn from_coo_balls(rows: usize, cols: usize, prec: i64, triples: Vec<Triple<Arb>>) -> Self {
        assert!(prec >= 2, "CsrArb: prec must be >= 2 bits; got {prec}");
        let (indptr, indices, data) = build_csr(
            rows,
            cols,
            triples,
            |acc, v| acc.add_assign(&v, prec),
            |v| v.is_exact_zero(),
        );
        Self {
            rows,
            cols,
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// Build directly from already-sorted, already-canonical rows.
    /// Skips the COO sort / dedupe pipeline. Used by [`Self::transpose`].
    ///
    /// Caller's contract (checked under `debug_assertions` only):
    /// `sorted_rows.len() == rows`; within each inner vec column
    /// indices are strictly increasing and lie in `[0, cols)`; every
    /// ball is non-exact-zero.
    fn from_sorted_rows(
        rows: usize,
        cols: usize,
        prec: i64,
        sorted_rows: Vec<Vec<(usize, Arb)>>,
    ) -> Self {
        debug_assert_eq!(sorted_rows.len(), rows);
        let nnz_estimate: usize = sorted_rows.iter().map(|r| r.len()).sum();
        let mut indptr = vec![0usize; rows + 1];
        let mut indices: Vec<usize> = Vec::with_capacity(nnz_estimate);
        let mut data: Vec<Arb> = Vec::with_capacity(nnz_estimate);
        for (r, row) in sorted_rows.into_iter().enumerate() {
            let mut last_col: Option<usize> = None;
            for (c, v) in row {
                debug_assert!(c < cols, "col index {c} out of bounds (cols={cols})");
                debug_assert!(
                    !v.is_exact_zero(),
                    "exact-zero ball passed to from_sorted_rows"
                );
                debug_assert!(
                    last_col.is_none_or(|lc| lc < c),
                    "column indices not strictly increasing within row {r}"
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
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// The working precision in bits shared across this matrix.
    pub fn prec(&self) -> i64 {
        self.prec
    }

    /// Look up entry `(row, col)`.
    ///
    /// Returns a borrowed reference when the cell is explicitly
    /// stored, otherwise `None` (a structural zero). `O(log
    /// nnz_in_row)` via binary search on the sorted index slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows` or `col >= cols`.
    pub fn get(&self, row: usize, col: usize) -> Option<&Arb> {
        assert!(row < self.rows, "row index {row} out of bounds");
        assert!(col < self.cols, "col index {col} out of bounds");
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
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, &Arb)> + '_ {
        assert!(row < self.rows, "row index {row} out of bounds");
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .iter()
            .copied()
            .zip(self.data[s..e].iter())
    }

    /// Sparse matrix–vector product `y = A · x` in ball arithmetic.
    ///
    /// See [`crate::spmv::spmv`] for the contract and panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparb::{Arb, CsrArb};
    /// let a = CsrArb::from_coo_f64(
    ///     2, 2, 64,
    ///     &[(0, 0, 1.0, 0.0), (1, 1, 2.0, 0.0)],
    /// );
    /// let y = a.spmv(&[Arb::from_f64(3.0), Arb::from_f64(4.0)]);
    /// assert_eq!(y[0].midpoint(), 3.0);
    /// assert_eq!(y[1].midpoint(), 8.0);
    /// ```
    pub fn spmv(&self, x: &[Arb]) -> Vec<Arb> {
        crate::spmv::spmv(self, x)
    }

    /// Element-wise sum `self + other` in ball arithmetic.
    ///
    /// Both matrices must have the same shape and the same working
    /// precision. The result's stored-entry pattern is the union of
    /// `self`'s and `other`'s; entries summing to the *exact* zero
    /// ball are dropped, but entries that merely contain zero are
    /// kept (dropping them would discard a rigorous enclosure).
    ///
    /// Cost: `O(nnz(self) + nnz(other))` ball adds at this matrix's
    /// precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparb::CsrArb;
    /// let a = CsrArb::from_coo_f64(
    ///     1, 3, 64, &[(0, 0, 1.0, 0.0), (0, 2, 3.0, 0.0)],
    /// );
    /// let b = CsrArb::from_coo_f64(1, 3, 64, &[(0, 1, 2.0, 0.0)]);
    /// let c = a.add(&b);
    /// assert_eq!(c.get(0, 0).unwrap().midpoint(), 1.0);
    /// assert_eq!(c.get(0, 1).unwrap().midpoint(), 2.0);
    /// assert_eq!(c.get(0, 2).unwrap().midpoint(), 3.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on shape mismatch or precision mismatch.
    pub fn add(&self, other: &CsrArb) -> CsrArb {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "CsrArb::add: shape mismatch ({}×{}) vs ({}×{})",
            self.rows,
            self.cols,
            other.rows,
            other.cols
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrArb::add: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        self.linear_combine(other, false)
    }

    /// Element-wise difference `self - other` in ball arithmetic.
    /// Mirrors [`Self::add`] in every respect — same shape /
    /// precision requirements, same exact-zero-drop rule, same
    /// `O(nnz(self) + nnz(other))` cost. Subtraction is *not* a
    /// rigorous shortcut for "negate then add" at the radius level
    /// because `arb_neg` is exact; this method runs FLINT's
    /// `arb_sub` per matched cell and `arb_neg` per `other`-only
    /// cell.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatch or precision mismatch.
    pub fn sub(&self, other: &CsrArb) -> CsrArb {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "CsrArb::sub: shape mismatch ({}×{}) vs ({}×{})",
            self.rows,
            self.cols,
            other.rows,
            other.cols
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrArb::sub: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        self.linear_combine(other, true)
    }

    /// Shared merge-pattern for [`Self::add`] / [`Self::sub`]. When
    /// `subtract` is true, `other`-only entries are emitted negated
    /// and matched entries use `arb_sub`; otherwise both code paths
    /// use plain copy / `arb_add`.
    fn linear_combine(&self, other: &CsrArb, subtract: bool) -> CsrArb {
        use std::cmp::Ordering::*;
        let prec = self.prec;
        let nnz_upper = self.nnz() + other.nnz();
        let mut indptr = Vec::with_capacity(self.rows + 1);
        let mut indices = Vec::with_capacity(nnz_upper);
        let mut data: Vec<Arb> = Vec::with_capacity(nnz_upper);
        indptr.push(0);
        for r in 0..self.rows {
            let (a_s, a_e) = (self.indptr[r], self.indptr[r + 1]);
            let (b_s, b_e) = (other.indptr[r], other.indptr[r + 1]);
            let (mut i, mut j) = (a_s, b_s);
            while i < a_e && j < b_e {
                match self.indices[i].cmp(&other.indices[j]) {
                    Less => {
                        indices.push(self.indices[i]);
                        data.push(self.data[i].clone());
                        i += 1;
                    }
                    Greater => {
                        indices.push(other.indices[j]);
                        data.push(if subtract {
                            other.data[j].neg()
                        } else {
                            other.data[j].clone()
                        });
                        j += 1;
                    }
                    Equal => {
                        let combined = if subtract {
                            self.data[i].sub(&other.data[j], prec)
                        } else {
                            let mut s = self.data[i].clone();
                            s.add_assign(&other.data[j], prec);
                            s
                        };
                        if !combined.is_exact_zero() {
                            indices.push(self.indices[i]);
                            data.push(combined);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            while i < a_e {
                indices.push(self.indices[i]);
                data.push(self.data[i].clone());
                i += 1;
            }
            while j < b_e {
                indices.push(other.indices[j]);
                data.push(if subtract {
                    other.data[j].neg()
                } else {
                    other.data[j].clone()
                });
                j += 1;
            }
            indptr.push(indices.len());
        }
        CsrArb {
            rows: self.rows,
            cols: self.cols,
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// Multiply every entry by the ball `alpha`, in place into a
    /// fresh matrix at the same shape and precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparb::{Arb, CsrArb};
    /// let a = CsrArb::from_coo_f64(
    ///     2, 2, 64, &[(0, 0, 3.0, 0.0), (1, 1, -7.0, 0.0)],
    /// );
    /// let c = a.scale(&Arb::from_f64(2.0));
    /// assert_eq!(c.get(0, 0).unwrap().midpoint(), 6.0);
    /// assert_eq!(c.get(1, 1).unwrap().midpoint(), -14.0);
    /// ```
    ///
    /// If `alpha` is the *exact* zero ball, the result is the zero
    /// matrix (no stored entries). Otherwise the result's stored
    /// pattern is identical to `self`'s — `alpha * entry` is never
    /// the *exact* zero ball when neither factor is (FLINT's
    /// `arb_mul` widens the radius on any cross-term cancellation,
    /// so a product whose midpoints happen to be zero still carries
    /// a non-zero radius and is preserved as a rigorous enclosure).
    ///
    /// Cost: `O(nnz)` ball multiplies at this matrix's precision.
    pub fn scale(&self, alpha: &Arb) -> CsrArb {
        let prec = self.prec;
        if alpha.is_exact_zero() {
            return CsrArb::zeros(self.rows, self.cols, prec);
        }
        let data: Vec<Arb> = self.data.iter().map(|v| v.mul(alpha, prec)).collect();
        CsrArb {
            rows: self.rows,
            cols: self.cols,
            prec,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data,
        }
    }

    /// Sparse × sparse ball matrix multiply `self · other` —
    /// Gustavson's row-by-row algorithm with `arb_addmul` as the
    /// inner-loop primitive.
    ///
    /// # Contract
    ///
    /// `self.cols()` must equal `other.rows()`. Both matrices must
    /// share the same working precision. Each output ball
    /// `C[i, j]` rigorously encloses the true mathematical sum
    /// `Σ_k A[i, k] · B[k, j]`. Pattern: `j` appears in row `i`
    /// of the result iff at least one `(k, A[i,k] != 0_exact)` /
    /// `(k → j, B[k,j] != 0_exact)` chain reaches it; balls that
    /// happen to round to the exact zero ball are dropped from
    /// storage per the storage invariant.
    ///
    /// # Algorithm
    ///
    /// Standard Gustavson — for each row `i` of `A`, walk its
    /// stored entries `(k, A[i,k])`; for each, walk row `k` of
    /// `B` and accumulate `A[i,k] · B[k,j]` into a dense workspace
    /// indexed by `j`. After a row of `A` is fully processed, the
    /// touched workspace slots are emitted in column-ascending
    /// order and reset. The accumulator uses `arb_addmul` so each
    /// `(j, k → j)` pair-contribution costs one FFI call and reuses
    /// the workspace ball's mantissa across the row.
    ///
    /// # Cost
    ///
    /// `O(flops)` ball fused-multiply-adds where `flops = Σ_i Σ_k
    /// nnz(B, row k) · [A[i,k] stored]` — the standard sparse
    /// matmul flop count, dominated by the structural fill of the
    /// product matrix. Memory `O(other.cols())` for the workspace,
    /// allocated once and reused across rows.
    ///
    /// # Panics
    ///
    /// Panics on inner-dimension mismatch (`self.cols() != other.rows()`)
    /// or precision mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use sparb::CsrArb;
    /// // [[1, 2], [3, 4]] · [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    /// let a = CsrArb::from_coo_f64(2, 2, 64, &[
    ///     (0, 0, 1.0, 0.0), (0, 1, 2.0, 0.0),
    ///     (1, 0, 3.0, 0.0), (1, 1, 4.0, 0.0),
    /// ]);
    /// let b = CsrArb::from_coo_f64(2, 2, 64, &[
    ///     (0, 0, 5.0, 0.0), (0, 1, 6.0, 0.0),
    ///     (1, 0, 7.0, 0.0), (1, 1, 8.0, 0.0),
    /// ]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.get(0, 0).unwrap().midpoint(), 19.0);
    /// assert_eq!(c.get(1, 1).unwrap().midpoint(), 50.0);
    /// ```
    pub fn matmul(&self, other: &CsrArb) -> CsrArb {
        assert_eq!(
            self.cols, other.rows,
            "CsrArb::matmul: inner-dimension mismatch — self is {}×{}, other is {}×{}",
            self.rows, self.cols, other.rows, other.cols
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrArb::matmul: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        let prec = self.prec;
        let m = self.rows;
        let n = other.cols;

        // Gustavson row-by-row. Workspace holds a sparse view of
        // the current row of C: `present[j]` flags which columns
        // have been touched in this row; `touched` lists them in
        // arrival order; `workspace[j]` is the accumulator slot.
        // All three are allocated once and reused across rows.
        let mut workspace: Vec<Arb> = (0..n).map(|_| Arb::zero()).collect();
        let mut touched: Vec<usize> = Vec::new();
        let mut present: Vec<bool> = vec![false; n];

        let mut indptr: Vec<usize> = Vec::with_capacity(m + 1);
        indptr.push(0);
        let mut indices: Vec<usize> = Vec::new();
        let mut data: Vec<Arb> = Vec::new();

        for i in 0..m {
            for (k, a_ik) in self.iter_row(i) {
                for (j, b_kj) in other.iter_row(k) {
                    if !present[j] {
                        present[j] = true;
                        touched.push(j);
                    }
                    workspace[j].addmul_assign(a_ik, b_kj, prec);
                }
            }

            // Emit row i in column-ascending order, drop exact zeros,
            // reset workspace slots.
            touched.sort();
            for &j in &touched {
                let val = std::mem::take(&mut workspace[j]);
                if !val.is_exact_zero() {
                    indices.push(j);
                    data.push(val);
                }
                present[j] = false;
            }
            touched.clear();
            indptr.push(indices.len());
        }

        CsrArb {
            rows: m,
            cols: n,
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// Frobenius (entrywise 2-) norm of the matrix — `√(Σ |A_ij|²)`
    /// as a rigorous `Arb` ball.
    ///
    /// Each entry's squared magnitude is computed via FLINT
    /// `arb_addmul`, summed into an accumulator at this matrix's
    /// precision, and the result is taken `√` at the same prec.
    /// Structural zeros contribute nothing.
    ///
    /// # Cost
    ///
    /// `O(nnz)` ball multiply-adds plus a single `arb_sqrt`. The
    /// final square root's precision matches the matrix's working
    /// precision.
    pub fn frobenius_norm(&self) -> Arb {
        let prec = self.prec;
        let mut sum_sq = Arb::zero();
        for v in &self.data {
            // arb_addmul(sum_sq, v, v, prec) computes the rigorous
            // ball-square: midpoint m², radius widened by 2|m|·r +
            // r² + rounding error. Correct enclosure of v² for any
            // real ball v.
            sum_sq.addmul_assign(v, v, prec);
        }
        sum_sq.sqrt(prec)
    }

    /// Operator infinity-norm `‖A‖_∞ = max_i Σ_j |A_ij|` as a
    /// rigorous `Arb` ball.
    ///
    /// For each row, sums `|entry|` balls into a per-row total;
    /// takes the rigorous max across rows. The "max of balls" step
    /// is implemented as `[max_i (mid_i - rad_i), max_i (mid_i + rad_i)]`,
    /// which contains the true max over the row sums: every row
    /// sum lies in its own ball, so the true max is at least the
    /// largest lower bound and at most the largest upper bound.
    ///
    /// # Cost
    ///
    /// `O(nnz)` ball adds plus one f64 max-reduction across `rows`.
    pub fn linf_norm(&self) -> Arb {
        let prec = self.prec;
        // |·| is non-negative, so row sums are >= 0; initialise the
        // lower / upper-bound accumulators at 0 to handle the
        // empty-matrix and all-zero-row cases without a special path.
        let mut max_upper = 0.0_f64;
        let mut max_lower = 0.0_f64;
        for r in 0..self.rows {
            let mut row_sum = Arb::zero();
            for (_, v) in self.iter_row(r) {
                row_sum.add_assign(&v.abs(), prec);
            }
            let (m, rad) = row_sum.to_pair(prec);
            max_upper = max_upper.max(m + rad);
            max_lower = max_lower.max(m - rad);
        }
        let mid = (max_upper + max_lower) * 0.5;
        // Guard against f64 wobble that could make the half-range
        // microscopically negative on a perfectly tight result.
        let rad = ((max_upper - max_lower) * 0.5).max(0.0);
        Arb::from_mid_rad(mid, rad)
    }

    /// Transpose: swap rows and columns.
    ///
    /// The result is `cols × rows` at the same precision and stores
    /// entry `(c, r)` for every stored entry `(r, c)` of `self`.
    /// Entries are **cloned exactly** — `arb_set` copies a ball
    /// without precision rounding, so the transposed entries have
    /// the same midpoints and radii as the originals (no widening).
    ///
    /// # Cost
    ///
    /// `O(rows + nnz)` — one pass to bucket entries by column of
    /// `self` (= row of the transpose), one pass to emit.
    pub fn transpose(&self) -> CsrArb {
        let m = self.rows;
        let n = self.cols;
        let mut tr_rows: Vec<Vec<(usize, Arb)>> = (0..n).map(|_| Vec::new()).collect();
        for r in 0..m {
            for (c, v) in self.iter_row(r) {
                tr_rows[c].push((r, v.clone()));
            }
        }
        CsrArb::from_sorted_rows(n, m, self.prec, tr_rows)
    }

    /// Materialise to a sorted-by-`(row, col)` list of
    /// `(row, col, midpoint, radius)` `f64` quadruples.
    ///
    /// The midpoint is `arf → f64` round-to-nearest; the radius
    /// is a conservative upper bound at this matrix's working
    /// precision (see [`Arb::radius`] for the readback caveats —
    /// large internal radii overflow to `f64::INFINITY`; very
    /// small ones may underflow to `0.0` or to f64 ulp of the
    /// midpoint).
    ///
    /// Round-trip with [`Self::from_coo_f64`] is *lossy* in general
    /// (a high-precision arb that doesn't fit in f64 mantissa is
    /// truncated on materialisation); it is *exact* for any matrix
    /// whose entries are exactly representable in f64 with
    /// f64-representable radii.
    pub fn to_coo_f64(&self) -> Vec<(usize, usize, f64, f64)> {
        let mut out = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                let (mid, rad) = v.to_pair(self.prec);
                out.push((r, c, mid, rad));
            }
        }
        out
    }

    /// Densify into a row-major `Vec<Vec<(midpoint, radius)>>`.
    ///
    /// Structural zeros materialise as the exact zero ball
    /// `(0.0, 0.0)`. Same readback caveats as [`Self::to_coo_f64`]
    /// (the radius is at the precision of this matrix; very large
    /// radii overflow to infinity, very tight ones may underflow).
    /// `O(rows · cols)` memory and time — only viable on small
    /// matrices, e.g. for parity tests against dense references.
    pub fn to_dense_f64(&self) -> Vec<Vec<(f64, f64)>> {
        let mut out: Vec<Vec<(f64, f64)>> = (0..self.rows)
            .map(|_| vec![(0.0, 0.0); self.cols])
            .collect();
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in self.iter_row(r) {
                row[c] = v.to_pair(self.prec);
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

    /// `true` if the matrix equals its transpose **as balls** —
    /// `A[i,j]` and `A[j,i]` must be the identical ball (same
    /// midpoint and radius), not merely overlapping enclosures.
    /// Requires a square matrix; non-square returns `false`.
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                match self.get(c, r) {
                    Some(w) if w.same_ball(v) => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Sum of the diagonal entries — `Σᵢ A[i, i]` — as a single
    /// `Arb` ball.
    ///
    /// The accumulator runs at this matrix's working precision;
    /// the output ball encloses the true trace, with radius the
    /// sum of the diagonal entries' radii plus the per-add
    /// rounding error (`min(rows, cols) · |trace| · 2⁻ᵖʳᵉᶜ`).
    /// Rectangular matrices: trace runs to `min(rows, cols)`.
    pub fn trace(&self) -> Arb {
        let mut acc = Arb::zero();
        let n = self.rows.min(self.cols);
        for i in 0..n {
            if let Some(v) = self.get(i, i) {
                acc.add_assign(v, self.prec);
            }
        }
        acc
    }
}

impl SparseRingStorage for CsrArb {
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

    /// `(row, col, mid, rad)` quad with radius 0 — an exact entry.
    fn e(r: usize, c: usize, v: f64) -> (usize, usize, f64, f64) {
        (r, c, v, 0.0)
    }

    #[test]
    fn zeros_has_no_entries() {
        let a = CsrArb::zeros(3, 4, 53);
        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 4);
        assert_eq!(a.nnz(), 0);
        assert!(a.is_zero());
        assert_eq!(a.density(), 0.0);
        assert!(!a.is_square());
        assert_eq!(a.prec(), 53);
    }

    #[test]
    #[should_panic(expected = "prec must be >= 2")]
    fn zeros_rejects_low_prec() {
        let _ = CsrArb::zeros(2, 2, 1);
    }

    #[test]
    fn from_coo_orders_indices_within_row() {
        let a = CsrArb::from_coo_f64(
            2,
            3,
            53,
            &[e(1, 2, 7.0), e(0, 2, 4.0), e(0, 0, 1.0), e(1, 0, -3.0)],
        );
        assert_eq!(a.nnz(), 4);
        assert_eq!(a.indptr, vec![0, 2, 4]);
        assert_eq!(a.indices, vec![0, 2, 0, 2]);
        assert_eq!(a.get(0, 0).unwrap().midpoint(), 1.0);
        assert_eq!(a.get(0, 2).unwrap().midpoint(), 4.0);
        assert_eq!(a.get(1, 0).unwrap().midpoint(), -3.0);
        assert_eq!(a.get(1, 2).unwrap().midpoint(), 7.0);
        assert!(a.get(0, 1).is_none()); // structural zero
        assert!(a.get(1, 1).is_none());
    }

    #[test]
    fn duplicate_entries_are_summed() {
        let a = CsrArb::from_coo_f64(1, 2, 53, &[e(0, 0, 10.0), e(0, 0, 5.0), e(0, 1, 1.0)]);
        assert_eq!(a.get(0, 0).unwrap().midpoint(), 15.0);
        assert_eq!(a.get(0, 1).unwrap().midpoint(), 1.0);
        assert_eq!(a.nnz(), 2);
    }

    #[test]
    fn duplicates_cancelling_to_exact_zero_are_dropped() {
        let a = CsrArb::from_coo_f64(1, 2, 53, &[e(0, 0, 4.0), e(0, 0, -4.0), e(0, 1, 9.0)]);
        assert_eq!(a.nnz(), 1);
        assert!(a.get(0, 0).is_none());
        assert_eq!(a.get(0, 1).unwrap().midpoint(), 9.0);
    }

    #[test]
    fn ball_containing_zero_is_kept() {
        // [0 ± 0.1] contains zero but is NOT the exact zero ball —
        // dropping it would discard a rigorous enclosure.
        let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 0.0, 0.1)]);
        assert_eq!(a.nnz(), 1);
        let cell = a.get(0, 0).unwrap();
        assert_eq!(cell.midpoint(), 0.0);
        assert!(cell.radius(53) >= 0.1);
    }

    #[test]
    fn iter_row_yields_sorted_entries() {
        let a = CsrArb::from_coo_f64(2, 3, 53, &[e(0, 2, 5.0), e(0, 0, 1.0), e(0, 1, 3.0)]);
        let row0: Vec<_> = a.iter_row(0).map(|(c, v)| (c, v.midpoint())).collect();
        assert_eq!(row0, vec![(0, 1.0), (1, 3.0), (2, 5.0)]);
        assert_eq!(a.iter_row(1).count(), 0);
    }

    #[test]
    fn to_coo_round_trips() {
        let original = CsrArb::from_coo_f64(3, 3, 53, &[e(0, 1, 2.0), e(2, 0, 7.0), e(1, 2, -1.0)]);
        let triples = original.to_coo_f64();
        let rebuilt = CsrArb::from_coo_f64(3, 3, 53, &triples);
        assert_eq!(rebuilt.indptr, original.indptr);
        assert_eq!(rebuilt.indices, original.indices);
        assert_eq!(rebuilt.to_dense_f64(), original.to_dense_f64());
    }

    #[test]
    fn to_dense_matches_inputs() {
        let a = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 0, 1.0), e(0, 1, 2.0), e(1, 1, -3.0)]);
        let d = a.to_dense_f64();
        assert_eq!(d[0][0], (1.0, 0.0));
        assert_eq!(d[0][1], (2.0, 0.0));
        assert_eq!(d[1][0], (0.0, 0.0)); // structural zero
        assert_eq!(d[1][1], (-3.0, 0.0));
    }

    #[test]
    fn density_is_nnz_over_total_cells() {
        let a = CsrArb::from_coo_f64(2, 4, 53, &[e(0, 0, 1.0), e(0, 1, 1.0)]);
        assert_eq!(a.density(), 2.0 / 8.0);
        assert_eq!(CsrArb::zeros(0, 5, 53).density(), 0.0);
    }

    #[test]
    fn transpose_swaps_indices() {
        // [[1, 0, 2], [0, 3, 0]] → [[1, 0], [0, 3], [2, 0]].
        let a = CsrArb::from_coo_f64(2, 3, 53, &[e(0, 0, 1.0), e(0, 2, 2.0), e(1, 1, 3.0)]);
        let at = a.transpose();
        assert_eq!(at.rows(), 3);
        assert_eq!(at.cols(), 2);
        assert_eq!(at.get(0, 0).unwrap().midpoint(), 1.0);
        assert_eq!(at.get(1, 1).unwrap().midpoint(), 3.0);
        assert_eq!(at.get(2, 0).unwrap().midpoint(), 2.0);
        // Round-trip.
        assert_eq!(at.transpose().to_dense_f64(), a.to_dense_f64());
    }

    #[test]
    fn transpose_empty_and_zero() {
        assert_eq!(CsrArb::zeros(0, 0, 53).transpose().rows(), 0);
        let z = CsrArb::zeros(3, 4, 53).transpose();
        assert_eq!((z.rows(), z.cols(), z.nnz()), (4, 3, 0));
    }

    #[test]
    fn is_diagonal_predicate() {
        let diag = CsrArb::from_coo_f64(3, 3, 53, &[e(0, 0, 1.0), e(1, 1, 2.0), e(2, 2, 3.0)]);
        assert!(diag.is_diagonal());
        let off = CsrArb::from_coo_f64(3, 3, 53, &[e(0, 0, 1.0), e(1, 2, 5.0)]);
        assert!(!off.is_diagonal());
        assert!(CsrArb::zeros(2, 2, 53).is_diagonal()); // vacuously
    }

    #[test]
    fn is_symmetric_predicate() {
        let sym = CsrArb::from_coo_f64(
            3,
            3,
            53,
            &[
                e(0, 1, 5.0),
                e(1, 0, 5.0),
                e(0, 2, 7.0),
                e(2, 0, 7.0),
                e(1, 1, 9.0),
            ],
        );
        assert!(sym.is_symmetric());
        let asym = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 1, 1.0), e(1, 0, 2.0)]);
        assert!(!asym.is_symmetric());
        // Non-square is never symmetric.
        assert!(!CsrArb::from_coo_f64(1, 2, 53, &[e(0, 1, 3.0)]).is_symmetric());
        // Same midpoint but different radius → not the same ball.
        let radius_mismatch = CsrArb::from_coo_f64(2, 2, 53, &[(0, 1, 1.0, 0.0), (1, 0, 1.0, 0.1)]);
        assert!(!radius_mismatch.is_symmetric());
    }

    #[test]
    fn trace_sums_diagonal() {
        let a = CsrArb::from_coo_f64(
            3,
            3,
            53,
            &[e(0, 0, 2.0), e(1, 1, 5.0), e(2, 2, -1.0), e(0, 1, 99.0)],
        );
        assert_eq!(a.trace().midpoint(), 6.0);
        // Rectangular: trace runs to min(rows, cols).
        let wide = CsrArb::from_coo_f64(2, 4, 53, &[e(0, 0, 3.0), e(1, 1, 4.0), e(1, 3, 100.0)]);
        assert_eq!(wide.trace().midpoint(), 7.0);
    }

    #[test]
    #[should_panic(expected = "row index 5 out of bounds")]
    fn from_coo_out_of_bounds_panics() {
        let _ = CsrArb::from_coo_f64(2, 2, 53, &[e(5, 0, 1.0)]);
    }

    // ── add / sub / scale ──────────────────────────────────────────

    #[test]
    fn add_merges_disjoint_patterns() {
        // A has cols {0, 2}, B has cols {1}; union is {0, 1, 2}.
        let a = CsrArb::from_coo_f64(1, 3, 53, &[e(0, 0, 1.0), e(0, 2, 3.0)]);
        let b = CsrArb::from_coo_f64(1, 3, 53, &[e(0, 1, 2.0)]);
        let c = a.add(&b);
        assert_eq!((c.rows(), c.cols(), c.nnz()), (1, 3, 3));
        assert_eq!(c.get(0, 0).unwrap().midpoint(), 1.0);
        assert_eq!(c.get(0, 1).unwrap().midpoint(), 2.0);
        assert_eq!(c.get(0, 2).unwrap().midpoint(), 3.0);
    }

    #[test]
    fn add_sums_overlapping_entries() {
        let a = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 0, 2.0), e(1, 1, 5.0)]);
        let b = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 0, 3.0), e(1, 1, -1.0)]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0).unwrap().midpoint(), 5.0);
        assert_eq!(c.get(1, 1).unwrap().midpoint(), 4.0);
    }

    #[test]
    fn add_drops_entries_summing_to_exact_zero() {
        let a = CsrArb::from_coo_f64(1, 2, 53, &[e(0, 0, 5.0), e(0, 1, 7.0)]);
        let b = CsrArb::from_coo_f64(1, 2, 53, &[e(0, 0, -5.0)]);
        let c = a.add(&b);
        assert!(c.get(0, 0).is_none());
        assert_eq!(c.get(0, 1).unwrap().midpoint(), 7.0);
        assert_eq!(c.nnz(), 1);
    }

    #[test]
    fn add_keeps_entry_whose_sum_only_contains_zero() {
        // Mid 5 ± 0.1 plus mid -5 ± 0.1: midpoint 0, radius >= 0.2.
        // Not the exact zero ball — must be kept.
        let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 5.0, 0.1)]);
        let b = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, -5.0, 0.1)]);
        let c = a.add(&b);
        assert_eq!(c.nnz(), 1);
        let ball = c.get(0, 0).unwrap();
        assert_eq!(ball.midpoint(), 0.0);
        assert!(ball.radius(53) >= 0.2);
    }

    #[test]
    fn add_widens_radius_under_uncertainty() {
        let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 2.0, 0.1)]);
        let b = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 3.0, 0.05)]);
        let c = a.add(&b);
        let ball = c.get(0, 0).unwrap();
        assert_eq!(ball.midpoint(), 5.0);
        assert!(ball.radius(53) >= 0.15);
    }

    #[test]
    fn sub_a_minus_a_is_all_exact_zero() {
        let a = CsrArb::from_coo_f64(2, 2, 64, &[e(0, 0, 1.0), e(0, 1, 2.0), e(1, 1, 3.0)]);
        let z = a.sub(&a);
        assert_eq!(z.nnz(), 0);
        assert!(z.is_zero());
    }

    #[test]
    fn sub_negates_entries_only_in_other() {
        let a = CsrArb::from_coo_f64(1, 3, 53, &[e(0, 0, 1.0)]);
        let b = CsrArb::from_coo_f64(1, 3, 53, &[e(0, 1, 2.0), e(0, 2, -4.0)]);
        let c = a.sub(&b);
        assert_eq!(c.get(0, 0).unwrap().midpoint(), 1.0);
        assert_eq!(c.get(0, 1).unwrap().midpoint(), -2.0);
        assert_eq!(c.get(0, 2).unwrap().midpoint(), 4.0);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_shape_mismatch_panics() {
        let a = CsrArb::zeros(2, 3, 53);
        let b = CsrArb::zeros(2, 4, 53);
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "precision mismatch")]
    fn add_prec_mismatch_panics() {
        let a = CsrArb::zeros(2, 2, 53);
        let b = CsrArb::zeros(2, 2, 64);
        let _ = a.add(&b);
    }

    #[test]
    fn scale_by_two_doubles_each_entry() {
        let a = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 0, 3.0), e(1, 1, -7.0)]);
        let two = Arb::from_f64(2.0);
        let c = a.scale(&two);
        assert_eq!((c.rows(), c.cols(), c.nnz()), (2, 2, 2));
        assert_eq!(c.get(0, 0).unwrap().midpoint(), 6.0);
        assert_eq!(c.get(1, 1).unwrap().midpoint(), -14.0);
    }

    #[test]
    fn scale_by_exact_zero_is_zero_matrix() {
        let a = CsrArb::from_coo_f64(3, 3, 53, &[e(0, 0, 1.0), e(2, 2, 5.0)]);
        let z = a.scale(&Arb::zero());
        assert!(z.is_zero());
        assert_eq!((z.rows(), z.cols(), z.nnz()), (3, 3, 0));
    }

    #[test]
    fn scale_by_ball_widens_entries() {
        // Scaling exact entries by an uncertain alpha must widen
        // every entry's radius to reflect alpha's uncertainty.
        let a = CsrArb::from_coo_f64(1, 1, 53, &[e(0, 0, 4.0)]);
        let alpha = Arb::from_mid_rad(2.0, 0.1);
        let c = a.scale(&alpha);
        let ball = c.get(0, 0).unwrap();
        assert_eq!(ball.midpoint(), 8.0);
        assert!(ball.radius(53) >= 0.4); // |entry| * rad(alpha) = 4 * 0.1
    }

    #[test]
    fn scale_preserves_pattern_for_nonzero_alpha() {
        let a = CsrArb::from_coo_f64(2, 3, 53, &[e(0, 1, 1.0), e(1, 0, 2.0)]);
        let c = a.scale(&Arb::from_f64(-3.0));
        assert_eq!(c.nnz(), a.nnz());
        assert_eq!(c.get(0, 1).unwrap().midpoint(), -3.0);
        assert_eq!(c.get(1, 0).unwrap().midpoint(), -6.0);
    }

    // ── matmul (spgemm) ─────────────────────────────────────────────

    #[test]
    fn matmul_2x2_classic() {
        // [[1, 2], [3, 4]] · [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = CsrArb::from_coo_f64(
            2,
            2,
            64,
            &[e(0, 0, 1.0), e(0, 1, 2.0), e(1, 0, 3.0), e(1, 1, 4.0)],
        );
        let b = CsrArb::from_coo_f64(
            2,
            2,
            64,
            &[e(0, 0, 5.0), e(0, 1, 6.0), e(1, 0, 7.0), e(1, 1, 8.0)],
        );
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 0).unwrap().midpoint(), 19.0);
        assert_eq!(c.get(0, 1).unwrap().midpoint(), 22.0);
        assert_eq!(c.get(1, 0).unwrap().midpoint(), 43.0);
        assert_eq!(c.get(1, 1).unwrap().midpoint(), 50.0);
    }

    #[test]
    fn matmul_times_identity_is_identity_on_midpoints() {
        let a = CsrArb::from_coo_f64(2, 3, 64, &[e(0, 0, 1.0), e(0, 2, 2.0), e(1, 1, 3.0)]);
        // 3×3 identity
        let i = CsrArb::from_coo_f64(3, 3, 64, &[e(0, 0, 1.0), e(1, 1, 1.0), e(2, 2, 1.0)]);
        let c = a.matmul(&i);
        assert_eq!(c.to_dense_f64(), a.to_dense_f64());
    }

    #[test]
    fn matmul_with_zero_matrix_is_zero() {
        let a = CsrArb::from_coo_f64(2, 3, 64, &[e(0, 0, 1.0), e(1, 2, 5.0)]);
        let z = CsrArb::zeros(3, 4, 64);
        let c = a.matmul(&z);
        assert!(c.is_zero());
        assert_eq!((c.rows(), c.cols()), (2, 4));
    }

    #[test]
    fn matmul_zero_inner_dim_yields_zero_result() {
        // 3×0 · 0×4 — inner dim is 0, no flops happen.
        let a = CsrArb::zeros(3, 0, 64);
        let b = CsrArb::zeros(0, 4, 64);
        let c = a.matmul(&b);
        assert_eq!((c.rows(), c.cols(), c.nnz()), (3, 4, 0));
    }

    #[test]
    fn matmul_rectangular_shapes() {
        // (2×3) · (3×4) = (2×4)
        let a = CsrArb::from_coo_f64(2, 3, 64, &[e(0, 1, 2.0), e(1, 0, 1.0), e(1, 2, 3.0)]);
        let b = CsrArb::from_coo_f64(3, 4, 64, &[e(0, 0, 1.0), e(1, 1, 4.0), e(2, 3, 5.0)]);
        let c = a.matmul(&b);
        assert_eq!((c.rows(), c.cols()), (2, 4));
        // Row 0: A[0,1]·B[1,1] = 2·4 = 8 at (0,1).
        assert_eq!(c.get(0, 1).unwrap().midpoint(), 8.0);
        // Row 1: A[1,0]·B[0,0] = 1 at (1,0); A[1,2]·B[2,3] = 15 at (1,3).
        assert_eq!(c.get(1, 0).unwrap().midpoint(), 1.0);
        assert_eq!(c.get(1, 3).unwrap().midpoint(), 15.0);
    }

    #[test]
    fn matmul_drops_entries_summing_to_exact_zero() {
        // A · B where two contributions cancel exactly to zero.
        // A = [[1, -1]], B = [[5, 0], [5, 0]].
        //   C[0, 0] = 1·5 + (-1)·5 = 0 (exact) → dropped.
        let a = CsrArb::from_coo_f64(1, 2, 64, &[e(0, 0, 1.0), e(0, 1, -1.0)]);
        let b = CsrArb::from_coo_f64(2, 2, 64, &[e(0, 0, 5.0), e(1, 0, 5.0)]);
        let c = a.matmul(&b);
        assert!(c.get(0, 0).is_none());
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn matmul_radius_propagates() {
        // Entry-radius times exact entries: |a_mid| * b_rad term.
        // [[2, 0]] · [[(3 ± 0.1)], [99]] = [[6 ± 0.2]] — only the
        // first row of B contributes since A[0, 1] is structural zero.
        let a = CsrArb::from_coo_f64(1, 2, 64, &[(0, 0, 2.0, 0.0)]);
        let b = CsrArb::from_coo_f64(2, 1, 64, &[(0, 0, 3.0, 0.1), (1, 0, 99.0, 0.0)]);
        let c = a.matmul(&b);
        let ball = c.get(0, 0).unwrap();
        assert_eq!(ball.midpoint(), 6.0);
        assert!(ball.radius(64) >= 0.2);
    }

    #[test]
    fn matmul_emits_row_in_column_ascending_order() {
        // Test that the touched-list sort actually emits in order
        // even when rows of A reach B's columns out of order.
        let a = CsrArb::from_coo_f64(
            1,
            3,
            64,
            // Walk A's row 0 in column order: hits B's rows 0, 1, 2.
            &[e(0, 0, 1.0), e(0, 1, 1.0), e(0, 2, 1.0)],
        );
        let b = CsrArb::from_coo_f64(
            3,
            4,
            64,
            // Each row of B writes to a non-monotone column of C.
            &[e(0, 3, 1.0), e(1, 0, 1.0), e(2, 2, 1.0)],
        );
        let c = a.matmul(&b);
        // C[0,:] should have entries at cols 0, 2, 3 in that order.
        let cols: Vec<usize> = c.iter_row(0).map(|(j, _)| j).collect();
        assert_eq!(cols, vec![0, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "inner-dimension mismatch")]
    fn matmul_inner_dim_mismatch_panics() {
        let a = CsrArb::zeros(2, 3, 64);
        let b = CsrArb::zeros(4, 5, 64);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "precision mismatch")]
    fn matmul_prec_mismatch_panics() {
        let a = CsrArb::zeros(2, 2, 64);
        let b = CsrArb::zeros(2, 2, 53);
        let _ = a.matmul(&b);
    }

    // ── norms ──────────────────────────────────────────────────────

    #[test]
    fn frobenius_norm_of_3_4_5_triangle() {
        // 3² + 4² = 25; √25 = 5.
        let a = CsrArb::from_coo_f64(1, 2, 64, &[e(0, 0, 3.0), e(0, 1, 4.0)]);
        let n = a.frobenius_norm();
        assert_eq!(n.midpoint(), 5.0);
        assert_eq!(n.radius(64), 0.0);
    }

    #[test]
    fn frobenius_norm_zero_matrix_is_zero() {
        let a = CsrArb::zeros(5, 5, 64);
        assert!(a.frobenius_norm().is_exact_zero());
    }

    #[test]
    fn frobenius_norm_widens_under_uncertainty() {
        // Single entry [2 ± 0.1]: |v|² = [3.99, 4.41], √ ≈ [1.997, 2.10].
        // Midpoint near 2, radius captures both the |v|² widening
        // and the sqrt-rounding contribution.
        let a = CsrArb::from_coo_f64(1, 1, 80, &[(0, 0, 2.0, 0.1)]);
        let n = a.frobenius_norm();
        let (mid, rad) = n.to_pair(80);
        // Rigorous bound: ball must enclose √4 = 2.
        assert!(mid - rad <= 2.0 && 2.0 <= mid + rad);
        // Tight bound: enclose √(2.1²) = 2.1 too.
        assert!(mid - rad <= 2.1 && 2.1 <= mid + rad);
    }

    #[test]
    fn frobenius_norm_sign_independence() {
        // |·|² is sign-blind; the Frobenius norm of A and -A should
        // be equal on midpoints (and identical as balls).
        let a = CsrArb::from_coo_f64(2, 2, 64, &[e(0, 0, 3.0), e(1, 1, 4.0)]);
        let neg_a = a.scale(&Arb::from_f64(-1.0));
        assert_eq!(
            a.frobenius_norm().midpoint(),
            neg_a.frobenius_norm().midpoint()
        );
    }

    #[test]
    fn linf_norm_max_row_sum() {
        // Rows: |1| + |-2| = 3; |3| + |4| = 7; |0| = 0. Max = 7.
        let a = CsrArb::from_coo_f64(
            3,
            2,
            64,
            &[e(0, 0, 1.0), e(0, 1, -2.0), e(1, 0, 3.0), e(1, 1, 4.0)],
        );
        let n = a.linf_norm();
        assert_eq!(n.midpoint(), 7.0);
        assert_eq!(n.radius(64), 0.0);
    }

    #[test]
    fn linf_norm_zero_matrix_is_zero() {
        let a = CsrArb::zeros(5, 5, 64);
        let (mid, rad) = a.linf_norm().to_pair(64);
        assert_eq!(mid, 0.0);
        assert_eq!(rad, 0.0);
    }

    #[test]
    fn linf_norm_single_dense_row() {
        // 1×n row with values 1..n: linf = sum = n(n+1)/2 for n=10
        // is 55. Exact on small integers at prec=64.
        let triples: Vec<(usize, usize, f64, f64)> =
            (0..10).map(|c| (0, c, (c + 1) as f64, 0.0)).collect();
        let a = CsrArb::from_coo_f64(1, 10, 64, &triples);
        assert_eq!(a.linf_norm().midpoint(), 55.0);
    }

    #[test]
    fn linf_norm_widens_under_uncertainty() {
        // Row 0: [1 ± 0.1, 2 ± 0.05]; row sum = [3 ± 0.15].
        // Row 1: [4 ± 0]; row sum = [4 ± 0].
        // Max ball must contain 4 (the true max) AND have radius
        // capturing the row-1 row being the maximizer.
        let a = CsrArb::from_coo_f64(
            2,
            2,
            64,
            &[(0, 0, 1.0, 0.1), (0, 1, 2.0, 0.05), (1, 0, 4.0, 0.0)],
        );
        let (mid, rad) = a.linf_norm().to_pair(64);
        assert!(mid - rad <= 4.0 && 4.0 <= mid + rad);
    }

    #[test]
    fn linf_norm_dominated_by_negative_entries() {
        // |·| means sign doesn't matter; row with all-negative
        // entries gets the same linf as the all-positive version.
        let pos = CsrArb::from_coo_f64(1, 3, 64, &[e(0, 0, 3.0), e(0, 1, 4.0), e(0, 2, 5.0)]);
        let neg = CsrArb::from_coo_f64(1, 3, 64, &[e(0, 0, -3.0), e(0, 1, -4.0), e(0, 2, -5.0)]);
        assert_eq!(pos.linf_norm().midpoint(), neg.linf_norm().midpoint());
    }
}
