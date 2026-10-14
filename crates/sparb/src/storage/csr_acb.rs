// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrix over complex balls in compressed-sparse-row layout.
//!
//! The complex sister of [`super::csr_arb::CsrArb`]. Entries are
//! [`Acb`] complex balls — each a rigorous enclosure of a complex
//! value as an axis-aligned rectangle
//! `[re_mid ± re_rad] + i [im_mid ± im_rad]`. The storage
//! invariant matches the real cell: strictly increasing column
//! indices per row, no *exact*-zero entries (but balls whose real
//! or imaginary part merely contains zero are kept).
//!
//! Every method that produces new balls runs at the matrix's
//! working precision (`prec`, in bits); operations that take an
//! `Acb` scalar argument inherit that precision. Where the real-
//! ball cell would return an `Arb`, the complex cell returns a
//! component-aware result type — notably [`frobenius_norm`] returns
//! an `Arb` because the squared modulus is real.
//!
//! [`Acb`]: crate::Acb
//! [`frobenius_norm`]: CsrAcb::frobenius_norm

use crate::ball::Arb;
use crate::Acb;

use super::coo::{build_csr, Triple};
use super::SparseRingStorage;

/// Sparse `rows × cols` matrix over complex balls (`Acb`) in CSR
/// layout, at a fixed working precision.
#[derive(Clone, Debug)]
pub struct CsrAcb {
    rows: usize,
    cols: usize,
    prec: i64,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<Acb>,
}

impl CsrAcb {
    /// The all-zero `rows × cols` matrix at `prec` bits (no stored
    /// entries).
    ///
    /// # Panics
    ///
    /// Panics if `prec < 2`.
    pub fn zeros(rows: usize, cols: usize, prec: i64) -> Self {
        assert!(prec >= 2, "CsrAcb: prec must be >= 2 bits; got {prec}");
        Self {
            rows,
            cols,
            prec,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Build from `(row, col, re_mid, re_rad, im_mid, im_rad)`
    /// `f64` sextuples.
    ///
    /// Duplicate `(row, col)` entries are summed in complex ball
    /// arithmetic at `prec`. Entries summing to the *exact* zero
    /// complex ball are dropped from storage; entries whose
    /// real or imaginary part merely contains zero are kept.
    /// Each radius must be finite and `>= 0`.
    ///
    /// # Panics
    ///
    /// Panics on `prec < 2`, on row / col out of bounds, or on a
    /// negative / non-finite radius.
    pub fn from_coo_f64(
        rows: usize,
        cols: usize,
        prec: i64,
        triples: &[(usize, usize, f64, f64, f64, f64)],
    ) -> Self {
        assert!(prec >= 2, "CsrAcb: prec must be >= 2 bits; got {prec}");
        let balls: Vec<Triple<Acb>> = triples
            .iter()
            .map(|&(r, c, re_mid, re_rad, im_mid, im_rad)| {
                (r, c, Acb::from_re_im_rad(re_mid, re_rad, im_mid, im_rad))
            })
            .collect();
        Self::from_coo_balls(rows, cols, prec, balls)
    }

    /// Build from owned `(row, col, Acb)` triples — the move-based
    /// ingest that avoids cloning each ball.
    pub fn from_coo_balls(rows: usize, cols: usize, prec: i64, triples: Vec<Triple<Acb>>) -> Self {
        assert!(prec >= 2, "CsrAcb: prec must be >= 2 bits; got {prec}");
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

    fn from_sorted_rows(
        rows: usize,
        cols: usize,
        prec: i64,
        sorted_rows: Vec<Vec<(usize, Acb)>>,
    ) -> Self {
        debug_assert_eq!(sorted_rows.len(), rows);
        let nnz_estimate: usize = sorted_rows.iter().map(|r| r.len()).sum();
        let mut indptr = vec![0usize; rows + 1];
        let mut indices: Vec<usize> = Vec::with_capacity(nnz_estimate);
        let mut data: Vec<Acb> = Vec::with_capacity(nnz_estimate);
        for (r, row) in sorted_rows.into_iter().enumerate() {
            for (c, v) in row {
                debug_assert!(c < cols, "col index {c} out of bounds (cols={cols})");
                debug_assert!(
                    !v.is_exact_zero(),
                    "exact-zero ball passed to from_sorted_rows"
                );
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

    /// The working precision shared across this matrix.
    pub fn prec(&self) -> i64 {
        self.prec
    }

    /// Look up entry `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> Option<&Acb> {
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
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, &Acb)> + '_ {
        assert!(row < self.rows, "row index {row} out of bounds");
        let s = self.indptr[row];
        let e = self.indptr[row + 1];
        self.indices[s..e]
            .iter()
            .copied()
            .zip(self.data[s..e].iter())
    }

    /// Sparse matrix-vector product `y = A · x` in complex ball
    /// arithmetic. `x.len()` must equal `self.cols()`.
    pub fn spmv(&self, x: &[Acb]) -> Vec<Acb> {
        assert_eq!(
            x.len(),
            self.cols,
            "CsrAcb::spmv: x has length {} but A has {} columns",
            x.len(),
            self.cols
        );
        let prec = self.prec;
        let mut y: Vec<Acb> = (0..self.rows).map(|_| Acb::zero()).collect();
        for (r, acc) in y.iter_mut().enumerate() {
            for (c, v) in self.iter_row(r) {
                acc.addmul_assign(v, &x[c], prec);
            }
        }
        y
    }

    /// Element-wise sum `self + other`.
    ///
    /// # Panics
    ///
    /// Panics on shape or precision mismatch.
    pub fn add(&self, other: &CsrAcb) -> CsrAcb {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "CsrAcb::add: shape mismatch"
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrAcb::add: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        self.linear_combine(other, false)
    }

    /// Element-wise difference `self - other`.
    pub fn sub(&self, other: &CsrAcb) -> CsrAcb {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "CsrAcb::sub: shape mismatch"
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrAcb::sub: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        self.linear_combine(other, true)
    }

    fn linear_combine(&self, other: &CsrAcb, subtract: bool) -> CsrAcb {
        use std::cmp::Ordering::*;
        let prec = self.prec;
        let nnz_upper = self.nnz() + other.nnz();
        let mut indptr = Vec::with_capacity(self.rows + 1);
        let mut indices = Vec::with_capacity(nnz_upper);
        let mut data: Vec<Acb> = Vec::with_capacity(nnz_upper);
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
        CsrAcb {
            rows: self.rows,
            cols: self.cols,
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// Multiply every entry by the complex ball `alpha`. If `alpha`
    /// is the exact zero complex ball, returns the zero matrix.
    pub fn scale(&self, alpha: &Acb) -> CsrAcb {
        let prec = self.prec;
        if alpha.is_exact_zero() {
            return CsrAcb::zeros(self.rows, self.cols, prec);
        }
        let data: Vec<Acb> = self.data.iter().map(|v| v.mul(alpha, prec)).collect();
        CsrAcb {
            rows: self.rows,
            cols: self.cols,
            prec,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data,
        }
    }

    /// Sparse × sparse complex-ball matrix multiply via Gustavson's
    /// row-by-row algorithm with `acb_addmul` as the inner-loop
    /// primitive. Direct analogue of `CsrArb::matmul`.
    ///
    /// # Panics
    ///
    /// Panics on inner-dim or precision mismatch.
    pub fn matmul(&self, other: &CsrAcb) -> CsrAcb {
        assert_eq!(
            self.cols, other.rows,
            "CsrAcb::matmul: inner-dimension mismatch"
        );
        assert_eq!(
            self.prec, other.prec,
            "CsrAcb::matmul: precision mismatch {} vs {}",
            self.prec, other.prec
        );
        let prec = self.prec;
        let m = self.rows;
        let n = other.cols;

        let mut workspace: Vec<Acb> = (0..n).map(|_| Acb::zero()).collect();
        let mut touched: Vec<usize> = Vec::new();
        let mut present: Vec<bool> = vec![false; n];

        let mut indptr: Vec<usize> = Vec::with_capacity(m + 1);
        indptr.push(0);
        let mut indices: Vec<usize> = Vec::new();
        let mut data: Vec<Acb> = Vec::new();

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

        CsrAcb {
            rows: m,
            cols: n,
            prec,
            indptr,
            indices,
            data,
        }
    }

    /// Transpose: swap rows and columns. **Not** the Hermitian
    /// adjoint — entries are *not* conjugated.
    pub fn transpose(&self) -> CsrAcb {
        let m = self.rows;
        let n = self.cols;
        let mut tr_rows: Vec<Vec<(usize, Acb)>> = (0..n).map(|_| Vec::new()).collect();
        for r in 0..m {
            for (c, v) in self.iter_row(r) {
                tr_rows[c].push((r, v.clone()));
            }
        }
        CsrAcb::from_sorted_rows(n, m, self.prec, tr_rows)
    }

    /// Materialise to a sorted-by-`(row, col)` list of
    /// `(row, col, re_mid, re_rad, im_mid, im_rad)` `f64` sextuples.
    /// Same readback caveats as the real cell (very large radii
    /// overflow to `INFINITY`).
    pub fn to_coo_f64(&self) -> Vec<(usize, usize, f64, f64, f64, f64)> {
        let mut out = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            for (c, v) in self.iter_row(r) {
                let (re_m, re_r, im_m, im_r) = v.to_4tuple(self.prec);
                out.push((r, c, re_m, re_r, im_m, im_r));
            }
        }
        out
    }

    /// Densify into a row-major `Vec<Vec<(re_mid, re_rad, im_mid, im_rad)>>`.
    pub fn to_dense_f64(&self) -> Vec<Vec<(f64, f64, f64, f64)>> {
        let mut out: Vec<Vec<(f64, f64, f64, f64)>> = (0..self.rows)
            .map(|_| vec![(0.0, 0.0, 0.0, 0.0); self.cols])
            .collect();
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in self.iter_row(r) {
                row[c] = v.to_4tuple(self.prec);
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

    /// Sum of the diagonal entries — `Σᵢ A[i, i]` — as a single
    /// `Acb` ball at this matrix's working precision.
    pub fn trace(&self) -> Acb {
        let mut acc = Acb::zero();
        let n = self.rows.min(self.cols);
        for i in 0..n {
            if let Some(v) = self.get(i, i) {
                acc.add_assign(v, self.prec);
            }
        }
        acc
    }

    /// Frobenius norm `‖A‖_F = √(Σ |A_ij|²)`, returned as a real
    /// `Arb` ball.
    ///
    /// Each entry's squared modulus `|A_ij|²` is real (FLINT's
    /// `acb_abs` gives the modulus as an `Arb`; squaring is then
    /// `arb_addmul` of the modulus with itself). Summed across all
    /// stored entries and `√` taken at the matrix's precision.
    ///
    /// # Cost
    ///
    /// `O(nnz)` `acb_abs` calls plus an `arb_addmul`-per-nnz and
    /// one final `arb_sqrt`.
    pub fn frobenius_norm(&self) -> Arb {
        let prec = self.prec;
        let mut sum_sq = Arb::zero();
        for v in &self.data {
            let modulus = v.abs(prec); // |v| as an Arb
            sum_sq.addmul_assign(&modulus, &modulus, prec);
        }
        sum_sq.sqrt(prec)
    }
}

impl SparseRingStorage for CsrAcb {
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

    /// `(row, col, re_mid, im_mid)` with zero radii — exact complex
    /// entry, convenience for tests.
    fn e(r: usize, c: usize, re: f64, im: f64) -> (usize, usize, f64, f64, f64, f64) {
        (r, c, re, 0.0, im, 0.0)
    }

    #[test]
    fn zeros_has_no_entries() {
        let a = CsrAcb::zeros(3, 4, 64);
        assert_eq!((a.rows(), a.cols(), a.nnz()), (3, 4, 0));
        assert!(a.is_zero());
    }

    #[test]
    fn from_coo_orders_within_row() {
        let a = CsrAcb::from_coo_f64(
            2,
            3,
            64,
            &[e(1, 2, 7.0, 0.0), e(0, 0, 1.0, 2.0), e(1, 0, -3.0, 0.0)],
        );
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.indptr, vec![0, 1, 3]);
        assert_eq!(a.indices, vec![0, 0, 2]);
        let e_00 = a.get(0, 0).unwrap();
        assert_eq!(e_00.re_midpoint(), 1.0);
        assert_eq!(e_00.im_midpoint(), 2.0);
        let e_12 = a.get(1, 2).unwrap();
        assert_eq!(e_12.re_midpoint(), 7.0);
        assert_eq!(e_12.im_midpoint(), 0.0);
    }

    #[test]
    fn duplicates_summed() {
        let a = CsrAcb::from_coo_f64(1, 1, 64, &[e(0, 0, 1.0, 2.0), e(0, 0, 3.0, 4.0)]);
        assert_eq!(a.nnz(), 1);
        let v = a.get(0, 0).unwrap();
        assert_eq!(v.re_midpoint(), 4.0);
        assert_eq!(v.im_midpoint(), 6.0);
    }

    #[test]
    fn duplicates_cancelling_to_exact_zero_dropped() {
        let a = CsrAcb::from_coo_f64(
            1,
            2,
            64,
            &[e(0, 0, 1.0, 1.0), e(0, 0, -1.0, -1.0), e(0, 1, 2.0, 0.0)],
        );
        assert_eq!(a.nnz(), 1);
        assert!(a.get(0, 0).is_none());
        assert_eq!(a.get(0, 1).unwrap().re_midpoint(), 2.0);
    }

    #[test]
    fn spmv_complex() {
        // [[1+i, 2], [0, 3i]] · [1+0i, 1+0i] = [1+i + 2, 0 + 3i] = [3+i, 3i].
        let a = CsrAcb::from_coo_f64(
            2,
            2,
            64,
            &[e(0, 0, 1.0, 1.0), e(0, 1, 2.0, 0.0), e(1, 1, 0.0, 3.0)],
        );
        let x: Vec<Acb> = vec![Acb::from_re_im(1.0, 0.0), Acb::from_re_im(1.0, 0.0)];
        let y = a.spmv(&x);
        assert_eq!(y[0].re_midpoint(), 3.0);
        assert_eq!(y[0].im_midpoint(), 1.0);
        assert_eq!(y[1].re_midpoint(), 0.0);
        assert_eq!(y[1].im_midpoint(), 3.0);
    }

    #[test]
    fn add_with_overlap() {
        let a = CsrAcb::from_coo_f64(1, 2, 64, &[e(0, 0, 1.0, 2.0)]);
        let b = CsrAcb::from_coo_f64(1, 2, 64, &[e(0, 0, 3.0, 4.0), e(0, 1, 5.0, 6.0)]);
        let c = a.add(&b);
        let c00 = c.get(0, 0).unwrap();
        assert_eq!(c00.re_midpoint(), 4.0);
        assert_eq!(c00.im_midpoint(), 6.0);
        let c01 = c.get(0, 1).unwrap();
        assert_eq!(c01.re_midpoint(), 5.0);
        assert_eq!(c01.im_midpoint(), 6.0);
    }

    #[test]
    fn sub_self_is_zero() {
        let a = CsrAcb::from_coo_f64(2, 2, 64, &[e(0, 0, 1.0, 1.0), e(1, 1, 2.0, -3.0)]);
        let z = a.sub(&a);
        assert!(z.is_zero());
    }

    #[test]
    fn scale_by_i_rotates_each_entry() {
        // A · i: each entry (a + bi) becomes -b + ai.
        let a = CsrAcb::from_coo_f64(1, 2, 64, &[e(0, 0, 3.0, 4.0), e(0, 1, -1.0, 2.0)]);
        let i = Acb::from_re_im(0.0, 1.0);
        let c = a.scale(&i);
        let c00 = c.get(0, 0).unwrap();
        assert_eq!(c00.re_midpoint(), -4.0);
        assert_eq!(c00.im_midpoint(), 3.0);
        let c01 = c.get(0, 1).unwrap();
        assert_eq!(c01.re_midpoint(), -2.0);
        assert_eq!(c01.im_midpoint(), -1.0);
    }

    #[test]
    fn scale_by_zero_is_zero_matrix() {
        let a = CsrAcb::from_coo_f64(2, 2, 64, &[e(0, 0, 1.0, 2.0), e(1, 1, 3.0, 4.0)]);
        let z = a.scale(&Acb::zero());
        assert!(z.is_zero());
    }

    #[test]
    fn matmul_2x2_complex() {
        // [[i, 0], [0, i]] · [[i, 0], [0, i]] = [[-1, 0], [0, -1]].
        let i_id = CsrAcb::from_coo_f64(2, 2, 64, &[e(0, 0, 0.0, 1.0), e(1, 1, 0.0, 1.0)]);
        let c = i_id.matmul(&i_id);
        let c00 = c.get(0, 0).unwrap();
        assert_eq!(c00.re_midpoint(), -1.0);
        assert_eq!(c00.im_midpoint(), 0.0);
        let c11 = c.get(1, 1).unwrap();
        assert_eq!(c11.re_midpoint(), -1.0);
        assert_eq!(c11.im_midpoint(), 0.0);
        assert!(c.get(0, 1).is_none());
        assert!(c.get(1, 0).is_none());
    }

    #[test]
    fn matmul_times_identity() {
        let a = CsrAcb::from_coo_f64(
            2,
            3,
            64,
            &[e(0, 0, 1.0, 2.0), e(0, 2, 3.0, 4.0), e(1, 1, 5.0, 6.0)],
        );
        let id = CsrAcb::from_coo_f64(
            3,
            3,
            64,
            &[e(0, 0, 1.0, 0.0), e(1, 1, 1.0, 0.0), e(2, 2, 1.0, 0.0)],
        );
        let c = a.matmul(&id);
        assert_eq!(c.to_dense_f64(), a.to_dense_f64());
    }

    #[test]
    fn transpose_swaps_indices() {
        let a = CsrAcb::from_coo_f64(2, 3, 64, &[e(0, 0, 1.0, 1.0), e(1, 2, 2.0, 3.0)]);
        let at = a.transpose();
        assert_eq!((at.rows(), at.cols()), (3, 2));
        let v = at.get(0, 0).unwrap();
        assert_eq!(v.re_midpoint(), 1.0);
        assert_eq!(v.im_midpoint(), 1.0);
        let v = at.get(2, 1).unwrap();
        assert_eq!(v.re_midpoint(), 2.0);
        assert_eq!(v.im_midpoint(), 3.0);
        // Transpose is plain (not Hermitian) — imag parts not
        // conjugated.
        assert_eq!(at.transpose().to_dense_f64(), a.to_dense_f64());
    }

    #[test]
    fn trace_sums_diagonal_complex() {
        // diag(1+i, 2-i, 3): trace = 6 + 0i.
        let a = CsrAcb::from_coo_f64(
            3,
            3,
            64,
            &[e(0, 0, 1.0, 1.0), e(1, 1, 2.0, -1.0), e(2, 2, 3.0, 0.0)],
        );
        let t = a.trace();
        assert_eq!(t.re_midpoint(), 6.0);
        assert_eq!(t.im_midpoint(), 0.0);
    }

    #[test]
    fn frobenius_norm_of_3_plus_4i() {
        // Single entry 3 + 4i: |·|² = 25; √25 = 5.
        let a = CsrAcb::from_coo_f64(1, 1, 64, &[e(0, 0, 3.0, 4.0)]);
        let n = a.frobenius_norm();
        assert_eq!(n.midpoint(), 5.0);
        assert_eq!(n.radius(64), 0.0);
    }

    #[test]
    fn frobenius_norm_two_entries() {
        // (3+4i) and (5+12i): moduli 5 and 13; squared sum 25 + 169
        // = 194; √194 ≈ 13.928.
        let a = CsrAcb::from_coo_f64(1, 2, 64, &[e(0, 0, 3.0, 4.0), e(0, 1, 5.0, 12.0)]);
        let n = a.frobenius_norm();
        let expected = (25.0_f64 + 169.0).sqrt();
        let (mid, rad) = n.to_pair(64);
        assert!(
            (mid - expected).abs() <= rad + 1e-12,
            "frobenius midpoint {mid} vs expected {expected}"
        );
    }

    #[test]
    fn frobenius_norm_zero_matrix() {
        let a = CsrAcb::zeros(5, 5, 64);
        assert!(a.frobenius_norm().is_exact_zero());
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_shape_mismatch_panics() {
        let a = CsrAcb::zeros(2, 3, 64);
        let b = CsrAcb::zeros(2, 4, 64);
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "inner-dimension mismatch")]
    fn matmul_inner_dim_mismatch_panics() {
        let a = CsrAcb::zeros(2, 3, 64);
        let b = CsrAcb::zeros(4, 5, 64);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "precision mismatch")]
    fn add_prec_mismatch_panics() {
        let a = CsrAcb::zeros(2, 2, 64);
        let b = CsrAcb::zeros(2, 2, 53);
        let _ = a.add(&b);
    }
}
