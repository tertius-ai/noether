// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! PLUQ decomposition over F_2.
//!
//! Given `A: m × n`, computes `P, L, U, Q` and rank `r` such that
//! `P · L · U · Q = A`, where:
//!
//! * `P` is `m × m`, a row permutation matrix,
//! * `L` is `m × k` (`k = min(m, n)`), unit lower triangular in its
//!   first `r` columns; trailing `k - r` columns are zero (no unit
//!   diagonal continuation past the rank),
//! * `U` is `k × n`, upper triangular in its first `r` rows;
//!   trailing `k - r` rows are zero,
//! * `Q` is `n × n`, a column permutation matrix.
//!
//! ## Algorithm
//!
//! Standard Gaussian elimination with full pivoting:
//!
//! 1. At iteration `r`, scan `work[r..m, r..n]` column-by-column for
//!    a nonzero entry. Choose the lexicographically-smallest `(i, c)`
//!    as the pivot.
//!
//! 2. Swap rows `pr ↔ r` and columns `pc ↔ r` in `work` so the pivot
//!    is at `(r, r)`. Mirror those swaps in `row_perm` / `col_perm`
//!    and in the already-accumulated columns of `L`.
//!
//! 3. Set `L[r, r] = 1`. For each `i > r` with `work[i, r] = 1`,
//!    XOR `work[r]` into `work[i]` and record `L[i, r] = 1`.
//!
//! 4. `r += 1`; loop until no pivot remains.
//!
//! ## Naivety
//!
//! This is the unblocked `O(m · n · min(m, n))` version.
//!
//! ## References
//!
//! * Albrecht, Pernet, *Efficient Decomposition of Dense Matrices
//!   over GF(2)*, 2010 — [arXiv:1006.1744](https://arxiv.org/abs/1006.1744).
//!   The blocked PLE algorithm and the equivalence to PLUQ (`A = PLUQ`) over
//!   F_2.

use crate::mzd::Mzd;

impl Mzd {
    /// Rank via the M4R-block PLE engine
    /// (`crate::pluq_kernels::ple_russian`). Doesn't allocate the
    /// P/L/U/Q output matrices — only returns the integer rank.
    /// Used by the libm4ri parity tests to validate the PLE engine against
    /// `mzd_pluq` at production sizes. This route uses `ple_russian` without
    /// allocating the four explicit decomposition matrices.
    pub fn ple_rank(&self) -> usize {
        let mut work = self.clone();
        let mut p = vec![0usize; self.rows().max(1)];
        let mut q = vec![0usize; self.cols().max(1)];
        crate::pluq_kernels::ple_russian(&mut work, &mut p, &mut q)
    }

    /// PLUQ decomposition. Returns `(P, L, U, Q, rank)` such that
    /// `P · L · U · Q == self` over F_2. See module docs for shapes
    /// and conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(16, 16, 1);
    /// let (p, l, u, q, _rank) = a.pluq();
    /// let lu = Mzd::matmul(&l, &u);
    /// let plu = Mzd::matmul(&p, &lu);
    /// let pluq = Mzd::matmul(&plu, &q);
    /// assert_eq!(pluq.as_words(), a.as_words());
    /// ```
    pub fn pluq(&self) -> (Mzd, Mzd, Mzd, Mzd, usize) {
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);
        let mut p = Mzd::new(m, m);
        let mut l = Mzd::new(m, k);
        let mut u = Mzd::new(k, n);
        let mut q = Mzd::new(n, n);
        let rank = self.pluq_core(&mut p, &mut l, &mut u, &mut q);
        (p, l, u, q, rank)
    }

    /// Caller-owned-output counterpart to [`Mzd::pluq`]. `p`/`l`/`u`/`q` must already
    /// be the right shape (`m×m`, `m×k`, `k×n`, `n×n` for
    /// `k = min(m, n)`); this fully overwrites every one of them
    /// (never accumulates into whatever they held before). `pluq` is
    /// infallible (every matrix has a PLUQ decomposition) — there is
    /// no error path to leave an output untouched. The working matrix and
    /// permutation vectors are allocated internally.
    ///
    /// # Panics
    ///
    /// If any of `p`/`l`/`u`/`q` has the wrong shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(16, 16, 1);
    /// let (mut p, mut l, mut u, mut q) =
    ///     (Mzd::new(16, 16), Mzd::new(16, 16), Mzd::new(16, 16), Mzd::new(16, 16));
    /// let rank = a.pluq_into(&mut p, &mut l, &mut u, &mut q);
    /// let (p2, l2, u2, q2, rank2) = a.pluq();
    /// assert_eq!(rank, rank2);
    /// assert_eq!(p.as_words(), p2.as_words());
    /// assert_eq!(l.as_words(), l2.as_words());
    /// assert_eq!(u.as_words(), u2.as_words());
    /// assert_eq!(q.as_words(), q2.as_words());
    /// ```
    pub fn pluq_into(&self, p: &mut Mzd, l: &mut Mzd, u: &mut Mzd, q: &mut Mzd) -> usize {
        use crate::view::MatrixWrite;
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);
        assert_eq!((p.rows(), p.cols()), (m, m), "pluq_into: P shape mismatch");
        assert_eq!((l.rows(), l.cols()), (m, k), "pluq_into: L shape mismatch");
        assert_eq!((u.rows(), u.cols()), (k, n), "pluq_into: U shape mismatch");
        assert_eq!((q.rows(), q.cols()), (n, n), "pluq_into: Q shape mismatch");
        p.fill_zero();
        l.fill_zero();
        u.fill_zero();
        q.fill_zero();
        self.pluq_core(p, l, u, q)
    }

    /// Shared core: writes the PLUQ decomposition into already-
    /// correctly-shaped, already-zeroed `p`/`l`/`u`/`q`, returns the
    /// rank. [`Mzd::pluq`] and [`Mzd::pluq_into`] are both thin
    /// wrappers around this.
    fn pluq_core(&self, p: &mut Mzd, l: &mut Mzd, u: &mut Mzd, q: &mut Mzd) -> usize {
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);

        let mut work = self.clone();
        let mut row_perm: Vec<usize> = (0..m).collect();
        let mut col_perm: Vec<usize> = (0..n).collect();

        let mut r = 0usize;
        while r < k {
            // Find a pivot in `work[r..m, r..n]`: smallest column with
            // a nonzero in any row r..m, and within that column the
            // smallest row with a 1.
            let mut pivot: Option<(usize, usize)> = None;
            for i in r..m {
                if let Some(c) = work.first_one_in_row(i, r) {
                    if pivot.is_none_or(|(best_row, best_col)| (c, i) < (best_col, best_row)) {
                        pivot = Some((i, c));
                    }
                }
            }
            let (pr, pc) = match pivot {
                Some(p) => p,
                None => break,
            };

            // Row swap.
            if pr != r {
                work.swap_rows(pr, r);
                row_perm.swap(pr, r);
                // Mirror the swap on L's already-built columns 0..r.
                for j in 0..r {
                    let a = l.get(pr, j);
                    let b = l.get(r, j);
                    l.set(pr, j, b);
                    l.set(r, j, a);
                }
            }

            // Column swap (on the whole working matrix).
            if pc != r {
                work.swap_cols(pc, r);
                col_perm.swap(pc, r);
            }

            // Pivot is at (r, r). Set unit diagonal and eliminate below.
            l.set(r, r, 1);
            for i in (r + 1)..m {
                if work.get(i, r) == 1 {
                    work.xor_row_self(i, r);
                    l.set(i, r, 1);
                }
            }

            r += 1;
        }

        // ---- Fill p/u/q from row_perm, col_perm, work (l is already
        // ---- built above) ------------------------------------------
        //
        // P, Q follow the convention `P · L · U · Q = A`, which means
        // P is the *inverse* of the row permutation we tracked
        // (transpose, since perm matrices are orthogonal). That works
        // out to `P[row_perm[i], i] = 1`. Similarly `Q[i, col_perm[i]] = 1`.
        for (i, &src) in row_perm.iter().enumerate() {
            p.set(src, i, 1);
        }
        for (i, &dst) in col_perm.iter().enumerate() {
            q.set(i, dst, 1);
        }

        // U is the first k rows of `work`, with rows past r already
        // zero from the elimination loop.
        for i in 0..r {
            for j in 0..n {
                if work.get(i, j) == 1 {
                    u.set(i, j, 1);
                }
            }
        }

        r
    }
}
