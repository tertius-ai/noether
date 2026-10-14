// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! PLUQ decomposition over GF(2^e).
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
//! This is the direct GF(2^e) generalization of [`m4rst::Mzd::pluq`]'s F_2
//! PLUQ, with the same four-explicit-matrices convention. `L` remains unit
//! lower triangular and `U` carries the pivot values, so elimination uses the
//! genuine field quotient `raw / pivot`.
//!
//! ## Algorithm
//!
//! Standard Gaussian elimination with full pivoting:
//!
//! 1. At iteration `r`, scan `work[r..m, r..n]` column-by-column for a
//!    nonzero entry. Choose the lexicographically-smallest `(i, c)` as
//!    the pivot.
//!
//! 2. Swap rows `pr ↔ r` and columns `pc ↔ r` in `work` so the pivot is
//!    at `(r, r)`. Mirror those swaps in `row_perm` / `col_perm` and in
//!    the already-accumulated columns of `L`.
//!
//! 3. Set `L[r, r] = 1`; `U`'s diagonal carries the pivot value. For each
//!    `i > r`, let `raw = work[i, r]`; if nonzero,
//!    the elimination **factor** is `raw / work[r, r]` (a genuine
//!    field division, not just `raw` — F_2's version gets this for
//!    free since its only nonzero element is 1). Subtract
//!    `factor · work[r]` from `work[i]` (subtract is XOR-add in
//!    characteristic 2) and record `L[i, r] = factor`.
//!
//! 4. `r += 1`; loop until no pivot remains.
//!
//! ## Naivety
//!
//! This is the unblocked `O(m · n · min(m, n))` implementation.

use crate::gf2e::Gf2e;
use crate::mzed::Mzed;

impl Mzed {
    /// PLUQ decomposition. Returns `(P, L, U, Q, rank)` such that
    /// `P · L · U · Q == self` over GF(2^e). See module docs for
    /// shapes and conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B); // GF(256), AES polynomial
    /// let a = Mzed::from_u16_buffer(f, 4, 4, &[
    ///     1, 2, 3, 4,
    ///     5, 6, 7, 8,
    ///     9, 10, 200, 12,
    ///     13, 14, 15, 250,
    /// ]);
    /// let (p, l, u, q, _rank) = a.pluq();
    /// let lu = Mzed::matmul(&l, &u);
    /// let plu = Mzed::matmul(&p, &lu);
    /// let pluq = Mzed::matmul(&plu, &q);
    /// assert_eq!(pluq.to_u16_buffer(), a.to_u16_buffer());
    /// ```
    pub fn pluq(&self) -> (Mzed, Mzed, Mzed, Mzed, usize) {
        let f: Gf2e = self.ctx();
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);
        let mut p = Mzed::zeros(f, m, m);
        let mut l = Mzed::zeros(f, m, k);
        let mut u = Mzed::zeros(f, k, n);
        let mut q = Mzed::zeros(f, n, n);
        let rank = self.pluq_core(&mut p, &mut l, &mut u, &mut q);
        (p, l, u, q, rank)
    }

    /// Caller-owned-output counterpart to [`Mzed::pluq`]. `p`/`l`/`u`/`q` must already
    /// be the right shape (`m×m`, `m×k`, `k×n`, `n×n` for
    /// `k = min(m, n)`) and field; this fully overwrites every one of
    /// them (never accumulates into whatever they held before) and
    /// reuses their storage. `pluq` is
    /// infallible (every matrix has a PLUQ decomposition) — there is
    /// no error path to leave an output untouched. Working buffers are
    /// allocated internally.
    ///
    /// # Panics
    ///
    /// If any of `p`/`l`/`u`/`q` has the wrong shape or field.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let a = Mzed::from_u16_buffer(f, 3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 200]);
    /// let (mut p, mut l, mut u, mut q) = (
    ///     Mzed::zeros(f, 3, 3),
    ///     Mzed::zeros(f, 3, 3),
    ///     Mzed::zeros(f, 3, 3),
    ///     Mzed::zeros(f, 3, 3),
    /// );
    /// let rank = a.pluq_into(&mut p, &mut l, &mut u, &mut q);
    /// let (p2, l2, u2, q2, rank2) = a.pluq();
    /// assert_eq!(rank, rank2);
    /// assert_eq!(p.to_u16_buffer(), p2.to_u16_buffer());
    /// assert_eq!(l.to_u16_buffer(), l2.to_u16_buffer());
    /// assert_eq!(u.to_u16_buffer(), u2.to_u16_buffer());
    /// assert_eq!(q.to_u16_buffer(), q2.to_u16_buffer());
    /// ```
    pub fn pluq_into(&self, p: &mut Mzed, l: &mut Mzed, u: &mut Mzed, q: &mut Mzed) -> usize {
        let f = self.ctx();
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);
        assert_eq!(p.ctx(), f, "pluq_into: P field mismatch");
        assert_eq!(l.ctx(), f, "pluq_into: L field mismatch");
        assert_eq!(u.ctx(), f, "pluq_into: U field mismatch");
        assert_eq!(q.ctx(), f, "pluq_into: Q field mismatch");
        assert_eq!((p.rows(), p.cols()), (m, m), "pluq_into: P shape mismatch");
        assert_eq!((l.rows(), l.cols()), (m, k), "pluq_into: L shape mismatch");
        assert_eq!((u.rows(), u.cols()), (k, n), "pluq_into: U shape mismatch");
        assert_eq!((q.rows(), q.cols()), (n, n), "pluq_into: Q shape mismatch");
        use m4rst::MatrixWrite;
        p.packed.fill_zero();
        l.packed.fill_zero();
        u.packed.fill_zero();
        q.packed.fill_zero();
        self.pluq_core(p, l, u, q)
    }

    /// Shared core: writes the PLUQ decomposition into already-
    /// correctly-shaped, already-zeroed `p`/`l`/`u`/`q`, returns the
    /// rank. [`Mzed::pluq`] and [`Mzed::pluq_into`] are both thin wrappers
    /// around this single implementation.
    fn pluq_core(&self, p: &mut Mzed, l: &mut Mzed, u: &mut Mzed, q: &mut Mzed) -> usize {
        let f: Gf2e = self.ctx();
        let m = self.rows();
        let n = self.cols();
        let k = m.min(n);
        let tables = f.tables();

        let mut work = self.to_u16_buffer(); // row-major m × n
        let mut row_perm: Vec<usize> = (0..m).collect();
        let mut col_perm: Vec<usize> = (0..n).collect();

        let mut r = 0usize;
        while r < k {
            // Find a pivot in `work[r..m, r..n]`: smallest column with
            // a nonzero in any row r..m, and within that column the
            // smallest row with a nonzero.
            let mut pivot: Option<(usize, usize)> = None;
            'outer: for c in r..n {
                for i in r..m {
                    if work[i * n + c] != 0 {
                        pivot = Some((i, c));
                        break 'outer;
                    }
                }
            }
            let Some((pr, pc)) = pivot else { break };

            // Row swap.
            if pr != r {
                for c in 0..n {
                    work.swap(pr * n + c, r * n + c);
                }
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
                for i in 0..m {
                    let a = work[i * n + pc];
                    let b = work[i * n + r];
                    work[i * n + pc] = b;
                    work[i * n + r] = a;
                }
                col_perm.swap(pc, r);
            }

            // Pivot is at (r, r). L keeps a genuine unit diagonal; the
            // pivot row is left as-is (not normalized), so U's diagonal
            // carries the real pivot value and each elimination factor
            // is a proper field division `raw / pivot_val`.
            let pivot_val = work[r * n + r];
            let inv = tables.inv(pivot_val);

            l.set(r, r, 1);
            for i in (r + 1)..m {
                let raw = work[i * n + r];
                if raw == 0 {
                    continue;
                }
                let factor = tables.mul(raw, inv);
                for c in 0..n {
                    let piv = work[r * n + c];
                    if piv != 0 {
                        // char-2: subtraction is addition is XOR.
                        work[i * n + c] ^= tables.mul(factor, piv);
                    }
                }
                l.set(i, r, factor);
            }

            r += 1;
        }

        // ---- Fill p/u/q from row_perm, col_perm, work (l is already
        // ---- built above) ------------------------------------------
        //
        // P, Q follow the convention `P · L · U · Q = A`, which means P
        // is the *inverse* of the row permutation we tracked (transpose,
        // since perm matrices are orthogonal). That works out to
        // `P[row_perm[i], i] = 1`. Similarly `Q[i, col_perm[i]] = 1`.
        for (i, &src) in row_perm.iter().enumerate() {
            p.set(src, i, 1);
        }
        for (i, &dst) in col_perm.iter().enumerate() {
            q.set(i, dst, 1);
        }

        // U is the first r rows of `work`, with rows past r already
        // zero from the elimination loop.
        for i in 0..r {
            for j in 0..n {
                let v = work[i * n + j];
                if v != 0 {
                    u.set(i, j, v);
                }
            }
        }

        r
    }
}
