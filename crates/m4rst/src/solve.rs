// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Linear systems and kernel computation over F_2.
//!
//! Both `solve_left` and `nullspace` are built on the PLUQ
//! decomposition (`pluq.rs`) followed by forward- and back-
//! substitution through the triangular factors. Naive variants;
//! correctness is the bar.
//!
//! `solve_left(A, B)`: returns `X` such that `A · X = B`, or `None`
//! if `B` lies outside the column space of `A`. When `A` is rank-
//! deficient and a solution exists, this routine returns the
//! "natural" solution with free variables set to zero. (To enumerate
//! the full affine solution set, combine with `nullspace(A)`.)
//!
//! `nullspace(A)`: returns an `n × (n - r)` matrix whose columns
//! are a basis for `ker(A)`.

use crate::mzd::Mzd;
use crate::view::MatrixWrite;

impl Mzd {
    /// Solve `A · X = B` over F_2. Returns `Some(X)` or `None` if
    /// inconsistent.
    ///
    /// `A` is `m × n`, `B` is `m × n_rhs`, `X` is `n × n_rhs`. When
    /// `A` is rank-deficient and the system is solvable, returns
    /// the "natural" solution with free variables set to zero;
    /// combine with [`Mzd::nullspace`] to enumerate the full
    /// affine solution set.
    ///
    /// # Panics
    ///
    /// On shape mismatch: `a.rows()` must equal `b.rows()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(16, 16, 1);
    /// let y = Mzd::random(16, 4, 2);
    /// let b = Mzd::matmul(&a, &y);
    /// let x = Mzd::solve_left(&a, &b).expect("rank 16 → solvable");
    /// assert_eq!(Mzd::matmul(&a, &x).as_words(), b.as_words());
    /// ```
    pub fn solve_left(a: &Mzd, b: &Mzd) -> Option<Mzd> {
        assert_eq!(
            a.rows(),
            b.rows(),
            "solve_left: A and B must have the same number of rows ({} vs {})",
            a.rows(),
            b.rows()
        );
        let m = a.rows();
        let n = a.cols();
        let n_rhs = b.cols();
        let k = m.min(n);

        let (p, l, u, q, rank) = a.pluq();

        // PT_B = P^T · B
        let pt = p.transpose();
        let pt_b = Mzd::matmul(&pt, b);

        // Forward-sub: L · Y = PT_B. L is unit lower triangular in its
        // first `rank` columns. Solve for Y[0..rank].
        let mut y = Mzd::new(k, n_rhs);
        for i in 0..rank {
            // Y[i] starts as PT_B[i].
            for c in 0..n_rhs {
                y.set(i, c, pt_b.get(i, c));
            }
            for j in 0..i {
                if l.get(i, j) == 1 {
                    y.xor_row_self(i, j);
                }
            }
        }

        // Consistency check on rows [rank, m): each of those rows of
        // PT_B must equal the corresponding linear combination of
        // Y[0..rank] selected by L's bottom block. Otherwise no
        // solution exists.
        if rank < m && n_rhs > 0 {
            let mut scratch = Mzd::new(1, n_rhs);
            for i in rank..m {
                for c in 0..n_rhs {
                    scratch.set(0, c, pt_b.get(i, c));
                }
                for j in 0..rank {
                    if l.get(i, j) == 1 {
                        scratch.xor_row_from(0, &y, j);
                    }
                }
                for c in 0..n_rhs {
                    if scratch.get(0, c) != 0 {
                        return None;
                    }
                }
            }
        }

        // Back-sub: U · Z = Y. U has unit pivots at (i, i) for i < rank
        // and is zero below the diagonal in the rank × rank block.
        // Set Z[rank..n] = 0 (free variables to zero) so columns of U
        // past rank contribute nothing.
        let mut z = Mzd::new(n, n_rhs);
        for i in (0..rank).rev() {
            for c in 0..n_rhs {
                z.set(i, c, y.get(i, c));
            }
            for j in (i + 1)..rank {
                if u.get(i, j) == 1 {
                    z.xor_row_self(i, j);
                }
            }
        }

        // X = Q^T · Z.
        let qt = q.transpose();
        Some(Mzd::matmul(&qt, &z))
    }

    /// Multiplicative inverse of an `n × n` matrix `A`, or `None`
    /// if `A` is singular. Defined as `solve_left(A, I_n)`.
    ///
    /// # Panics
    ///
    /// On non-square input: `a.rows()` must equal `a.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let i = Mzd::identity(8);
    /// let inv = Mzd::inverse(&i).expect("I is invertible");
    /// assert_eq!(inv.as_words(), i.as_words());
    /// ```
    pub fn inverse(a: &Mzd) -> Option<Mzd> {
        assert_eq!(
            a.rows(),
            a.cols(),
            "Mzd::inverse: matrix must be square ({}×{})",
            a.rows(),
            a.cols()
        );
        let n = a.rows();
        let i = Mzd::identity(n);
        let x = Mzd::solve_left(a, &i)?;
        // solve_left returns _some_ solution for any consistent system.
        // For full-rank A, A · X = I uniquely determines X = A^{-1}. For
        // rank-deficient A, A · X = I is inconsistent (the rows of I past
        // rank can't be hit), and we've already returned None above.
        Some(x)
    }

    /// Compute a basis for the (right) nullspace of `A`. Returns an
    /// `n × (n - r)` matrix whose columns span `ker(A) = { v : A · v = 0 }`.
    /// If `A` has full column rank, returns an `n × 0` matrix.
    ///
    /// Rank-nullity: `a.rank() + nullspace(a).cols() == a.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(20, 30, 1);
    /// let k = Mzd::nullspace(&a);
    /// // Every column of K is in ker(A): A · K = 0.
    /// let ak = Mzd::matmul(&a, &k);
    /// for &w in ak.as_words() { assert_eq!(w, 0); }
    /// // Rank-nullity holds.
    /// assert_eq!(a.rank() + k.cols(), a.cols());
    /// ```
    /// If `A` has full column rank, returns an `n × 0` matrix.
    pub fn nullspace(a: &Mzd) -> Mzd {
        let n = a.cols();
        let (_p, _l, u, q, rank) = a.pluq();
        let nullity = n - rank;
        if nullity == 0 {
            return Mzd::new(n, 0);
        }
        // Build the n × nullity matrix `zs` whose columns are the
        // pre-permutation basis vectors: for each k ∈ [rank, n), one
        // basis vector has z[k] = 1, the other free entries (also in
        // [rank, n)) set to zero, and the top `rank` entries
        // back-substituted from U · z = 0.
        let mut zs = Mzd::new(n, nullity);
        for (col, free_var) in (rank..n).enumerate() {
            zs.set(free_var, col, 1);
        }
        // For i = rank-1 down to 0: z[i] = sum_{j > i} U[i, j] · z[j].
        // (j ranges over all of [i+1, n) — including columns past
        // rank, since those columns of U can have non-zero entries in
        // the rank-r top block.)
        for i in (0..rank).rev() {
            for j in (i + 1)..n {
                if u.get(i, j) == 1 {
                    zs.xor_row_self(i, j);
                }
            }
        }
        let qt = q.transpose();
        Mzd::matmul(&qt, &zs)
    }
}
