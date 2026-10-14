// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Linear systems and kernel computation over GF(2^e).
//!
//! Direct generalization of [`m4rst::Mzd::solve_left`] /
//! [`m4rst::Mzd::nullspace`] — same algorithm shape (PLUQ, then
//! forward- and back-substitution through the triangular factors),
//! with the F_2 XOR-if-set-bit substitution step replaced by a
//! genuine field scaled-row-add (`row_i += factor · row_j`, via
//! [`crate::gf2e::Gf2eTables`]).
//!
//! Operates on flat `Vec<u16>` buffers (via `to_u16_buffer` /
//! `from_u16_buffer`) rather than `Mzed` method calls for the
//! substitution loops, matching `pluq.rs`'s own style and avoiding a
//! dependency on any new `Mzed` row-mutation primitive beyond
//! [`Mzed::transpose`] (used, like `m4rst`'s version, to apply `Pᵀ`
//! and `Qᵀ`).
//!
//! `solve_left(A, B)`: returns `X` such that `A · X = B`, or `None`
//! if `B` lies outside the column space of `A`. When `A` is rank-
//! deficient and a solution exists, this routine returns the
//! "natural" solution with free variables set to zero. (To enumerate
//! the full affine solution set, combine with [`Mzed::nullspace`].)
//!
//! `nullspace(A)`: returns an `n × (n - r)` matrix whose columns are
//! a basis for `ker(A)`.

use crate::mzed::Mzed;

impl Mzed {
    /// Solve `A · X = B` over GF(2^e). Returns `Some(X)` or `None` if
    /// inconsistent.
    ///
    /// `A` is `m × n`, `B` is `m × n_rhs`, `X` is `n × n_rhs`. When
    /// `A` is rank-deficient and the system is solvable, returns the
    /// "natural" solution with free variables set to zero; combine
    /// with [`Mzed::nullspace`] to enumerate the full affine solution
    /// set.
    ///
    /// # Panics
    ///
    /// On shape mismatch (`a.rows() != b.rows()`) or field mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let a = Mzed::from_u16_buffer(f, 3, 3, &[1, 2, 3, 0, 1, 2, 1, 2, 4]);
    /// let y = Mzed::from_u16_buffer(f, 3, 2, &[5, 6, 7, 8, 9, 10]);
    /// let b = Mzed::matmul(&a, &y);
    /// let x = Mzed::solve_left(&a, &b).expect("full rank → solvable");
    /// assert_eq!(Mzed::matmul(&a, &x).to_u16_buffer(), b.to_u16_buffer());
    /// ```
    pub fn solve_left(a: &Mzed, b: &Mzed) -> Option<Mzed> {
        assert_eq!(a.ctx(), b.ctx(), "Mzed::solve_left: field mismatch");
        assert_eq!(
            a.rows(),
            b.rows(),
            "Mzed::solve_left: A and B must have the same number of rows ({} vs {})",
            a.rows(),
            b.rows()
        );
        let f = a.ctx();
        let tables = f.tables();
        let m = a.rows();
        let n = a.cols();
        let n_rhs = b.cols();
        let k = m.min(n);

        let (p, l, u, q, rank) = a.pluq();

        // PT_B = Pᵀ · B.
        let pt_b = Mzed::matmul(&p.transpose(), b).to_u16_buffer();
        let l_buf = l.to_u16_buffer(); // m × k
        let u_buf = u.to_u16_buffer(); // k × n

        // Forward-sub: L · Y = PT_B. L is unit lower triangular in
        // its first `rank` columns. Solve for Y[0..rank].
        let mut y = vec![0u16; k * n_rhs];
        for i in 0..rank {
            for c in 0..n_rhs {
                y[i * n_rhs + c] = pt_b[i * n_rhs + c];
            }
            for j in 0..i {
                let factor = l_buf[i * k + j];
                if factor != 0 {
                    for c in 0..n_rhs {
                        y[i * n_rhs + c] ^= tables.mul(factor, y[j * n_rhs + c]);
                    }
                }
            }
        }

        // Consistency check on rows [rank, m): each of those rows of
        // PT_B must equal the corresponding linear combination of
        // Y[0..rank] selected by L's bottom block. Otherwise no
        // solution exists.
        if rank < m && n_rhs > 0 {
            let mut scratch = vec![0u16; n_rhs];
            for i in rank..m {
                for c in 0..n_rhs {
                    scratch[c] = pt_b[i * n_rhs + c];
                }
                for j in 0..rank {
                    let factor = l_buf[i * k + j];
                    if factor != 0 {
                        for c in 0..n_rhs {
                            scratch[c] ^= tables.mul(factor, y[j * n_rhs + c]);
                        }
                    }
                }
                if scratch.iter().any(|&v| v != 0) {
                    return None;
                }
            }
        }

        // Back-sub: U · Z = Y. U has genuine pivots (not necessarily
        // 1) at (i, i) for i < rank — divide them out — and is zero
        // below the diagonal in the rank × rank block. Z[rank..n] = 0
        // (free variables to zero) so columns of U past rank
        // contribute nothing.
        let mut z = vec![0u16; n * n_rhs];
        for i in (0..rank).rev() {
            for c in 0..n_rhs {
                z[i * n_rhs + c] = y[i * n_rhs + c];
            }
            for j in (i + 1)..rank {
                let factor = u_buf[i * n + j];
                if factor != 0 {
                    for c in 0..n_rhs {
                        z[i * n_rhs + c] ^= tables.mul(factor, z[j * n_rhs + c]);
                    }
                }
            }
            let pivot = u_buf[i * n + i];
            let inv = tables.inv(pivot);
            for c in 0..n_rhs {
                let v = z[i * n_rhs + c];
                if v != 0 {
                    z[i * n_rhs + c] = tables.mul(v, inv);
                }
            }
        }

        // X = Qᵀ · Z.
        let z_mat = Mzed::from_u16_buffer(f, n, n_rhs, &z);
        Some(Mzed::matmul(&q.transpose(), &z_mat))
    }

    /// Multiplicative inverse of an `n × n` matrix `A`, or `None` if
    /// `A` is singular. Defined as `solve_left(A, I_n)`.
    ///
    /// # Panics
    ///
    /// On non-square input: `a.rows()` must equal `a.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let id = Mzed::identity(f, 8);
    /// let inv = Mzed::inverse(&id).expect("I is invertible");
    /// assert_eq!(inv.to_u16_buffer(), id.to_u16_buffer());
    /// ```
    pub fn inverse(a: &Mzed) -> Option<Mzed> {
        assert_eq!(
            a.rows(),
            a.cols(),
            "Mzed::inverse: matrix must be square ({}×{})",
            a.rows(),
            a.cols()
        );
        let i = Mzed::identity(a.ctx(), a.rows());
        // solve_left returns _some_ solution for any consistent
        // system. For full-rank A, A·X = I uniquely determines
        // X = A⁻¹. For rank-deficient A, A·X = I is inconsistent (the
        // rows of I past rank can't be hit), and solve_left already
        // returns None in that case.
        Mzed::solve_left(a, &i)
    }

    /// Compute a basis for the (right) nullspace of `A`. Returns an
    /// `n × (n - r)` matrix whose columns span
    /// `ker(A) = { v : A · v = 0 }`. If `A` has full column rank,
    /// returns an `n × 0` matrix.
    ///
    /// Rank-nullity: `a.rank() + nullspace(a).cols() == a.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let a = Mzed::from_u16_buffer(f, 2, 3, &[1, 2, 3, 2, 4, 6]); // row 1 = 2 · row 0
    /// let k = Mzed::nullspace(&a);
    /// let ak = Mzed::matmul(&a, &k);
    /// assert!(ak.to_u16_buffer().iter().all(|&v| v == 0));
    /// assert_eq!(a.rank() + k.cols(), a.cols());
    /// ```
    pub fn nullspace(a: &Mzed) -> Mzed {
        let f = a.ctx();
        let tables = f.tables();
        let n = a.cols();
        let (_p, _l, u, q, rank) = a.pluq();
        let nullity = n - rank;
        if nullity == 0 {
            return Mzed::zeros(f, n, 0);
        }
        // Build the n × nullity matrix `zs` whose columns are the
        // pre-permutation basis vectors: for each k ∈ [rank, n), one
        // basis vector has z[k] = 1, the other free entries (also in
        // [rank, n)) set to zero, and the top `rank` entries
        // back-substituted from U · z = 0 (dividing out U's genuine
        // pivot values, unlike the F_2 version where every pivot
        // already is 1).
        let u_buf = u.to_u16_buffer(); // k × n
        let mut zs = vec![0u16; n * nullity];
        for (col, free_var) in (rank..n).enumerate() {
            zs[free_var * nullity + col] = 1;
        }
        for i in (0..rank).rev() {
            for j in (i + 1)..n {
                let factor = u_buf[i * n + j];
                if factor != 0 {
                    for c in 0..nullity {
                        zs[i * nullity + c] ^= tables.mul(factor, zs[j * nullity + c]);
                    }
                }
            }
            let pivot = u_buf[i * n + i];
            let inv = tables.inv(pivot);
            for c in 0..nullity {
                let v = zs[i * nullity + c];
                if v != 0 {
                    zs[i * nullity + c] = tables.mul(v, inv);
                }
            }
        }
        let zs_mat = Mzed::from_u16_buffer(f, n, nullity, &zs);
        Mzed::matmul(&q.transpose(), &zs_mat)
    }
}
