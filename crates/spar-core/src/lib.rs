// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! # spar-core — shared contracts for the Tertius sparse-LA kernels
//!
//! The sparse linear-algebra kernels — `sparla` (exact ℤ / F_p),
//! and `sparb` (Arb/Acb ball) — are sibling
//! crates that cannot depend on one another. `spar-core` is the crate
//! they all depend *up* into for the surface they genuinely share.
//! Everyone depends up into it; nobody sideways.
//!
//! Today that shared surface is exactly one trait: [`SparseRingStorage`],
//! the ring-independent structural view of a sparse matrix — shape,
//! sparsity, and the cheap structural predicates. It is deliberately
//! narrow: anything that touches *values* (`get`, row iteration,
//! arithmetic, SpMV) is scalar-specific and stays on each kernel's own
//! types, not here. The trait grows only when a new member is genuinely
//! ring-independent and has more than one implementor.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// The ring-independent structural surface of a sparse matrix: shape,
/// sparsity, and the cheap structural predicates.
///
/// Implemented across the sparse kernels — `sparla`'s `CsrZ` / `CsrFp`,
/// and `sparb`'s `CsrArb` / `CsrAcb` — so that
/// structural code can be written once without caring whether the
/// entries are integers, prime-field residues, or real/complex balls.
pub trait SparseRingStorage {
    /// Number of rows.
    fn rows(&self) -> usize;

    /// Number of columns.
    fn cols(&self) -> usize;

    /// Number of explicitly stored entries.
    ///
    /// By the storage invariant, every stored entry is structurally
    /// significant, so `nnz()` is the count of stored cells. (A ball
    /// that merely *contains* zero is still stored — dropping it would
    /// lose its rigorous enclosure — so for `sparb` this counts the
    /// structurally significant cells, not the exact nonzeros.)
    fn nnz(&self) -> usize;

    /// Fraction of cells that are stored, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` for the empty matrix (zero rows or zero
    /// columns), which is the only case where `rows * cols == 0`.
    fn density(&self) -> f64 {
        let total = self.rows().saturating_mul(self.cols());
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// `true` if no entries are stored — equivalently, the matrix is
    /// the (exact) zero matrix.
    fn is_zero(&self) -> bool {
        self.nnz() == 0
    }

    /// `true` if the matrix is square.
    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }
}

#[cfg(test)]
mod tests {
    use super::SparseRingStorage;

    struct Fake {
        rows: usize,
        cols: usize,
        nnz: usize,
    }

    impl SparseRingStorage for Fake {
        fn rows(&self) -> usize {
            self.rows
        }
        fn cols(&self) -> usize {
            self.cols
        }
        fn nnz(&self) -> usize {
            self.nnz
        }
    }

    #[test]
    fn defaults_are_derived_from_shape_and_nnz() {
        let a = Fake {
            rows: 4,
            cols: 5,
            nnz: 5,
        };
        assert_eq!(a.density(), 5.0 / 20.0);
        assert!(!a.is_zero());
        assert!(!a.is_square());

        let sq = Fake {
            rows: 3,
            cols: 3,
            nnz: 0,
        };
        assert!(sq.is_square());
        assert!(sq.is_zero());
        assert_eq!(sq.density(), 0.0);
    }

    #[test]
    fn empty_matrix_has_zero_density_not_nan() {
        let empty = Fake {
            rows: 0,
            cols: 7,
            nnz: 0,
        };
        assert_eq!(empty.density(), 0.0);
        assert!(empty.is_zero());
    }
}
