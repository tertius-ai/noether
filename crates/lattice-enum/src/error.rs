// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed errors for the enumeration surface. Adversarial inputs return one of
//! these variants rather than panicking or producing a partial result.

use core::fmt;
use flint_core::Integer;

/// The enumeration error type. All fallible entry points return
/// `Result<_, EnumError>`; the sister pyo3 wrapper maps each variant onto a
/// `PyValueError` / `PyRuntimeError`.
///
/// (`Eq` is intentionally not derived — [`EnumError::EnumerationOverflow`]
/// carries an `f64` estimate; `PartialEq` is enough for tests and matching.)
#[derive(Debug, Clone, PartialEq)]
pub enum EnumError {
    /// The Gram matrix is not square, or a coordinate/target vector has a
    /// length that disagrees with the Gram dimension.
    InvalidShape {
        /// Human-readable detail (dimensions that disagreed).
        detail: String,
    },
    /// The Gram matrix is not symmetric (`G[i][j] != G[j][i]`).
    Asymmetric {
        /// The `(i, j)` position at which symmetry first failed.
        at: (usize, usize),
    },
    /// The Gram matrix is not positive definite (Sylvester / Cholesky
    /// rejects it): some leading principal minor is `≤ 0`.
    NotPositiveDefinite,
    /// A norm bound was negative.
    InvalidBound {
        /// The offending bound.
        bound: Integer,
    },
    /// An unbounded minimum-shell query was made in rank zero, where there is
    /// no nonzero vector and hence no attained minimum.
    NoNonzeroVectors,
    /// The bounding-ellipsoid integer-point estimate exceeded the sanity
    /// cap (`2·10⁷`) and `max_count` was `None`.
    EnumerationOverflow {
        /// The estimated integer-point count that tripped the cap.
        estimate: f64,
    },
    /// A heuristic could not be projected to finite floating point, or an
    /// enumerated coordinate exceeded the word-sized public representation.
    GramOverflow,
    /// A pruning vector's dimension disagreed with the Gram dimension, or a
    /// pruning coefficient fell outside `(0, 1]` / was non-monotone.
    PruningShape {
        /// Human-readable detail.
        detail: String,
    },
}

impl fmt::Display for EnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnumError::InvalidShape { detail } => write!(f, "invalid shape: {detail}"),
            EnumError::Asymmetric { at } => {
                write!(f, "Gram matrix not symmetric at {:?}", at)
            }
            EnumError::NotPositiveDefinite => {
                write!(f, "Gram matrix is not positive definite")
            }
            EnumError::InvalidBound { bound } => write!(f, "bound must be ≥ 0, got {bound}"),
            EnumError::NoNonzeroVectors => {
                write!(
                    f,
                    "the rank-zero lattice has no nonzero vector or minimum shell"
                )
            }
            EnumError::EnumerationOverflow { estimate } => write!(
                f,
                "enumeration estimate {estimate:.3e} exceeds the 2e7 sanity cap; \
                 pass max_count to cap, or reduce the bound"
            ),
            EnumError::GramOverflow => write!(
                f,
                "a lattice coordinate is outside i64 or a floating heuristic is unavailable"
            ),
            EnumError::PruningShape { detail } => write!(f, "invalid pruning vector: {detail}"),
        }
    }
}

impl std::error::Error for EnumError {}

/// Crate result alias.
pub type EnumResult<T> = Result<T, EnumError>;
