// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed construction and conversion failures.

use core::fmt;

/// Everything that can go wrong in the convex-hull kernel.
///
/// A degenerate-but-valid input (fewer than `dim + 1` affinely
/// independent points, duplicate / interior points) is **not** an error
/// — it yields a valid lower-dimensional [`crate::Polytope`] with the
/// redundant points silently absorbed. Only the four cases below are
/// errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HullError {
    /// An empty point set was handed to a hull constructor.
    EmptyInput,
    /// An H-representation describes an unbounded polyhedron (a recession
    /// ray / lineality exists). This kernel builds **bounded** polytopes
    /// only; cones and unbounded polyhedra are outside its scope.
    Unbounded,
    /// An H-representation is infeasible — the described polytope is
    /// empty (no point satisfies every inequality).
    Infeasible,
    /// Input vectors disagree on their coordinate count.
    DimensionMismatch {
        /// The ambient dimension established by the first vector.
        expected: usize,
        /// The offending vector's length.
        got: usize,
    },
    /// A zero-dimensional ambient space (points with no coordinates) was
    /// supplied. There is nothing to hull.
    ZeroAmbientDim,
}

impl fmt::Display for HullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HullError::EmptyInput => write!(f, "empty input point set"),
            HullError::Unbounded => {
                write!(f, "unbounded polyhedron (bounded polytopes only)")
            }
            HullError::Infeasible => write!(f, "infeasible H-representation (empty polytope)"),
            HullError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            HullError::ZeroAmbientDim => write!(f, "zero-dimensional ambient space"),
        }
    }
}

impl std::error::Error for HullError {}
