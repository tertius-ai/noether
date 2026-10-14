// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed errors for `graph-core`.
//!
//! Every fallible entry point returns [`Result<T, GraphError>`]. Malformed
//! input maps to a variant here rather than causing a panic.

use core::fmt;

/// The error type for every fallible `graph-core` operation.
///
/// A typed error whose variants carry the data needed to diagnose the fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// An edge endpoint, source, or target referenced a vertex id
    /// that is `>= n` (the vertex count). Carries the offending id
    /// and the vertex count it violated.
    VertexOutOfRange {
        /// The out-of-range vertex id.
        vertex: usize,
        /// The graph's vertex count `n`; valid ids are `0..n`.
        order: usize,
    },
    /// A negative edge weight was handed to the Dijkstra path.
    /// Dijkstra is only correct for non-negative weights; rather
    /// than return a silently wrong distance, this is surfaced.
    /// Carries the offending weight.
    NegativeWeight {
        /// The offending negative weight.
        weight: i64,
    },
    /// A label table's length did not match the graph it annotates —
    /// a vertex-label slice whose length is not the vertex count, or
    /// an edge-label slice whose length is not the edge count.
    LabelCountMismatch {
        /// The number of labels supplied.
        got: usize,
        /// The number of labels expected (vertex count or edge count).
        expected: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::VertexOutOfRange { vertex, order } => write!(
                f,
                "vertex {vertex} out of range for a graph of order {order} (valid ids are 0..{order})"
            ),
            GraphError::NegativeWeight { weight } => write!(
                f,
                "negative edge weight {weight}: Dijkstra requires non-negative weights"
            ),
            GraphError::LabelCountMismatch { got, expected } => write!(
                f,
                "label count mismatch: got {got} labels, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// Shorthand for `Result<T, GraphError>`.
pub type Result<T> = core::result::Result<T, GraphError>;
