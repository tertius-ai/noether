// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed failures for distributed matrix operations.

use std::fmt;

/// Failure from MPI initialization, layout, bounded materialization or
/// workspace, or distributed persistence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistributedError {
    /// MPI was already initialized outside this crate's explicit world.
    MpiAlreadyInitialized,
    /// The MPI implementation supplied weaker thread support than requested.
    InsufficientThreadSupport,
    /// A process rank is outside the active world.
    InvalidMpiRank {
        /// Rejected rank.
        rank: i32,
        /// Active communicator size.
        world_size: i32,
    },
    /// Tile or process-grid geometry is invalid.
    InvalidLayout(String),
    /// Matrix shapes are incompatible with the requested operation.
    ShapeMismatch(String),
    /// A row, column, or half-open range is outside the logical matrix.
    OutOfBounds(String),
    /// Operands do not retain the same initialized MPI world.
    UnequalWorlds,
    /// A checked size calculation overflowed.
    SizeOverflow,
    /// A checked heap reservation was rejected before the operation began.
    AllocationFailure {
        /// Requested bytes when that value was representable.
        requested_bytes: Option<usize>,
    },
    /// A bounded gather or root-buffer construction exceeds its byte ceiling.
    MaterializationLimit {
        /// Packed bytes required by the requested local value.
        required: usize,
        /// Caller-supplied maximum packed bytes.
        limit: usize,
    },
    /// A bounded distributed algorithm exceeds its per-process workspace ceiling.
    WorkspaceLimit {
        /// Estimated bytes required by the requested rank-local workspace.
        required: usize,
        /// Caller-supplied maximum rank-local workspace bytes.
        limit: usize,
    },
    /// The root process did not supply the exact declared input buffer.
    RootBuffer(String),
    /// Processes entered a collective with inconsistent arguments.
    CollectiveMismatch(&'static str),
    /// At least one process failed a collective filesystem or validation step.
    CollectiveFailure(String),
    /// A filesystem operation failed.
    Io(String),
    /// A manifest or packed tile is invalid.
    Manifest(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MpiAlreadyInitialized => write!(
                formatter,
                "MPI is already initialized; m4rst-distributed must initialize its explicit World"
            ),
            Self::InsufficientThreadSupport => write!(
                formatter,
                "the MPI implementation does not provide serialized thread support"
            ),
            Self::InvalidMpiRank { rank, world_size } => {
                write!(
                    formatter,
                    "MPI rank {rank} is outside world size {world_size}"
                )
            }
            Self::InvalidLayout(message) => {
                write!(formatter, "invalid distributed layout: {message}")
            }
            Self::ShapeMismatch(message) => write!(formatter, "shape mismatch: {message}"),
            Self::OutOfBounds(message) => write!(formatter, "out of bounds: {message}"),
            Self::UnequalWorlds => write!(formatter, "distributed operands use unequal MPI worlds"),
            Self::SizeOverflow => write!(formatter, "distributed matrix size overflows usize"),
            Self::AllocationFailure { requested_bytes } => match requested_bytes {
                Some(bytes) => write!(formatter, "allocator rejected a {bytes}-byte reservation"),
                None => write!(
                    formatter,
                    "allocator rejected a distributed workspace reservation"
                ),
            },
            Self::MaterializationLimit { required, limit } => write!(
                formatter,
                "materialization requires {required} bytes, exceeding the {limit}-byte limit"
            ),
            Self::WorkspaceLimit { required, limit } => write!(
                formatter,
                "distributed workspace requires {required} bytes, exceeding the {limit}-byte limit"
            ),
            Self::RootBuffer(message) => write!(formatter, "invalid root buffer: {message}"),
            Self::CollectiveMismatch(argument) => {
                write!(
                    formatter,
                    "MPI processes disagree on collective argument {argument}"
                )
            }
            Self::CollectiveFailure(message) => {
                write!(formatter, "collective operation failed: {message}")
            }
            Self::Io(message) => write!(formatter, "distributed filesystem error: {message}"),
            Self::Manifest(message) => write!(formatter, "invalid distributed manifest: {message}"),
        }
    }
}

impl std::error::Error for DistributedError {}
