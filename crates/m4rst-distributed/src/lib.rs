// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![deny(unused_crate_dependencies)]

//! System-MPI dense bit-packed linear algebra over `F_2`.
//!
//! [`DistributedMzd`] splits one logical matrix into word-aligned rectangular
//! tiles and assigns them block-cyclically over an explicit [`World`]. Each
//! rank stores only its ordinary [`m4rst::Mzd`] tiles. Manifests are
//! process-grid-independent, so the same files reopen under a different valid
//! MPI world size.
//!
//! ```no_run
//! use m4rst_distributed::{DistributedMzd, World};
//!
//! let world = World::initialize()?;
//! let matrix = DistributedMzd::random(&world, 1024, 1024, 1729)?;
//! let hash = matrix.logical_hash()?;
//! if world.is_root() {
//!     println!("{hash:016x}");
//! }
//! # Ok::<(), m4rst_distributed::DistributedError>(())
//! ```

mod error;
mod factorization;
mod layout;
mod manifest;
mod matrix;
mod random;
mod world;

pub use error::DistributedError;
pub use factorization::DistributedPLUQFactorization;
pub use layout::{Layout, ProcessGrid, TileCoord, DEFAULT_TILE_COLS, DEFAULT_TILE_ROWS};
pub use manifest::{MatrixManifest, TileRecord};
pub use matrix::{DistributedMzd, LocalTile};
pub use world::{
    active_mpi_runtime, ActiveMpiRuntime, CommunicationCounters, MatmulPhaseCounters,
    MpiThreadSupport, TimedPhaseCounters, World,
};
