// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse storage types and the COO ingestion path.
//!
//! Storage surface:
//! - [`CsrZ`] — CSR over `Z` (FLINT-backed [`crate::Integer`] entries).
//! - [`CsrFp`] — CSR over `F_p` for `p ≤ 2³¹` (`u32` entries).
//! - [`SparseRingStorage`] — the shape-and-size trait shared by both.
//!
//! See [`coo`] for the canonical CSR invariant that both constructors enforce.

pub mod coo;
pub mod csr_fp;
pub mod csr_z;

pub use coo::Triple;
pub use csr_fp::{CsrFp, MAX_PRIME};
pub use csr_z::CsrZ;
// Re-export the shared structural contract from the storage namespace.
pub use spar_core::SparseRingStorage;
