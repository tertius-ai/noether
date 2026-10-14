// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse storage types and the COO ingestion path.
//!
//! Storage surface:
//! - [`CsrArb`] — CSR over real balls (`Arb` entries) at a fixed
//!   working precision.
//! - [`CsrAcb`] — CSR over complex balls (`Acb` entries) at a fixed
//!   working precision.
//! - [`SparseRingStorage`] — the shape-and-size trait.
//!
//! See [`coo`] for the canonical CSR invariant that the
//! constructors enforce.

pub mod coo;
pub mod csr_acb;
pub mod csr_arb;

pub use coo::Triple;
pub use csr_acb::CsrAcb;
pub use csr_arb::CsrArb;
// Re-export the shared structural contract from the storage namespace.
pub use spar_core::SparseRingStorage;
