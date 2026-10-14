// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! sparb — sparse interval (Arb ball) linear algebra.
//!
//! # Quick example
//!
//! ```
//! use sparb::{Arb, CsrArb};
//!
//! // A 2×2 sparse matrix at 64-bit working precision. One exact
//! // entry, one uncertain by ±0.01.
//! let a = CsrArb::from_coo_f64(
//!     2, 2, 64,
//!     &[(0, 0, 3.0, 0.0), (1, 1, 1.0, 0.01)],
//! );
//!
//! // Rigorous sparse matrix-vector product against an exact x.
//! let y = a.spmv(&[Arb::from_f64(2.0), Arb::from_f64(5.0)]);
//!
//! // Row 0 is exact: 3 · 2 = 6.
//! assert_eq!(y[0].midpoint(), 6.0);
//! assert_eq!(y[0].radius(64), 0.0);
//!
//! // Row 1 inherits the entry uncertainty: 1.0 · 5 = 5, with
//! // a radius ≥ 5 · 0.01 = 0.05.
//! assert_eq!(y[1].midpoint(), 5.0);
//! assert!(y[1].radius(64) >= 0.05);
//! ```
//!
//! The rigorous-ball counterpart to the sparse-exact [`sparla`] and
//! dense-`F_2` [`m4rst`] kernels. `sparb` owns CSR matrices over
//! rigorous real and complex ball enclosures.
//!
//! # Design
//!
//! sparb owns the **outer sparse layer** — CSR storage, COO
//! ingestion, SpMV orchestration — and delegates **scalar ball
//! arithmetic** to FLINT's `arb` through the workspace `flint-sys`
//! crate. FLINT is a *mandatory* link dependency: a ball matrix with
//! no ball arithmetic is nothing.
//!
//! Scalar ball arithmetic comes from `flint-core`, which owns the FFI boundary;
//! this crate contains no unsafe code.
//!
//! [`sparla`]: https://github.com/tertius-ai/sparla
//! [`m4rst`]: https://github.com/tertius-ai/m4rst

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
// Public algorithm documentation occasionally links to private helpers.
#![allow(rustdoc::private_intra_doc_links)]

pub mod acb;
pub mod ball;
pub mod spectrum;
pub mod spmv;
pub mod storage;

pub use acb::Acb;
pub use ball::Arb;
#[cfg(feature = "parallel")]
pub use spmv::par_spmv;
pub use spmv::spmv;
pub use storage::{CsrAcb, CsrArb, SparseRingStorage, Triple};
