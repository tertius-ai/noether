// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse exact linear algebra over Z and F_p.
//!
//! Sparse-native counterpart to [`m4rst`] (dense F_2) and `pwvla`
//! (packed Z/p^n). The crate provides CSR storage, multiplication,
//! elimination, solve, Hermite and Smith forms, CRT, and prime selection.
//!
//! [`m4rst`]: https://github.com/tertius-ai/m4rst

#![deny(unsafe_code)]
#![deny(missing_docs)]
// Public algorithm documentation occasionally links to private helpers.
#![allow(rustdoc::private_intra_doc_links)]

pub mod crt;
pub mod elim;
pub mod hnf;
pub mod matmul;
pub mod prime_set;
pub mod snf;
pub mod solve;
pub mod storage;

pub use elim::RrefFp;
pub use flint_core::{Integer, ParseIntegerError};
pub use storage::{CsrFp, CsrZ, SparseRingStorage};
