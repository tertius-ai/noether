// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Smith Normal Form kernels.
//!
//! [`direct::snf`] performs sparse column reduction followed by exact dense
//! Smith normal form on the pivot block. The dense subroutine remains private.

pub mod direct;

pub use direct::snf;
