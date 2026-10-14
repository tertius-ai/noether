// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse Gaussian elimination kernels.
//!
//! Current surface:
//! - [`gauss_fp::rref`] — sparse Gaussian elimination over `F_p` to
//!   reduced row echelon form, driving rank, kernel, image, RREF on
//!   [`CsrFp`].
//! - [`pivot::select_pivot_row`] — the shared pivot-row heuristic.
//! - [`reduce_z`] — exact integer row primitives used by normal forms.
//!
//! [`CsrFp`]: crate::storage::CsrFp

pub mod gauss_fp;
pub mod pivot;
pub mod reduce_z;

pub use gauss_fp::{rref, RrefFp};
