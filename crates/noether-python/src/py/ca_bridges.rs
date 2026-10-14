// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Private availability probes for optional commutative-algebra subprocesses.
//!
//! The Rust facade owns input translation, candidate parsing, exact
//! verification, and routine dispatch. The Python transport exposes no raw
//! evaluation primitive and therefore cannot turn backend stdout into a
//! mathematical result.

#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use pyo3::prelude::*;

/// Whether the msolve subprocess feature is compiled and its executable is on
/// `PATH`.
#[pyfunction]
pub fn engine_msolve_available() -> bool {
    #[cfg(feature = "msolve-subprocess")]
    {
        noether_bridges::msolve::msolve_available()
    }
    #[cfg(not(feature = "msolve-subprocess"))]
    {
        false
    }
}

/// Whether the Macaulay2 subprocess feature is compiled and its executable is
/// on `PATH`.
#[pyfunction]
pub fn engine_m2_available() -> bool {
    #[cfg(feature = "macaulay2-subprocess")]
    {
        noether_bridges::macaulay2::m2_available()
    }
    #[cfg(not(feature = "macaulay2-subprocess"))]
    {
        false
    }
}

/// Whether the Singular subprocess feature is compiled and its executable is
/// on `PATH`.
#[pyfunction]
pub fn engine_singular_available() -> bool {
    #[cfg(feature = "singular-subprocess")]
    {
        noether_bridges::singular::singular_available()
    }
    #[cfg(not(feature = "singular-subprocess"))]
    {
        false
    }
}
