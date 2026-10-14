// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! `lattice-enum` — pure-Rust **short-vector (SVP) /
//! closest-vector (CVP) enumeration** kernel.
//!
//! Given a symmetric **positive-definite integer Gram matrix** `G`, this
//! kernel enumerates the integer vectors `v ∈ Zⁿ` that are short (SVP) or
//! close to a target (CVP) under the `G`-inner product `⟨u, w⟩ = uᵀ G w`.
//!
//! # Scope
//!
//! * [`svp_enumerate`] — all `v` with `vᵀGv ≤ B`, one rep per ± pair.
//! * [`cvp_enumerate`] — integer `v` closest to a (rational) target.
//! * [`representation_vectors_collect`] — all `v` with `vᵀGv = k`
//!   (theta-series coefficient support).
//! * [`gaussian_heuristic_bound`] / [`enumeration_radius_set`] — Gaussian-
//!   heuristic sizing utilities.
//! * [`pruned_enumeration`] — linear (Gama–Nguyen–Regev) / extreme
//!   (Aono–Wang) pruned enumeration (crypto-scale path).
//! * [`schnorr_euchner::enumerate`] — the shared inner loop.
//!
//! # Correctness posture
//!
//! The exact FLINT-rational `LDLᵀ` GSO (see [`gso`]) drives the recursion:
//! pivot signs, partial norms, centers, and integer intervals are exact. Every
//! emitted candidate is independently re-verified in arbitrary-precision
//! FLINT integer arithmetic. Floating point is limited to optional Babai seeds
//! and Gaussian work estimates, so it cannot exclude a valid vector. The
//! kernel returns a typed [`EnumError`] on adversarial inputs.
//!
//! Noether's `linalg::Lattice` facade consumes this kernel.
//!
//! # Kernel discipline
//!
//! The algorithm crate uses `#![forbid(unsafe_code)]` and has no GPL runtime
//! dependency. Exact arithmetic is supplied by `flint-core` over dynamically
//! linked LGPL FLINT.

use flint_core::Integer;

pub mod cvp;
pub mod error;
pub mod gso;
pub mod pruned;
pub mod radius;
mod reduction;
pub mod schnorr_euchner;
pub mod svp;
pub mod theta;

#[cfg(test)]
mod tests;

pub use cvp::cvp_enumerate;
pub use error::{EnumError, EnumResult};
pub use pruned::{pruned_enumeration, PruningMode};
pub use radius::{enumeration_radius_set, gaussian_heuristic_bound};
pub use svp::svp_enumerate;
pub use theta::representation_vectors_collect;

/// Result of [`svp_enumerate`] — the short vectors of `G` up to a bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortVectors {
    /// The signed count: the total number of vectors including both members
    /// of every ± pair, so `count == 2 * vectors.len()`.
    pub count: Integer,
    /// The largest `vᵀGv` actually attained over the returned vectors. For
    /// the no-bound query this is `min(G)`. Zero when there are no vectors.
    pub max_norm: Integer,
    /// The requested norm bound. `None` denotes the minimum shell.
    pub bound: Option<Integer>,
    /// Whether the search proved that no qualifying `±` pair was omitted.
    ///
    /// A capped result is incomplete when the search encountered more than
    /// `max_count` qualifying pairs. The retained vectors remain a
    /// deterministic prefix.
    pub complete: bool,
    /// One representative per ± pair, canonical sign (first nonzero
    /// coordinate positive). Each is a length-`n` integer vector.
    pub vectors: Vec<Vec<i64>>,
}

impl ShortVectors {
    /// The number of ± pair representatives (`count / 2`).
    pub fn pair_count(&self) -> usize {
        self.vectors.len()
    }
}

/// Result of [`cvp_enumerate`] — the lattice vector(s) closest to a target.
///
/// The exact squared `G`-distance is the rational `distance_sq_num /
/// distance_sq_den`; `distance_sq_den` is the square of the target's common
/// denominator, so an integer (lattice-point) target reports
/// `distance_sq_den == 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosestVectors {
    /// Numerator of the exact minimal squared `G`-distance.
    pub distance_sq_num: Integer,
    /// Denominator of the exact minimal squared `G`-distance (`q²`).
    pub distance_sq_den: Integer,
    /// Every integer `v` attaining that minimal distance (there can be
    /// several — a target equidistant from a coset). Deterministically
    /// sorted.
    pub vectors: Vec<Vec<i64>>,
}
