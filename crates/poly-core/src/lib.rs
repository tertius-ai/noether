// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed monomial, sparse polynomial, and fixed-truncation kernels.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod accumulator;
mod arena_polynomial;
pub mod f2;
mod monomial;
mod polynomial;
mod terms;
mod truncation;

pub use accumulator::PolynomialAccumulator;
pub use arena_polynomial::{ArenaPolynomial, PolynomialWorkspace};
pub use monomial::{
    ExponentOverflow, Monomial, MonomialArena, MonomialArenaError, MonomialArenaId, MonomialId,
    MonomialMarks, MonomialOrder,
};
pub use polynomial::{CoefficientRing, SparsePolynomial};
pub use terms::{Term, TermArena};
pub use truncation::{DenseMonomialRegion, RegionError, Truncation};
