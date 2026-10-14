// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Hermite Normal Form kernels.
//!
//! [`direct::hnf`] computes sparse HNF over `Z` by column-wise Euclidean row
//! reduction followed by sign normalization and reduction above pivots into
//! `[0, pivot)`. The implementation favors a small exact correctness floor
//! over a separate modular dispatch.

pub mod direct;

pub use direct::{hnf, hnf_with_transforms, rank};
