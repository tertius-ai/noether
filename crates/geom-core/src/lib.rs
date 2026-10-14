// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Pure-Rust Euclidean geometry kernel.
//!
//! The geometric substrate under Noether's algebraic topology: point clouds
//! and metrics, circumspheres and minimum enclosing balls, exact orientation
//! and in-sphere predicates, Delaunay and regular triangulations, and the
//! Voronoi dual. It computes the *geometry of a sample*;
//! `noether::topology` consumes that and computes the
//! *topology of a filtration*. Simplicial complexes, simplicial sets,
//! persistence and discrete Morse theory stay on the other side of that line
//! — they are about abstract vertex sets, not embedded points.
//!
//! # Signs are exact; numbers need not be
//!
//! The asymmetry the whole kernel is built on. A triangulation is a
//! *combinatorial* object, and the algorithm that builds it branches on
//! predicate answers — is this point inside that circumsphere, is this
//! simplex positively oriented. Those answers must be mutually consistent or
//! the result is not a triangulation at all: overlapping cells, a cavity that
//! is not star-shaped, adjacency that does not close. Rounding error near
//! zero does not perturb the output, it corrupts it — and it does so on
//! exactly the inputs people have, since points on a grid, a circle or a
//! sphere are precisely the degenerate configurations.
//!
//! So [`predicate`] decides signs **exactly**, while the constructions that
//! hang off them ([`ball`]'s circumcentres and radii) stay in `f64`, where an
//! ULP of error moves a filtration value imperceptibly and nothing branches
//! on it.
//!
//! Exactness is nearly free because the predicates are *filtered*: evaluate
//! in `f64`, carry a bound on the accumulated rounding error, and escalate to
//! exact integer arithmetic only when the bound fails to separate the result
//! from zero. Generic input takes the fast path essentially always.
//!
//! # Layout
//!
//! - [`predicate`] — orientation and in-sphere, general dimension, filtered.
//! - [`cloud`] — [`PointCloud`], [`Metric`], [`DistanceMatrix`], affine hull.
//! - [`ball`] — circumspheres (plain and weighted), minimum enclosing balls.
//! - [`mod@delaunay`] — Delaunay and regular triangulations.
//!
//! # Quick example
//!
//! ```
//! use geom_core::predicate::{in_sphere, Sign};
//!
//! // The unit square's four corners are cocircular, so the fourth point is
//! // exactly *on* the circle through the first three — a degeneracy that
//! // f64 alone reports as a random sign near zero.
//! let square: [&[f64]; 3] = [
//!     &[0.0, 0.0],
//!     &[1.0, 0.0],
//!     &[1.0, 1.0],
//! ];
//! assert_eq!(in_sphere(&square, &[0.0, 1.0]), Sign::Zero);
//! ```

pub mod ball;
pub mod cloud;
pub mod delaunay;
pub mod predicate;

pub use ball::{circumsphere, min_enclosing_ball, Ball};
pub use cloud::{affine_rank, project_to_affine_hull, DistanceMatrix, Metric, PointCloud};
pub use delaunay::{delaunay, regular, Triangulation};
pub use predicate::{in_sphere, in_sphere_weighted, orientation, Sign};
