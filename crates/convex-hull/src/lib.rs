// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! A pure-Rust exact-rational convex-polytope kernel.
//!
//! Every coordinate and facet inequality is an exact
//! rational ([`Q`] = [`flint_core::Rational`]), so a degenerate
//! incidence is **decided**, never rounded.
//!
//! # Scope
//!
//! * [`Polytope`] — a bounded convex polytope over `Q` in both
//!   representations (V: vertices, H: facet inequalities) with the
//!   vertex/facet incidence and face lattice.
//! * [`convex_hull`] — the hull of a finite rational point set.
//! * [`double_description`] — H ⇄ V by the double-description method
//!   ([`cone::extreme_rays`]).
//! * [`beneath_beyond()`] — the incremental beneath–beyond hull.
//! * [`vertex_enumeration`] (H→V) / [`facet_enumeration`] (V→H) — the two
//!   directions, dispatching to the algorithms above.
//! * [`FaceLattice`] — the face lattice; [`Polytope::f_vector`],
//!   [`Polytope::is_vertex`], [`Polytope::is_facet`].
//!
//! # Consumer
//!
//! `noether::geometry::Polytope` curates this kernel over the same shared
//! FLINT-backed rational values. The private PyO3 transport projects that
//! facade as `noether.geometry.Polytope` with `fractions.Fraction` coordinates.
//!
//! # Kernel discipline
//!
//! The algorithm layer is safe Rust and has no GPL runtime dependency. Exact
//! scalar arithmetic is delegated through `flint-core` to dynamically linked
//! FLINT. Zero `unsafe` lives in this crate — `#![forbid(unsafe_code)]` at the
//! crate root.
//!
//! # Quick example
//!
//! ```
//! use convex_hull::{convex_hull, q_point};
//!
//! // The unit triangle in the plane, plus an interior point that is
//! // silently absorbed.
//! let pts = vec![
//!     q_point(&[0, 0]),
//!     q_point(&[2, 0]),
//!     q_point(&[0, 2]),
//!     q_point(&[1, 1]), // interior — not a vertex
//! ];
//! let p = convex_hull(&pts).unwrap();
//! assert_eq!(p.dim(), 2);
//! assert_eq!(p.n_vertices(), 3);
//! assert_eq!(p.n_facets(), 3);
//! assert_eq!(p.f_vector(), vec![1, 3, 3, 1]); // (empty, verts, edges, whole)
//! ```

pub mod beneath_beyond;
pub mod cone;
pub mod error;
pub mod face_lattice;
pub mod linalg;
pub mod polytope;

#[cfg(test)]
mod tests;

/// The exact scalar: an arbitrary-precision rational.
pub type Q = flint_core::Rational;

pub use beneath_beyond::beneath_beyond;
pub use error::HullError;
pub use face_lattice::{Face, FaceLattice};
pub use polytope::{
    convex_hull, double_description, facet_enumeration, vertex_enumeration, Facet, Polytope,
};

/// Build a rational point from integer coordinates — a convenience for
/// callers and doctests.
pub fn q_point(coords: &[i64]) -> Vec<Q> {
    coords.iter().map(|&c| Q::from(c)).collect()
}

/// Build a single exact rational `num / den`.
///
/// # Panics
/// Panics if `den == 0`.
pub fn q(num: i64, den: i64) -> Q {
    assert!(den != 0, "zero denominator");
    Q::new(num.into(), den.into()).expect("denominator was checked nonzero")
}
