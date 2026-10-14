// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite strong Steiner strict omega-categories.
//!
//! The owning representation is a finite unital, strongly loop-free based
//! augmented directed complex.  Composite cells, mapping categories, and
//! homological constructions are derived from that one exact value.

mod canonical;
mod homotopy;
mod model;
mod nerve;

pub use canonical::SteinerCanonicalLabel;
pub use homotopy::{FiniteOmegaCategory, OrientedBasepoint};
pub use model::{
    AtomId, CategoryError, SignedBoundary, SteinerCategory, SteinerCell, MAX_ENUMERATED_CELLS,
};
pub use nerve::{
    CategoricalHomologyComponentSlice, CategoricalHomologyPresentation, StreetNerveSkeleton,
    StreetSimplex, MAX_CATEGORICAL_HOMOLOGY_PERMUTATIONS, MAX_STREET_NERVE_SIMPLICES_PER_DEGREE,
};
