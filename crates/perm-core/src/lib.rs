// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Safe-Rust permutation-group kernel for Noether.
//!
//! # Scope
//!
//! * **Type** — [`PermGroup`], a permutation group given by generators,
//!   carrying its base + strong generating set + stabilizer chain.
//! * **BSGS construction** — [`base_and_strong_generating_set`] runs
//!   either deterministic Schreier–Sims or the randomized
//!   (Monte-Carlo, verified) variant.
//! * **Order** — [`group_order`] reads the order off the transversal
//!   sizes.
//! * **Membership** — [`element_membership_test`] sifts a permutation
//!   down the stabilizer chain.
//! * **Orbit-stabilizer family** — [`orbit`], [`transversal`],
//!   [`schreier_tree`], [`stabilizer_chain`], [`pointwise_stabilizer`],
//!   [`orbit_stabilizer`], [`schreier_generators`].
//! * **Enumeration** — [`iterate_elements`], a lazy odometer over the
//!   product of transversals.
//! * **Sampling** — [`uniform_random_element`], exact uniform
//!   sampling via per-level transversal choice; takes an explicit
//!   `seed` and records it on the result. Seed expansion, bounded sampling,
//!   and test shuffling are owned here rather than delegated to changing
//!   distribution APIs.
//!
//! # Domain convention
//!
//! Permutations act on `0..degree`. A [`Perm`] is a length-`degree`
//! vector of `Point`s with `perm[i]` the image of `i`. All operations
//! act on the right: `(g * h)[i] = h[g[i]]`. This matches GAP's right
//! action convention and matches the Schreier–Sims literature.
//!
//! # Graceful failure
//!
//! Every fallible entry point returns [`Result<T, GroupError>`]. Mismatched
//! degree, invalid generator images, and wrong-degree membership queries map
//! to typed errors rather than panics.
//!
//! # Getting started
//!
//! ```
//! use perm_core::{PermGroup, Perm};
//!
//! // Symmetric group S_4 from the standard pair (1 2), (1 2 3 4).
//! let s4 = PermGroup::from_generators(
//!     4,
//!     &[
//!         Perm::from_images(&[1, 0, 2, 3]).unwrap(),
//!         Perm::from_images(&[1, 2, 3, 0]).unwrap(),
//!     ],
//! )
//! .unwrap();
//!
//! assert_eq!(s4.order(), num_to_big(24));
//! # fn num_to_big(n: u64) -> perm_core::BigUInt { perm_core::BigUInt::from(n) }
//! ```

mod backtrack;
mod bigint;
mod conjugacy;
mod error;
mod group;
mod perm;
mod random;
mod schreier;
mod structure;
mod subgroup;
mod subset;

pub use backtrack::{
    containment_up_to_conjugacy, normalizer, normalizer_in_symmetric_group, set_stabilizer,
    subgroup_conjugacy,
};
pub use bigint::BigUInt;
pub use conjugacy::{ConjugacyClass, CONJUGACY_ENUMERATION_CAP};
pub use error::{GroupError, NotAPermutationReason, Result};
pub use group::{
    base_and_strong_generating_set, element_membership_test, group_order, iterate_elements, orbit,
    orbit_stabilizer, pointwise_stabilizer, schreier_generators, schreier_tree, stabilizer_chain,
    transversal, uniform_random_element, ElementIter, OrbitStabilizer, PermGroup, RandomElement,
    SchreierGenerator, SchreierTree, StabilizerChain, Transversal,
};
pub use perm::{Perm, Point};
pub use structure::{
    derived_series_orders, derived_subgroup, is_nilpotent, is_perfect, is_solvable,
    lower_central_series_orders, normal_closure,
};
pub use subgroup::{
    coset_transversal, double_coset_triangle, double_cosets, is_subgroup, subdegrees,
    visit_double_coset_triangle, DoubleCoset, DOUBLE_COSET_AMBIENT_CAP, DOUBLE_COSET_QUOTIENT_CAP,
    SUBGROUP_ENUMERATION_CAP,
};
pub use subset::{orbit_of_subset, orbits_on_subsets, pair_orbits, PairOrbit, SubsetOrbit};
