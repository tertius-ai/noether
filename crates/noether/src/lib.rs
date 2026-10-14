// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Noether's public Rust facade.
//!
//! Mathematical objects and algorithms live under nine owning domains. The
//! package root contains only cross-domain traits, errors, and release
//! metadata. Python bindings are implemented by a separate private transport
//! crate and are not a dependency of this library.

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(unused_crate_dependencies)]

#[cfg(test)]
use criterion as _;
#[cfg(test)]
use proptest as _;

mod ball;
mod coefficient_maps;
mod concurrency;
mod discovery;
mod error;
#[cfg(test)]
mod kernel_routes;
mod point_count;
mod public_surface;
mod scalar;
mod traits;
mod validity;

pub mod category;
pub mod combin;
pub mod commalg;
#[cfg(feature = "distributed")]
mod distributed;
pub mod geometry;
pub mod groups;
pub mod homalg;
pub mod linalg;
pub mod numthy;
pub mod topology;

pub use ball::{BallError, ComplexBall, ComplexBallField, RealBall, RealBallField};
pub use coefficient_maps::{
    CompositeCoefficientMap, CompositeCoefficientMapError, IdentityCoefficientMap,
    IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction, PrimeSubfieldEmbedding,
    PrimeSubfieldEmbeddingError,
};
pub use concurrency::{local_concurrency, LocalConcurrency, THREAD_BUDGET_ENV};
pub use discovery::{accepted_type_schemas, type_schema, TypeSchemaRegistration};
pub use error::{
    CoefficientMappingError, ContextMismatchError, DivisionByZeroError, InversionError,
    NoetherError, UnitError, UnsupportedOperationError, VerificationError,
};
pub use public_surface::{
    accepted_domains, accepted_operations, accepted_public_types, accepted_public_values,
    available_domains, available_operations, available_public_types, available_public_values,
};
pub use scalar::{Integer, IntegerRing, Rational, RationalField, Q, Z};
pub use traits::{
    BezoutIdentity, CheckpointableEnumeration, CoefficientFrobeniusContext, CoefficientMap,
    CommutativeRing, EuclideanDomain, Field, FiniteContext, IntegralDomain, PrecisionContext,
    PrincipalIdealDomain, Ring, SemanticType, TypeDescription, UnitInversion, ValuedRing,
};
pub use validity::{Assumption, MathematicalValidity};

/// The Noether release version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Implementation backends compiled into the public Rust facade.
///
/// This is release metadata, not an algorithm-selection API. Mathematical
/// operations dispatch internally and preserve the same public result type.
pub const COMPILED_BACKENDS: &[&str] = &[
    "convex-hull",
    "flint",
    "geom-core",
    "graph-core",
    "lattice-enum",
    "m4rst",
    #[cfg(feature = "distributed")]
    "m4rst-distributed",
    "m4rste",
    "mzs",
    "poly-core",
    "groebner-kernel",
    "perm-core",
    "pwvla",
    "sparb",
    "sparla",
];

/// Optional facade features compiled into this artifact.
pub const COMPILED_FEATURES: &[&str] = &[
    #[cfg(feature = "distributed")]
    "distributed",
];
