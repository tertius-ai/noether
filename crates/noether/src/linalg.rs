// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Linear algebra over explicit coefficient contexts.

mod exact;
mod factorization;
mod field;
mod flint;
mod generic_matmul;
mod kernels;
mod lattice;
mod line_configuration;
mod line_realization;
mod line_symmetry;
mod matrix;
mod normal_form;
mod operator;
mod padic_normal_form;
mod semilinear;
mod sparse_matrix;
mod vector;

#[cfg(feature = "distributed")]
pub use crate::distributed::{
    active_mpi_runtime, DistributedError, DistributedMatrix, DistributedPLUQFactorization,
    MpiRuntimeFacts, MpiThreadSupport, World,
};
pub(crate) use exact::{canonical_congruence_columns, canonical_integer_columns};
pub use exact::{
    CongruenceInconsistencyCertificate, CongruenceSolutionModule, CongruenceSystemResult,
    IntegerInconsistencyCertificate, IntegerLinearSystemResult, IntegerSolutionModule,
};
pub use factorization::PLUQFactorization;
pub use field::SolutionSpace;
pub use lattice::{ClosestLatticeVectors, Lattice, LatticeEnumeration, LatticeError};
pub use line_configuration::{
    IntegralLineConfiguration, LineConfigurationError, MinimumVectorConfiguration,
};
pub use line_realization::{
    MinimumCompletion, MinimumCompletionCertificate, MinimumNonrealizabilityCertificate,
    MinimumRealization, MinimumRealizationDiagnostic,
};
pub use line_symmetry::{
    IntegralLineAutomorphismGroup, IntegralLineEquivalence, IntegralLineOrbitAction,
    IntegralLineSubgroup,
};
pub use matrix::{IndexError, LinalgError, Matrix};
pub use normal_form::{SmithNormalForm, SmithNormalFormWithTransforms, SmithVerification};
pub use operator::LinearOperator;
pub use padic_normal_form::PadicSmithInvariants;
pub use semilinear::{FrobeniusSemilinearMap, SemilinearMapError};
pub use sparse_matrix::SparseMatrix;
pub use vector::Vector;
