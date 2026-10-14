// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Chain complexes, derived invariants, and resolutions.
//!
//! [`ChainComplex<R>`] is the one complex type for homological and
//! cohomological indexing. Boundary and coboundary constructors normalize to
//! an internal lowering differential; the display convention only controls
//! which degree a caller sees.

use core::fmt;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::linalg::{LinalgError, Matrix, SolutionSpace, SparseMatrix, Vector};
use crate::{
    CoefficientMap, CoefficientMappingError, CommutativeRing, ContextMismatchError, Field, Integer,
    IntegerRing, MathematicalValidity, NoetherError, Ring, SemanticType, TypeDescription,
    VerificationError,
};

mod abelian;
mod associated_primes;
mod betti;
mod derived;
mod exterior;
mod finite_group_cohomology;
mod invariants;
mod local;
mod module;
mod module_groebner;
mod padic;
mod power_maps;
mod powers;
mod primary;
mod reduction;
mod resolution;

pub use abelian::{AbelianGroupError, FinitelyGeneratedAbelianGroup};
pub use associated_primes::ModuleAssociatedPrimeSet;
pub use betti::{GradedBettiTable, GradedBettiTableError};
pub use derived::{DerivedError, DerivedFunctor, DerivedModule, DerivedModuleHomomorphism};
pub use exterior::{ExteriorAlgebra, ExteriorAlgebraError, ExteriorElement};
pub use finite_group_cohomology::{
    FiniteAbelianGroupCohomology, FiniteAbelianGroupCohomologyError,
    FiniteAbelianGroupCohomologyPresentation, FiniteAbelianGroupIntegralCohomology,
    FiniteAbelianGroupIntegralCohomologyPresentation, FiniteCyclicGroupIntegralCohomology,
    FiniteCyclicGroupIntegralCohomologyError, FiniteCyclicGroupIntegralCohomologyPresentation,
    FiniteGroupCohomology, FiniteGroupCohomologyAlgebraGenerator,
    FiniteGroupCohomologyAlgebraTruncation, FiniteGroupCohomologyCompletePresentation,
    FiniteGroupCohomologyCompletionStrategy, FiniteGroupCohomologyElementaryAbelianRestriction,
    FiniteGroupCohomologyError, FiniteGroupCohomologyGeneratorOperationWitness,
    FiniteGroupCohomologySymondsParameterCertificate, FiniteGroupCohomologyTruncation,
    FiniteGroupIntegralCohomology, FiniteGroupIntegralCohomologyError,
    FiniteGroupIntegralCohomologyTruncation, FiniteGroupIntegralResolutionStrategy,
    FiniteGroupResolutionStrategy,
};
pub use invariants::{GradedModuleInvariants, ModuleInvariantError};
pub use local::{
    LocalModuleError, LocalModuleOrder, LocalModuleReduction, LocalModuleStandardBasis,
};
pub use module::{
    FreeModule, ImageSubmodule, KernelSubmodule, ModuleElement, ModuleError, ModuleHomomorphism,
    ModuleMonomialOrder, PresentedModule, PresentedModuleHomology, PresentedModuleHomomorphism,
    Submodule, SubmoduleMembership,
};
pub use module_groebner::{
    ModuleGroebnerBasis, ModuleGroebnerError, ModuleMonomial, ModuleReduction,
    StrongModuleGroebnerBasis, StrongModuleReduction, StrongModuleTerm,
};
pub use padic::{PadicCyclicSummand, PadicModuleInvariants, RankBounds};
pub use power_maps::{ModulePowerHomomorphism, PresentedModulePowerHomomorphism};
pub use powers::{FreeModulePower, ModulePowerError, ModulePowerKind, PresentedModulePower};
pub use primary::{
    ModulePrimaryComponent, ModulePrimaryDecomposition, ModulePrimaryDecompositionError,
};
pub use reduction::{
    ChainCancellation, ChainReduction, PadicChainCancellation, PadicChainReduction, UniversalUnit,
};
pub use resolution::{
    FreeResolution, ResolutionCompleteness, ResolutionError, ResolutionMinimality,
};

static CHAIN_COMPLEX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ChainComplex@1",
    rust_path: "noether::homalg::ChainComplex",
    python_path: Some("noether.homalg.ChainComplex"),
};

static HOMOLOGY_GROUP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/HomologyGroup@1",
    rust_path: "noether::homalg::HomologyGroup",
    python_path: Some("noether.homalg.HomologyGroup"),
};

static GRADED_HOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/GradedHomology@1",
    rust_path: "noether::homalg::GradedHomology",
    python_path: Some("noether.homalg.GradedHomology"),
};

// Pinned `homology_playbooks` benchmarks put direct Smith ahead below this
// dense matrix-cell budget and algebraic Morse reduction decisively ahead on
// the larger collapsible cases. The reduction still has an explicit public
// operation below this threshold.
const ALGEBRAIC_MORSE_MIN_MATRIX_CELLS: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegralInvariantStrategy {
    DirectSmith,
    AlgebraicMorseThenSmith,
}

/// Failure to construct a complex or compute a derived invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HomologyError {
    /// One differential belongs to another coefficient context.
    ContextMismatch {
        /// Displayed degree of the rejected differential.
        degree: i32,
        /// Underlying context error.
        error: ContextMismatchError,
    },
    /// Adjacent differentials imply inconsistent dimensions for one module.
    DimensionConflict {
        /// Degree whose dimension is inconsistent.
        degree: i32,
        /// Dimension implied by the first differential.
        expected: usize,
        /// Dimension implied by the next differential.
        actual: usize,
    },
    /// A degree could not be reindexed inside the public signed degree type.
    DegreeOverflow {
        /// Degree that could not be reindexed.
        degree: i32,
    },
    /// Consecutive differentials have nonzero composite.
    NotAComplex {
        /// Degree of the first map in `d_(degree-1) d_degree`.
        degree: i32,
    },
    /// A homology operation was requested from a coboundary presentation, or
    /// a cohomology operation from a boundary presentation.
    GradingConvention {
        /// Stable operation name.
        operation: &'static str,
        /// Constructor that gives the operation its intended meaning.
        required_constructor: &'static str,
    },
    /// Exact linear algebra failed.
    LinearAlgebra(LinalgError),
    /// Adjacent ranks exceed the dimension of the middle free module.
    RankInconsistency {
        /// Degree of the middle module.
        degree: i32,
        /// Rank of the middle free module.
        module_rank: usize,
        /// Rank of the outgoing differential.
        outgoing_rank: usize,
        /// Rank of the incoming differential.
        incoming_rank: usize,
    },
    /// Supplied exact rank evidence lies outside the capped Smith bounds.
    RankOutsideBounds {
        /// Displayed degree of the differential.
        degree: i32,
        /// Supplied exact rank.
        rank: usize,
        /// Certified lower bound from capped Smith reduction.
        lower: usize,
        /// Certified upper bound from capped Smith reduction.
        upper: usize,
    },
    /// A graded rank map omitted one represented differential.
    MissingRankEvidence {
        /// Displayed degree of the missing differential.
        degree: i32,
    },
    /// A graded rank map named no represented differential.
    UnexpectedRankEvidence {
        /// Displayed degree not present in the complex.
        degree: i32,
    },
    /// An exact integral lift does not reduce to the capped p-adic presentation.
    ExactLiftMismatch {
        /// Displayed differential degree, or `None` for grading/module data.
        degree: Option<i32>,
    },
    /// A computed result failed its own verifier.
    Verification(VerificationError),
}

impl fmt::Display for HomologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch { degree, error } => {
                write!(f, "differential in degree {degree}: {error}")
            }
            Self::DimensionConflict {
                degree,
                expected,
                actual,
            } => write!(
                f,
                "module in degree {degree} has conflicting dimensions {expected} and {actual}"
            ),
            Self::DegreeOverflow { degree } => {
                write!(f, "degree {degree} cannot be reindexed as an i32")
            }
            Self::NotAComplex { degree } => write!(
                f,
                "differentials do not form a complex: d_{} * d_{degree} is nonzero",
                degree - 1
            ),
            Self::GradingConvention {
                operation,
                required_constructor,
            } => write!(
                f,
                "{operation} requires a complex constructed with {required_constructor}"
            ),
            Self::RankInconsistency {
                degree,
                module_rank,
                outgoing_rank,
                incoming_rank,
            } => write!(
                f,
                "adjacent differential ranks {outgoing_rank} + {incoming_rank} exceed module rank {module_rank} in degree {degree}"
            ),
            Self::RankOutsideBounds {
                degree,
                rank,
                lower,
                upper,
            } => write!(
                f,
                "differential rank {rank} in degree {degree} lies outside certified bounds {lower}..={upper}"
            ),
            Self::MissingRankEvidence { degree } => {
                write!(f, "rank evidence is missing for differential degree {degree}")
            }
            Self::UnexpectedRankEvidence { degree } => {
                write!(f, "rank evidence names absent differential degree {degree}")
            }
            Self::ExactLiftMismatch { degree: Some(degree) } => write!(
                f,
                "exact integral differential in degree {degree} does not reduce to the p-adic presentation"
            ),
            Self::ExactLiftMismatch { degree: None } => f.write_str(
                "exact integral lift has different grading, modules, or represented differential degrees",
            ),
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::Verification(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HomologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch { error, .. } => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinalgError> for HomologyError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<VerificationError> for HomologyError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GradingConvention {
    Homological,
    Cohomological,
}

#[derive(Clone)]
enum StoredDifferential<R: Ring> {
    Dense(Matrix<R>),
    Sparse(SparseMatrix<R>),
}

impl<R: Ring> StoredDifferential<R> {
    fn ring(&self) -> &R {
        match self {
            Self::Dense(matrix) => matrix.ring(),
            Self::Sparse(matrix) => matrix.ring(),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Dense(matrix) => matrix.shape(),
            Self::Sparse(matrix) => matrix.shape(),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    fn to_dense(&self) -> Matrix<R> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix.to_dense(),
        }
    }

    fn rows(&self) -> std::vec::IntoIter<Vec<R::Element>> {
        match self {
            Self::Dense(matrix) => matrix.rows().collect::<Vec<_>>().into_iter(),
            Self::Sparse(matrix) => matrix.to_dense().rows().collect::<Vec<_>>().into_iter(),
        }
    }

    fn nonzero_entries(&self) -> Vec<(usize, usize, R::Element)> {
        match self {
            Self::Dense(matrix) => {
                let zero = matrix.ring().zero();
                matrix
                    .rows()
                    .enumerate()
                    .flat_map(|(row, values)| {
                        let zero = zero.clone();
                        values
                            .into_iter()
                            .enumerate()
                            .filter(move |(_, value)| value != &zero)
                            .map(move |(column, value)| (row, column, value))
                    })
                    .collect()
            }
            Self::Sparse(matrix) => {
                let mut entries = Vec::with_capacity(matrix.nnz());
                for row in 0..matrix.shape().0 {
                    for index in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        entries.push((
                            row,
                            matrix.column_indices()[index],
                            matrix.values()[index].clone(),
                        ));
                    }
                }
                entries
            }
        }
    }

    fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<StoredDifferential<M::Target>, CoefficientMappingError>
    where
        R: CommutativeRing + fmt::Display,
        M: CoefficientMap<Source = R>,
    {
        match self {
            Self::Dense(matrix) => matrix.map_coefficients(map).map(StoredDifferential::Dense),
            Self::Sparse(matrix) => matrix.map_coefficients(map).map(StoredDifferential::Sparse),
        }
    }
}

impl<R: Ring> PartialEq for StoredDifferential<R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left == right,
            (Self::Sparse(left), Self::Sparse(right)) => left == right,
            _ => self.to_dense() == other.to_dense(),
        }
    }
}

impl<R: Ring> Eq for StoredDifferential<R> {}

impl StoredDifferential<IntegerRing> {
    fn integral_smith_invariant_factors(&self) -> Vec<Integer> {
        match self {
            Self::Dense(matrix) => matrix.smith_normal_form().invariant_factors().to_vec(),
            Self::Sparse(matrix) => matrix.integral_smith_invariant_factors(),
        }
    }
}

/// A finite complex of free modules over one explicit coefficient context.
///
/// Internally every differential lowers degree. A boundary supplied at key
/// `n` is `d_n: C_n -> C_(n-1)`. A coboundary supplied at key `n` is
/// `delta^n: C^n -> C^(n+1)` and is stored at internal degree `-n`.
/// Adjacent matrix shapes determine the free-module dimensions.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use noether::homalg::ChainComplex;
/// use noether::linalg::Matrix;
/// use noether::{Rational, Q};
///
/// // Cellular chains of a circle: C_1 -> C_0 is the zero map.
/// let boundary = Matrix::from_rows(Q, [[Rational::from(0)]])?;
/// let circle = ChainComplex::from_boundaries(Q, BTreeMap::from([(1, boundary)]))?;
/// assert_eq!(circle.homology(0)?.dimension(), 1);
/// assert_eq!(circle.homology(1)?.dimension(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct ChainComplex<R: Ring> {
    ring: R,
    dimensions: BTreeMap<i32, usize>,
    differentials: BTreeMap<i32, StoredDifferential<R>>,
    display: GradingConvention,
    differential_ranks: BTreeMap<i32, Arc<OnceLock<usize>>>,
}

impl<R: Ring> PartialEq for ChainComplex<R> {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring
            && self.dimensions == other.dimensions
            && self.differentials == other.differentials
            && self.display == other.display
    }
}

impl<R: Ring> Eq for ChainComplex<R> {}

impl<R: Ring> ChainComplex<R> {
    /// Construct from lowering maps `d_n: C_n -> C_(n-1)`.
    pub fn from_boundaries(
        ring: R,
        boundaries: BTreeMap<i32, Matrix<R>>,
    ) -> Result<Self, HomologyError> {
        Self::from_lowering(
            ring,
            boundaries
                .into_iter()
                .map(|(degree, matrix)| (degree, StoredDifferential::Dense(matrix)))
                .collect(),
            GradingConvention::Homological,
        )
    }

    /// Construct from lowering maps retained in canonical sparse storage.
    pub fn from_sparse_boundaries(
        ring: R,
        boundaries: BTreeMap<i32, SparseMatrix<R>>,
    ) -> Result<Self, HomologyError> {
        Self::from_lowering(
            ring,
            boundaries
                .into_iter()
                .map(|(degree, matrix)| (degree, StoredDifferential::Sparse(matrix)))
                .collect(),
            GradingConvention::Homological,
        )
    }

    /// Construct from raising maps `delta^n: C^n -> C^(n+1)`.
    ///
    /// Degree negation turns these into the same internal lowering form used
    /// by [`Self::from_boundaries`].
    pub fn from_coboundaries(
        ring: R,
        coboundaries: BTreeMap<i32, Matrix<R>>,
    ) -> Result<Self, HomologyError> {
        let mut lowering = BTreeMap::new();
        for (degree, differential) in coboundaries {
            degree
                .checked_add(1)
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            let internal_degree = degree
                .checked_neg()
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            lowering.insert(internal_degree, StoredDifferential::Dense(differential));
        }
        Self::from_lowering(ring, lowering, GradingConvention::Cohomological)
    }

    /// Construct from raising maps retained in canonical sparse storage.
    pub fn from_sparse_coboundaries(
        ring: R,
        coboundaries: BTreeMap<i32, SparseMatrix<R>>,
    ) -> Result<Self, HomologyError> {
        let mut lowering = BTreeMap::new();
        for (degree, differential) in coboundaries {
            degree
                .checked_add(1)
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            let internal_degree = degree
                .checked_neg()
                .ok_or(HomologyError::DegreeOverflow { degree })?;
            lowering.insert(internal_degree, StoredDifferential::Sparse(differential));
        }
        Self::from_lowering(ring, lowering, GradingConvention::Cohomological)
    }

    /// Exact coefficient context.
    pub fn ring(&self) -> &R {
        &self.ring
    }

    /// Degrees carrying a nonzero explicitly inferred free module, in order.
    ///
    /// Cohomological presentations are returned in their displayed degrees.
    pub fn degrees(&self) -> Vec<i32> {
        match self.display {
            GradingConvention::Homological => self
                .dimensions
                .iter()
                .filter_map(|(degree, dimension)| (*dimension != 0).then_some(*degree))
                .collect(),
            GradingConvention::Cohomological => self
                .dimensions
                .iter()
                .rev()
                .filter_map(|(degree, dimension)| (*dimension != 0).then_some(-*degree))
                .collect(),
        }
    }

    /// Dimension of the free module in a displayed degree.
    pub fn module_dimension(&self, degree: i32) -> usize {
        self.internal_degree(degree)
            .ok()
            .and_then(|degree| self.dimensions.get(&degree).copied())
            .unwrap_or(0)
    }

    /// Storage layout of a represented differential in the displayed degree.
    pub fn differential_layout(&self, degree: i32) -> Option<&'static str> {
        self.internal_degree(degree)
            .ok()
            .and_then(|internal_degree| self.differentials.get(&internal_degree))
            .map(|differential| {
                if differential.is_sparse() {
                    "sparse"
                } else {
                    "dense"
                }
            })
    }

    /// Apply one checked coefficient map to every differential.
    ///
    /// The result preserves the boundary or coboundary convention, represented
    /// degrees, and every matrix shape. It is reconstructed and rechecked as a
    /// complex over the map's explicit target context.
    pub fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<ChainComplex<M::Target>, CoefficientMappingError>
    where
        R: CommutativeRing + fmt::Display,
        M: CoefficientMap<Source = R>,
    {
        if &self.ring != map.source() {
            return Err(ContextMismatchError {
                expected: map.source().to_string(),
                actual: self.ring.to_string(),
            }
            .into());
        }
        let differentials = self
            .differentials
            .iter()
            .map(|(&degree, differential)| Ok((degree, differential.map_coefficients(map)?)))
            .collect::<Result<BTreeMap<_, _>, CoefficientMappingError>>()?;
        ChainComplex::<M::Target>::from_lowering(map.target().clone(), differentials, self.display)
            .map_err(|error| CoefficientMappingError::Value(error.to_string()))
    }

    /// Differential supplied at one displayed degree.
    ///
    /// For a homological presentation this is `d_degree`; for a
    /// cohomological presentation it is `delta^degree`. Missing degrees return
    /// `None`, including degrees represented only by an inferred free module.
    /// Sparse storage is materialized only when this inspection method is
    /// called; algorithms continue to operate on the retained layout.
    pub fn differential(&self, degree: i32) -> Option<Matrix<R>> {
        self.internal_degree(degree)
            .ok()
            .and_then(|degree| self.differentials.get(&degree))
            .map(StoredDifferential::to_dense)
    }

    /// Recheck contexts, adjacent shapes, and `d^2 = 0`.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.verify_complex()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "chain-complex contexts, shapes, and d^2 = 0",
            })
    }

    fn from_lowering(
        ring: R,
        differentials: BTreeMap<i32, StoredDifferential<R>>,
        display: GradingConvention,
    ) -> Result<Self, HomologyError> {
        let mut dimensions = BTreeMap::new();
        for (degree, differential) in &differentials {
            if differential.ring() != &ring {
                return Err(HomologyError::ContextMismatch {
                    degree: *degree,
                    error: ContextMismatchError {
                        expected: R::TYPE_ID.to_owned(),
                        actual: "a different coefficient context".to_owned(),
                    },
                });
            }
            let lower_degree = degree
                .checked_sub(1)
                .ok_or(HomologyError::DegreeOverflow { degree: *degree })?;
            record_dimension(&mut dimensions, *degree, differential.shape().1)?;
            record_dimension(&mut dimensions, lower_degree, differential.shape().0)?;
        }
        let differential_ranks = differentials
            .keys()
            .map(|&degree| (degree, Arc::new(OnceLock::new())))
            .collect();
        let result = Self {
            ring,
            dimensions,
            differentials,
            display,
            differential_ranks,
        };
        result.verify_complex()?;
        Ok(result)
    }

    fn internal_degree(&self, displayed_degree: i32) -> Result<i32, HomologyError> {
        match self.display {
            GradingConvention::Homological => Ok(displayed_degree),
            GradingConvention::Cohomological => {
                displayed_degree
                    .checked_neg()
                    .ok_or(HomologyError::DegreeOverflow {
                        degree: displayed_degree,
                    })
            }
        }
    }

    fn verify_complex(&self) -> Result<(), HomologyError> {
        let mut dimensions = BTreeMap::new();
        for (degree, differential) in &self.differentials {
            if differential.ring() != &self.ring {
                return Err(HomologyError::ContextMismatch {
                    degree: *degree,
                    error: ContextMismatchError {
                        expected: R::TYPE_ID.to_owned(),
                        actual: "a different coefficient context".to_owned(),
                    },
                });
            }
            let lower_degree = degree
                .checked_sub(1)
                .ok_or(HomologyError::DegreeOverflow { degree: *degree })?;
            record_dimension(&mut dimensions, *degree, differential.shape().1)?;
            record_dimension(&mut dimensions, lower_degree, differential.shape().0)?;
            if let Some(lower) = self.differentials.get(&lower_degree) {
                if !differentials_compose_to_zero(lower, differential)? {
                    return Err(HomologyError::NotAComplex { degree: *degree });
                }
            }
        }
        if dimensions != self.dimensions {
            return Err(HomologyError::Verification(
                VerificationError::InvalidWitness {
                    claim: "stored module dimensions equal differential dimensions",
                },
            ));
        }
        Ok(())
    }
}

fn differentials_compose_to_zero<R: Ring>(
    left: &StoredDifferential<R>,
    right: &StoredDifferential<R>,
) -> Result<bool, LinalgError> {
    match (left, right) {
        (StoredDifferential::Sparse(left), StoredDifferential::Sparse(right)) => {
            Ok(left.matmul(right)?.nnz() == 0)
        }
        _ => {
            let left = left.to_dense();
            let right = right.to_dense();
            let composite = left.matmul(&right)?;
            Ok(composite == Matrix::zeros(left.ring().clone(), left.shape().0, right.shape().1))
        }
    }
}

impl<R: Field> ChainComplex<R> {
    fn differential_rank(&self, degree: i32) -> Result<usize, HomologyError> {
        let Some(differential) = self.differentials.get(&degree) else {
            return Ok(0);
        };
        let cache = self
            .differential_ranks
            .get(&degree)
            .expect("every stored differential has a rank cache");
        if let Some(&rank) = cache.get() {
            return Ok(rank);
        }
        let rank = differential.to_dense().rank()?;
        let _ = cache.set(rank);
        Ok(rank)
    }

    /// Return the Betti number in one displayed degree.
    ///
    /// This is the invariants-only path: it computes
    /// `dim(C_n) - rank(d_n) - rank(d_(n+1))` without constructing cycle,
    /// boundary, or quotient bases.  For a boundary presentation it is the
    /// dimension of homology; for a coboundary presentation it is the
    /// dimension of cohomology.
    pub fn betti_number(&self, degree: i32) -> Result<usize, HomologyError> {
        let internal_degree = self.internal_degree(degree)?;
        let module_rank = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let outgoing_rank = self.differential_rank(internal_degree)?;
        let incoming_rank = match internal_degree.checked_add(1) {
            Some(incoming_degree) => self.differential_rank(incoming_degree)?,
            None => 0,
        };
        quotient_rank(degree, module_rank, outgoing_rank, incoming_rank)
    }

    /// Compute `H_n = ker(d_n) / im(d_(n+1))` over an exact field.
    pub fn homology(&self, degree: i32) -> Result<HomologyGroup<R>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.homology_at(degree, degree, false)
    }

    /// Compute `H^n = ker(delta^n) / im(delta^(n-1))` over an exact field.
    pub fn cohomology(&self, degree: i32) -> Result<HomologyGroup<R>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let internal_degree = self.internal_degree(degree)?;
        self.homology_at(internal_degree, degree, true)
    }

    /// Compute homology in every nonzero explicitly inferred degree.
    pub fn graded_homology(&self) -> Result<GradedHomology<R>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        let groups = self
            .degrees()
            .into_iter()
            .map(|degree| self.homology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let result = GradedHomology {
            ring: self.ring.clone(),
            groups,
        };
        result.verify()?;
        Ok(result)
    }

    /// Compute cohomology in every explicitly inferred cochain degree.
    pub fn graded_cohomology(&self) -> Result<GradedHomology<R>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let groups = self
            .degrees()
            .into_iter()
            .map(|degree| self.cohomology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let result = GradedHomology {
            ring: self.ring.clone(),
            groups,
        };
        result.verify()?;
        Ok(result)
    }

    fn homology_at(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
        cohomological: bool,
    ) -> Result<HomologyGroup<R>, HomologyError> {
        let dimension = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let lower_dimension = internal_degree
            .checked_sub(1)
            .and_then(|degree| self.dimensions.get(&degree).copied())
            .unwrap_or(0);
        let upper_dimension = internal_degree
            .checked_add(1)
            .and_then(|degree| self.dimensions.get(&degree).copied())
            .unwrap_or(0);
        let outgoing = self
            .differentials
            .get(&internal_degree)
            .map(StoredDifferential::to_dense)
            .unwrap_or_else(|| Matrix::zeros(self.ring.clone(), lower_dimension, dimension));
        let incoming = internal_degree
            .checked_add(1)
            .and_then(|degree| self.differentials.get(&degree))
            .map(StoredDifferential::to_dense)
            .unwrap_or_else(|| Matrix::zeros(self.ring.clone(), dimension, upper_dimension));
        // `ChainComplex` already established d² = 0, and the RREF
        // construction itself gives a complete kernel basis.  The public
        // `Matrix::kernel` operation performs a deliberately independent deep
        // recheck; doing that here would repeat the dominant elimination before
        // the homology quotient is even formed.
        let (cycles, free_coordinates) = outgoing.kernel_from_elimination_with_free_columns()?;
        let boundaries = incoming.image()?;
        let representatives =
            quotient_representatives(&self.ring, &cycles, &boundaries, &free_coordinates)?;
        let free_rank = representatives.len();
        // The chain-complex constructor has already checked d² = 0.  RREF
        // constructs the complete cycle and boundary spaces, and
        // `quotient_representatives` checks their exact quotient. Do not
        // immediately repeat all three eliminations through the public deep
        // `verify()` operation; callers can invoke it when they need an
        // independent recomputation of the returned witness.
        Ok(HomologyGroup {
            ring: self.ring.clone(),
            degree: displayed_degree,
            cohomological,
            outgoing,
            incoming,
            cycles: Some(cycles),
            boundaries: Some(boundaries),
            representatives,
            free_rank,
            torsion_coefficients: Vec::new(),
            validity: MathematicalValidity::unconditional(),
        })
    }
}

impl ChainComplex<IntegerRing> {
    /// Compute only the canonical finitely generated abelian group underlying
    /// `H_n`, without retaining adjacent differentials or lattice witnesses.
    ///
    /// This is the lean integral analogue of [`ChainComplex::betti_number`].
    /// It computes the two adjacent Smith ranks and the incoming invariant
    /// factors, then returns only the decomposition of the resulting group.
    pub fn homology_invariants(
        &self,
        degree: i32,
    ) -> Result<FinitelyGeneratedAbelianGroup, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology_invariants",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.integral_invariants_at(degree, degree)
    }

    /// Compute lean integral homology invariants in every nonzero chain degree.
    ///
    /// Each differential is Smith-reduced exactly once. The result is keyed by
    /// homological degree and retains neither differentials nor lattice
    /// witnesses. Use [`Self::graded_homology`] when independently verifiable
    /// source-retaining groups are required.
    pub fn graded_homology_invariants(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology_invariants",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.graded_integral_invariants()
    }

    /// Compute only the canonical finitely generated abelian group underlying
    /// integral cohomology, without retaining adjacent coboundaries.
    pub fn cohomology_invariants(
        &self,
        degree: i32,
    ) -> Result<FinitelyGeneratedAbelianGroup, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology_invariants",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let internal_degree = self.internal_degree(degree)?;
        self.integral_invariants_at(internal_degree, degree)
    }

    /// Compute lean integral cohomology invariants in every nonzero degree.
    ///
    /// Each stored coboundary is Smith-reduced exactly once. The result is
    /// keyed by displayed cohomological degree.
    pub fn graded_cohomology_invariants(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology_invariants",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        self.graded_integral_invariants()
    }

    /// Compute `H_n = ker(d_n) / im(d_(n+1))` as a finitely generated
    /// abelian group.
    ///
    /// This uses the fast Smith forms without transformation matrices.  For a
    /// chain complex of free abelian groups, the free rank follows from the
    /// two adjacent ranks, while the torsion subgroup is the torsion subgroup
    /// of `coker(d_(n+1))` because `im(d_n)` is free.
    pub fn homology(&self, degree: i32) -> Result<HomologyGroup<IntegerRing>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "homology",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        self.integral_homology_at(degree, degree, false)
    }

    /// Compute integral cohomology from a coboundary presentation.
    pub fn cohomology(&self, degree: i32) -> Result<HomologyGroup<IntegerRing>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "cohomology",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let internal_degree = self.internal_degree(degree)?;
        self.integral_homology_at(internal_degree, degree, true)
    }

    /// Compute integral homology in every nonzero chain degree.
    pub fn graded_homology(&self) -> Result<GradedHomology<IntegerRing>, HomologyError> {
        if self.display != GradingConvention::Homological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_homology",
                required_constructor: "ChainComplex::from_boundaries",
            });
        }
        let groups = self
            .degrees()
            .into_iter()
            .map(|degree| self.homology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let result = GradedHomology {
            ring: self.ring,
            groups,
        };
        result.verify()?;
        Ok(result)
    }

    /// Compute integral cohomology in every nonzero cochain degree.
    pub fn graded_cohomology(&self) -> Result<GradedHomology<IntegerRing>, HomologyError> {
        if self.display != GradingConvention::Cohomological {
            return Err(HomologyError::GradingConvention {
                operation: "graded_cohomology",
                required_constructor: "ChainComplex::from_coboundaries",
            });
        }
        let groups = self
            .degrees()
            .into_iter()
            .map(|degree| self.cohomology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let result = GradedHomology {
            ring: self.ring,
            groups,
        };
        result.verify()?;
        Ok(result)
    }

    fn integral_homology_at(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
        cohomological: bool,
    ) -> Result<HomologyGroup<IntegerRing>, HomologyError> {
        let invariants = self.integral_invariants_at(internal_degree, displayed_degree)?;
        let module_rank = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let lower_rank = internal_degree
            .checked_sub(1)
            .and_then(|degree| self.dimensions.get(&degree).copied())
            .unwrap_or(0);
        let upper_rank = internal_degree
            .checked_add(1)
            .and_then(|degree| self.dimensions.get(&degree).copied())
            .unwrap_or(0);
        let outgoing = self
            .differentials
            .get(&internal_degree)
            .map(StoredDifferential::to_dense)
            .unwrap_or_else(|| Matrix::zeros(self.ring, lower_rank, module_rank));
        let incoming = internal_degree
            .checked_add(1)
            .and_then(|degree| self.differentials.get(&degree))
            .map(StoredDifferential::to_dense)
            .unwrap_or_else(|| Matrix::zeros(self.ring, module_rank, upper_rank));
        // The lean calculation above establishes the invariant factors and
        // ranks. This richer result additionally retains the adjacent maps so
        // its public verifier can recompute the claim independently.
        // The public verifier deliberately recomputes both forms and d²; do
        // not double the dominant work on every successful construction.
        Ok(HomologyGroup::from_integral_invariants(
            displayed_degree,
            cohomological,
            outgoing,
            incoming,
            invariants.free_rank,
            invariants.torsion_coefficients,
        ))
    }

    fn integral_invariants_at(
        &self,
        internal_degree: i32,
        displayed_degree: i32,
    ) -> Result<FinitelyGeneratedAbelianGroup, HomologyError> {
        let module_rank = self.dimensions.get(&internal_degree).copied().unwrap_or(0);
        let outgoing = self.differentials.get(&internal_degree).map(|outgoing| {
            IntegralDifferentialInvariants::from_factors(
                &outgoing.integral_smith_invariant_factors(),
            )
        });
        let incoming = internal_degree
            .checked_add(1)
            .and_then(|degree| self.differentials.get(&degree))
            .map(|incoming| {
                IntegralDifferentialInvariants::from_factors(
                    &incoming.integral_smith_invariant_factors(),
                )
            });
        integral_group_from_adjacent(
            displayed_degree,
            module_rank,
            outgoing.as_ref(),
            incoming.as_ref(),
        )
    }

    fn graded_integral_invariants(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
        if self.integral_invariant_strategy() == IntegralInvariantStrategy::DirectSmith {
            return self.graded_integral_invariants_direct();
        }
        let reduction = self.algebraic_morse_reduction()?;
        if reduction.removed_pairs() == 0 {
            return self.graded_integral_invariants_direct();
        }

        let mut reduced = reduction.reduced().graded_integral_invariants_direct()?;
        for degree in self.degrees() {
            reduced
                .entry(degree)
                .or_insert_with(|| FinitelyGeneratedAbelianGroup::from_invariants(0, Vec::new()));
        }
        Ok(reduced)
    }

    fn integral_invariant_strategy(&self) -> IntegralInvariantStrategy {
        let matrix_cells = self.differentials.values().fold(0usize, |total, matrix| {
            let (rows, columns) = matrix.shape();
            total.saturating_add(rows.saturating_mul(columns))
        });
        if matrix_cells < ALGEBRAIC_MORSE_MIN_MATRIX_CELLS {
            return IntegralInvariantStrategy::DirectSmith;
        }
        let zero = Integer::zero();
        let one = Integer::one();
        let negative_one = Integer::from(-1);
        let mut has_universal_unit = false;
        let mut every_differential_is_monomial = true;
        for matrix in self.differentials.values() {
            let rows = matrix.rows().collect::<Vec<_>>();
            let mut column_nonzeros = vec![0usize; matrix.shape().1];
            for row in &rows {
                let row_nonzeros = row.iter().filter(|value| *value != &zero).count();
                every_differential_is_monomial &= row_nonzeros <= 1;
                for (column, value) in row.iter().enumerate() {
                    if value != &zero {
                        column_nonzeros[column] += 1;
                    }
                    has_universal_unit |= value == &one || value == &negative_one;
                }
            }
            every_differential_is_monomial &= column_nonzeros.into_iter().all(|count| count <= 1);
        }
        if has_universal_unit && !every_differential_is_monomial {
            IntegralInvariantStrategy::AlgebraicMorseThenSmith
        } else {
            IntegralInvariantStrategy::DirectSmith
        }
    }

    fn graded_integral_invariants_direct(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
        let differentials = self
            .differentials
            .iter()
            .map(|(&degree, differential)| {
                (
                    degree,
                    IntegralDifferentialInvariants::from_factors(
                        &differential.integral_smith_invariant_factors(),
                    ),
                )
            })
            .collect();
        graded_integral_invariants_from_data(&self.dimensions, &differentials, self.display)
    }
}

#[derive(Clone, Debug)]
struct IntegralDifferentialInvariants {
    rank: usize,
    torsion_coefficients: Vec<Integer>,
}

impl IntegralDifferentialInvariants {
    fn from_factors(factors: &[Integer]) -> Self {
        let one = Integer::one();
        Self {
            rank: factors.len(),
            torsion_coefficients: factors
                .iter()
                .filter(|factor| *factor != &one)
                .cloned()
                .collect(),
        }
    }
}

fn integral_group_from_adjacent(
    degree: i32,
    module_rank: usize,
    outgoing: Option<&IntegralDifferentialInvariants>,
    incoming: Option<&IntegralDifferentialInvariants>,
) -> Result<FinitelyGeneratedAbelianGroup, HomologyError> {
    let outgoing_rank = outgoing.map_or(0, |data| data.rank);
    let incoming_rank = incoming.map_or(0, |data| data.rank);
    let free_rank = quotient_rank(degree, module_rank, outgoing_rank, incoming_rank)?;
    Ok(FinitelyGeneratedAbelianGroup::from_invariants(
        free_rank,
        incoming.map_or_else(Vec::new, |data| data.torsion_coefficients.clone()),
    ))
}

fn graded_integral_invariants_from_data(
    dimensions: &BTreeMap<i32, usize>,
    differentials: &BTreeMap<i32, IntegralDifferentialInvariants>,
    display: GradingConvention,
) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
    dimensions
        .iter()
        .filter(|(_, module_rank)| **module_rank != 0)
        .map(|(&internal_degree, &module_rank)| {
            let displayed_degree = match display {
                GradingConvention::Homological => internal_degree,
                GradingConvention::Cohomological => {
                    internal_degree
                        .checked_neg()
                        .ok_or(HomologyError::DegreeOverflow {
                            degree: internal_degree,
                        })?
                }
            };
            let incoming = internal_degree
                .checked_add(1)
                .and_then(|degree| differentials.get(&degree));
            let group = integral_group_from_adjacent(
                displayed_degree,
                module_rank,
                differentials.get(&internal_degree),
                incoming,
            )?;
            Ok((displayed_degree, group))
        })
        .collect()
}

/// Compute one integral homology group from trusted adjacent sparse boundaries.
///
/// The caller owns construction-time verification that the adjacent maps
/// compose to zero.
pub(crate) fn integral_homology_invariants_from_sparse_boundaries(
    degree: i32,
    module_rank: usize,
    outgoing: &SparseMatrix<IntegerRing>,
    incoming: &SparseMatrix<IntegerRing>,
) -> Result<FinitelyGeneratedAbelianGroup, HomologyError> {
    let outgoing = IntegralDifferentialInvariants {
        rank: outgoing.integral_rank(),
        torsion_coefficients: Vec::new(),
    };
    let incoming =
        IntegralDifferentialInvariants::from_factors(&incoming.integral_smith_invariant_factors());
    integral_group_from_adjacent(degree, module_rank, Some(&outgoing), Some(&incoming))
}

/// Homalg-owned invariants-only route for a trusted sparse boundary presentation.
///
/// The caller owns construction-time verification that adjacent boundaries
/// compose to zero; this route shares the homological quotient calculation
/// with [`ChainComplex<IntegerRing>`] while preserving sparse storage.
pub(crate) fn graded_homology_invariants_from_sparse_boundaries(
    boundaries: &BTreeMap<i32, SparseMatrix<IntegerRing>>,
) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, HomologyError> {
    let mut dimensions = BTreeMap::new();
    let mut differentials = BTreeMap::new();
    for (&degree, differential) in boundaries {
        let lower_degree = degree
            .checked_sub(1)
            .ok_or(HomologyError::DegreeOverflow { degree })?;
        record_dimension(&mut dimensions, degree, differential.shape().1)?;
        record_dimension(&mut dimensions, lower_degree, differential.shape().0)?;
        differentials.insert(
            degree,
            IntegralDifferentialInvariants::from_factors(
                &differential.integral_smith_invariant_factors(),
            ),
        );
    }
    graded_integral_invariants_from_data(
        &dimensions,
        &differentials,
        GradingConvention::Homological,
    )
}

impl<R: Ring> fmt::Debug for ChainComplex<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChainComplex")
            .field("ring_type", &R::TYPE_ID)
            .field("grading", &self.display)
            .field("degrees", &self.degrees())
            .field("differentials", &self.differentials.len())
            .finish()
    }
}

impl<R: Ring> fmt::Display for ChainComplex<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grading = match self.display {
            GradingConvention::Homological => "chain",
            GradingConvention::Cohomological => "cochain",
        };
        write!(
            f,
            "ChainComplex<{}>({grading}, degrees={}, differentials={})",
            R::TYPE_ID,
            self.dimensions.len(),
            self.differentials.len()
        )
    }
}

impl<R: Ring> SemanticType for ChainComplex<R> {
    const TYPE_ID: &'static str = "noether/homalg/ChainComplex@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = vec![match self.display {
            GradingConvention::Homological => 0,
            GradingConvention::Cohomological => 1,
        }];
        push_frame(&mut output, &self.ring.canonical_bytes()?)?;
        output.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for (degree, dimension) in &self.dimensions {
            output.extend_from_slice(&degree.to_be_bytes());
            output.extend_from_slice(&(*dimension as u64).to_be_bytes());
        }
        output.extend_from_slice(&(self.differentials.len() as u64).to_be_bytes());
        for (degree, differential) in &self.differentials {
            output.extend_from_slice(&degree.to_be_bytes());
            push_frame(&mut output, &differential.to_dense().canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CHAIN_COMPLEX_DESCRIPTION
    }
}

/// A checked single-degree homology or cohomology group.
///
/// Over a field this is a vector space with complete cycle and boundary
/// spaces plus deterministic quotient representatives.  Over the integers it
/// is a finitely generated abelian group recorded by its free rank and
/// positive Smith invariant factors.  Both retain the adjacent differentials
/// from which their claims can be recomputed.
#[derive(Clone, PartialEq, Eq)]
pub struct HomologyGroup<R: Ring> {
    ring: R,
    degree: i32,
    cohomological: bool,
    outgoing: Matrix<R>,
    incoming: Matrix<R>,
    cycles: Option<SolutionSpace<R>>,
    boundaries: Option<SolutionSpace<R>>,
    representatives: Vec<Vector<R>>,
    free_rank: usize,
    torsion_coefficients: Vec<Integer>,
    validity: MathematicalValidity,
}

impl<R: Ring> HomologyGroup<R> {
    /// Displayed homological or cohomological degree.
    pub fn degree(&self) -> i32 {
        self.degree
    }

    /// Whether this result was computed from a coboundary presentation.
    pub fn is_cohomology(&self) -> bool {
        self.cohomological
    }

    /// Coefficient context.
    pub fn coefficient_context(&self) -> &R {
        &self.ring
    }

    /// Rank of the free part.
    ///
    /// Over a field this is the vector-space dimension.  Over `Z` it is the
    /// number of infinite cyclic summands.
    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    /// Positive nonunit invariant factors of the torsion subgroup.
    ///
    /// These are empty over a field.  Over `Z`, a factor `d` denotes one
    /// summand `Z/d`, and the factors occur in divisibility order.
    pub fn torsion_coefficients(&self) -> &[Integer] {
        &self.torsion_coefficients
    }

    /// Mathematical assumptions under which the result is valid.
    pub fn validity(&self) -> &MathematicalValidity {
        &self.validity
    }

    fn verify_common(&self) -> Result<(), VerificationError> {
        if self.outgoing.ring() != &self.ring
            || self.incoming.ring() != &self.ring
            || self
                .cycles
                .as_ref()
                .is_some_and(|cycles| cycles.ring() != &self.ring)
            || self
                .boundaries
                .as_ref()
                .is_some_and(|boundaries| boundaries.ring() != &self.ring)
            || self
                .representatives
                .iter()
                .any(|representative| representative.ring() != &self.ring)
        {
            return Err(VerificationError::ContextMismatch(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "a homology witness over another context".to_owned(),
            }));
        }
        let composite = self.outgoing.matmul(&self.incoming).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "adjacent homology differentials are composable",
            }
        })?;
        if composite
            != Matrix::zeros(
                self.ring.clone(),
                self.outgoing.shape().0,
                self.incoming.shape().1,
            )
        {
            return Err(VerificationError::InvalidWitness {
                claim: "adjacent homology differentials compose to zero",
            });
        }
        if !self.validity.is_unconditional() {
            return Err(VerificationError::InvalidWitness {
                claim: "exact additive homology is unconditional",
            });
        }
        Ok(())
    }
}

impl HomologyGroup<IntegerRing> {
    pub(crate) fn from_integral_invariants(
        degree: i32,
        cohomological: bool,
        outgoing: Matrix<IntegerRing>,
        incoming: Matrix<IntegerRing>,
        free_rank: usize,
        torsion_coefficients: Vec<Integer>,
    ) -> Self {
        Self {
            ring: IntegerRing,
            degree,
            cohomological,
            outgoing,
            incoming,
            cycles: None,
            boundaries: None,
            representatives: Vec::new(),
            free_rank,
            torsion_coefficients,
            validity: MathematicalValidity::unconditional(),
        }
    }
}

impl<R: Field> HomologyGroup<R> {
    /// Dimension over the coefficient field.
    pub fn dimension(&self) -> usize {
        self.free_rank
    }

    /// Complete kernel representing the cycle space.
    pub fn cycles(&self) -> &SolutionSpace<R> {
        self.cycles
            .as_ref()
            .expect("field homology always carries its cycle space")
    }

    /// Complete image representing the boundary space.
    pub fn boundaries(&self) -> &SolutionSpace<R> {
        self.boundaries
            .as_ref()
            .expect("field homology always carries its boundary space")
    }

    /// Deterministic cycle representatives forming a basis modulo boundaries.
    pub fn representatives(&self) -> &[Vector<R>] {
        &self.representatives
    }

    /// Recompute adjacent kernels/images and check the quotient basis.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.verify_common()?;
        let cycles = self
            .cycles
            .as_ref()
            .ok_or(VerificationError::InvalidWitness {
                claim: "field homology carries a cycle space",
            })?;
        let boundaries = self
            .boundaries
            .as_ref()
            .ok_or(VerificationError::InvalidWitness {
                claim: "field homology carries a boundary space",
            })?;
        if !self.torsion_coefficients.is_empty() || self.free_rank != self.representatives.len() {
            return Err(VerificationError::InvalidWitness {
                claim: "field homology has a dimension and no torsion invariants",
            });
        }
        cycles.verify()?;
        boundaries.verify()?;
        let expected_cycles =
            self.outgoing
                .kernel()
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "cycle space is recomputable",
                })?;
        let expected_boundaries =
            self.incoming
                .image()
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "boundary space is recomputable",
                })?;
        if *cycles != expected_cycles || *boundaries != expected_boundaries {
            return Err(VerificationError::InvalidWitness {
                claim: "reported cycle and boundary spaces match adjacent differentials",
            });
        }
        let mut quotient_basis = boundaries.basis().to_vec();
        quotient_basis.extend(self.representatives.iter().cloned());
        let span = matrix_from_columns(&self.ring, cycles.ambient_dimension(), &quotient_basis)
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "quotient representatives have compatible dimensions",
            })?;
        let span_rank = span.rank().map_err(|_| VerificationError::InvalidWitness {
            claim: "quotient-representative rank is computable",
        })?;
        if span_rank != cycles.dimension()
            || quotient_basis.len() != cycles.dimension()
            || boundaries.dimension() + self.representatives.len() != cycles.dimension()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "boundaries and representatives form a basis of cycles",
            });
        }
        for representative in &self.representatives {
            let image = self
                .outgoing
                .matmul(&representative.to_column_matrix())
                .map_err(|_| VerificationError::InvalidWitness {
                    claim: "homology representatives can be tested as cycles",
                })?;
            if image != Matrix::zeros(self.ring.clone(), self.outgoing.shape().0, 1) {
                return Err(VerificationError::InvalidWitness {
                    claim: "every homology representative is a cycle",
                });
            }
        }
        Ok(())
    }
}

impl HomologyGroup<IntegerRing> {
    /// Recompute both adjacent Smith forms and check the abelian invariants.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.verify_common()?;
        if self.cycles.is_some() || self.boundaries.is_some() || !self.representatives.is_empty() {
            return Err(VerificationError::InvalidWitness {
                claim: "integral homology stores Smith invariants rather than field-space data",
            });
        }
        let outgoing_form = self.outgoing.smith_normal_form();
        let incoming_form = self.incoming.smith_normal_form();
        let module_rank = self.outgoing.shape().1;
        let expected_free_rank = module_rank
            .checked_sub(outgoing_form.rank())
            .and_then(|rank| rank.checked_sub(incoming_form.rank()))
            .ok_or(VerificationError::InvalidWitness {
                claim: "adjacent integral differential ranks fit the middle free module",
            })?;
        let one = Integer::one();
        let expected_torsion = incoming_form
            .invariant_factors()
            .iter()
            .filter(|factor| *factor != &one)
            .cloned()
            .collect::<Vec<_>>();
        if self.free_rank != expected_free_rank || self.torsion_coefficients != expected_torsion {
            return Err(VerificationError::InvalidWitness {
                claim: "integral free rank and torsion match the adjacent Smith forms",
            });
        }
        Ok(())
    }
}

impl<R: Ring> fmt::Debug for HomologyGroup<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HomologyGroup")
            .field("context", &R::TYPE_ID)
            .field("degree", &self.degree)
            .field("cohomological", &self.cohomological)
            .field("free_rank", &self.free_rank)
            .field("torsion_coefficients", &self.torsion_coefficients)
            .finish()
    }
}

impl<R: Ring> fmt::Display for HomologyGroup<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let marker = if self.cohomological { "H^" } else { "H_" };
        write!(
            f,
            "{marker}{}({}; rank={}",
            self.degree,
            R::TYPE_ID,
            self.free_rank
        )?;
        if !self.torsion_coefficients.is_empty() {
            f.write_str(", torsion=[")?;
            for (index, factor) in self.torsion_coefficients.iter().enumerate() {
                if index != 0 {
                    f.write_str(", ")?;
                }
                factor.fmt(f)?;
            }
            f.write_str("]")?;
        }
        f.write_str(")")
    }
}

impl<R: Ring> SemanticType for HomologyGroup<R> {
    const TYPE_ID: &'static str = "noether/homalg/HomologyGroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = vec![u8::from(self.cohomological)];
        output.extend_from_slice(&self.degree.to_be_bytes());
        push_frame(&mut output, &self.ring.canonical_bytes()?)?;
        push_frame(&mut output, &self.outgoing.canonical_bytes()?)?;
        push_frame(&mut output, &self.incoming.canonical_bytes()?)?;
        output.extend_from_slice(&(self.free_rank as u64).to_be_bytes());
        output.extend_from_slice(&(self.torsion_coefficients.len() as u64).to_be_bytes());
        for factor in &self.torsion_coefficients {
            push_frame(&mut output, &factor.canonical_bytes()?)?;
        }
        for representative in &self.representatives {
            push_frame(&mut output, &representative.canonical_bytes()?)?;
        }
        push_frame(&mut output, &self.validity.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &HOMOLOGY_GROUP_DESCRIPTION
    }
}

/// A checked collection of homology groups in deterministic degree order.
#[derive(Clone, PartialEq, Eq)]
pub struct GradedHomology<R: Ring> {
    ring: R,
    groups: Vec<HomologyGroup<R>>,
}

impl<R: Ring> GradedHomology<R> {
    pub(crate) fn from_groups(ring: R, groups: Vec<HomologyGroup<R>>) -> Self {
        debug_assert!(groups
            .windows(2)
            .all(|pair| pair[0].degree() < pair[1].degree()));
        debug_assert!(groups
            .iter()
            .all(|group| group.coefficient_context() == &ring));
        Self { ring, groups }
    }

    /// Coefficient context shared by every group.
    pub fn coefficient_context(&self) -> &R {
        &self.ring
    }

    /// Groups in increasing homological degree.
    pub fn groups(&self) -> &[HomologyGroup<R>] {
        &self.groups
    }

    /// Group in one degree, if that degree was explicitly present.
    pub fn get(&self, degree: i32) -> Option<&HomologyGroup<R>> {
        self.groups
            .binary_search_by_key(&degree, HomologyGroup::degree)
            .ok()
            .map(|index| &self.groups[index])
    }

    fn verify_structure(&self) -> Result<(), VerificationError> {
        let mut previous = None;
        for group in &self.groups {
            if group.coefficient_context() != &self.ring
                || previous.is_some_and(|degree| degree >= group.degree())
            {
                return Err(VerificationError::InvalidWitness {
                    claim: "graded homology has one ordered group per degree and context",
                });
            }
            previous = Some(group.degree());
        }
        Ok(())
    }
}

impl<R: Field> GradedHomology<R> {
    /// Recheck ordering, coefficient contexts, and every field-space witness.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.verify_structure()?;
        for group in &self.groups {
            group.verify()?;
        }
        Ok(())
    }
}

impl GradedHomology<IntegerRing> {
    /// Recheck ordering, coefficient context, and every integral Smith claim.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.verify_structure()?;
        for group in &self.groups {
            group.verify()?;
        }
        Ok(())
    }
}

impl<R: Ring> fmt::Debug for GradedHomology<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GradedHomology")
            .field("context", &R::TYPE_ID)
            .field("groups", &self.groups.len())
            .finish()
    }
}

impl<R: Ring> fmt::Display for GradedHomology<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradedHomology<{}>(degrees={})",
            R::TYPE_ID,
            self.groups.len()
        )
    }
}

impl<R: Ring> SemanticType for GradedHomology<R> {
    const TYPE_ID: &'static str = "noether/homalg/GradedHomology@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        for group in &self.groups {
            push_frame(&mut output, &group.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_HOMOLOGY_DESCRIPTION
    }
}

fn quotient_representatives<R: Field>(
    ring: &R,
    cycles: &SolutionSpace<R>,
    boundaries: &SolutionSpace<R>,
    free_coordinates: &[usize],
) -> Result<Vec<Vector<R>>, HomologyError> {
    if cycles.ring() != ring
        || boundaries.ring() != ring
        || cycles.ambient_dimension() != boundaries.ambient_dimension()
    {
        return Err(HomologyError::ContextMismatch {
            degree: 0,
            error: ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "incompatible cycle and boundary spaces".to_owned(),
            },
        });
    }
    if free_coordinates.len() != cycles.dimension() {
        return Err(HomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "cycle basis has one free coordinate per vector",
            },
        ));
    }

    // The RREF kernel basis is the identity on its ordered free coordinates.
    // Restricting each boundary vector to those positions therefore gives its
    // exact coordinates in the cycle basis.  Pivot columns of the transposed
    // coordinate matrix identify which cycle-basis vectors are needed for the
    // boundary span; the remaining cycle vectors form a quotient basis.  This
    // eliminates a boundary_dimension x cycle_dimension matrix instead of an
    // ambient_dimension x (boundary_dimension + cycle_dimension) matrix.
    let coordinates = Matrix::from_rows(
        ring.clone(),
        boundaries.basis().iter().map(|boundary| {
            free_coordinates.iter().map(|coordinate| {
                boundary
                    .get(*coordinate)
                    .expect("free coordinate lies in the homology ambient space")
            })
        }),
    )?;
    let boundary_pivots = coordinates.pivot_columns_from_elimination()?;
    if boundary_pivots.len() != boundaries.dimension() {
        return Err(HomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "boundaries are a subspace of cycles",
            },
        ));
    }
    let mut pivot = vec![false; cycles.dimension()];
    for coordinate in boundary_pivots {
        pivot[coordinate] = true;
    }
    Ok(cycles
        .basis()
        .iter()
        .enumerate()
        .filter(|(index, _)| !pivot[*index])
        .map(|(_, representative)| representative.clone())
        .collect())
}

fn matrix_from_columns<R: Ring>(
    ring: &R,
    rows: usize,
    columns: &[Vector<R>],
) -> Result<Matrix<R>, LinalgError> {
    if columns.is_empty() {
        return Ok(Matrix::zeros(ring.clone(), rows, 0));
    }
    if columns
        .iter()
        .any(|column| column.ring() != ring || column.dimension() != rows)
    {
        return Err(LinalgError::DimensionMismatch {
            operation: "matrix construction from columns",
            left: rows,
            right: columns
                .iter()
                .find(|column| column.ring() != ring || column.dimension() != rows)
                .map_or(rows, Vector::dimension),
        });
    }
    Matrix::from_rows(
        ring.clone(),
        (0..rows).map(|row| {
            columns
                .iter()
                .map(move |column| column.get(row).expect("validated column dimension"))
        }),
    )
}

fn record_dimension(
    dimensions: &mut BTreeMap<i32, usize>,
    degree: i32,
    dimension: usize,
) -> Result<(), HomologyError> {
    if let Some(expected) = dimensions.insert(degree, dimension) {
        if expected != dimension {
            dimensions.insert(degree, expected);
            return Err(HomologyError::DimensionConflict {
                degree,
                expected,
                actual: dimension,
            });
        }
    }
    Ok(())
}

fn quotient_rank(
    degree: i32,
    module_rank: usize,
    outgoing_rank: usize,
    incoming_rank: usize,
) -> Result<usize, HomologyError> {
    module_rank
        .checked_sub(outgoing_rank)
        .and_then(|rank| rank.checked_sub(incoming_rank))
        .ok_or(HomologyError::RankInconsistency {
            degree,
            module_rank,
            outgoing_rank,
            incoming_rank,
        })
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        NoetherError::Serialization("homological component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;
    use crate::{Integer, Rational, Q, Z};

    fn q(value: i64) -> Rational {
        Rational::from(value)
    }

    #[test]
    fn circle_has_one_dimensional_homology_in_degrees_zero_and_one() {
        let boundary = Matrix::from_rows(Q, [[q(0)]]).unwrap();
        let complex = ChainComplex::from_boundaries(Q, BTreeMap::from([(1, boundary)])).unwrap();

        let h0 = complex.homology(0).unwrap();
        let h1 = complex.homology(1).unwrap();
        assert_eq!(h0.dimension(), 1);
        assert_eq!(h1.dimension(), 1);
        assert_eq!(h0.representatives().len(), 1);
        assert_eq!(h1.representatives().len(), 1);
        h0.verify().unwrap();
        h1.verify().unwrap();
    }

    #[test]
    fn interval_has_only_degree_zero_homology() {
        let boundary = Matrix::from_rows(Q, [[q(-1)], [q(1)]]).unwrap();
        let complex = ChainComplex::from_boundaries(Q, BTreeMap::from([(1, boundary)])).unwrap();

        assert_eq!(complex.betti_number(-1).unwrap(), 0);
        assert_eq!(complex.betti_number(0).unwrap(), 1);
        assert_eq!(complex.betti_number(1).unwrap(), 0);
        assert_eq!(complex.betti_number(2).unwrap(), 0);
        assert_eq!(complex.homology(1).unwrap().dimension(), 0);
        assert_eq!(complex.homology(0).unwrap().dimension(), 1);
        let graded = complex.graded_homology().unwrap();
        assert_eq!(graded.groups().len(), 2);
        assert_eq!(graded.get(0).unwrap().dimension(), 1);
        graded.verify().unwrap();
    }

    #[test]
    fn representatives_are_a_basis_modulo_nonzero_boundaries() {
        let d1 = Matrix::from_rows(Q, [[q(0), q(0)]]).unwrap();
        let d2 = Matrix::from_rows(Q, [[q(1)], [q(0)]]).unwrap();
        let complex = ChainComplex::from_boundaries(Q, BTreeMap::from([(1, d1), (2, d2)])).unwrap();

        let h1 = complex.homology(1).unwrap();
        assert_eq!(h1.cycles().dimension(), 2);
        assert_eq!(h1.boundaries().dimension(), 1);
        assert_eq!(h1.dimension(), 1);
        assert_eq!(h1.representatives()[0].get(0).unwrap(), q(0));
        assert_eq!(h1.representatives()[0].get(1).unwrap(), q(1));
        h1.verify().unwrap();
    }

    #[test]
    fn coboundaries_reindex_to_the_same_lowering_shape() {
        let coboundary = Matrix::from_rows(Q, [[q(-1), q(1)]]).unwrap();
        let complex =
            ChainComplex::from_coboundaries(Q, BTreeMap::from([(0, coboundary)])).unwrap();

        assert_eq!(complex.degrees(), vec![0, 1]);
        assert_eq!(complex.module_dimension(0), 2);
        assert_eq!(complex.module_dimension(1), 1);
        assert_eq!(complex.betti_number(0).unwrap(), 1);
        assert_eq!(complex.betti_number(1).unwrap(), 0);
        assert_eq!(complex.cohomology(0).unwrap().dimension(), 1);
        assert_eq!(complex.cohomology(1).unwrap().dimension(), 0);
        let graded = complex.graded_cohomology().unwrap();
        assert_eq!(graded.groups().len(), 2);
        assert!(graded.groups().iter().all(HomologyGroup::is_cohomology));
        graded.verify().unwrap();
        assert!(matches!(
            complex.homology(0),
            Err(HomologyError::GradingConvention { .. })
        ));
    }

    #[test]
    fn rejects_unrepresentable_terminal_coboundary_degree() {
        let differential = Matrix::zeros(Q, 0, 0);
        let error = ChainComplex::from_coboundaries(Q, BTreeMap::from([(i32::MAX, differential)]))
            .unwrap_err();
        assert_eq!(error, HomologyError::DegreeOverflow { degree: i32::MAX });
    }

    #[test]
    fn rejects_nonzero_d_squared_in_the_rank_one_case() {
        let one = Matrix::from_rows(Q, [[q(1)]]).unwrap();
        let error = ChainComplex::from_boundaries(Q, BTreeMap::from([(1, one.clone()), (2, one)]))
            .unwrap_err();
        assert_eq!(error, HomologyError::NotAComplex { degree: 2 });
    }

    #[test]
    fn rejects_conflicting_adjacent_dimensions() {
        let d1 = Matrix::zeros(Q, 2, 1);
        let d2 = Matrix::zeros(Q, 3, 1);
        let error =
            ChainComplex::from_boundaries(Q, BTreeMap::from([(1, d1), (2, d2)])).unwrap_err();
        assert_eq!(
            error,
            HomologyError::DimensionConflict {
                degree: 1,
                expected: 1,
                actual: 3,
            }
        );
    }

    #[test]
    fn exact_extension_field_homology_uses_the_generic_field_path() {
        let field = FiniteField::extension(3, 2).unwrap();
        let zero = field.element(0);
        let boundary = Matrix::from_rows(field.clone(), [[zero]]).unwrap();
        let complex =
            ChainComplex::from_boundaries(field, BTreeMap::from([(1, boundary)])).unwrap();

        assert_eq!(complex.homology(0).unwrap().dimension(), 1);
        assert_eq!(complex.homology(1).unwrap().dimension(), 1);
    }

    #[test]
    fn adjacent_field_betti_queries_share_exact_differential_ranks() {
        let field = FiniteField::prime(3_u8).unwrap();
        let d1 = Matrix::from_rows(
            field.clone(),
            [
                [
                    field.element(1_u8),
                    field.element(0_u8),
                    field.element(0_u8),
                ],
                [
                    field.element(0_u8),
                    field.element(0_u8),
                    field.element(0_u8),
                ],
            ],
        )
        .unwrap();
        let d2 = Matrix::from_rows(
            field.clone(),
            [
                [field.element(0_u8)],
                [field.element(1_u8)],
                [field.element(0_u8)],
            ],
        )
        .unwrap();
        let complex =
            ChainComplex::from_boundaries(field, BTreeMap::from([(1, d1), (2, d2)])).unwrap();
        let clone = complex.clone();

        assert_eq!(complex.betti_number(1).unwrap(), 1);
        assert_eq!(complex.differential_ranks[&1].get(), Some(&1));
        assert_eq!(complex.differential_ranks[&2].get(), Some(&1));
        assert_eq!(clone.betti_number(0).unwrap(), 1);
        assert_eq!(clone.betti_number(2).unwrap(), 0);
        assert_eq!(complex, clone);
    }

    #[test]
    fn rank_one_integral_complex_reports_torsion_without_transforms() {
        // Z --2--> Z has H_0 = Z/2 and H_1 = 0.
        let twice = Matrix::from_rows(Z, [[Integer::from(2)]]).unwrap();
        let complex = ChainComplex::from_boundaries(Z, BTreeMap::from([(1, twice)])).unwrap();

        let invariants = complex.homology_invariants(0).unwrap();
        assert_eq!(invariants.free_rank(), 0);
        assert_eq!(invariants.torsion_coefficients(), &[Integer::from(2)]);
        assert_eq!(invariants.to_string(), "Z/2");
        invariants.verify().unwrap();

        let h0 = complex.homology(0).unwrap();
        assert_eq!(h0.free_rank(), 0);
        assert_eq!(h0.torsion_coefficients(), &[Integer::from(2)]);
        h0.verify().unwrap();

        let h1 = complex.homology(1).unwrap();
        assert_eq!(h1.free_rank(), 0);
        assert!(h1.torsion_coefficients().is_empty());
        h1.verify().unwrap();

        let graded = complex.graded_homology_invariants().unwrap();
        assert_eq!(graded.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(graded.get(&0), Some(&invariants));
        assert_eq!(
            graded.get(&1),
            Some(&complex.homology_invariants(1).unwrap())
        );

        let sparse = SparseMatrix::from_triplets(Z, 1, 1, [(0, 0, Integer::from(2))]).unwrap();
        assert_eq!(
            graded_homology_invariants_from_sparse_boundaries(&BTreeMap::from([(1, sparse)]))
                .unwrap(),
            graded
        );
    }

    #[test]
    fn integral_circle_has_one_free_class_in_degrees_zero_and_one() {
        let boundary = Matrix::from_rows(Z, [[Integer::zero()]]).unwrap();
        let complex = ChainComplex::from_boundaries(Z, BTreeMap::from([(1, boundary)])).unwrap();
        let graded = complex.graded_homology().unwrap();

        assert_eq!(graded.get(0).unwrap().free_rank(), 1);
        assert_eq!(graded.get(1).unwrap().free_rank(), 1);
        assert!(graded
            .groups()
            .iter()
            .all(|group| group.torsion_coefficients().is_empty()));
        graded.verify().unwrap();
    }

    #[test]
    fn integral_graded_cohomology_preserves_torsion() {
        // Z --2--> Z as a cochain complex has H^0 = 0 and H^1 = Z/2.
        let twice = Matrix::from_rows(Z, [[Integer::from(2)]]).unwrap();
        let complex = ChainComplex::from_coboundaries(Z, BTreeMap::from([(0, twice)])).unwrap();
        let invariants = complex.cohomology_invariants(1).unwrap();
        assert_eq!(invariants.torsion_coefficients(), &[Integer::from(2)]);
        assert!(matches!(
            complex.homology_invariants(1),
            Err(HomologyError::GradingConvention { .. })
        ));
        let graded = complex.graded_cohomology().unwrap();
        let lean = complex.graded_cohomology_invariants().unwrap();

        assert_eq!(graded.get(0).unwrap().free_rank(), 0);
        assert_eq!(
            graded.get(1).unwrap().torsion_coefficients(),
            &[Integer::from(2)]
        );
        assert_eq!(lean.get(&0).unwrap().free_rank(), 0);
        assert_eq!(
            lean.get(&1).unwrap().torsion_coefficients(),
            &[Integer::from(2)]
        );
        assert!(graded.groups().iter().all(HomologyGroup::is_cohomology));
        graded.verify().unwrap();
    }

    #[test]
    fn algebraic_reduction_preserves_graded_integral_invariants() {
        let boundary = Matrix::from_rows(
            Z,
            [
                [Integer::from(1), Integer::from(2)],
                [Integer::from(3), Integer::from(8)],
            ],
        )
        .unwrap();
        let complex = ChainComplex::from_boundaries(Z, BTreeMap::from([(1, boundary)])).unwrap();
        let direct = complex.graded_integral_invariants_direct().unwrap();
        let reduction = complex.algebraic_morse_reduction().unwrap();
        reduction.verify().unwrap();
        let mut reduced = reduction
            .reduced()
            .graded_integral_invariants_direct()
            .unwrap();
        for degree in complex.degrees() {
            reduced
                .entry(degree)
                .or_insert_with(|| FinitelyGeneratedAbelianGroup::from_invariants(0, Vec::new()));
        }
        assert_eq!(reduced, direct);
    }

    #[test]
    fn algebraic_reduction_playbook_uses_the_benchmarked_cutover() {
        let small =
            ChainComplex::from_boundaries(Z, BTreeMap::from([(1, Matrix::zeros(Z, 32, 32))]))
                .unwrap();
        let large_boundary = Matrix::from_rows(
            Z,
            (0..128).map(|row| {
                (0..127)
                    .map(|column| {
                        Integer::from(if row == column {
                            -1
                        } else if row == column + 1 {
                            1
                        } else {
                            0
                        })
                    })
                    .collect::<Vec<_>>()
            }),
        )
        .unwrap();
        let large =
            ChainComplex::from_boundaries(Z, BTreeMap::from([(1, large_boundary)])).unwrap();
        assert_eq!(
            small.integral_invariant_strategy(),
            IntegralInvariantStrategy::DirectSmith
        );
        assert_eq!(
            large.integral_invariant_strategy(),
            IntegralInvariantStrategy::AlgebraicMorseThenSmith
        );
    }

    #[test]
    fn finitely_generated_abelian_group_checks_canonical_invariants() {
        let group = FinitelyGeneratedAbelianGroup::from_invariants(
            2,
            vec![Integer::from(2), Integer::from(6)],
        );
        assert_eq!(group.to_string(), "Z^2 + Z/2 + Z/6");
        group.verify().unwrap();
        let json = serde_json::to_string(&group).unwrap();
        assert_eq!(
            serde_json::from_str::<FinitelyGeneratedAbelianGroup>(&json).unwrap(),
            group
        );

        let unit = FinitelyGeneratedAbelianGroup {
            free_rank: 0,
            torsion_coefficients: vec![Integer::one()],
        };
        assert!(unit.verify().is_err());
        let nonchain = FinitelyGeneratedAbelianGroup {
            free_rank: 0,
            torsion_coefficients: vec![Integer::from(4), Integer::from(6)],
        };
        assert!(nonchain.verify().is_err());
        assert!(serde_json::from_str::<FinitelyGeneratedAbelianGroup>(
            r#"{"type":"noether/homalg/FinitelyGeneratedAbelianGroup@1","free_rank":0,"torsion_coefficients":["4","6"]}"#,
        )
        .is_err());
    }
}
