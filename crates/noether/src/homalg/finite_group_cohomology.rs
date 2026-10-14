// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite-group cohomology from checked free resolutions.

use core::fmt;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::commalg::{
    GradedCommutativeError, GradedCommutativePolynomialRing, GradedCommutativePresentation,
    GradedGenerator, GradedMonomial,
};
use crate::groups::{
    iterate_elements, FiniteAbelianInvariantError, FiniteAbelianInvariants, FiniteAbelianization,
    GroupError, Permutation, PermutationGroup,
};
use crate::linalg::{LinalgError, LinearOperator, Matrix, SparseMatrix, Vector};
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::topology::{
    AdmissibleCohomologyOperation, EilenbergMacLaneCohomologyGenerator,
    EilenbergMacLaneCohomologyOperation, EilenbergMacLaneError,
    EilenbergMacLaneIntegralCohomologyPresentation, EilenbergMacLaneModularCohomology,
    EilenbergMacLaneSpace, SimplicialFace, SimplicialSet, TopologyError,
};
use crate::{
    EuclideanDomain, Integer, IntegerRing, NoetherError, Ring, SemanticType, TypeDescription,
    VerificationError, Z,
};

use super::reduction::SparseChainContraction;
use super::{AbelianGroupError, FinitelyGeneratedAbelianGroup};

/// Work ceiling selecting the modular normalized-bar correctness oracle.
///
/// This retains the measured small-oracle regime independently of storage:
/// the implementation itself is sparse, but larger non-p-groups are better
/// served by the reduced Sylow stable-elements strategy.
const MODULAR_BAR_ORACLE_WORK_LIMIT: usize = 25_000_000;
/// Maximum number of sparse boundary contributions materialized by one
/// normalized integral or modular bar differential.
///
/// Sparse storage removes the irrelevant rows-times-columns ceiling, but the
/// bar basis itself still grows exponentially.  Refuse an infeasible request
/// before reserving or iterating that basis.
const NORMALIZED_BAR_SPARSE_ENTRY_LIMIT: usize = 25_000_000;
/// Defensive ceiling for the exact elementary-abelian subgroup search used by
/// the parameter completion certificate. This is an implementation limit,
/// not a theorem or census bound.
const MAX_ELEMENTARY_ABELIAN_SUBGROUPS: usize = 100_000;
type NormalizedMultiplicationTable = Vec<Vec<Option<usize>>>;

static DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteGroupCohomologyTruncation@1",
    rust_path: "noether::homalg::FiniteGroupCohomologyTruncation",
    python_path: Some("noether.homalg.FiniteGroupCohomologyTruncation"),
};

static INTEGRAL_TRUNCATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteGroupIntegralCohomologyTruncation@1",
    rust_path: "noether::homalg::FiniteGroupIntegralCohomologyTruncation",
    python_path: Some("noether.homalg.FiniteGroupIntegralCohomologyTruncation"),
};

static ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteGroupCohomologyAlgebraTruncation@1",
    rust_path: "noether::homalg::FiniteGroupCohomologyAlgebraTruncation",
    python_path: Some("noether.homalg.FiniteGroupCohomologyAlgebraTruncation"),
};

static COMPLETE_PRESENTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteGroupCohomologyCompletePresentation@1",
    rust_path: "noether::homalg::FiniteGroupCohomologyCompletePresentation",
    python_path: Some("noether.homalg.FiniteGroupCohomologyCompletePresentation"),
};

static FINITE_ABELIAN_GROUP_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteAbelianGroupCohomologyPresentation@1",
    rust_path: "noether::homalg::FiniteAbelianGroupCohomologyPresentation",
    python_path: Some("noether.homalg.FiniteAbelianGroupCohomologyPresentation"),
};

static FINITE_CYCLIC_GROUP_INTEGRAL_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteCyclicGroupIntegralCohomologyPresentation@1",
    rust_path: "noether::homalg::FiniteCyclicGroupIntegralCohomologyPresentation",
    python_path: Some("noether.homalg.FiniteCyclicGroupIntegralCohomologyPresentation"),
};

static FINITE_ABELIAN_GROUP_INTEGRAL_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FiniteAbelianGroupIntegralCohomologyPresentation@1",
    rust_path: "noether::homalg::FiniteAbelianGroupIntegralCohomologyPresentation",
    python_path: Some("noether.homalg.FiniteAbelianGroupIntegralCohomologyPresentation"),
};

/// Failure to construct a complete classifying-space presentation from a
/// concrete finite abelian group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteAbelianGroupCohomologyError {
    /// The supplied permutation group is nonabelian or exceeds the exact
    /// decomposition implementation's safety ceiling.
    AbelianInvariants(FiniteAbelianInvariantError),
    /// The derived symbolic `K(A,1)` computation failed.
    EilenbergMacLane(EilenbergMacLaneError),
    /// Retained evidence failed exact reconstruction.
    Verification(VerificationError),
}

impl fmt::Display for FiniteAbelianGroupCohomologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbelianInvariants(error) => error.fmt(formatter),
            Self::EilenbergMacLane(error) => error.fmt(formatter),
            Self::Verification(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FiniteAbelianGroupCohomologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AbelianInvariants(error) => Some(error),
            Self::EilenbergMacLane(error) => Some(error),
            Self::Verification(error) => Some(error),
        }
    }
}

impl From<FiniteAbelianInvariantError> for FiniteAbelianGroupCohomologyError {
    fn from(value: FiniteAbelianInvariantError) -> Self {
        Self::AbelianInvariants(value)
    }
}

impl From<EilenbergMacLaneError> for FiniteAbelianGroupCohomologyError {
    fn from(value: EilenbergMacLaneError) -> Self {
        Self::EilenbergMacLane(value)
    }
}

impl From<VerificationError> for FiniteAbelianGroupCohomologyError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

/// Failure to construct the complete integral cohomology ring of a finite
/// cyclic group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteCyclicGroupIntegralCohomologyError {
    /// The concrete group was not finite abelian or exceeded the checked
    /// invariant-factor implementation's safety ceiling.
    AbelianInvariants(FiniteAbelianInvariantError),
    /// The concrete finite abelian group has more than one invariant factor.
    NotCyclic {
        /// Canonical invariant factors proving that the group is not cyclic.
        invariant_factors: Vec<Integer>,
    },
    /// Retained periodic-resolution evidence failed exact reconstruction.
    Verification(VerificationError),
}

impl fmt::Display for FiniteCyclicGroupIntegralCohomologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbelianInvariants(error) => error.fmt(formatter),
            Self::NotCyclic { invariant_factors } => write!(
                formatter,
                "finite abelian group with invariant factors {invariant_factors:?} is not cyclic"
            ),
            Self::Verification(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FiniteCyclicGroupIntegralCohomologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AbelianInvariants(error) => Some(error),
            Self::NotCyclic { .. } => None,
            Self::Verification(error) => Some(error),
        }
    }
}

impl From<FiniteAbelianInvariantError> for FiniteCyclicGroupIntegralCohomologyError {
    fn from(value: FiniteAbelianInvariantError) -> Self {
        Self::AbelianInvariants(value)
    }
}

impl From<VerificationError> for FiniteCyclicGroupIntegralCohomologyError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

/// Complete integral cohomology of the classifying space of a supplied finite
/// cyclic permutation group.
///
/// For a nontrivial cyclic group `G`, the result certifies
/// `H^*(BG; Z) = Z[u]/(|G| u)` with `|u| = 2`. The named class `u` is the
/// integral Bockstein of the degree-one tautological class with coefficients
/// in `Z/|G|`. The trivial group has coefficient ring `Z` and no positive-
/// degree generator.
#[derive(Clone)]
pub struct FiniteCyclicGroupIntegralCohomologyPresentation {
    group: PermutationGroup,
    order: Integer,
}

impl PartialEq for FiniteCyclicGroupIntegralCohomologyPresentation {
    fn eq(&self, other: &Self) -> bool {
        self.group.degree() == other.group.degree()
            && self.group.generators() == other.group.generators()
            && self.order == other.order
    }
}

impl Eq for FiniteCyclicGroupIntegralCohomologyPresentation {}

impl FiniteCyclicGroupIntegralCohomologyPresentation {
    /// Concrete cyclic permutation group whose classifying space is represented.
    pub fn group(&self) -> &PermutationGroup {
        &self.group
    }

    /// Exact order of the cyclic group.
    pub fn order(&self) -> &Integer {
        &self.order
    }

    /// Stable id of the degree-two generator, absent for the trivial group.
    pub fn generator_id(&self) -> Option<&'static str> {
        (!self.order.is_one()).then_some("u2_1")
    }

    /// Cohomological degree of the generator, absent for the trivial group.
    pub fn generator_degree(&self) -> Option<u32> {
        (!self.order.is_one()).then_some(2)
    }

    /// Reader-facing operation name for the generator.
    pub fn generator_display_name(&self) -> Option<String> {
        (!self.order.is_one()).then(|| format!("beta_{}(iota_1)", self.order))
    }

    /// Coefficient in the sole relation `|G| u = 0`, absent for the trivial group.
    pub fn relation_coefficient(&self) -> Option<&Integer> {
        (!self.order.is_one()).then_some(&self.order)
    }

    /// Additive integral cohomology group in one degree.
    pub fn additive_group(&self, degree: usize) -> FinitelyGeneratedAbelianGroup {
        if degree == 0 {
            FinitelyGeneratedAbelianGroup::free(1)
        } else if degree.is_multiple_of(2) && !self.order.is_one() {
            FinitelyGeneratedAbelianGroup::from_invariants(0, vec![self.order.clone()])
        } else {
            FinitelyGeneratedAbelianGroup::trivial()
        }
    }

    /// The exact theorem used to promote the presentation.
    pub fn completion_strategy(&self) -> &'static str {
        "cyclic_periodic_resolution"
    }

    /// Recompute cyclicity, order, and the periodic-resolution presentation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let order =
            cyclic_group_order(&self.group).map_err(|_| VerificationError::InvalidWitness {
                claim: "finite cyclic integral cohomology retains a cyclic source group",
            })?;
        if order != self.order
            || self.additive_group(0) != FinitelyGeneratedAbelianGroup::free(1)
            || self.additive_group(1) != FinitelyGeneratedAbelianGroup::trivial()
            || (!order.is_one()
                && self.additive_group(2)
                    != FinitelyGeneratedAbelianGroup::from_invariants(0, vec![order]))
        {
            return Err(VerificationError::InvalidWitness {
                claim: "cyclic periodic resolution gives Z in degree zero, Z/|G| in positive even degrees, and zero in odd degrees",
            });
        }
        Ok(())
    }
}

impl fmt::Debug for FiniteCyclicGroupIntegralCohomologyPresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteCyclicGroupIntegralCohomologyPresentation")
            .field("order", &self.order)
            .field("generator_id", &self.generator_id())
            .field("completion_strategy", &self.completion_strategy())
            .finish_non_exhaustive()
    }
}

impl SemanticType for FiniteCyclicGroupIntegralCohomologyPresentation {
    const TYPE_ID: &'static str =
        "noether/homalg/FiniteCyclicGroupIntegralCohomologyPresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.group.canonical_bytes()?;
        let order = self.order.to_string();
        let length = u64::try_from(order.len()).map_err(|_| {
            NoetherError::Serialization("cyclic group order exceeds u64 framing".to_owned())
        })?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(order.as_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_CYCLIC_GROUP_INTEGRAL_COHOMOLOGY_DESCRIPTION
    }
}

/// Closed-form integral classifying-space cohomology for finite cyclic groups.
pub trait FiniteCyclicGroupIntegralCohomology {
    /// Return the complete periodic-resolution presentation.
    fn complete_cyclic_integral_cohomology(
        &self,
    ) -> Result<
        FiniteCyclicGroupIntegralCohomologyPresentation,
        FiniteCyclicGroupIntegralCohomologyError,
    >;
}

impl FiniteCyclicGroupIntegralCohomology for PermutationGroup {
    fn complete_cyclic_integral_cohomology(
        &self,
    ) -> Result<
        FiniteCyclicGroupIntegralCohomologyPresentation,
        FiniteCyclicGroupIntegralCohomologyError,
    > {
        let result = FiniteCyclicGroupIntegralCohomologyPresentation {
            group: self.clone(),
            order: cyclic_group_order(self)?,
        };
        result.verify()?;
        Ok(result)
    }
}

fn cyclic_group_order(
    group: &PermutationGroup,
) -> Result<Integer, FiniteCyclicGroupIntegralCohomologyError> {
    let invariant_factors = group.abelian_invariant_factors()?;
    if invariant_factors.len() > 1 {
        return Err(FiniteCyclicGroupIntegralCohomologyError::NotCyclic { invariant_factors });
    }
    let order = invariant_factors
        .first()
        .cloned()
        .unwrap_or_else(Integer::one);
    if order.to_string() != group.order().to_decimal() {
        return Err(FiniteCyclicGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "cyclic invariant factor equals the concrete group order",
            },
        ));
    }
    Ok(order)
}

/// Complete integral cohomology of `BG` for a supplied finite abelian group
/// in an exact classifying-space family.
///
/// The retained invariant factors certify `G ~= A`, while the symbolic
/// `K(A,1)` presentation records the theorem-backed generators, relations,
/// and additive groups. The current exact families have at most two
/// nontrivial invariant factors; unsupported higher ranks fail through the
/// typed Eilenberg--Mac Lane capability error.
#[derive(Clone)]
pub struct FiniteAbelianGroupIntegralCohomologyPresentation {
    group: PermutationGroup,
    invariant_factors: Vec<Integer>,
    cohomology: EilenbergMacLaneIntegralCohomologyPresentation,
}

impl PartialEq for FiniteAbelianGroupIntegralCohomologyPresentation {
    fn eq(&self, other: &Self) -> bool {
        self.group.degree() == other.group.degree()
            && self.group.generators() == other.group.generators()
            && self.invariant_factors == other.invariant_factors
            && self.cohomology == other.cohomology
    }
}

impl Eq for FiniteAbelianGroupIntegralCohomologyPresentation {}

impl FiniteAbelianGroupIntegralCohomologyPresentation {
    /// Concrete finite group whose classifying space is represented.
    pub fn group(&self) -> &PermutationGroup {
        &self.group
    }

    /// Canonical factors `d_1 | ... | d_r` of the finite abelian group.
    pub fn invariant_factors(&self) -> &[Integer] {
        &self.invariant_factors
    }

    /// Complete symbolic `K(A,1)` calculation retained as theorem evidence.
    pub fn classifying_space_cohomology(&self) -> &EilenbergMacLaneIntegralCohomologyPresentation {
        &self.cohomology
    }

    /// Additive integral cohomology group in one degree.
    pub fn additive_group(
        &self,
        degree: usize,
    ) -> Result<FinitelyGeneratedAbelianGroup, EilenbergMacLaneError> {
        self.cohomology.additive_group(degree)
    }

    /// The exact family theorem used to promote the presentation.
    pub fn completion_strategy(&self) -> &'static str {
        "finite_abelian_classifying_space"
    }

    /// Recompute the abelian decomposition and complete `K(A,1)` result.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = finite_abelian_group_integral_cohomology(&self.group).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "finite-abelian integral classifying-space cohomology is exactly reconstructible",
            }
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim:
                    "finite-abelian decomposition and integral classifying-space presentation agree",
            })
        }
    }
}

impl fmt::Debug for FiniteAbelianGroupIntegralCohomologyPresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteAbelianGroupIntegralCohomologyPresentation")
            .field("invariant_factors", &self.invariant_factors)
            .field("generators", &self.cohomology.generators())
            .field("completion_strategy", &self.completion_strategy())
            .finish_non_exhaustive()
    }
}

impl SemanticType for FiniteAbelianGroupIntegralCohomologyPresentation {
    const TYPE_ID: &'static str =
        "noether/homalg/FiniteAbelianGroupIntegralCohomologyPresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        for component in [
            self.group.canonical_bytes()?,
            self.cohomology.canonical_bytes()?,
        ] {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "finite-abelian integral cohomology component exceeds u64".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_ABELIAN_GROUP_INTEGRAL_COHOMOLOGY_DESCRIPTION
    }
}

/// Closed-form integral classifying-space cohomology for a supplied finite
/// abelian permutation group.
pub trait FiniteAbelianGroupIntegralCohomology {
    /// Return a complete integral presentation with exact abelian
    /// decomposition evidence. Nonabelian and unsupported higher-rank groups
    /// fail with typed errors.
    fn complete_abelian_integral_cohomology(
        &self,
    ) -> Result<FiniteAbelianGroupIntegralCohomologyPresentation, FiniteAbelianGroupCohomologyError>;
}

impl FiniteAbelianGroupIntegralCohomology for PermutationGroup {
    fn complete_abelian_integral_cohomology(
        &self,
    ) -> Result<FiniteAbelianGroupIntegralCohomologyPresentation, FiniteAbelianGroupCohomologyError>
    {
        let result = finite_abelian_group_integral_cohomology(self)?;
        result.verify()?;
        Ok(result)
    }
}

fn finite_abelian_group_integral_cohomology(
    group: &PermutationGroup,
) -> Result<FiniteAbelianGroupIntegralCohomologyPresentation, FiniteAbelianGroupCohomologyError> {
    let invariant_factors = group.abelian_invariant_factors()?;
    let abelian_group = FinitelyGeneratedAbelianGroup::new(0, invariant_factors.clone())
        .map_err(EilenbergMacLaneError::AbelianGroup)?;
    let space = EilenbergMacLaneSpace::new(abelian_group, 1)?;
    let cohomology = space.integral_cohomology_presentation()?;
    Ok(FiniteAbelianGroupIntegralCohomologyPresentation {
        group: group.clone(),
        invariant_factors,
        cohomology,
    })
}

/// Complete prime-field cohomology of `BG` for a concrete finite abelian
/// permutation group.
///
/// The retained invariant factors certify `G ~= A`; the complete presentation
/// is then the exactly replayable `K(A,1)` closed form. This is a family theorem,
/// not a promotion inferred from a short truncation.
#[derive(Clone)]
pub struct FiniteAbelianGroupCohomologyPresentation {
    group: PermutationGroup,
    invariant_factors: Vec<Integer>,
    cohomology: EilenbergMacLaneModularCohomology,
}

impl PartialEq for FiniteAbelianGroupCohomologyPresentation {
    fn eq(&self, other: &Self) -> bool {
        self.group.degree() == other.group.degree()
            && self.group.generators() == other.group.generators()
            && self.invariant_factors == other.invariant_factors
            && self.cohomology == other.cohomology
    }
}

impl Eq for FiniteAbelianGroupCohomologyPresentation {}

impl FiniteAbelianGroupCohomologyPresentation {
    /// Concrete finite group whose classifying space is represented.
    pub fn group(&self) -> &PermutationGroup {
        &self.group
    }

    /// Canonical factors `d_1 | ... | d_r` of the finite abelian group.
    pub fn invariant_factors(&self) -> &[Integer] {
        &self.invariant_factors
    }

    /// Complete symbolic `K(A,1)` calculation retained as theorem evidence.
    pub fn classifying_space_cohomology(&self) -> &EilenbergMacLaneModularCohomology {
        &self.cohomology
    }

    /// Exact coefficient prime field.
    pub fn field(&self) -> &FiniteField {
        self.cohomology.field()
    }

    /// Complete graded-commutative quotient presentation.
    pub fn presentation(&self) -> &GradedCommutativePresentation {
        self.cohomology.presentation()
    }

    /// Stable generator metadata.
    pub fn generators(&self) -> &[EilenbergMacLaneCohomologyGenerator] {
        self.cohomology.generators()
    }

    /// Primary operations on presentation generators.
    pub fn operations(&self) -> &[EilenbergMacLaneCohomologyOperation] {
        self.cohomology.operations()
    }

    /// Recompute the abelian decomposition and the complete `K(A,1)` result.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt =
            finite_abelian_group_cohomology(&self.group, self.field().clone()).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim: "finite-abelian classifying-space cohomology is exactly reconstructible",
                }
            })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "finite-abelian decomposition and classifying-space presentation agree",
            })
        }
    }
}

impl fmt::Debug for FiniteAbelianGroupCohomologyPresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteAbelianGroupCohomologyPresentation")
            .field("invariant_factors", &self.invariant_factors)
            .field("field", self.field())
            .field("generators", &self.generators())
            .finish_non_exhaustive()
    }
}

impl SemanticType for FiniteAbelianGroupCohomologyPresentation {
    const TYPE_ID: &'static str = "noether/homalg/FiniteAbelianGroupCohomologyPresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        for component in [
            self.group.canonical_bytes()?,
            self.cohomology.canonical_bytes()?,
        ] {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "finite-abelian cohomology component exceeds u64".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_ABELIAN_GROUP_COHOMOLOGY_DESCRIPTION
    }
}

/// Closed-form classifying-space cohomology for a supplied finite abelian
/// permutation group.
pub trait FiniteAbelianGroupCohomology {
    /// Return a complete prime-field presentation with exact abelian
    /// decomposition evidence. Nonabelian groups fail with a typed error.
    fn complete_abelian_modular_cohomology(
        &self,
        field: FiniteField,
    ) -> Result<FiniteAbelianGroupCohomologyPresentation, FiniteAbelianGroupCohomologyError>;
}

impl FiniteAbelianGroupCohomology for PermutationGroup {
    fn complete_abelian_modular_cohomology(
        &self,
        field: FiniteField,
    ) -> Result<FiniteAbelianGroupCohomologyPresentation, FiniteAbelianGroupCohomologyError> {
        let result = finite_abelian_group_cohomology(self, field)?;
        result.verify()?;
        Ok(result)
    }
}

fn finite_abelian_group_cohomology(
    group: &PermutationGroup,
    field: FiniteField,
) -> Result<FiniteAbelianGroupCohomologyPresentation, FiniteAbelianGroupCohomologyError> {
    let invariant_factors = group.abelian_invariant_factors()?;
    let abelian_group = FinitelyGeneratedAbelianGroup::new(0, invariant_factors.clone())
        .map_err(EilenbergMacLaneError::AbelianGroup)?;
    let space = EilenbergMacLaneSpace::new(abelian_group, 1)?;
    let cohomology = space.modular_cohomology(field)?;
    Ok(FiniteAbelianGroupCohomologyPresentation {
        group: group.clone(),
        invariant_factors,
        cohomology,
    })
}

/// Resolution strategy used to obtain a finite-group cohomology result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteGroupResolutionStrategy {
    /// A minimal free resolution over the local group algebra of a p-group.
    MinimalPGroup,
    /// The normalized inhomogeneous bar resolution with trivial coefficients.
    NormalizedBar,
    /// Stable elements inside the cohomology of a Sylow subgroup.
    SylowStableElements,
}

/// Failure to compute or query finite-group cohomology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteGroupCohomologyError {
    /// The requested characteristic is not prime.
    InvalidPrime {
        /// Rejected integer.
        prime: u64,
    },
    /// A normalized bar differential would exceed the selected work ceiling.
    BarComplexTooLarge {
        /// Differential degree.
        degree: usize,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
        /// Current implementation work ceiling.
        cell_limit: usize,
    },
    /// A normalized bar basis cardinality overflowed `usize`.
    BasisCardinalityOverflow {
        /// Basis degree.
        degree: usize,
        /// Number of nonidentity group elements forming the alphabet.
        alphabet_size: usize,
    },
    /// A requested degree lies outside the exact truncation.
    DegreeOutOfRange {
        /// Requested degree.
        degree: usize,
        /// Largest computed cohomological degree.
        max_degree: usize,
    },
    /// A requested basis vector does not exist in the named degree.
    BasisIndexOutOfRange {
        /// Cohomological degree.
        degree: usize,
        /// Requested zero-based basis index.
        index: usize,
        /// Cohomology dimension in that degree.
        dimension: usize,
    },
    /// The concrete permutation-group operation failed.
    Group(GroupError),
    /// Exact field linear algebra failed.
    LinearAlgebra(LinalgError),
    /// Exact graded-commutative algebra failed.
    GradedAlgebra(GradedCommutativeError),
    /// A cohomological degree cannot be represented by the graded-algebra API.
    AlgebraDegreeOverflow {
        /// Degree that did not fit the public graded degree type.
        degree: usize,
    },
    /// Extracted generators and relations did not reproduce the supplied
    /// multiplication table through its exact bound.
    InvalidAlgebraTruncation {
        /// Failed invariant.
        reason: &'static str,
    },
    /// A theorem-backed completion criterion requires a larger exact range.
    CompletionBoundNotReached {
        /// Largest degree computed exactly.
        computed_through_degree: usize,
        /// Degree required by the selected completion theorem.
        required_through_degree: usize,
    },
    /// A theorem-backed completion bound exceeded the public degree type.
    CompletionBoundOverflow,
    /// The parameter criterion cannot construct its exact restriction
    /// evidence from the supplied resolution model.
    ParameterEvidenceUnavailable {
        /// Stable explanation of the unsupported model or exhausted search.
        reason: &'static str,
    },
    /// Retained evidence failed exact recomputation.
    Verification(VerificationError),
    /// A topology-level cochain operation failed.
    Topology(TopologyError),
    /// The chosen reduced resolution cannot yet supply a genuinely lower
    /// Steenrod operation and the normalized bar comparison is unavailable.
    CohomologyOperationUnavailable {
        /// Stable explanation of the missing comparison evidence.
        reason: &'static str,
    },
}

impl fmt::Display for FiniteGroupCohomologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrime { prime } => write!(f, "{prime} is not a prime characteristic"),
            Self::BarComplexTooLarge {
                degree,
                rows,
                columns,
                cell_limit,
            } => write!(
                f,
                "normalized bar differential d_{degree} has shape {rows} x {columns}, exceeding the implementation work ceiling of {cell_limit}"
            ),
            Self::BasisCardinalityOverflow {
                degree,
                alphabet_size,
            } => write!(
                f,
                "normalized bar basis cardinality {alphabet_size}^{degree} exceeds usize"
            ),
            Self::DegreeOutOfRange { degree, max_degree } => write!(
                f,
                "cohomological degree {degree} lies above the exact truncation degree {max_degree}"
            ),
            Self::BasisIndexOutOfRange {
                degree,
                index,
                dimension,
            } => write!(
                f,
                "basis index {index} is outside the dimension-{dimension} cohomology group in degree {degree}"
            ),
            Self::Group(error) => error.fmt(f),
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::GradedAlgebra(error) => error.fmt(f),
            Self::AlgebraDegreeOverflow { degree } => {
                write!(f, "cohomological degree {degree} exceeds the graded-algebra degree type")
            }
            Self::InvalidAlgebraTruncation { reason } => {
                write!(f, "invalid finite-group cohomology algebra truncation: {reason}")
            }
            Self::CompletionBoundNotReached {
                computed_through_degree,
                required_through_degree,
            } => write!(
                f,
                "cohomology is exact through degree {computed_through_degree}, but the completion theorem requires degree {required_through_degree}"
            ),
            Self::CompletionBoundOverflow => {
                f.write_str("finite-group cohomology completion bound exceeds usize")
            }
            Self::ParameterEvidenceUnavailable { reason } => write!(
                f,
                "finite-group cohomology parameter evidence is unavailable: {reason}"
            ),
            Self::Verification(error) => error.fmt(f),
            Self::Topology(error) => error.fmt(f),
            Self::CohomologyOperationUnavailable { reason } => {
                write!(f, "finite-group cohomology operation is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for FiniteGroupCohomologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Group(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::GradedAlgebra(error) => Some(error),
            Self::Verification(error) => Some(error),
            Self::Topology(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GroupError> for FiniteGroupCohomologyError {
    fn from(value: GroupError) -> Self {
        Self::Group(value)
    }
}

impl From<LinalgError> for FiniteGroupCohomologyError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<GradedCommutativeError> for FiniteGroupCohomologyError {
    fn from(value: GradedCommutativeError) -> Self {
        Self::GradedAlgebra(value)
    }
}

impl From<VerificationError> for FiniteGroupCohomologyError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

impl From<TopologyError> for FiniteGroupCohomologyError {
    fn from(value: TopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Failure to compute or query integral finite-group cohomology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteGroupIntegralCohomologyError {
    /// A sparse normalized-bar differential would exceed its work ceiling.
    BarComplexTooLarge {
        /// Differential degree.
        degree: usize,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
        /// Upper bound on materialized sparse contributions.
        estimated_entries: usize,
        /// Current implementation ceiling.
        entry_limit: usize,
    },
    /// A normalized bar basis cardinality overflowed `usize`.
    BasisCardinalityOverflow {
        /// Basis degree.
        degree: usize,
        /// Number of nonidentity group elements forming the alphabet.
        alphabet_size: usize,
    },
    /// A requested degree lies outside the exact truncation.
    DegreeOutOfRange {
        /// Requested degree.
        degree: usize,
        /// Largest computed cohomological degree.
        max_degree: usize,
    },
    /// A requested basis vector does not exist in the named degree.
    BasisIndexOutOfRange {
        /// Cohomological degree.
        degree: usize,
        /// Requested zero-based basis index.
        index: usize,
        /// Number of nontrivial cyclic and free basis summands.
        basis_size: usize,
    },
    /// Exact abelianization failed.
    AbelianInvariants(FiniteAbelianInvariantError),
    /// Canonical abelian-group construction failed.
    AbelianGroup(AbelianGroupError),
    /// Exact integer linear algebra failed.
    LinearAlgebra(LinalgError),
    /// The concrete permutation-group operation failed.
    Group(GroupError),
    /// Retained evidence failed exact recomputation.
    Verification(VerificationError),
}

impl fmt::Display for FiniteGroupIntegralCohomologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BarComplexTooLarge {
                degree,
                rows,
                columns,
                estimated_entries,
                entry_limit,
            } => write!(
                f,
                "normalized integral bar differential d_{degree} has shape {rows} x {columns} and up to {estimated_entries} sparse contributions, exceeding the implementation ceiling of {entry_limit}"
            ),
            Self::BasisCardinalityOverflow {
                degree,
                alphabet_size,
            } => write!(
                f,
                "normalized bar basis cardinality {alphabet_size}^{degree} exceeds usize"
            ),
            Self::DegreeOutOfRange { degree, max_degree } => write!(
                f,
                "integral cohomological degree {degree} lies above the exact truncation degree {max_degree}"
            ),
            Self::BasisIndexOutOfRange {
                degree,
                index,
                basis_size,
            } => write!(
                f,
                "basis index {index} is outside the {basis_size}-summand integral cohomology group in degree {degree}"
            ),
            Self::AbelianInvariants(error) => error.fmt(f),
            Self::AbelianGroup(error) => error.fmt(f),
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::Group(error) => error.fmt(f),
            Self::Verification(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for FiniteGroupIntegralCohomologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AbelianInvariants(error) => Some(error),
            Self::AbelianGroup(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::Group(error) => Some(error),
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FiniteAbelianInvariantError> for FiniteGroupIntegralCohomologyError {
    fn from(value: FiniteAbelianInvariantError) -> Self {
        Self::AbelianInvariants(value)
    }
}

impl From<AbelianGroupError> for FiniteGroupIntegralCohomologyError {
    fn from(value: AbelianGroupError) -> Self {
        Self::AbelianGroup(value)
    }
}

impl From<LinalgError> for FiniteGroupIntegralCohomologyError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<GroupError> for FiniteGroupIntegralCohomologyError {
    fn from(value: GroupError) -> Self {
        Self::Group(value)
    }
}

impl From<VerificationError> for FiniteGroupIntegralCohomologyError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

#[derive(Clone, PartialEq, Eq)]
struct CohomologyDegree {
    coboundary_basis: Vec<Vector<FiniteField>>,
    representatives: Vec<Vector<FiniteField>>,
    coordinate_basis: Matrix<FiniteField>,
}

#[derive(Clone, PartialEq, Eq)]
struct MinimalPGroupResolution {
    elements: Vec<Permutation>,
    multiplication: Vec<Vec<usize>>,
    identity_index: usize,
    ranks: Vec<usize>,
    differentials: Vec<SparseMatrix<FiniteField>>,
}

#[derive(Clone)]
struct StableElementModel {
    sylow_cohomology: Box<FiniteGroupCohomologyTruncation>,
    stability_operators: Vec<Matrix<FiniteField>>,
}

/// Exact strategy used for an integral finite-group cohomology truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteGroupIntegralResolutionStrategy {
    /// The universal description in degrees zero through two, using
    /// `H^2(BG; Z) = Ext(G_ab, Z)` for finite `G`.
    UniversalLowDegree,
    /// The sparse normalized inhomogeneous integral bar resolution, followed
    /// by algebraic Morse reduction and exact Hermite/Smith witnesses.
    NormalizedBar,
}

#[derive(Clone, PartialEq, Eq)]
struct IntegralCohomologyDegree {
    additive_group: FinitelyGeneratedAbelianGroup,
    basis_orders: Vec<Option<Integer>>,
    representatives: Vec<Vector<IntegerRing>>,
    coordinate_map: Matrix<IntegerRing>,
}

/// Exact initial segment of the integral cohomology ring `H^*(BG; Z)`.
///
/// Every additive group is in canonical invariant-factor form. When the
/// normalized bar strategy is used, the object retains sparse bar
/// differentials, a sparse algebraic-Morse contraction, deterministic integer
/// cocycles, and Smith transformation maps on the reduced complex. Cup
/// products are formed on normalized bar cochains and then projected to
/// canonical torsion/free coordinates. The universal low-degree strategy
/// avoids constructing the third bar differential and is exact for every
/// finite permutation group through degree two.
#[derive(Clone)]
pub struct FiniteGroupIntegralCohomologyTruncation {
    group: PermutationGroup,
    max_degree: usize,
    strategy: FiniteGroupIntegralResolutionStrategy,
    alphabet_size: usize,
    boundaries: Vec<SparseMatrix<IntegerRing>>,
    contraction: Option<SparseChainContraction<IntegerRing>>,
    cohomology: Vec<IntegralCohomologyDegree>,
}

impl PartialEq for FiniteGroupIntegralCohomologyTruncation {
    fn eq(&self, other: &Self) -> bool {
        self.group.degree() == other.group.degree()
            && self.group.generators() == other.group.generators()
            && self.max_degree == other.max_degree
            && self.strategy == other.strategy
            && self.alphabet_size == other.alphabet_size
            && self.boundaries == other.boundaries
            && self.cohomology == other.cohomology
    }
}

impl Eq for FiniteGroupIntegralCohomologyTruncation {}

/// Integral cohomology of a concrete finite permutation group.
pub trait FiniteGroupIntegralCohomology {
    /// Compute the exact integral cohomology ring through `max_degree`.
    fn integral_cohomology(
        &self,
        max_degree: usize,
    ) -> Result<FiniteGroupIntegralCohomologyTruncation, FiniteGroupIntegralCohomologyError>;
}

impl FiniteGroupIntegralCohomology for PermutationGroup {
    fn integral_cohomology(
        &self,
        max_degree: usize,
    ) -> Result<FiniteGroupIntegralCohomologyTruncation, FiniteGroupIntegralCohomologyError> {
        FiniteGroupIntegralCohomologyTruncation::build(self, max_degree)
    }
}

/// Exact initial segment of `H^*(BG; F_p)` with canonical resolution cocycles.
///
/// The result contains every group through [`Self::max_degree`], not a claim
/// that the cohomology ring is generated or presented there. Cup products are
/// available whenever the sum of the input degrees remains in the truncation.
/// The retained free-resolution differentials and deterministic cocycles make
/// the calculation independently replayable. The mathematical strategy is
/// explicit: p-groups use a minimal resolution over their local group algebra,
/// while other groups use the normalized bar resolution when it fits and a
/// certified Sylow stable-elements calculation otherwise.
#[derive(Clone)]
pub struct FiniteGroupCohomologyTruncation {
    group: PermutationGroup,
    field: FiniteField,
    max_degree: usize,
    basis_dimensions: Vec<usize>,
    boundaries: Vec<SparseMatrix<FiniteField>>,
    bar_contraction: Option<SparseChainContraction<FiniteField>>,
    cohomology: Vec<CohomologyDegree>,
    strategy: FiniteGroupResolutionStrategy,
    minimal_p_group: Option<MinimalPGroupResolution>,
    stable_elements: Option<StableElementModel>,
}

/// One minimal homogeneous algebra generator selected from the canonical
/// cohomology basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteGroupCohomologyAlgebraGenerator {
    name: String,
    display_name: String,
    degree: usize,
    basis_index: usize,
    operation_witness: Option<FiniteGroupCohomologyGeneratorOperationWitness>,
}

/// Replayable evidence that a chosen algebra generator is named by a primary
/// cohomology operation modulo decomposable classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteGroupCohomologyGeneratorOperationWitness {
    source_generator_index: usize,
    operation: AdmissibleCohomologyOperation,
    operation_coefficient: FiniteFieldElement,
    decomposable_correction: Vector<FiniteField>,
}

impl FiniteGroupCohomologyAlgebraGenerator {
    /// Stable name used by the associated graded-commutative presentation.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Mathematical display name, using a cohomology operation when the
    /// retained witness supplies one.
    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Cohomological degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Zero-based index in the canonical basis of this degree.
    #[must_use]
    pub const fn basis_index(&self) -> usize {
        self.basis_index
    }

    /// Operation witness used to name this indecomposable class, when one was
    /// found deterministically.
    #[must_use]
    pub const fn operation_witness(
        &self,
    ) -> Option<&FiniteGroupCohomologyGeneratorOperationWitness> {
        self.operation_witness.as_ref()
    }
}

impl FiniteGroupCohomologyGeneratorOperationWitness {
    /// Index of the earlier algebra generator to which the operation applies.
    #[must_use]
    pub const fn source_generator_index(&self) -> usize {
        self.source_generator_index
    }

    /// Typed admissible primary operation.
    #[must_use]
    pub const fn operation(&self) -> &AdmissibleCohomologyOperation {
        &self.operation
    }

    /// Nonzero scalar multiplying the operation image.
    #[must_use]
    pub const fn operation_coefficient(&self) -> &FiniteFieldElement {
        &self.operation_coefficient
    }

    /// Decomposable class added to the scaled operation image to obtain the
    /// selected canonical basis generator exactly.
    #[must_use]
    pub const fn decomposable_correction(&self) -> &Vector<FiniteField> {
        &self.decomposable_correction
    }
}

/// Minimal generators and all homogeneous relations visible in an exact
/// initial segment of a finite group's mod-p cohomology ring.
///
/// This is deliberately not a complete-presentation claim. The quotient has
/// exactly the supplied cohomology dimensions and multiplication table through
/// [`Self::max_degree`]; its behavior above that degree is unspecified until a
/// separate completion theorem is certified.
#[derive(Clone)]
pub struct FiniteGroupCohomologyAlgebraTruncation {
    cohomology: FiniteGroupCohomologyTruncation,
    generators: Vec<FiniteGroupCohomologyAlgebraGenerator>,
    presentation: GradedCommutativePresentation,
}

/// The theorem used to promote an exact algebra truncation to a complete
/// presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteGroupCohomologyCompletionStrategy {
    /// Symonds's universal generator bound `|G|-1` and relation bound
    /// `2(|G|-1)`.
    SymondsUniversalBound,
    /// Symonds's parameter criterion, with parameterhood certified by exact
    /// restriction to every maximal elementary abelian p-subgroup.
    SymondsParameterCriterion,
}

/// One maximal elementary abelian restriction used to prove that a selected
/// family consists of parameters for the full cohomology ring.
#[derive(Clone, Debug)]
pub struct FiniteGroupCohomologyElementaryAbelianRestriction {
    subgroup: PermutationGroup,
    rank: usize,
    restricted_parameters: Vec<crate::commalg::GradedCommutativePolynomial>,
    quotient_generator_power_bounds: Vec<u32>,
}

impl PartialEq for FiniteGroupCohomologyElementaryAbelianRestriction {
    fn eq(&self, other: &Self) -> bool {
        self.subgroup.degree() == other.subgroup.degree()
            && self.subgroup.generators() == other.subgroup.generators()
            && self.rank == other.rank
            && self.restricted_parameters == other.restricted_parameters
            && self.quotient_generator_power_bounds == other.quotient_generator_power_bounds
    }
}

impl Eq for FiniteGroupCohomologyElementaryAbelianRestriction {}

impl FiniteGroupCohomologyElementaryAbelianRestriction {
    /// Literal maximal elementary abelian subgroup of the retained Sylow
    /// subgroup.
    #[must_use]
    pub const fn subgroup(&self) -> &PermutationGroup {
        &self.subgroup
    }

    /// Elementary abelian rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Restrictions of the selected parameters in a deterministic complete
    /// presentation of this subgroup's cohomology.
    #[must_use]
    pub fn restricted_parameters(&self) -> &[crate::commalg::GradedCommutativePolynomial] {
        &self.restricted_parameters
    }

    /// A pure power of each non-exterior presentation generator killed by the
    /// restricted parameter ideal. Exterior generators record the intrinsic
    /// square-zero bound `2`.
    #[must_use]
    pub fn quotient_generator_power_bounds(&self) -> &[u32] {
        &self.quotient_generator_power_bounds
    }
}

/// Replayable evidence for Symonds's parameter completion criterion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteGroupCohomologySymondsParameterCertificate {
    parameter_generator_indices: Vec<usize>,
    elementary_abelian_restrictions: Vec<FiniteGroupCohomologyElementaryAbelianRestriction>,
    quotient_generator_power_bounds: Vec<u32>,
    maximum_module_generator_degree: usize,
    parameter_degree_bound: usize,
    completion_degree_bound: usize,
}

impl FiniteGroupCohomologySymondsParameterCertificate {
    /// Indices in the complete presentation's deterministic algebra-generator
    /// family.
    #[must_use]
    pub fn parameter_generator_indices(&self) -> &[usize] {
        &self.parameter_generator_indices
    }

    /// Restriction certificates for all maximal elementary abelian
    /// subgroups of the retained Sylow subgroup.
    #[must_use]
    pub fn elementary_abelian_restrictions(
        &self,
    ) -> &[FiniteGroupCohomologyElementaryAbelianRestriction] {
        &self.elementary_abelian_restrictions
    }

    /// Pure-power bounds proving that the algebra approximation is finite over
    /// the selected parameter subalgebra.
    #[must_use]
    pub fn quotient_generator_power_bounds(&self) -> &[u32] {
        &self.quotient_generator_power_bounds
    }

    /// Largest degree needed to generate the approximation as a module over
    /// the selected parameters.
    #[must_use]
    pub const fn maximum_module_generator_degree(&self) -> usize {
        self.maximum_module_generator_degree
    }

    /// `sum_i (degree(parameter_i) - 1) + 1` in Symonds's theorem.
    #[must_use]
    pub const fn parameter_degree_bound(&self) -> usize {
        self.parameter_degree_bound
    }

    /// Maximum of the module-generator and parameter-degree bounds.
    #[must_use]
    pub const fn completion_degree_bound(&self) -> usize {
        self.completion_degree_bound
    }
}

/// A complete graded-commutative presentation of `H^*(BG; F_p)` with a
/// theorem-backed completion certificate.
#[derive(Clone)]
pub struct FiniteGroupCohomologyCompletePresentation {
    algebra: FiniteGroupCohomologyAlgebraTruncation,
    strategy: FiniteGroupCohomologyCompletionStrategy,
    group_order: Integer,
    generator_degree_bound: usize,
    relation_degree_bound: usize,
    parameter_certificate: Option<FiniteGroupCohomologySymondsParameterCertificate>,
}

/// Cohomology operations on a supplied concrete finite group.
pub trait FiniteGroupCohomology {
    /// Compute `H^*(BG; F_p)` exactly through `max_degree`.
    ///
    /// The playbook uses a minimal free resolution when the supplied group is
    /// a p-group, the normalized inhomogeneous bar resolution as a small exact
    /// oracle, and Sylow stable elements beyond that oracle's measured work
    /// regime. Both the p-group and normalized-bar lanes retain sparse
    /// differentials.
    fn modular_cohomology(
        &self,
        prime: u64,
        max_degree: usize,
    ) -> Result<FiniteGroupCohomologyTruncation, FiniteGroupCohomologyError>;
}

impl FiniteGroupCohomology for PermutationGroup {
    fn modular_cohomology(
        &self,
        prime: u64,
        max_degree: usize,
    ) -> Result<FiniteGroupCohomologyTruncation, FiniteGroupCohomologyError> {
        FiniteGroupCohomologyTruncation::build(self, prime, max_degree)
    }
}

impl FiniteGroupCohomologyTruncation {
    fn build(
        group: &PermutationGroup,
        prime: u64,
        max_degree: usize,
    ) -> Result<Self, FiniteGroupCohomologyError> {
        let field = FiniteField::prime(prime)
            .map_err(|_| FiniteGroupCohomologyError::InvalidPrime { prime })?;
        let elements = ordered_group_elements(group);
        if is_prime_power(elements.len(), prime) {
            let minimal = build_minimal_p_group_resolution(group, &field, &elements, max_degree)?;
            let cohomology = minimal
                .ranks
                .iter()
                .map(|rank| CohomologyDegree {
                    coboundary_basis: Vec::new(),
                    representatives: standard_basis(&field, *rank),
                    coordinate_basis: Matrix::identity(field.clone(), *rank),
                })
                .collect();
            return Ok(Self {
                group: group.clone(),
                field,
                max_degree,
                basis_dimensions: Vec::new(),
                boundaries: Vec::new(),
                bar_contraction: None,
                cohomology,
                strategy: FiniteGroupResolutionStrategy::MinimalPGroup,
                minimal_p_group: Some(minimal),
                stable_elements: None,
            });
        }
        if !normalized_bar_fits(elements.len().saturating_sub(1), max_degree) {
            return Self::build_from_stable_elements(group, field, max_degree);
        }
        let (nonidentity, multiplication) = normalized_alphabet(group)?;
        let alphabet_size = nonidentity.len();
        let basis_dimensions = (0..=max_degree.saturating_add(1))
            .map(|degree| {
                normalized_basis_dimension(alphabet_size, degree).ok_or(
                    FiniteGroupCohomologyError::BasisCardinalityOverflow {
                        degree,
                        alphabet_size,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        for degree in 1..=max_degree.saturating_add(1) {
            let rows = basis_dimensions[degree - 1];
            let columns = basis_dimensions[degree];
            let estimated_entries = normalized_bar_entry_bound(columns, degree).ok_or(
                FiniteGroupCohomologyError::BarComplexTooLarge {
                    degree,
                    rows,
                    columns,
                    cell_limit: NORMALIZED_BAR_SPARSE_ENTRY_LIMIT,
                },
            )?;
            if estimated_entries > NORMALIZED_BAR_SPARSE_ENTRY_LIMIT {
                return Err(FiniteGroupCohomologyError::BarComplexTooLarge {
                    degree,
                    rows,
                    columns,
                    cell_limit: NORMALIZED_BAR_SPARSE_ENTRY_LIMIT,
                });
            }
        }
        let mut boundaries = Vec::with_capacity(max_degree.saturating_add(2));
        boundaries.push(SparseMatrix::zeros(field.clone(), 0, 1));
        for degree in 1..=max_degree.saturating_add(1) {
            boundaries.push(normalized_boundary(
                &field,
                alphabet_size,
                &multiplication,
                degree,
            ));
        }

        let mut coboundaries = BTreeMap::new();
        for degree in 0..=max_degree {
            let displayed_degree = i32::try_from(degree).map_err(|_| {
                FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                    claim: "modular bar truncation degree fits the signed grading type",
                })
            })?;
            coboundaries.insert(-displayed_degree, boundaries[degree + 1].transpose());
        }
        let contraction =
            SparseChainContraction::from_trusted_lowering(field.clone(), coboundaries).map_err(
                |_| {
                    FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                        claim: "sparse modular bar contraction is exact and replayable",
                    })
                },
            )?;
        let mut cohomology = Vec::with_capacity(max_degree.saturating_add(1));
        for degree in 0..=max_degree {
            let internal_degree = -i32::try_from(degree).map_err(|_| {
                FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                    claim: "modular bar truncation degree fits the signed grading type",
                })
            })?;
            let ambient_dimension = contraction.reduced_dimension(internal_degree);
            let incoming = if degree == 0 {
                Matrix::zeros(field.clone(), ambient_dimension, 0)
            } else {
                contraction
                    .reduced_differential(internal_degree + 1)
                    .expect("the sparse truncation contains the incoming coboundary")
                    .to_dense()
            };
            let outgoing = contraction
                .reduced_differential(internal_degree)
                .expect("the sparse truncation contains the outgoing coboundary")
                .to_dense();
            let coboundary_basis = incoming.image()?.basis().to_vec();
            let cocycle_basis = outgoing.kernel()?.basis().to_vec();
            let all_columns = coboundary_basis
                .iter()
                .chain(cocycle_basis.iter())
                .cloned()
                .collect::<Vec<_>>();
            let span = matrix_from_columns(&field, ambient_dimension, &all_columns)?;
            let pivots = span.pivot_columns_from_elimination()?;
            let coboundary_dimension = coboundary_basis.len();
            let reduced_representatives = pivots
                .into_iter()
                .filter_map(|column| {
                    column
                        .checked_sub(coboundary_dimension)
                        .and_then(|index| cocycle_basis.get(index).cloned())
                })
                .collect::<Vec<_>>();
            let coordinate_columns = coboundary_basis
                .iter()
                .chain(reduced_representatives.iter())
                .cloned()
                .collect::<Vec<_>>();
            let coordinate_basis =
                matrix_from_columns(&field, ambient_dimension, &coordinate_columns)?;
            let representatives = reduced_representatives
                .iter()
                .map(|representative| contraction.lift(internal_degree, representative))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                        claim: "reduced modular cocycles lift to normalized bar cocycles",
                    })
                })?;
            cohomology.push(CohomologyDegree {
                coboundary_basis,
                representatives,
                coordinate_basis,
            });
        }

        Ok(Self {
            group: group.clone(),
            field,
            max_degree,
            basis_dimensions,
            boundaries,
            bar_contraction: Some(contraction),
            cohomology,
            strategy: FiniteGroupResolutionStrategy::NormalizedBar,
            minimal_p_group: None,
            stable_elements: None,
        })
    }

    fn build_from_stable_elements(
        group: &PermutationGroup,
        field: FiniteField,
        max_degree: usize,
    ) -> Result<Self, FiniteGroupCohomologyError> {
        let prime = field
            .characteristic()
            .to_string()
            .parse::<u64>()
            .map_err(|_| FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
        let sylow = group.sylow_subgroup(prime)?;
        let sylow_cohomology = Box::new(Self::build(&sylow, prime, max_degree)?);
        let sylow_resolution =
            sylow_cohomology
                .minimal_p_group
                .as_ref()
                .ok_or(LinalgError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "a Sylow p-subgroup uses the minimal p-group resolution lane",
                    },
                ))?;
        let group_elements = ordered_group_elements(group);
        let mut operators_by_degree = (0..=max_degree)
            .map(|degree| Matrix::zeros(field.clone(), 0, sylow_resolution.ranks[degree]))
            .collect::<Vec<_>>();

        for conjugator in group_elements {
            let inverse = conjugator.inverse();
            let conjugated = PermutationGroup::from_generators(
                group.degree(),
                &sylow
                    .generators()
                    .iter()
                    .map(|generator| generator.conjugate(&inverse))
                    .collect::<Result<Vec<_>, GroupError>>()?,
            )?;
            let intersection = sylow.intersection(&conjugated)?;
            let intersection_elements = ordered_group_elements(&intersection);
            let intersection_resolution = build_minimal_p_group_resolution(
                &intersection,
                &field,
                &intersection_elements,
                max_degree,
            )?;
            let inclusion = resolution_restriction_maps(
                &field,
                &intersection_resolution,
                sylow_resolution,
                &intersection_resolution
                    .elements
                    .iter()
                    .map(|element| sylow_element_index(sylow_resolution, element))
                    .collect::<Result<Vec<_>, _>>()?,
                max_degree,
            )?;
            let conjugation = resolution_restriction_maps(
                &field,
                &intersection_resolution,
                sylow_resolution,
                &intersection_resolution
                    .elements
                    .iter()
                    .map(|element| {
                        sylow_element_index(sylow_resolution, &element.conjugate(&conjugator)?)
                    })
                    .collect::<Result<Vec<_>, FiniteGroupCohomologyError>>()?,
                max_degree,
            )?;
            for degree in 0..=max_degree {
                let difference = inclusion[degree].sub(&conjugation[degree])?;
                operators_by_degree[degree] =
                    stack_matrix_rows(&field, &operators_by_degree[degree], &difference)?;
            }
        }

        let mut cohomology = Vec::with_capacity(max_degree.saturating_add(1));
        for operator in &operators_by_degree {
            let representatives = operator.kernel()?.basis().to_vec();
            let coordinate_basis =
                matrix_from_columns(&field, operator.shape().1, &representatives)?;
            cohomology.push(CohomologyDegree {
                coboundary_basis: Vec::new(),
                representatives,
                coordinate_basis,
            });
        }

        Ok(Self {
            group: group.clone(),
            field,
            max_degree,
            basis_dimensions: Vec::new(),
            boundaries: Vec::new(),
            bar_contraction: None,
            cohomology,
            strategy: FiniteGroupResolutionStrategy::SylowStableElements,
            minimal_p_group: None,
            stable_elements: Some(StableElementModel {
                sylow_cohomology,
                stability_operators: operators_by_degree,
            }),
        })
    }

    /// Supplied concrete permutation group.
    #[must_use]
    pub const fn group(&self) -> &PermutationGroup {
        &self.group
    }

    /// Exact prime coefficient field.
    #[must_use]
    pub const fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Largest cohomological degree computed exactly.
    #[must_use]
    pub const fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Mathematical strategy used by this calculation.
    #[must_use]
    pub const fn resolution_strategy(&self) -> FiniteGroupResolutionStrategy {
        self.strategy
    }

    /// Dimension of `H^degree(BG; F_p)`.
    pub fn dimension(&self, degree: usize) -> Result<usize, FiniteGroupCohomologyError> {
        self.cohomology
            .get(degree)
            .map(|value| value.representatives.len())
            .ok_or(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            })
    }

    /// Deterministic model cocycles representing a cohomology basis.
    ///
    /// The minimal and bar lanes return cochains in their named resolutions.
    /// The stable-elements lane returns coordinates in the canonical basis of
    /// the retained Sylow-resolution cohomology.
    pub fn basis_cocycles(
        &self,
        degree: usize,
    ) -> Result<&[Vector<FiniteField>], FiniteGroupCohomologyError> {
        self.cohomology
            .get(degree)
            .map(|value| value.representatives.as_slice())
            .ok_or(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            })
    }

    /// Coordinates of the cup product of two canonical basis classes.
    ///
    /// Coordinates use the canonical basis returned by [`Self::basis_cocycles`]
    /// in degree `left_degree + right_degree`.
    pub fn basis_cup_product(
        &self,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
        right_index: usize,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        let target_degree = left_degree.checked_add(right_degree).ok_or(
            FiniteGroupCohomologyError::DegreeOutOfRange {
                degree: usize::MAX,
                max_degree: self.max_degree,
            },
        )?;
        if target_degree > self.max_degree {
            return Err(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree: target_degree,
                max_degree: self.max_degree,
            });
        }
        let left = self.basis_vector(left_degree, left_index)?;
        let right = self.basis_vector(right_degree, right_index)?;
        if let Some(resolution) = &self.minimal_p_group {
            return minimal_p_group_basis_cup_product(
                &self.field,
                resolution,
                left_degree,
                left_index,
                right_degree,
                right_index,
            );
        }
        if let Some(stable) = &self.stable_elements {
            let ambient_dimension = stable.sylow_cohomology.dimension(target_degree)?;
            let mut ambient = vec![self.field.zero(); ambient_dimension];
            for left_ambient in 0..left.dimension() {
                let left_coefficient = left
                    .get(left_ambient)
                    .expect("stable-element coordinate exists");
                if left_coefficient.is_zero() {
                    continue;
                }
                for right_ambient in 0..right.dimension() {
                    let right_coefficient = right
                        .get(right_ambient)
                        .expect("stable-element coordinate exists");
                    if right_coefficient.is_zero() {
                        continue;
                    }
                    let coefficient = self
                        .field
                        .mul(&left_coefficient, &right_coefficient)
                        .map_err(LinalgError::from)?;
                    let product = stable.sylow_cohomology.basis_cup_product(
                        left_degree,
                        left_ambient,
                        right_degree,
                        right_ambient,
                    )?;
                    for (coordinate, value) in ambient.iter_mut().zip(product.values()) {
                        let term = self
                            .field
                            .mul(&coefficient, &value)
                            .map_err(LinalgError::from)?;
                        *coordinate = self
                            .field
                            .add(coordinate, &term)
                            .map_err(LinalgError::from)?;
                    }
                }
            }
            let ambient = Vector::from_values(self.field.clone(), ambient)?;
            return self.cohomology_coordinates(target_degree, &ambient);
        }
        let product = normalized_cup_product(
            &self.field,
            self.basis_dimensions[1],
            left_degree,
            left,
            right_degree,
            right,
        )?;
        self.cohomology_coordinates(target_degree, &product)
    }

    /// Matrix of multiplication on the left by one canonical basis class.
    ///
    /// Columns are indexed by the canonical basis in `right_degree`; rows are
    /// indexed by the canonical basis in `left_degree + right_degree`. This
    /// batched operation reuses one resolution lift across all right-hand
    /// basis vectors.
    pub fn basis_left_multiplication(
        &self,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
    ) -> Result<Matrix<FiniteField>, FiniteGroupCohomologyError> {
        let target_degree = left_degree.checked_add(right_degree).ok_or(
            FiniteGroupCohomologyError::DegreeOutOfRange {
                degree: usize::MAX,
                max_degree: self.max_degree,
            },
        )?;
        if target_degree > self.max_degree {
            return Err(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree: target_degree,
                max_degree: self.max_degree,
            });
        }
        self.basis_vector(left_degree, left_index)?;
        let right_dimension = self.dimension(right_degree)?;
        if let Some(resolution) = &self.minimal_p_group {
            return minimal_p_group_basis_left_multiplication(
                &self.field,
                resolution,
                left_degree,
                left_index,
                right_degree,
            );
        }
        let columns = (0..right_dimension)
            .map(|right_index| {
                self.basis_cup_product(left_degree, left_index, right_degree, right_index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        matrix_from_columns(&self.field, self.dimension(target_degree)?, &columns)
            .map_err(Into::into)
    }

    /// Apply one typed admissible Steenrod word to a canonical basis class.
    ///
    /// Identity, instability, and top-operation formulas are evaluated in
    /// every resolution strategy. Genuinely lower operations are evaluated
    /// cochain-level on the normalized nerve of the group when this
    /// truncation uses the normalized bar resolution. Reduced-resolution
    /// lanes fail closed until a checked bar comparison is available.
    pub fn apply_operation(
        &self,
        operation: &AdmissibleCohomologyOperation,
        source_degree: usize,
        source_index: usize,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        let characteristic = self
            .field
            .characteristic()
            .to_u64_checked()
            .ok_or(FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
        if characteristic != operation.prime() {
            return Err(FiniteGroupCohomologyError::InvalidPrime {
                prime: operation.prime(),
            });
        }
        let dimension = self.dimension(source_degree)?;
        if source_index >= dimension {
            return Err(FiniteGroupCohomologyError::BasisIndexOutOfRange {
                degree: source_degree,
                index: source_index,
                dimension,
            });
        }
        let mut degree = source_degree;
        let mut value = standard_basis_vector(&self.field, dimension, source_index)?;
        if let Some(sequence) = operation.square_sequence() {
            for square in sequence.iter().rev() {
                (degree, value) = self.apply_square_to_class(
                    degree,
                    &value,
                    usize::try_from(*square).map_err(|_| {
                        FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
                    })?,
                )?;
            }
            return Ok(value);
        }
        let bocksteins = operation
            .bocksteins()
            .expect("an odd-primary operation has Bockstein exponents");
        let powers = operation
            .reduced_powers()
            .expect("an odd-primary operation has reduced powers");
        if bocksteins.last().copied().unwrap_or(false) {
            (degree, value) = self.apply_odd_operation_to_class(degree, &value, 0, true)?;
        }
        for index in (0..powers.len()).rev() {
            let power = usize::try_from(powers[index]).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?;
            (degree, value) =
                self.apply_odd_operation_to_class(degree, &value, power, bocksteins[index])?;
        }
        Ok(value)
    }

    fn apply_square_to_class(
        &self,
        degree: usize,
        value: &Vector<FiniteField>,
        square: usize,
    ) -> Result<(usize, Vector<FiniteField>), FiniteGroupCohomologyError> {
        let target_degree =
            degree
                .checked_add(square)
                .ok_or(FiniteGroupCohomologyError::DegreeOutOfRange {
                    degree: usize::MAX,
                    max_degree: self.max_degree,
                })?;
        self.require_target_degree(target_degree)?;
        if square == 0 {
            return Ok((degree, value.clone()));
        }
        if square > degree {
            return Ok((
                target_degree,
                Vector::zeros(self.field.clone(), self.dimension(target_degree)?),
            ));
        }
        if square == degree {
            return Ok((
                target_degree,
                self.multiply_classes(degree, value, degree, value)?,
            ));
        }
        let atomic =
            AdmissibleCohomologyOperation::steenrod_square(u32::try_from(square).map_err(
                |_| FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX },
            )?);
        if let Some(result) = self.apply_stable_operation(degree, value, &atomic)? {
            return Ok((target_degree, result));
        }
        if let Some(result) = self.apply_cyclic_minimal_operation(degree, value, square, false)? {
            return Ok((target_degree, result));
        }
        self.apply_bar_cochain_operation(degree, value, target_degree, |nerve, cocycle| {
            nerve.steenrod_square(square, degree, cocycle)
        })
        .map(|value| (target_degree, value))
    }

    fn apply_odd_operation_to_class(
        &self,
        degree: usize,
        value: &Vector<FiniteField>,
        power: usize,
        bockstein: bool,
    ) -> Result<(usize, Vector<FiniteField>), FiniteGroupCohomologyError> {
        let prime = self
            .field
            .characteristic()
            .to_u64_checked()
            .ok_or(FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
        let reduced_shift = power
            .checked_mul(usize::try_from(prime.saturating_sub(1)).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?)
            .and_then(|value| value.checked_mul(2))
            .ok_or(FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX })?;
        let target_degree = degree
            .checked_add(reduced_shift)
            .and_then(|value| value.checked_add(usize::from(bockstein)))
            .ok_or(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree: usize::MAX,
                max_degree: self.max_degree,
            })?;
        self.require_target_degree(target_degree)?;
        let doubled_power = power.saturating_mul(2);
        if bockstein && doubled_power >= degree {
            return Ok((
                target_degree,
                Vector::zeros(self.field.clone(), self.dimension(target_degree)?),
            ));
        }
        if !bockstein && power == 0 {
            return Ok((degree, value.clone()));
        }
        if !bockstein && doubled_power > degree {
            return Ok((
                target_degree,
                Vector::zeros(self.field.clone(), self.dimension(target_degree)?),
            ));
        }
        if !bockstein && doubled_power == degree {
            let exponent = usize::try_from(prime).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?;
            return Ok((target_degree, self.power_class(degree, value, exponent)?));
        }
        let power_u32 = u32::try_from(power).map_err(|_| {
            FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
        })?;
        let atomic = if bockstein {
            AdmissibleCohomologyOperation::bockstein_reduced_power(prime, power_u32)?
        } else {
            AdmissibleCohomologyOperation::reduced_power(prime, power_u32)?
        };
        if let Some(result) = self.apply_stable_operation(degree, value, &atomic)? {
            return Ok((target_degree, result));
        }
        if let Some(result) =
            self.apply_cyclic_minimal_operation(degree, value, power, bockstein)?
        {
            return Ok((target_degree, result));
        }
        self.apply_bar_cochain_operation(degree, value, target_degree, |nerve, cocycle| {
            if bockstein {
                nerve.bockstein_reduced_power(power, degree, cocycle)
            } else {
                nerve.steenrod_reduced_power(power, degree, cocycle)
            }
        })
        .map(|value| (target_degree, value))
    }

    fn apply_stable_operation(
        &self,
        degree: usize,
        value: &Vector<FiniteField>,
        operation: &AdmissibleCohomologyOperation,
    ) -> Result<Option<Vector<FiniteField>>, FiniteGroupCohomologyError> {
        let Some(stable) = &self.stable_elements else {
            return Ok(None);
        };
        let target_degree = degree
            .checked_add(usize::try_from(operation.degree_shift()).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?)
            .ok_or(FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX })?;
        let source = self.class_representative(degree, value)?;
        let ambient_dimension = stable.sylow_cohomology.dimension(target_degree)?;
        let mut ambient = vec![self.field.zero(); ambient_dimension];
        for source_index in 0..source.dimension() {
            let coefficient = source
                .get(source_index)
                .expect("stable-element coordinate exists");
            if coefficient.is_zero() {
                continue;
            }
            let image = stable
                .sylow_cohomology
                .apply_operation(operation, degree, source_index)?;
            for (target, coordinate) in ambient.iter_mut().zip(image.values()) {
                let term = self
                    .field
                    .mul(&coefficient, &coordinate)
                    .map_err(LinalgError::from)?;
                *target = self.field.add(target, &term).map_err(LinalgError::from)?;
            }
        }
        Ok(Some(self.cohomology_coordinates(
            target_degree,
            &Vector::from_values(self.field.clone(), ambient)?,
        )?))
    }

    fn apply_cyclic_minimal_operation(
        &self,
        degree: usize,
        value: &Vector<FiniteField>,
        index: usize,
        bockstein: bool,
    ) -> Result<Option<Vector<FiniteField>>, FiniteGroupCohomologyError> {
        if self.strategy != FiniteGroupResolutionStrategy::MinimalPGroup
            || self.dimension(degree)? != 1
        {
            return Ok(None);
        }
        let Ok(invariant_factors) = self.group.abelian_invariant_factors() else {
            return Ok(None);
        };
        if invariant_factors.len() != 1 {
            return Ok(None);
        }
        let prime = self
            .field
            .characteristic()
            .to_u64_checked()
            .ok_or(FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
        let coefficient = if prime == 2 {
            let is_order_two = invariant_factors[0] == 2;
            if !is_order_two && index % 2 == 1 {
                return Ok(Some(Vector::zeros(
                    self.field.clone(),
                    self.dimension(degree + index)?,
                )));
            }
            binomial_mod_prime(
                if is_order_two { degree } else { degree / 2 },
                if is_order_two { index } else { index / 2 },
                prime,
            )
        } else {
            let reduced_coefficient = binomial_mod_prime(degree / 2, index, prime);
            if bockstein {
                let order = invariant_factors[0].to_u64_checked().unwrap_or(u64::MAX);
                if degree.is_multiple_of(2) || order != prime {
                    0
                } else {
                    reduced_coefficient
                }
            } else {
                reduced_coefficient
            }
        };
        let source = value.get(0).expect("cyclic cohomology basis exists");
        let scalar = self.field.element(coefficient);
        Ok(Some(Vector::from_values(
            self.field.clone(),
            [self
                .field
                .mul(&source, &scalar)
                .map_err(LinalgError::from)?],
        )?))
    }

    fn apply_bar_cochain_operation<F>(
        &self,
        source_degree: usize,
        value: &Vector<FiniteField>,
        target_degree: usize,
        operation: F,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError>
    where
        F: FnOnce(
            &SimplicialSet,
            &Vector<FiniteField>,
        ) -> Result<Vector<FiniteField>, TopologyError>,
    {
        if self.strategy != FiniteGroupResolutionStrategy::NormalizedBar {
            return Err(
                FiniteGroupCohomologyError::CohomologyOperationUnavailable {
                    reason: "a lower operation on a reduced resolution requires a checked comparison with the normalized bar resolution",
                },
            );
        }
        let cocycle = self.class_representative(source_degree, value)?;
        let nerve = normalized_group_nerve(&self.group, target_degree)?;
        let result = operation(&nerve, &cocycle)?;
        self.cohomology_coordinates(target_degree, &result)
    }

    fn class_representative(
        &self,
        degree: usize,
        coordinates: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        if coordinates.dimension() != self.dimension(degree)? {
            return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                reason: "cohomology coordinate vector has the wrong dimension",
            });
        }
        let ambient_dimension = self.cohomology[degree].representatives.first().map_or_else(
            || self.basis_dimensions.get(degree).copied().unwrap_or(0),
            Vector::dimension,
        );
        let mut result = vec![self.field.zero(); ambient_dimension];
        for (coefficient, representative) in coordinates
            .values()
            .zip(&self.cohomology[degree].representatives)
        {
            if coefficient.is_zero() {
                continue;
            }
            for (target, entry) in result.iter_mut().zip(representative.values()) {
                let term = self
                    .field
                    .mul(&coefficient, &entry)
                    .map_err(LinalgError::from)?;
                *target = self.field.add(target, &term).map_err(LinalgError::from)?;
            }
        }
        Vector::from_values(self.field.clone(), result).map_err(Into::into)
    }

    fn multiply_classes(
        &self,
        left_degree: usize,
        left: &Vector<FiniteField>,
        right_degree: usize,
        right: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        let target_degree = left_degree + right_degree;
        let mut result = vec![self.field.zero(); self.dimension(target_degree)?];
        for left_index in 0..left.dimension() {
            let left_coefficient = left.get(left_index).expect("class coordinate exists");
            if left_coefficient.is_zero() {
                continue;
            }
            for right_index in 0..right.dimension() {
                let right_coefficient = right.get(right_index).expect("class coordinate exists");
                if right_coefficient.is_zero() {
                    continue;
                }
                let scalar = self
                    .field
                    .mul(&left_coefficient, &right_coefficient)
                    .map_err(LinalgError::from)?;
                let product =
                    self.basis_cup_product(left_degree, left_index, right_degree, right_index)?;
                for (target, coordinate) in result.iter_mut().zip(product.values()) {
                    let term = self
                        .field
                        .mul(&scalar, &coordinate)
                        .map_err(LinalgError::from)?;
                    *target = self.field.add(target, &term).map_err(LinalgError::from)?;
                }
            }
        }
        Vector::from_values(self.field.clone(), result).map_err(Into::into)
    }

    fn power_class(
        &self,
        degree: usize,
        value: &Vector<FiniteField>,
        exponent: usize,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        let mut result = Vector::from_values(self.field.clone(), [self.field.one()])?;
        let mut result_degree = 0;
        for _ in 0..exponent {
            result = self.multiply_classes(result_degree, &result, degree, value)?;
            result_degree = result_degree
                .checked_add(degree)
                .ok_or(FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX })?;
        }
        Ok(result)
    }

    fn require_target_degree(&self, degree: usize) -> Result<(), FiniteGroupCohomologyError> {
        if degree <= self.max_degree {
            Ok(())
        } else {
            Err(FiniteGroupCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            })
        }
    }

    /// Extract minimal homogeneous generators and every relation determined
    /// through the exact cohomological degree bound.
    ///
    /// The returned presentation is a truncated algebra invariant. It is not
    /// asserted to present the ring in higher degrees.
    pub fn algebra_truncation(
        &self,
    ) -> Result<FiniteGroupCohomologyAlgebraTruncation, FiniteGroupCohomologyError> {
        FiniteGroupCohomologyAlgebraTruncation::build(self)
    }

    /// Recompute every differential, quotient basis, and retained cocycle.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let prime = self
            .field
            .characteristic()
            .to_string()
            .parse::<u64>()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "finite-group cohomology coefficient prime fits the public operation",
            })?;
        let rebuilt = Self::build(&self.group, prime, self.max_degree).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "finite-group cohomology truncation is exactly recomputable",
            }
        })?;
        let stable_elements_match = match (&self.stable_elements, &rebuilt.stable_elements) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                left.stability_operators == right.stability_operators
                    && matches!(
                        (
                            left.sylow_cohomology.canonical_bytes(),
                            right.sylow_cohomology.canonical_bytes(),
                        ),
                        (Ok(left_bytes), Ok(right_bytes)) if left_bytes == right_bytes
                    )
            }
            _ => false,
        };
        if self.field != rebuilt.field
            || self.basis_dimensions != rebuilt.basis_dimensions
            || self.boundaries != rebuilt.boundaries
            || self.bar_contraction.is_some() != rebuilt.bar_contraction.is_some()
            || self.cohomology != rebuilt.cohomology
            || self.strategy != rebuilt.strategy
            || self.minimal_p_group != rebuilt.minimal_p_group
            || !stable_elements_match
        {
            return Err(VerificationError::InvalidWitness {
                claim:
                    "finite-group cohomology differentials and quotient bases match recomputation",
            });
        }
        Ok(())
    }

    fn basis_vector(
        &self,
        degree: usize,
        index: usize,
    ) -> Result<&Vector<FiniteField>, FiniteGroupCohomologyError> {
        let values = self.basis_cocycles(degree)?;
        values
            .get(index)
            .ok_or(FiniteGroupCohomologyError::BasisIndexOutOfRange {
                degree,
                index,
                dimension: values.len(),
            })
    }

    fn cohomology_coordinates(
        &self,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
        let data = &self.cohomology[degree];
        let reduced = if let Some(contraction) = &self.bar_contraction {
            let internal_degree = -i32::try_from(degree).map_err(|_| {
                FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                    claim: "modular bar coordinate degree fits the signed grading type",
                })
            })?;
            contraction.project(internal_degree, cocycle).map_err(|_| {
                FiniteGroupCohomologyError::Verification(VerificationError::InvalidWitness {
                    claim: "normalized modular cocycles project to the reduced complex",
                })
            })?
        } else {
            cocycle.clone()
        };
        let solution = data.coordinate_basis.solve(&reduced)?;
        let offset = data.coboundary_basis.len();
        Vector::from_values(
            self.field.clone(),
            solution.particular().values().skip(offset),
        )
        .map_err(Into::into)
    }
}

impl FiniteGroupIntegralCohomologyTruncation {
    fn build(
        group: &PermutationGroup,
        max_degree: usize,
    ) -> Result<Self, FiniteGroupIntegralCohomologyError> {
        if max_degree <= 2 {
            return Self::build_universal_low_degree(group, max_degree);
        }
        Self::build_normalized_bar(group, max_degree)
    }

    fn build_universal_low_degree(
        group: &PermutationGroup,
        max_degree: usize,
    ) -> Result<Self, FiniteGroupIntegralCohomologyError> {
        let mut cohomology = Vec::with_capacity(max_degree.saturating_add(1));
        cohomology.push(integral_degree_without_cochains(
            FinitelyGeneratedAbelianGroup::free(1),
            vec![None],
        ));
        if max_degree >= 1 {
            cohomology.push(integral_degree_without_cochains(
                FinitelyGeneratedAbelianGroup::trivial(),
                Vec::new(),
            ));
        }
        if max_degree >= 2 {
            let invariant_factors = group.abelianization_invariant_factors()?;
            let additive_group = FinitelyGeneratedAbelianGroup::new(0, invariant_factors.clone())?;
            cohomology.push(integral_degree_without_cochains(
                additive_group,
                invariant_factors.into_iter().map(Some).collect(),
            ));
        }
        Ok(Self {
            group: group.clone(),
            max_degree,
            strategy: FiniteGroupIntegralResolutionStrategy::UniversalLowDegree,
            alphabet_size: 0,
            boundaries: Vec::new(),
            contraction: None,
            cohomology,
        })
    }

    fn build_normalized_bar(
        group: &PermutationGroup,
        max_degree: usize,
    ) -> Result<Self, FiniteGroupIntegralCohomologyError> {
        let (_, multiplication) = normalized_alphabet(group).map_err(|error| match error {
            FiniteGroupCohomologyError::Group(error) => {
                FiniteGroupIntegralCohomologyError::Group(error)
            }
            _ => FiniteGroupIntegralCohomologyError::Verification(
                VerificationError::InvalidWitness {
                    claim: "normalized group alphabet construction has one group-error mode",
                },
            ),
        })?;
        let alphabet_size = multiplication.len();
        let basis_dimensions = (0..=max_degree.saturating_add(1))
            .map(|degree| {
                normalized_basis_dimension(alphabet_size, degree).ok_or(
                    FiniteGroupIntegralCohomologyError::BasisCardinalityOverflow {
                        degree,
                        alphabet_size,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for degree in 1..=max_degree.saturating_add(1) {
            let rows = basis_dimensions[degree - 1];
            let columns = basis_dimensions[degree];
            let estimated_entries = normalized_bar_entry_bound(columns, degree).ok_or(
                FiniteGroupIntegralCohomologyError::BarComplexTooLarge {
                    degree,
                    rows,
                    columns,
                    estimated_entries: usize::MAX,
                    entry_limit: NORMALIZED_BAR_SPARSE_ENTRY_LIMIT,
                },
            )?;
            if estimated_entries > NORMALIZED_BAR_SPARSE_ENTRY_LIMIT {
                return Err(FiniteGroupIntegralCohomologyError::BarComplexTooLarge {
                    degree,
                    rows,
                    columns,
                    estimated_entries,
                    entry_limit: NORMALIZED_BAR_SPARSE_ENTRY_LIMIT,
                });
            }
        }
        let mut boundaries = Vec::with_capacity(max_degree.saturating_add(2));
        boundaries.push(SparseMatrix::zeros(Z, 0, 1));
        for degree in 1..=max_degree.saturating_add(1) {
            boundaries.push(normalized_integral_boundary(
                alphabet_size,
                &multiplication,
                degree,
            ));
        }

        let mut coboundaries = BTreeMap::new();
        for degree in 0..=max_degree {
            let displayed_degree = i32::try_from(degree).map_err(|_| {
                FiniteGroupIntegralCohomologyError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "integral bar truncation degree fits the signed grading type",
                    },
                )
            })?;
            coboundaries.insert(-displayed_degree, boundaries[degree + 1].transpose());
        }
        let contraction =
            SparseChainContraction::from_trusted_lowering(Z, coboundaries).map_err(|_| {
                FiniteGroupIntegralCohomologyError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "sparse integral bar contraction is exact and replayable",
                    },
                )
            })?;
        let mut cohomology = Vec::with_capacity(max_degree.saturating_add(1));
        for degree in 0..=max_degree {
            let internal_degree = -i32::try_from(degree).map_err(|_| {
                FiniteGroupIntegralCohomologyError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "integral bar truncation degree fits the signed grading type",
                    },
                )
            })?;
            let ambient_dimension = contraction.reduced_dimension(internal_degree);
            let incoming = if degree == 0 {
                Matrix::zeros(Z, ambient_dimension, 0)
            } else {
                contraction
                    .reduced_differential(internal_degree + 1)
                    .expect("the sparse truncation contains the incoming coboundary")
                    .to_dense()
            };
            let outgoing = contraction
                .reduced_differential(internal_degree)
                .expect("the sparse truncation contains the outgoing coboundary")
                .to_dense();
            let mut degree_data = integral_cohomology_degree(&incoming, &outgoing)?;
            degree_data.representatives = degree_data
                .representatives
                .iter()
                .map(|representative| contraction.lift(internal_degree, representative))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    FiniteGroupIntegralCohomologyError::Verification(
                        VerificationError::InvalidWitness {
                            claim: "reduced integral cocycles lift to normalized bar cocycles",
                        },
                    )
                })?;
            cohomology.push(degree_data);
        }
        Ok(Self {
            group: group.clone(),
            max_degree,
            strategy: FiniteGroupIntegralResolutionStrategy::NormalizedBar,
            alphabet_size,
            boundaries,
            contraction: Some(contraction),
            cohomology,
        })
    }

    /// Supplied concrete permutation group.
    #[must_use]
    pub const fn group(&self) -> &PermutationGroup {
        &self.group
    }

    /// Largest cohomological degree computed exactly.
    #[must_use]
    pub const fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Exact mathematical strategy used by this calculation.
    #[must_use]
    pub const fn resolution_strategy(&self) -> FiniteGroupIntegralResolutionStrategy {
        self.strategy
    }

    /// Canonical additive group in one cohomological degree.
    pub fn additive_group(
        &self,
        degree: usize,
    ) -> Result<&FinitelyGeneratedAbelianGroup, FiniteGroupIntegralCohomologyError> {
        self.cohomology
            .get(degree)
            .map(|value| &value.additive_group)
            .ok_or(FiniteGroupIntegralCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            })
    }

    /// Orders of the canonical basis summands; `None` denotes a free summand.
    pub fn basis_orders(
        &self,
        degree: usize,
    ) -> Result<&[Option<Integer>], FiniteGroupIntegralCohomologyError> {
        self.cohomology
            .get(degree)
            .map(|value| value.basis_orders.as_slice())
            .ok_or(FiniteGroupIntegralCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            })
    }

    /// Deterministic normalized-bar cocycles for the canonical summands.
    ///
    /// The scalable universal low-degree strategy returns `None`: it certifies
    /// the groups theoremically without materializing bar cochains.
    pub fn basis_cocycles(
        &self,
        degree: usize,
    ) -> Result<Option<&[Vector<IntegerRing>]>, FiniteGroupIntegralCohomologyError> {
        let value = self.cohomology.get(degree).ok_or(
            FiniteGroupIntegralCohomologyError::DegreeOutOfRange {
                degree,
                max_degree: self.max_degree,
            },
        )?;
        Ok(
            (self.strategy == FiniteGroupIntegralResolutionStrategy::NormalizedBar)
                .then_some(value.representatives.as_slice()),
        )
    }

    /// Canonical coordinates of the cup product of two basis summands.
    ///
    /// Torsion coordinates are reduced to the interval `[0, order)`. Free
    /// coordinates are exact integers.
    pub fn basis_cup_product(
        &self,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
        right_index: usize,
    ) -> Result<Vec<Integer>, FiniteGroupIntegralCohomologyError> {
        let target_degree = left_degree.checked_add(right_degree).ok_or(
            FiniteGroupIntegralCohomologyError::DegreeOutOfRange {
                degree: usize::MAX,
                max_degree: self.max_degree,
            },
        )?;
        if target_degree > self.max_degree {
            return Err(FiniteGroupIntegralCohomologyError::DegreeOutOfRange {
                degree: target_degree,
                max_degree: self.max_degree,
            });
        }
        self.require_basis_index(left_degree, left_index)?;
        self.require_basis_index(right_degree, right_index)?;
        if self.strategy == FiniteGroupIntegralResolutionStrategy::UniversalLowDegree {
            let target_size = self.cohomology[target_degree].basis_orders.len();
            if left_degree == 0 {
                return Ok((0..target_size)
                    .map(|index| Integer::from(u8::from(index == right_index)))
                    .collect());
            }
            if right_degree == 0 {
                return Ok((0..target_size)
                    .map(|index| Integer::from(u8::from(index == left_index)))
                    .collect());
            }
            return Ok(vec![Integer::zero(); target_size]);
        }
        let left = &self.cohomology[left_degree].representatives[left_index];
        let right = &self.cohomology[right_degree].representatives[right_index];
        let product = normalized_integral_cup_product(
            self.alphabet_size,
            left_degree,
            left,
            right_degree,
            right,
        )?;
        self.cohomology_coordinates(target_degree, &product)
    }

    /// Recompute all exact evidence retained by this object.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = Self::build(&self.group, self.max_degree).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "integral finite-group cohomology is exactly recomputable",
            }
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "integral finite-group cohomology matches exact recomputation",
            })
        }
    }

    fn require_basis_index(
        &self,
        degree: usize,
        index: usize,
    ) -> Result<(), FiniteGroupIntegralCohomologyError> {
        let basis_size = self.basis_orders(degree)?.len();
        if index < basis_size {
            Ok(())
        } else {
            Err(FiniteGroupIntegralCohomologyError::BasisIndexOutOfRange {
                degree,
                index,
                basis_size,
            })
        }
    }

    fn cohomology_coordinates(
        &self,
        degree: usize,
        cocycle: &Vector<IntegerRing>,
    ) -> Result<Vec<Integer>, FiniteGroupIntegralCohomologyError> {
        let data = &self.cohomology[degree];
        let internal_degree = -i32::try_from(degree).map_err(|_| {
            FiniteGroupIntegralCohomologyError::Verification(VerificationError::InvalidWitness {
                claim: "integral cohomology degree fits the signed grading type",
            })
        })?;
        let reduced = self
            .contraction
            .as_ref()
            .ok_or(FiniteGroupIntegralCohomologyError::Verification(
                VerificationError::InvalidWitness {
                    claim: "normalized integral bar data retains its sparse contraction",
                },
            ))?
            .project(internal_degree, cocycle)
            .map_err(|_| {
                FiniteGroupIntegralCohomologyError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "normalized integral cocycles project to the reduced complex",
                    },
                )
            })?;
        let coordinates = data.coordinate_map.apply(&reduced)?;
        Ok(coordinates
            .values()
            .zip(&data.basis_orders)
            .map(|(coordinate, order)| {
                order.as_ref().map_or(coordinate.clone(), |order| {
                    Z.div_rem(&coordinate, order)
                        .expect("a torsion order is nonzero")
                        .1
                })
            })
            .collect())
    }
}

impl fmt::Debug for FiniteGroupIntegralCohomologyTruncation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteGroupIntegralCohomologyTruncation")
            .field("group_order", &self.group.order().to_decimal())
            .field("max_degree", &self.max_degree)
            .field("resolution_strategy", &self.strategy)
            .field(
                "additive_groups",
                &self
                    .cohomology
                    .iter()
                    .map(|degree| &degree.additive_group)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SemanticType for FiniteGroupIntegralCohomologyTruncation {
    const TYPE_ID: &'static str = "noether/homalg/FiniteGroupIntegralCohomologyTruncation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.group.canonical_bytes()?;
        output.extend_from_slice(
            &u64::try_from(self.max_degree)
                .map_err(|_| NoetherError::Serialization("degree exceeds u64".to_owned()))?
                .to_be_bytes(),
        );
        output.push(match self.strategy {
            FiniteGroupIntegralResolutionStrategy::UniversalLowDegree => 0,
            FiniteGroupIntegralResolutionStrategy::NormalizedBar => 1,
        });
        for boundary in &self.boundaries {
            push_frame(&mut output, &boundary.canonical_bytes()?)?;
        }
        for degree in &self.cohomology {
            push_frame(&mut output, &degree.additive_group.canonical_bytes()?)?;
            for representative in &degree.representatives {
                push_frame(&mut output, &representative.canonical_bytes()?)?;
            }
            push_frame(&mut output, &degree.coordinate_map.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_TRUNCATION_DESCRIPTION
    }
}

impl FiniteGroupCohomologyAlgebraTruncation {
    fn build(
        cohomology: &FiniteGroupCohomologyTruncation,
    ) -> Result<Self, FiniteGroupCohomologyError> {
        let max_degree = u32::try_from(cohomology.max_degree).map_err(|_| {
            FiniteGroupCohomologyError::AlgebraDegreeOverflow {
                degree: cohomology.max_degree,
            }
        })?;
        if cohomology.dimension(0)? != 1 {
            return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                reason: "degree-zero cohomology is not the coefficient field",
            });
        }

        let mut generators = Vec::new();
        let mut multiplication_cache = BTreeMap::new();
        for degree in 1..=max_degree {
            let degree_usize = usize::try_from(degree).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?;
            let ring = algebra_polynomial_ring(cohomology.field(), &generators)?;
            let free = GradedCommutativePresentation::new(ring, Vec::new())?;
            let monomials = free.basis(degree)?;
            let mut span = Vec::with_capacity(monomials.len());
            for monomial in &monomials {
                span.push(evaluate_algebra_monomial(
                    cohomology,
                    &generators,
                    monomial,
                    &mut multiplication_cache,
                )?);
            }
            let ambient_dimension = cohomology.dimension(degree_usize)?;
            let decomposable_span = span.clone();
            let operation_candidates =
                finite_group_operation_candidates(cohomology, &generators, degree_usize)?;
            let mut rank =
                matrix_from_columns(cohomology.field(), ambient_dimension, &span)?.rank()?;
            for basis_index in 0..ambient_dimension {
                let candidate =
                    standard_basis_vector(cohomology.field(), ambient_dimension, basis_index)?;
                span.push(candidate.clone());
                let extended_rank =
                    matrix_from_columns(cohomology.field(), ambient_dimension, &span)?.rank()?;
                if extended_rank > rank {
                    let operation_witness = operation_witness_for_basis(
                        cohomology.field(),
                        ambient_dimension,
                        &candidate,
                        &decomposable_span,
                        &operation_candidates,
                    )?;
                    let stable_name = format!("x{degree}_{}", basis_index + 1);
                    let display_name = operation_witness.as_ref().map_or_else(
                        || stable_name.clone(),
                        |witness| {
                            format!(
                                "{}({})",
                                witness.operation.display_name(),
                                generators[witness.source_generator_index].display_name
                            )
                        },
                    );
                    generators.push(FiniteGroupCohomologyAlgebraGenerator {
                        name: stable_name,
                        display_name,
                        degree: degree_usize,
                        basis_index,
                        operation_witness,
                    });
                    rank = extended_rank;
                } else {
                    span.pop();
                }
            }
            if rank != ambient_dimension {
                return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                    reason: "selected generators do not span one exact cohomological degree",
                });
            }
        }

        let ring = algebra_polynomial_ring(cohomology.field(), &generators)?;
        let free = GradedCommutativePresentation::new(ring.clone(), Vec::new())?;
        let mut presentation = free.clone();
        for degree in 1..=max_degree {
            let degree_usize = usize::try_from(degree).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?;
            let monomials = free.basis(degree)?;
            let mut images = Vec::with_capacity(monomials.len());
            for monomial in &monomials {
                images.push(evaluate_algebra_monomial(
                    cohomology,
                    &generators,
                    monomial,
                    &mut multiplication_cache,
                )?);
            }
            let evaluation = matrix_from_columns(
                cohomology.field(),
                cohomology.dimension(degree_usize)?,
                &images,
            )?;
            for relation in evaluation.kernel()?.basis() {
                let polynomial = ring.polynomial(
                    monomials
                        .iter()
                        .zip(relation.values())
                        .filter(|(_, coefficient)| !coefficient.is_zero())
                        .map(|(monomial, coefficient)| {
                            (coefficient, monomial.exponents().to_vec())
                        }),
                )?;
                let reduced = presentation.normal_form(&polynomial)?;
                if reduced.is_zero() {
                    continue;
                }
                let mut relations = presentation.relations().to_vec();
                relations.push(reduced);
                presentation = GradedCommutativePresentation::new(ring.clone(), relations)?;
            }
            if presentation.dimension(degree)? != cohomology.dimension(degree_usize)? {
                return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                    reason: "relation quotient has the wrong dimension inside the exact range",
                });
            }
        }

        Ok(Self {
            cohomology: cohomology.clone(),
            generators,
            presentation,
        })
    }

    /// Source exact cohomology truncation.
    #[must_use]
    pub const fn cohomology(&self) -> &FiniteGroupCohomologyTruncation {
        &self.cohomology
    }

    /// Largest degree in which the generators and relation ideal are exact.
    #[must_use]
    pub const fn max_degree(&self) -> usize {
        self.cohomology.max_degree
    }

    /// Deterministic minimal homogeneous generator family.
    #[must_use]
    pub fn generators(&self) -> &[FiniteGroupCohomologyAlgebraGenerator] {
        &self.generators
    }

    /// Graded-commutative quotient whose relation ideal is exact through the
    /// retained degree bound.
    #[must_use]
    pub const fn presentation(&self) -> &GradedCommutativePresentation {
        &self.presentation
    }

    /// Certify this quotient as the complete cohomology ring using Symonds's
    /// universal generator and relation bounds.
    ///
    /// For a finite group `G`, generators occur in degrees at most `|G|-1`
    /// and relations in degrees at most `2(|G|-1)`. This operation fails
    /// closed unless the retained exact range reaches the latter bound.
    pub fn certify_symonds_universal_bound(
        &self,
    ) -> Result<FiniteGroupCohomologyCompletePresentation, FiniteGroupCohomologyError> {
        self.verify()?;
        self.symonds_universal_certificate()
    }

    /// Certify this quotient as the complete cohomology ring using Symonds's
    /// parameter criterion.
    ///
    /// Candidate parameters are selected deterministically from the minimal
    /// algebra generators. Parameterhood in the actual group cohomology is
    /// proved by exact restriction to every maximal elementary abelian
    /// p-subgroup of a Sylow subgroup. The operation then proves that the
    /// current algebra approximation is finite over those parameters and
    /// checks Symonds's stopping inequality. It fails closed if the retained
    /// exact range or resolution model cannot supply all of that evidence.
    pub fn certify_symonds_parameter_criterion(
        &self,
    ) -> Result<FiniteGroupCohomologyCompletePresentation, FiniteGroupCohomologyError> {
        self.verify()?;
        self.symonds_parameter_certificate()
    }

    /// Recompute generator selection and every relation from the retained
    /// cohomology multiplication table.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.cohomology.verify()?;
        let rebuilt =
            Self::build(&self.cohomology).map_err(|_| VerificationError::InvalidWitness {
                claim: "finite-group cohomology algebra truncation is exactly recomputable",
            })?;
        if rebuilt.generators == self.generators && rebuilt.presentation == self.presentation {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "finite-group cohomology generators and relations match recomputation",
            })
        }
    }

    fn symonds_universal_certificate(
        &self,
    ) -> Result<FiniteGroupCohomologyCompletePresentation, FiniteGroupCohomologyError> {
        let group_order = self
            .cohomology
            .group
            .order()
            .to_decimal()
            .parse::<Integer>()
            .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        let order = group_order
            .to_u64_checked()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        let generator_degree_bound = order.saturating_sub(1);
        let relation_degree_bound = generator_degree_bound
            .checked_mul(2)
            .ok_or(FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        if self.max_degree() < relation_degree_bound {
            return Err(FiniteGroupCohomologyError::CompletionBoundNotReached {
                computed_through_degree: self.max_degree(),
                required_through_degree: relation_degree_bound,
            });
        }
        if self
            .generators
            .iter()
            .any(|generator| generator.degree > generator_degree_bound)
        {
            return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                reason: "a selected generator exceeds the Symonds universal bound",
            });
        }
        for relation in self.presentation.relations() {
            let degree = relation.homogeneous_degree()?.ok_or(
                FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                    reason: "a stored defining relation is zero or nonhomogeneous",
                },
            )?;
            if usize::try_from(degree)
                .ok()
                .is_none_or(|degree| degree > relation_degree_bound)
            {
                return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                    reason: "a selected relation exceeds the Symonds universal bound",
                });
            }
        }
        Ok(FiniteGroupCohomologyCompletePresentation {
            algebra: self.clone(),
            strategy: FiniteGroupCohomologyCompletionStrategy::SymondsUniversalBound,
            group_order,
            generator_degree_bound,
            relation_degree_bound,
            parameter_certificate: None,
        })
    }

    fn symonds_parameter_certificate(
        &self,
    ) -> Result<FiniteGroupCohomologyCompletePresentation, FiniteGroupCohomologyError> {
        let parameter_certificate = build_symonds_parameter_certificate(self)?;
        let group_order = self
            .cohomology
            .group
            .order()
            .to_decimal()
            .parse::<Integer>()
            .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        let generator_degree_bound = self
            .generators
            .iter()
            .map(FiniteGroupCohomologyAlgebraGenerator::degree)
            .max()
            .unwrap_or(0);
        let relation_degree_bound = self
            .presentation
            .relations()
            .iter()
            .map(|relation| {
                relation
                    .homogeneous_degree()
                    .map_err(FiniteGroupCohomologyError::from)
                    .and_then(|degree| {
                        degree
                            .and_then(|degree| usize::try_from(degree).ok())
                            .ok_or(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                                reason: "a stored defining relation has no representable degree",
                            })
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        Ok(FiniteGroupCohomologyCompletePresentation {
            algebra: self.clone(),
            strategy: FiniteGroupCohomologyCompletionStrategy::SymondsParameterCriterion,
            group_order,
            generator_degree_bound,
            relation_degree_bound,
            parameter_certificate: Some(parameter_certificate),
        })
    }
}

impl FiniteGroupCohomologyCompletePresentation {
    /// Exact algebra data promoted by the completion theorem.
    #[must_use]
    pub const fn algebra(&self) -> &FiniteGroupCohomologyAlgebraTruncation {
        &self.algebra
    }

    /// Complete graded-commutative quotient presentation.
    #[must_use]
    pub const fn presentation(&self) -> &GradedCommutativePresentation {
        self.algebra.presentation()
    }

    /// The theorem discharging completeness.
    #[must_use]
    pub const fn completion_strategy(&self) -> FiniteGroupCohomologyCompletionStrategy {
        self.strategy
    }

    /// Exact order of the finite group appearing in the certificate.
    #[must_use]
    pub const fn group_order(&self) -> &Integer {
        &self.group_order
    }

    /// Universal degree bound for ring generators.
    #[must_use]
    pub const fn generator_degree_bound(&self) -> usize {
        self.generator_degree_bound
    }

    /// Universal degree bound for defining relations.
    #[must_use]
    pub const fn relation_degree_bound(&self) -> usize {
        self.relation_degree_bound
    }

    /// Parameter evidence when completion used Symonds's parameter criterion.
    #[must_use]
    pub const fn symonds_parameter_certificate(
        &self,
    ) -> Option<&FiniteGroupCohomologySymondsParameterCertificate> {
        self.parameter_certificate.as_ref()
    }

    /// Recompute the algebra and every hypothesis of the completion theorem.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.algebra.verify()?;
        let rebuilt = match self.strategy {
            FiniteGroupCohomologyCompletionStrategy::SymondsUniversalBound => {
                self.algebra.symonds_universal_certificate()
            }
            FiniteGroupCohomologyCompletionStrategy::SymondsParameterCriterion => {
                self.algebra.symonds_parameter_certificate()
            }
        }
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "the selected finite-group cohomology completion theorem applies",
        })?;
        if self.strategy == rebuilt.strategy
            && self.group_order == rebuilt.group_order
            && self.generator_degree_bound == rebuilt.generator_degree_bound
            && self.relation_degree_bound == rebuilt.relation_degree_bound
            && self.parameter_certificate == rebuilt.parameter_certificate
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "finite-group cohomology completion certificate matches recomputation",
            })
        }
    }
}

impl fmt::Debug for FiniteGroupCohomologyCompletePresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteGroupCohomologyCompletePresentation")
            .field("strategy", &self.strategy)
            .field("group_order", &self.group_order)
            .field("generator_degree_bound", &self.generator_degree_bound)
            .field("relation_degree_bound", &self.relation_degree_bound)
            .finish()
    }
}

impl SemanticType for FiniteGroupCohomologyCompletePresentation {
    const TYPE_ID: &'static str = "noether/homalg/FiniteGroupCohomologyCompletePresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, &self.algebra.canonical_bytes()?)?;
        output.push(match self.strategy {
            FiniteGroupCohomologyCompletionStrategy::SymondsUniversalBound => 0,
            FiniteGroupCohomologyCompletionStrategy::SymondsParameterCriterion => 1,
        });
        push_frame(&mut output, &self.group_order.canonical_bytes()?)?;
        output.extend_from_slice(
            &u64::try_from(self.generator_degree_bound)
                .map_err(|_| {
                    NoetherError::Serialization("generator degree bound exceeds u64".to_owned())
                })?
                .to_be_bytes(),
        );
        output.extend_from_slice(
            &u64::try_from(self.relation_degree_bound)
                .map_err(|_| {
                    NoetherError::Serialization("relation degree bound exceeds u64".to_owned())
                })?
                .to_be_bytes(),
        );
        match &self.parameter_certificate {
            None => output.push(0),
            Some(certificate) => {
                output.push(1);
                push_usizes(&mut output, &certificate.parameter_generator_indices)?;
                output.extend_from_slice(
                    &u64::try_from(certificate.elementary_abelian_restrictions.len())
                        .map_err(|_| {
                            NoetherError::Serialization(
                                "elementary-abelian restriction count exceeds u64".to_owned(),
                            )
                        })?
                        .to_be_bytes(),
                );
                for restriction in &certificate.elementary_abelian_restrictions {
                    push_frame(&mut output, &restriction.subgroup.canonical_bytes()?)?;
                    output.extend_from_slice(
                        &u64::try_from(restriction.rank)
                            .map_err(|_| {
                                NoetherError::Serialization(
                                    "elementary-abelian rank exceeds u64".to_owned(),
                                )
                            })?
                            .to_be_bytes(),
                    );
                    output.extend_from_slice(
                        &u64::try_from(restriction.restricted_parameters.len())
                            .map_err(|_| {
                                NoetherError::Serialization(
                                    "restricted parameter count exceeds u64".to_owned(),
                                )
                            })?
                            .to_be_bytes(),
                    );
                    for parameter in &restriction.restricted_parameters {
                        push_frame(&mut output, &parameter.canonical_bytes()?)?;
                    }
                    push_u32s(&mut output, &restriction.quotient_generator_power_bounds)?;
                }
                push_u32s(&mut output, &certificate.quotient_generator_power_bounds)?;
                push_usizes(
                    &mut output,
                    &[
                        certificate.maximum_module_generator_degree,
                        certificate.parameter_degree_bound,
                        certificate.completion_degree_bound,
                    ],
                )?;
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &COMPLETE_PRESENTATION_DESCRIPTION
    }
}

impl fmt::Debug for FiniteGroupCohomologyAlgebraTruncation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteGroupCohomologyAlgebraTruncation")
            .field("max_degree", &self.max_degree())
            .field("generators", &self.generators)
            .field("relation_count", &self.presentation.relations().len())
            .finish()
    }
}

impl SemanticType for FiniteGroupCohomologyAlgebraTruncation {
    const TYPE_ID: &'static str = "noether/homalg/FiniteGroupCohomologyAlgebraTruncation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, &self.cohomology.canonical_bytes()?)?;
        output.extend_from_slice(
            &u64::try_from(self.generators.len())
                .map_err(|_| NoetherError::Serialization("generator count exceeds u64".to_owned()))?
                .to_be_bytes(),
        );
        for generator in &self.generators {
            push_frame(&mut output, generator.name.as_bytes())?;
            push_frame(&mut output, generator.display_name.as_bytes())?;
            output.extend_from_slice(
                &u64::try_from(generator.degree)
                    .map_err(|_| {
                        NoetherError::Serialization("generator degree exceeds u64".to_owned())
                    })?
                    .to_be_bytes(),
            );
            output.extend_from_slice(
                &u64::try_from(generator.basis_index)
                    .map_err(|_| NoetherError::Serialization("basis index exceeds u64".to_owned()))?
                    .to_be_bytes(),
            );
            match &generator.operation_witness {
                None => output.push(0),
                Some(witness) => {
                    output.push(1);
                    output.extend_from_slice(
                        &u64::try_from(witness.source_generator_index)
                            .map_err(|_| {
                                NoetherError::Serialization(
                                    "operation source generator index exceeds u64".to_owned(),
                                )
                            })?
                            .to_be_bytes(),
                    );
                    push_frame(
                        &mut output,
                        &cohomology_operation_canonical_bytes(&witness.operation)?,
                    )?;
                    push_frame(
                        &mut output,
                        &witness.operation_coefficient.canonical_bytes()?,
                    )?;
                    push_frame(
                        &mut output,
                        &witness.decomposable_correction.canonical_bytes()?,
                    )?;
                }
            }
        }
        push_frame(&mut output, &self.presentation.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &ALGEBRA_DESCRIPTION
    }
}

fn cohomology_operation_canonical_bytes(
    operation: &AdmissibleCohomologyOperation,
) -> Result<Vec<u8>, NoetherError> {
    let mut output = operation.prime().to_be_bytes().to_vec();
    if let Some(sequence) = operation.square_sequence() {
        output.push(0);
        output.extend_from_slice(
            &u64::try_from(sequence.len())
                .map_err(|_| {
                    NoetherError::Serialization(
                        "Steenrod-square word length exceeds u64".to_owned(),
                    )
                })?
                .to_be_bytes(),
        );
        for square in sequence {
            output.extend_from_slice(&square.to_be_bytes());
        }
        return Ok(output);
    }
    output.push(1);
    let bocksteins = operation
        .bocksteins()
        .expect("an odd-primary word has Bockstein exponents");
    let powers = operation
        .reduced_powers()
        .expect("an odd-primary word has reduced powers");
    output.extend_from_slice(
        &u64::try_from(bocksteins.len())
            .map_err(|_| {
                NoetherError::Serialization("Bockstein word length exceeds u64".to_owned())
            })?
            .to_be_bytes(),
    );
    output.extend(bocksteins.iter().map(|value| u8::from(*value)));
    output.extend_from_slice(
        &u64::try_from(powers.len())
            .map_err(|_| {
                NoetherError::Serialization("reduced-power word length exceeds u64".to_owned())
            })?
            .to_be_bytes(),
    );
    for power in powers {
        output.extend_from_slice(&power.to_be_bytes());
    }
    Ok(output)
}

impl fmt::Debug for FiniteGroupCohomologyTruncation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteGroupCohomologyTruncation")
            .field("group_order", &self.group.order().to_decimal())
            .field("field", &self.field)
            .field("max_degree", &self.max_degree)
            .field("resolution_strategy", &self.strategy)
            .field(
                "dimensions",
                &self
                    .cohomology
                    .iter()
                    .map(|degree| degree.representatives.len())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SemanticType for FiniteGroupCohomologyTruncation {
    const TYPE_ID: &'static str = "noether/homalg/FiniteGroupCohomologyTruncation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.group.canonical_bytes()?;
        push_frame(&mut output, &self.field.canonical_bytes()?)?;
        output.extend_from_slice(
            &u64::try_from(self.max_degree)
                .map_err(|_| NoetherError::Serialization("degree exceeds u64".to_owned()))?
                .to_be_bytes(),
        );
        output.push(match self.strategy {
            FiniteGroupResolutionStrategy::MinimalPGroup => 0,
            FiniteGroupResolutionStrategy::NormalizedBar => 1,
            FiniteGroupResolutionStrategy::SylowStableElements => 2,
        });
        for boundary in &self.boundaries {
            push_frame(&mut output, &boundary.canonical_bytes()?)?;
        }
        if let Some(resolution) = &self.minimal_p_group {
            for differential in &resolution.differentials {
                push_frame(&mut output, &differential.canonical_bytes()?)?;
            }
        }
        if let Some(stable) = &self.stable_elements {
            push_frame(&mut output, &stable.sylow_cohomology.canonical_bytes()?)?;
            for operator in &stable.stability_operators {
                push_frame(&mut output, &operator.canonical_bytes()?)?;
            }
        }
        for degree in &self.cohomology {
            for representative in &degree.representatives {
                push_frame(&mut output, &representative.canonical_bytes()?)?;
            }
            output.push(0xff);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &DESCRIPTION
    }
}

fn ordered_group_elements(group: &PermutationGroup) -> Vec<Permutation> {
    let mut elements = iterate_elements(group).collect::<Vec<_>>();
    elements.sort_by(|left, right| left.images().cmp(right.images()));
    elements.dedup_by(|left, right| left.images() == right.images());
    elements
}

fn is_prime_power(mut order: usize, prime: u64) -> bool {
    let Ok(prime) = usize::try_from(prime) else {
        return order == 1;
    };
    if prime < 2 {
        return false;
    }
    while order > 1 && order.is_multiple_of(prime) {
        order /= prime;
    }
    order == 1
}

fn binomial_mod_prime(n: usize, k: usize, prime: u64) -> u64 {
    if k > n {
        return 0;
    }
    let mut row = vec![0_u64; k.saturating_add(1)];
    row[0] = 1;
    for current in 1..=n {
        for index in (1..=k.min(current)).rev() {
            row[index] =
                ((u128::from(row[index]) + u128::from(row[index - 1])) % u128::from(prime)) as u64;
        }
    }
    row[k]
}

#[derive(Clone)]
struct FiniteQuotientWitness {
    generator_power_bounds: Vec<u32>,
    top_nonzero_degree: usize,
}

#[derive(Clone)]
struct ElementaryAbelianParameterData {
    subgroup: PermutationGroup,
    rank: usize,
    presentation: GradedCommutativePresentation,
    generator_restrictions: Vec<crate::commalg::GradedCommutativePolynomial>,
}

fn build_symonds_parameter_certificate(
    algebra: &FiniteGroupCohomologyAlgebraTruncation,
) -> Result<FiniteGroupCohomologySymondsParameterCertificate, FiniteGroupCohomologyError> {
    if algebra.generators.is_empty() {
        return Err(FiniteGroupCohomologyError::CompletionBoundNotReached {
            computed_through_degree: algebra.max_degree(),
            required_through_degree: algebra.max_degree().saturating_add(1),
        });
    }
    if algebra.generators.len() >= usize::BITS as usize {
        return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
            reason: "the deterministic parameter-subset search exceeds the bit-mask index type",
        });
    }
    let elementary = elementary_abelian_parameter_data(algebra)?;
    let mut candidates = (1usize..(1usize << algebra.generators.len()))
        .map(|mask| {
            (0..algebra.generators.len())
                .filter(|index| mask & (1usize << index) != 0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|indices| {
        let excess_degree = indices
            .iter()
            .map(|index| algebra.generators[*index].degree.saturating_sub(1))
            .sum::<usize>();
        let degree_one_count = indices
            .iter()
            .filter(|index| algebra.generators[**index].degree == 1)
            .count();
        (
            excess_degree,
            usize::MAX - degree_one_count,
            indices.len(),
            indices.clone(),
        )
    });

    for parameter_generator_indices in candidates {
        let parameter_degree_bound = parameter_generator_indices
            .iter()
            .try_fold(1usize, |bound, index| {
                bound.checked_add(algebra.generators[*index].degree.saturating_sub(1))
            })
            .ok_or(FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        if parameter_degree_bound > algebra.max_degree() {
            continue;
        }
        let mut restriction_certificates = Vec::with_capacity(elementary.len());
        let mut restrictions_are_parameters = true;
        for data in &elementary {
            let restricted_parameters = parameter_generator_indices
                .iter()
                .map(|index| data.generator_restrictions[*index].clone())
                .collect::<Vec<_>>();
            let Some(quotient) = finite_quotient_witness(
                &data.presentation,
                &restricted_parameters,
                algebra.max_degree(),
            )?
            else {
                restrictions_are_parameters = false;
                break;
            };
            restriction_certificates.push(FiniteGroupCohomologyElementaryAbelianRestriction {
                subgroup: data.subgroup.clone(),
                rank: data.rank,
                restricted_parameters,
                quotient_generator_power_bounds: quotient.generator_power_bounds,
            });
        }
        if !restrictions_are_parameters {
            continue;
        }

        let parameters = parameter_generator_indices
            .iter()
            .map(|index| algebra.presentation.ring().generator(*index))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(quotient) =
            finite_quotient_witness(&algebra.presentation, &parameters, algebra.max_degree())?
        else {
            continue;
        };
        let completion_degree_bound = quotient.top_nonzero_degree.max(parameter_degree_bound);
        if completion_degree_bound > algebra.max_degree() {
            continue;
        }
        return Ok(FiniteGroupCohomologySymondsParameterCertificate {
            parameter_generator_indices,
            elementary_abelian_restrictions: restriction_certificates,
            quotient_generator_power_bounds: quotient.generator_power_bounds,
            maximum_module_generator_degree: quotient.top_nonzero_degree,
            parameter_degree_bound,
            completion_degree_bound,
        });
    }

    Err(FiniteGroupCohomologyError::CompletionBoundNotReached {
        computed_through_degree: algebra.max_degree(),
        required_through_degree: algebra.max_degree().saturating_add(1),
    })
}

fn finite_quotient_witness(
    presentation: &GradedCommutativePresentation,
    parameters: &[crate::commalg::GradedCommutativePolynomial],
    search_through_degree: usize,
) -> Result<Option<FiniteQuotientWitness>, FiniteGroupCohomologyError> {
    let mut relations = presentation.relations().to_vec();
    relations.extend(parameters.iter().cloned());
    let quotient = GradedCommutativePresentation::new(presentation.ring().clone(), relations)?;
    let mut generator_power_bounds = Vec::with_capacity(quotient.ring().arity());
    let mut bounding_degree = 0usize;
    for (index, generator) in quotient.ring().generators().iter().enumerate() {
        let generator_degree = usize::try_from(generator.degree())
            .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        let maximum_power = if !quotient.ring().has_characteristic_two() && generator.is_odd() {
            2
        } else {
            search_through_degree / generator_degree
        };
        let mut power_bound = None;
        for power in 1..=maximum_power.max(1) {
            let power = u32::try_from(power)
                .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?;
            let mut exponents = vec![0_u32; quotient.ring().arity()];
            exponents[index] = power;
            let value = quotient
                .ring()
                .term(quotient.ring().field().one(), exponents)?;
            if quotient.normal_form(&value)?.is_zero() {
                power_bound = Some(power);
                break;
            }
        }
        let Some(power_bound) = power_bound else {
            return Ok(None);
        };
        bounding_degree = bounding_degree
            .checked_add(
                usize::try_from(power_bound.saturating_sub(1))
                    .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?
                    .checked_mul(generator_degree)
                    .ok_or(FiniteGroupCohomologyError::CompletionBoundOverflow)?,
            )
            .ok_or(FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        generator_power_bounds.push(power_bound);
    }
    let mut top_nonzero_degree = 0usize;
    for degree in 0..=bounding_degree {
        let degree_u32 = u32::try_from(degree)
            .map_err(|_| FiniteGroupCohomologyError::CompletionBoundOverflow)?;
        if quotient.dimension(degree_u32)? != 0 {
            top_nonzero_degree = degree;
        }
    }
    Ok(Some(FiniteQuotientWitness {
        generator_power_bounds,
        top_nonzero_degree,
    }))
}

fn elementary_abelian_parameter_data(
    algebra: &FiniteGroupCohomologyAlgebraTruncation,
) -> Result<Vec<ElementaryAbelianParameterData>, FiniteGroupCohomologyError> {
    let cohomology = &algebra.cohomology;
    let prime = cohomology
        .field
        .characteristic()
        .to_string()
        .parse::<u64>()
        .map_err(|_| FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
    let (sylow, target_resolution, target_vectors) =
        if let Some(resolution) = &cohomology.minimal_p_group {
            (
                cohomology.group.clone(),
                resolution,
                algebra
                    .generators
                    .iter()
                    .map(|generator| {
                        standard_basis_vector(
                            &cohomology.field,
                            resolution.ranks[generator.degree],
                            generator.basis_index,
                        )
                        .map_err(FiniteGroupCohomologyError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else if let Some(stable) = &cohomology.stable_elements {
            let resolution = stable.sylow_cohomology.minimal_p_group.as_ref().ok_or(
                FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
                    reason: "the stable-element model does not retain a minimal Sylow resolution",
                },
            )?;
            (
                stable.sylow_cohomology.group.clone(),
                resolution,
                algebra
                    .generators
                    .iter()
                    .map(|generator| {
                        cohomology.cohomology[generator.degree]
                            .representatives
                            .get(generator.basis_index)
                            .cloned()
                            .ok_or(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
                                reason: "an algebra generator has no stable-element representative",
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
                reason: "normalized-bar restrictions are not retained by this completion playbook",
            });
        };
    let maximum_generator_degree = algebra
        .generators
        .iter()
        .map(FiniteGroupCohomologyAlgebraGenerator::degree)
        .max()
        .unwrap_or(0);
    let subgroups = maximal_elementary_abelian_subgroups(&sylow, prime)?;
    let mut output = Vec::with_capacity(subgroups.len());
    for subgroup in subgroups {
        let subgroup_elements = ordered_group_elements(&subgroup);
        let rank = elementary_abelian_rank(subgroup_elements.len(), prime)?;
        let invariant_factors = subgroup.abelian_invariant_factors().map_err(|_| {
            FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
                reason: "a proposed elementary abelian subgroup failed abelian decomposition",
            }
        })?;
        if invariant_factors.len() != rank
            || invariant_factors
                .iter()
                .any(|factor| factor != &Integer::from(prime))
        {
            return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
                reason:
                    "a proposed maximal elementary abelian subgroup has the wrong invariant factors",
            });
        }
        let source_resolution = build_minimal_p_group_resolution(
            &subgroup,
            &cohomology.field,
            &subgroup_elements,
            maximum_generator_degree.max(2),
        )?;
        let source_action = source_resolution
            .elements
            .iter()
            .map(|element| sylow_element_index(target_resolution, element))
            .collect::<Result<Vec<_>, _>>()?;
        let restrictions = resolution_restriction_maps(
            &cohomology.field,
            &source_resolution,
            target_resolution,
            &source_action,
            maximum_generator_degree,
        )?;
        let subgroup_cohomology = FiniteGroupCohomologyTruncation::build(
            &subgroup,
            prime,
            maximum_generator_degree.max(2),
        )?;
        let subgroup_algebra = subgroup_cohomology.algebra_truncation()?;
        validate_elementary_abelian_algebra(&subgroup_algebra, rank, prime)?;
        let mut multiplication_cache = BTreeMap::new();
        let generator_restrictions = algebra
            .generators
            .iter()
            .zip(target_vectors.iter())
            .map(|(generator, vector)| {
                let restricted = restrictions[generator.degree].apply(vector)?;
                cohomology_class_as_polynomial(
                    &subgroup_cohomology,
                    &subgroup_algebra,
                    generator.degree,
                    &restricted,
                    &mut multiplication_cache,
                )
            })
            .collect::<Result<Vec<_>, FiniteGroupCohomologyError>>()?;
        output.push(ElementaryAbelianParameterData {
            subgroup,
            rank,
            presentation: subgroup_algebra.presentation.clone(),
            generator_restrictions,
        });
    }
    Ok(output)
}

fn validate_elementary_abelian_algebra(
    algebra: &FiniteGroupCohomologyAlgebraTruncation,
    rank: usize,
    prime: u64,
) -> Result<(), FiniteGroupCohomologyError> {
    let expected_generators = if prime == 2 {
        rank
    } else {
        rank.saturating_mul(2)
    };
    let expected_degrees = if prime == 2 {
        vec![1usize; rank]
    } else {
        (0..rank)
            .map(|_| 1usize)
            .chain((0..rank).map(|_| 2usize))
            .collect()
    };
    if algebra.generators.len() != expected_generators
        || algebra
            .generators
            .iter()
            .map(FiniteGroupCohomologyAlgebraGenerator::degree)
            .collect::<Vec<_>>()
            != expected_degrees
        || !algebra.presentation.relations().is_empty()
    {
        return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
            reason:
                "an elementary abelian resolution did not recover its free cohomology presentation",
        });
    }
    Ok(())
}

fn cohomology_class_as_polynomial(
    cohomology: &FiniteGroupCohomologyTruncation,
    algebra: &FiniteGroupCohomologyAlgebraTruncation,
    degree: usize,
    value: &Vector<FiniteField>,
    multiplication_cache: &mut BTreeMap<(usize, usize), Matrix<FiniteField>>,
) -> Result<crate::commalg::GradedCommutativePolynomial, FiniteGroupCohomologyError> {
    let degree_u32 = u32::try_from(degree)
        .map_err(|_| FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree })?;
    let monomials = algebra.presentation.basis(degree_u32)?;
    let images = monomials
        .iter()
        .map(|monomial| {
            evaluate_algebra_monomial(
                cohomology,
                &algebra.generators,
                monomial,
                multiplication_cache,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let evaluation = matrix_from_columns(&cohomology.field, value.dimension(), &images)?;
    let coordinates = evaluation.solve(value)?.particular().clone();
    algebra
        .presentation
        .ring()
        .polynomial(
            monomials
                .iter()
                .zip(coordinates.values())
                .filter(|(_, coefficient)| !coefficient.is_zero())
                .map(|(monomial, coefficient)| (coefficient, monomial.exponents().to_vec())),
        )
        .map_err(Into::into)
}

fn maximal_elementary_abelian_subgroups(
    group: &PermutationGroup,
    prime: u64,
) -> Result<Vec<PermutationGroup>, FiniteGroupCohomologyError> {
    let elements = ordered_group_elements(group);
    let prime_text = prime.to_string();
    let eligible = elements
        .iter()
        .filter(|element| element.is_identity() || element.order().to_decimal() == prime_text)
        .cloned()
        .collect::<Vec<_>>();
    let trivial = PermutationGroup::from_generators(group.degree(), &[])?;
    let trivial_key = subgroup_element_key(&trivial);
    let mut groups = vec![trivial];
    let mut known = BTreeMap::from([(trivial_key, 0usize)]);
    let mut cursor = 0usize;
    while cursor < groups.len() {
        let subgroup = groups[cursor].clone();
        cursor += 1;
        for element in &eligible {
            if subgroup.contains(element)? {
                continue;
            }
            let mut centralizes = true;
            for generator in subgroup.generators() {
                if generator.compose(element)? != element.compose(generator)? {
                    centralizes = false;
                    break;
                }
            }
            if !centralizes {
                continue;
            }
            let generators = subgroup
                .generators()
                .iter()
                .cloned()
                .chain(core::iter::once(element.clone()))
                .collect::<Vec<_>>();
            let candidate = PermutationGroup::from_generators(group.degree(), &generators)?;
            let key = subgroup_element_key(&candidate);
            if known.contains_key(&key) {
                continue;
            }
            if groups.len() >= MAX_ELEMENTARY_ABELIAN_SUBGROUPS {
                return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
                    reason: "elementary-abelian subgroup enumeration exceeded its safety ceiling",
                });
            }
            known.insert(key, groups.len());
            groups.push(candidate);
        }
    }
    let keys = groups
        .iter()
        .map(|subgroup| {
            subgroup_element_key(subgroup)
                .into_iter()
                .collect::<BTreeSet<Vec<u32>>>()
        })
        .collect::<Vec<_>>();
    let mut maximal = groups
        .into_iter()
        .enumerate()
        .filter_map(|(index, subgroup)| {
            (!keys.iter().enumerate().any(|(other_index, other)| {
                other_index != index
                    && keys[index].len() < other.len()
                    && keys[index].is_subset(other)
            }))
            .then_some(subgroup)
        })
        .collect::<Vec<_>>();
    maximal.sort_by_key(subgroup_element_key);
    Ok(maximal)
}

fn subgroup_element_key(group: &PermutationGroup) -> Vec<Vec<u32>> {
    ordered_group_elements(group)
        .into_iter()
        .map(|element| element.images().to_vec())
        .collect()
}

fn elementary_abelian_rank(
    mut order: usize,
    prime: u64,
) -> Result<usize, FiniteGroupCohomologyError> {
    let prime =
        usize::try_from(prime).map_err(|_| FiniteGroupCohomologyError::InvalidPrime { prime })?;
    let mut rank = 0usize;
    while order > 1 && order.is_multiple_of(prime) {
        order /= prime;
        rank += 1;
    }
    if order != 1 || rank == 0 {
        return Err(FiniteGroupCohomologyError::ParameterEvidenceUnavailable {
            reason: "a maximal elementary abelian subgroup has invalid prime-power order",
        });
    }
    Ok(rank)
}

fn normalized_bar_fits(alphabet_size: usize, max_degree: usize) -> bool {
    let Some(last_boundary) = max_degree.checked_add(1) else {
        return false;
    };
    (1..=last_boundary).all(|degree| {
        let Some(rows) = normalized_basis_dimension(alphabet_size, degree - 1) else {
            return false;
        };
        let Some(columns) = normalized_basis_dimension(alphabet_size, degree) else {
            return false;
        };
        rows.checked_mul(columns)
            .is_some_and(|work| work <= MODULAR_BAR_ORACLE_WORK_LIMIT)
    })
}

fn full_multiplication_table(
    elements: &[Permutation],
) -> Result<Vec<Vec<usize>>, FiniteGroupCohomologyError> {
    let indices = elements
        .iter()
        .enumerate()
        .map(|(index, element)| (element.images().to_vec(), index))
        .collect::<BTreeMap<_, _>>();
    elements
        .iter()
        .map(|left| {
            elements
                .iter()
                .map(|right| {
                    let product = left.compose(right)?;
                    indices
                        .get(product.images())
                        .copied()
                        .ok_or(GroupError::ElementNotInGroup)
                })
                .collect::<Result<Vec<_>, GroupError>>()
        })
        .collect::<Result<Vec<_>, GroupError>>()
        .map_err(Into::into)
}

fn sylow_element_index(
    resolution: &MinimalPGroupResolution,
    element: &Permutation,
) -> Result<usize, FiniteGroupCohomologyError> {
    resolution
        .elements
        .binary_search_by(|candidate| candidate.images().cmp(element.images()))
        .map_err(|_| GroupError::ElementNotInGroup.into())
}

fn stack_matrix_rows(
    field: &FiniteField,
    upper: &Matrix<FiniteField>,
    lower: &Matrix<FiniteField>,
) -> Result<Matrix<FiniteField>, LinalgError> {
    if upper.shape().1 != lower.shape().1 {
        return Err(LinalgError::ShapeMismatch {
            operation: "matrix row stacking",
            left: upper.shape(),
            right: lower.shape(),
        });
    }
    let rows = upper
        .shape()
        .0
        .checked_add(lower.shape().0)
        .ok_or(LinalgError::ShapeTooLarge {
            rows: usize::MAX,
            cols: upper.shape().1,
        })?;
    if rows == 0 {
        return Ok(Matrix::zeros(field.clone(), 0, upper.shape().1));
    }
    Matrix::from_rows(
        field.clone(),
        (0..upper.shape().0)
            .map(|row| {
                (0..upper.shape().1)
                    .map(|column| {
                        upper
                            .get(row, column)
                            .expect("stacked upper coordinate exists")
                    })
                    .collect::<Vec<_>>()
            })
            .chain((0..lower.shape().0).map(|row| {
                (0..lower.shape().1)
                    .map(|column| {
                        lower
                            .get(row, column)
                            .expect("stacked lower coordinate exists")
                    })
                    .collect::<Vec<_>>()
            })),
    )
}

fn build_minimal_p_group_resolution(
    group: &PermutationGroup,
    field: &FiniteField,
    elements: &[Permutation],
    max_degree: usize,
) -> Result<MinimalPGroupResolution, FiniteGroupCohomologyError> {
    let order = elements.len();
    let identity = Permutation::identity(group.degree());
    let identity_index = elements
        .iter()
        .position(|element| element == &identity)
        .ok_or(GroupError::ElementNotInGroup)?;
    let indices = elements
        .iter()
        .enumerate()
        .map(|(index, element)| (element.images().to_vec(), index))
        .collect::<BTreeMap<_, _>>();
    let generator_indices = group
        .generators()
        .iter()
        .filter(|generator| !generator.is_identity())
        .map(|generator| {
            indices
                .get(generator.images())
                .copied()
                .ok_or(GroupError::ElementNotInGroup)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let multiplication = full_multiplication_table(elements)?;
    let augmentation = SparseMatrix::from_triplets(
        field.clone(),
        1,
        order,
        (0..order).map(|column| (0, column, field.one())),
    )?;
    let mut ranks = vec![1usize];
    let mut differentials = vec![augmentation];

    for degree in 0..max_degree {
        let kernel_basis = differentials[degree].prime_field_kernel_basis()?;
        let radical_columns = augmentation_radical_columns(
            field,
            &kernel_basis,
            ranks[degree],
            order,
            &generator_indices,
            &multiplication,
        )?;
        let radical_basis = radical_columns.prime_field_image_basis()?;
        let span = concatenate_sparse_columns(field, &[&radical_basis, &kernel_basis])?;
        let radical_dimension = radical_basis.shape().1;
        let lifts = span
            .prime_field_pivot_columns()?
            .into_iter()
            .filter_map(|column| column.checked_sub(radical_dimension))
            .collect::<Vec<_>>();
        let next_rank = lifts.len();
        let lift_columns = select_sparse_columns(field, &kernel_basis, &lifts)?;
        let next = sparse_module_map_from_generator_images(
            field,
            order,
            ranks[degree],
            &lift_columns,
            &multiplication,
        )?;
        let composite = differentials[degree].matmul(&next)?;
        if composite.nnz() != 0 {
            return Err(
                LinalgError::Verification(VerificationError::InvalidWitness {
                    claim: "successive minimal p-group resolution differentials compose to zero",
                })
                .into(),
            );
        }
        ranks.push(next_rank);
        differentials.push(next);
    }

    Ok(MinimalPGroupResolution {
        elements: elements.to_vec(),
        multiplication,
        identity_index,
        ranks,
        differentials,
    })
}

fn standard_basis(field: &FiniteField, dimension: usize) -> Vec<Vector<FiniteField>> {
    (0..dimension)
        .map(|index| {
            Vector::from_values(
                field.clone(),
                (0..dimension).map(|coordinate| {
                    if coordinate == index {
                        field.one()
                    } else {
                        field.zero()
                    }
                }),
            )
            .expect("standard basis entries share one field")
        })
        .collect()
}

fn augmentation_radical_columns(
    field: &FiniteField,
    kernel_basis: &SparseMatrix<FiniteField>,
    module_rank: usize,
    order: usize,
    generator_indices: &[usize],
    multiplication: &[Vec<usize>],
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    debug_assert_eq!(kernel_basis.shape().0, module_rank * order);
    let column_count = kernel_basis
        .shape()
        .1
        .checked_mul(generator_indices.len())
        .ok_or(LinalgError::ShapeTooLarge {
            rows: kernel_basis.shape().0,
            cols: usize::MAX,
        })?;
    let negative_one = field.neg(&field.one())?;
    let mut entries = Vec::with_capacity(kernel_basis.nnz() * generator_indices.len() * 2);
    for row in 0..kernel_basis.shape().0 {
        let block = row / order;
        let element = row % order;
        for position in kernel_basis.row_offsets()[row]..kernel_basis.row_offsets()[row + 1] {
            let kernel_column = kernel_basis.column_indices()[position];
            let value = &kernel_basis.values()[position];
            for (generator_position, generator) in generator_indices.iter().enumerate() {
                let column = kernel_column * generator_indices.len() + generator_position;
                entries.push((row, column, field.mul(value, &negative_one)?));
                entries.push((
                    block * order + multiplication[element][*generator],
                    column,
                    value.clone(),
                ));
            }
        }
    }
    SparseMatrix::from_triplets(field.clone(), kernel_basis.shape().0, column_count, entries)
}

fn concatenate_sparse_columns(
    field: &FiniteField,
    matrices: &[&SparseMatrix<FiniteField>],
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    let rows = matrices.first().map_or(0, |matrix| matrix.shape().0);
    let columns = matrices.iter().try_fold(0usize, |columns, matrix| {
        debug_assert_eq!(matrix.ring(), field);
        debug_assert_eq!(matrix.shape().0, rows);
        columns
            .checked_add(matrix.shape().1)
            .ok_or(LinalgError::ShapeTooLarge {
                rows,
                cols: usize::MAX,
            })
    })?;
    let mut entries = Vec::new();
    let mut column_offset = 0;
    for matrix in matrices {
        entries.reserve(matrix.nnz());
        for row in 0..rows {
            for position in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                entries.push((
                    row,
                    column_offset + matrix.column_indices()[position],
                    matrix.values()[position].clone(),
                ));
            }
        }
        column_offset += matrix.shape().1;
    }
    SparseMatrix::from_triplets(field.clone(), rows, columns, entries)
}

fn select_sparse_columns(
    field: &FiniteField,
    matrix: &SparseMatrix<FiniteField>,
    columns: &[usize],
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    let mut destinations = vec![None; matrix.shape().1];
    for (destination, source) in columns.iter().copied().enumerate() {
        if source >= matrix.shape().1 {
            return Err(LinalgError::ShapeMismatch {
                operation: "sparse column selection",
                left: matrix.shape(),
                right: (matrix.shape().1, source.saturating_add(1)),
            });
        }
        destinations[source] = Some(destination);
    }
    let mut entries = Vec::new();
    for row in 0..matrix.shape().0 {
        for position in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            if let Some(column) = destinations[matrix.column_indices()[position]] {
                entries.push((row, column, matrix.values()[position].clone()));
            }
        }
    }
    SparseMatrix::from_triplets(field.clone(), matrix.shape().0, columns.len(), entries)
}

fn sparse_module_map_from_generator_images(
    field: &FiniteField,
    order: usize,
    target_rank: usize,
    images: &SparseMatrix<FiniteField>,
    multiplication: &[Vec<usize>],
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    sparse_module_map_from_generator_images_along(
        field,
        order,
        target_rank,
        order,
        images,
        multiplication,
        &(0..order).collect::<Vec<_>>(),
    )
}

fn sparse_module_map_from_generator_images_along(
    field: &FiniteField,
    target_order: usize,
    target_rank: usize,
    source_order: usize,
    images: &SparseMatrix<FiniteField>,
    target_multiplication: &[Vec<usize>],
    source_action: &[usize],
) -> Result<SparseMatrix<FiniteField>, LinalgError> {
    debug_assert_eq!(source_action.len(), source_order);
    debug_assert_eq!(images.shape().0, target_rank * target_order);
    let source_rank = images.shape().1;
    let columns = source_rank
        .checked_mul(source_order)
        .ok_or(LinalgError::ShapeTooLarge {
            rows: images.shape().0,
            cols: usize::MAX,
        })?;
    let mut entries = Vec::with_capacity(images.nnz().saturating_mul(source_order));
    for row in 0..images.shape().0 {
        let block = row / target_order;
        let element = row % target_order;
        for position in images.row_offsets()[row]..images.row_offsets()[row + 1] {
            let source_generator = images.column_indices()[position];
            let value = &images.values()[position];
            for (right, target_right) in source_action.iter().copied().enumerate() {
                entries.push((
                    block * target_order + target_multiplication[element][target_right],
                    source_generator * source_order + right,
                    value.clone(),
                ));
            }
        }
    }
    SparseMatrix::from_triplets(field.clone(), target_rank * target_order, columns, entries)
}

fn resolution_restriction_maps(
    field: &FiniteField,
    source: &MinimalPGroupResolution,
    target: &MinimalPGroupResolution,
    source_action: &[usize],
    max_degree: usize,
) -> Result<Vec<Matrix<FiniteField>>, FiniteGroupCohomologyError> {
    let source_order = source.elements.len();
    let target_order = target.elements.len();
    if source_action.len() != source_order
        || source_action[source.identity_index] != target.identity_index
    {
        return Err(
            LinalgError::Verification(VerificationError::InvalidWitness {
                claim: "a resolution comparison is induced by a unital group homomorphism",
            })
            .into(),
        );
    }
    for left in 0..source_order {
        for right in 0..source_order {
            if target.multiplication[source_action[left]][source_action[right]]
                != source_action[source.multiplication[left][right]]
            {
                return Err(
                    LinalgError::Verification(VerificationError::InvalidWitness {
                        claim: "a resolution comparison preserves group multiplication",
                    })
                    .into(),
                );
            }
        }
    }

    let initial_images = SparseMatrix::from_triplets(
        field.clone(),
        target_order,
        1,
        [(target.identity_index, 0, field.one())],
    )?;
    let mut comparisons = vec![sparse_module_map_from_generator_images_along(
        field,
        target_order,
        1,
        source_order,
        &initial_images,
        &target.multiplication,
        source_action,
    )?];
    for degree in 1..=max_degree {
        let rhs = comparisons[degree - 1].matmul(&source.differentials[degree])?;
        let generator_columns = (0..source.ranks[degree])
            .map(|generator| generator * source_order + source.identity_index)
            .collect::<Vec<_>>();
        let rhs_images = select_sparse_columns(field, &rhs, &generator_columns)?;
        let images = target.differentials[degree].prime_field_particular_solutions(&rhs_images)?;
        let comparison = sparse_module_map_from_generator_images_along(
            field,
            target_order,
            target.ranks[degree],
            source_order,
            &images,
            &target.multiplication,
            source_action,
        )?;
        if target.differentials[degree].matmul(&comparison)? != rhs {
            return Err(
                LinalgError::Verification(VerificationError::InvalidWitness {
                    claim: "a lifted resolution comparison commutes with differentials",
                })
                .into(),
            );
        }
        comparisons.push(comparison);
    }

    comparisons
        .iter()
        .enumerate()
        .map(|(degree, comparison)| {
            if source.ranks[degree] == 0 || target.ranks[degree] == 0 {
                return Ok(Matrix::zeros(
                    field.clone(),
                    source.ranks[degree],
                    target.ranks[degree],
                ));
            }
            Matrix::from_rows(
                field.clone(),
                (0..source.ranks[degree]).map(|source_generator| {
                    (0..target.ranks[degree])
                        .map(|target_generator| {
                            (0..target_order).fold(field.zero(), |sum, element| {
                                field
                                    .add(
                                        &sum,
                                        &comparison
                                            .get(
                                                target_generator * target_order + element,
                                                source_generator * source_order
                                                    + source.identity_index,
                                            )
                                            .expect("comparison coordinate exists"),
                                    )
                                    .expect("comparison coefficients share one field")
                            })
                        })
                        .collect::<Vec<_>>()
                }),
            )
            .map_err(FiniteGroupCohomologyError::from)
        })
        .collect()
}

fn minimal_p_group_basis_cup_product(
    field: &FiniteField,
    resolution: &MinimalPGroupResolution,
    left_degree: usize,
    left_index: usize,
    right_degree: usize,
    right_index: usize,
) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
    let multiplication = minimal_p_group_basis_left_multiplication(
        field,
        resolution,
        left_degree,
        left_index,
        right_degree,
    )?;
    Vector::from_values(
        field.clone(),
        (0..multiplication.shape().0).map(|row| {
            multiplication
                .get(row, right_index)
                .expect("right basis index was checked by the public operation")
        }),
    )
    .map_err(Into::into)
}

fn minimal_p_group_basis_left_multiplication(
    field: &FiniteField,
    resolution: &MinimalPGroupResolution,
    left_degree: usize,
    left_index: usize,
    right_degree: usize,
) -> Result<Matrix<FiniteField>, FiniteGroupCohomologyError> {
    let order = resolution.multiplication.len();
    let source_rank = resolution.ranks[left_degree];
    let initial_images = SparseMatrix::from_triplets(
        field.clone(),
        order,
        source_rank,
        [(resolution.identity_index, left_index, field.one())],
    )?;
    let mut lift = sparse_module_map_from_generator_images(
        field,
        order,
        1,
        &initial_images,
        &resolution.multiplication,
    )?;

    for target_degree in 1..=right_degree {
        let lifted_source_degree = left_degree + target_degree;
        let rhs = lift.matmul(&resolution.differentials[lifted_source_degree])?;
        let lifted_source_rank = resolution.ranks[lifted_source_degree];
        let generator_columns = (0..lifted_source_rank)
            .map(|generator| generator * order + resolution.identity_index)
            .collect::<Vec<_>>();
        let rhs_images = select_sparse_columns(field, &rhs, &generator_columns)?;
        let images = resolution.differentials[target_degree]
            .prime_field_particular_solutions(&rhs_images)?;
        lift = sparse_module_map_from_generator_images(
            field,
            order,
            resolution.ranks[target_degree],
            &images,
            &resolution.multiplication,
        )?;
    }

    let product_rank = resolution.ranks[left_degree + right_degree];
    let right_rank = resolution.ranks[right_degree];
    if product_rank == 0 || right_rank == 0 {
        return Ok(Matrix::zeros(field.clone(), product_rank, right_rank));
    }
    Matrix::from_rows(
        field.clone(),
        (0..product_rank).map(|generator| {
            (0..right_rank)
                .map(|right_index| {
                    (0..order).fold(field.zero(), |sum, element| {
                        field
                            .add(
                                &sum,
                                &lift
                                    .get(
                                        right_index * order + element,
                                        generator * order + resolution.identity_index,
                                    )
                                    .expect("lifted generator coordinate exists"),
                            )
                            .expect("cochain coefficients share one field")
                    })
                })
                .collect::<Vec<_>>()
        }),
    )
    .map_err(Into::into)
}

fn normalized_alphabet(
    group: &PermutationGroup,
) -> Result<(Vec<Permutation>, NormalizedMultiplicationTable), FiniteGroupCohomologyError> {
    let elements = ordered_group_elements(group);
    let identity = Permutation::identity(group.degree());
    let nonidentity = elements
        .into_iter()
        .filter(|element| element != &identity)
        .collect::<Vec<_>>();
    let indices = nonidentity
        .iter()
        .enumerate()
        .map(|(index, element)| (element.images().to_vec(), index))
        .collect::<BTreeMap<_, _>>();
    let multiplication = nonidentity
        .iter()
        .map(|left| {
            nonidentity
                .iter()
                .map(|right| {
                    let product = left.compose(right)?;
                    Ok(indices.get(product.images()).copied())
                })
                .collect::<Result<Vec<_>, GroupError>>()
        })
        .collect::<Result<Vec<_>, GroupError>>()?;
    Ok((nonidentity, multiplication))
}

fn normalized_group_nerve(
    group: &PermutationGroup,
    max_dimension: usize,
) -> Result<SimplicialSet, FiniteGroupCohomologyError> {
    let (alphabet, multiplication) = normalized_alphabet(group)?;
    let alphabet_size = alphabet.len();
    let mut faces = Vec::with_capacity(max_dimension.saturating_add(1));
    faces.push(vec![Vec::new()]);
    for dimension in 1..=max_dimension {
        let count = normalized_basis_dimension(alphabet_size, dimension).ok_or(
            FiniteGroupCohomologyError::BasisCardinalityOverflow {
                degree: dimension,
                alphabet_size,
            },
        )?;
        let simplices = (0..count)
            .map(|simplex| {
                let tuple = decode_tuple(simplex, dimension, alphabet_size);
                let mut simplex_faces = Vec::with_capacity(dimension.saturating_add(1));
                simplex_faces.push(SimplicialFace::cell(encode_tuple(
                    &tuple[1..],
                    alphabet_size,
                )));
                for face in 1..dimension {
                    let pair = face - 1;
                    if let Some(product) = multiplication[tuple[pair]][tuple[pair + 1]] {
                        let mut face_tuple = Vec::with_capacity(dimension - 1);
                        face_tuple.extend_from_slice(&tuple[..pair]);
                        face_tuple.push(product);
                        face_tuple.extend_from_slice(&tuple[pair + 2..]);
                        simplex_faces.push(SimplicialFace::cell(encode_tuple(
                            &face_tuple,
                            alphabet_size,
                        )));
                    } else {
                        let mut underlying = Vec::with_capacity(dimension.saturating_sub(2));
                        underlying.extend_from_slice(&tuple[..pair]);
                        underlying.extend_from_slice(&tuple[pair + 2..]);
                        let surjection = (0..dimension)
                            .map(|vertex| if vertex <= pair { vertex } else { vertex - 1 })
                            .collect::<Vec<_>>();
                        simplex_faces.push(SimplicialFace::degenerate(
                            dimension - 2,
                            encode_tuple(&underlying, alphabet_size),
                            surjection,
                        ));
                    }
                }
                simplex_faces.push(SimplicialFace::cell(encode_tuple(
                    &tuple[..dimension - 1],
                    alphabet_size,
                )));
                simplex_faces
            })
            .collect::<Vec<_>>();
        faces.push(simplices);
    }
    SimplicialSet::new(faces).map_err(Into::into)
}

fn normalized_basis_dimension(alphabet_size: usize, degree: usize) -> Option<usize> {
    if degree == 0 {
        Some(1)
    } else {
        alphabet_size.checked_pow(u32::try_from(degree).ok()?)
    }
}

fn normalized_bar_entry_bound(columns: usize, degree: usize) -> Option<usize> {
    columns.checked_mul(degree.checked_add(1)?)
}

fn normalized_boundary(
    field: &FiniteField,
    alphabet_size: usize,
    multiplication: &[Vec<Option<usize>>],
    degree: usize,
) -> SparseMatrix<FiniteField> {
    let rows = normalized_basis_dimension(alphabet_size, degree - 1)
        .expect("basis dimensions were checked before boundary construction");
    let columns = normalized_basis_dimension(alphabet_size, degree)
        .expect("basis dimensions were checked before boundary construction");
    let mut entries = Vec::with_capacity(
        normalized_bar_entry_bound(columns, degree)
            .expect("the sparse bar work estimate was checked before construction"),
    );
    let positive = field.one();
    let negative = field
        .neg(&positive)
        .expect("one belongs to the normalized-bar coefficient field");
    for column in 0..columns {
        let tuple = decode_tuple(column, degree, alphabet_size);
        entries.push((
            encode_tuple(&tuple[1..], alphabet_size),
            column,
            positive.clone(),
        ));
        for index in 0..degree.saturating_sub(1) {
            if let Some(product) = multiplication[tuple[index]][tuple[index + 1]] {
                let mut face = Vec::with_capacity(degree - 1);
                face.extend_from_slice(&tuple[..index]);
                face.push(product);
                face.extend_from_slice(&tuple[index + 2..]);
                entries.push((
                    encode_tuple(&face, alphabet_size),
                    column,
                    if (index + 1) % 2 == 1 {
                        negative.clone()
                    } else {
                        positive.clone()
                    },
                ));
            }
        }
        entries.push((
            encode_tuple(&tuple[..degree - 1], alphabet_size),
            column,
            if degree % 2 == 1 {
                negative.clone()
            } else {
                positive.clone()
            },
        ));
    }
    SparseMatrix::from_triplets(field.clone(), rows, columns, entries)
        .expect("normalized boundary entries have checked sparse coordinates")
}

fn normalized_integral_boundary(
    alphabet_size: usize,
    multiplication: &[Vec<Option<usize>>],
    degree: usize,
) -> SparseMatrix<IntegerRing> {
    let rows = normalized_basis_dimension(alphabet_size, degree - 1)
        .expect("basis dimensions were checked before boundary construction");
    let columns = normalized_basis_dimension(alphabet_size, degree)
        .expect("basis dimensions were checked before boundary construction");
    let mut entries = Vec::with_capacity(
        normalized_bar_entry_bound(columns, degree)
            .expect("the sparse integral bar work estimate was checked before construction"),
    );
    for column in 0..columns {
        let tuple = decode_tuple(column, degree, alphabet_size);
        entries.push((
            encode_tuple(&tuple[1..], alphabet_size),
            column,
            Integer::one(),
        ));
        for index in 0..degree.saturating_sub(1) {
            if let Some(product) = multiplication[tuple[index]][tuple[index + 1]] {
                let mut face = Vec::with_capacity(degree - 1);
                face.extend_from_slice(&tuple[..index]);
                face.push(product);
                face.extend_from_slice(&tuple[index + 2..]);
                entries.push((
                    encode_tuple(&face, alphabet_size),
                    column,
                    Integer::from(if (index + 1) % 2 == 1 { -1 } else { 1 }),
                ));
            }
        }
        entries.push((
            encode_tuple(&tuple[..degree - 1], alphabet_size),
            column,
            Integer::from(if degree % 2 == 1 { -1 } else { 1 }),
        ));
    }
    SparseMatrix::from_triplets(Z, rows, columns, entries)
        .expect("normalized integral boundary has checked sparse coordinates")
}

fn integral_degree_without_cochains(
    additive_group: FinitelyGeneratedAbelianGroup,
    basis_orders: Vec<Option<Integer>>,
) -> IntegralCohomologyDegree {
    IntegralCohomologyDegree {
        additive_group,
        coordinate_map: Matrix::zeros(Z, basis_orders.len(), 0),
        basis_orders,
        representatives: Vec::new(),
    }
}

fn integral_cohomology_degree(
    incoming: &Matrix<IntegerRing>,
    outgoing: &Matrix<IntegerRing>,
) -> Result<IntegralCohomologyDegree, FiniteGroupIntegralCohomologyError> {
    let ambient_dimension = outgoing.shape().1;
    if incoming.shape().0 != ambient_dimension {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "adjacent integral bar coboundaries share their middle module",
            },
        ));
    }
    // Row HNF on the transpose produces exactly the column-basis change
    // needed to split the outgoing kernel.  Unlike a full Smith witness, its
    // unimodular transform has ambient-domain size, not target size; this is
    // essential for tall reduced bar coboundaries.
    let outgoing_transpose = outgoing.transpose();
    let (outgoing_hnf_transpose, left) = outgoing_transpose.hermite_normal_form_with_transform();
    if left.matmul(&outgoing_transpose)? != outgoing_hnf_transpose {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "Hermite transformation reconstructs the outgoing coboundary",
            },
        ));
    }
    let hnf_rows = outgoing_hnf_transpose.rows().collect::<Vec<_>>();
    let outgoing_rank = hnf_rows
        .iter()
        .take_while(|row| row.iter().any(|value| !value.is_zero()))
        .count();
    if hnf_rows[outgoing_rank..]
        .iter()
        .any(|row| row.iter().any(|value| !value.is_zero()))
    {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "row Hermite form places all zero rows after its pivots",
            },
        ));
    }
    let kernel_dimension = ambient_dimension.saturating_sub(outgoing_rank);
    let right = left.transpose();
    let right_inverse = unimodular_inverse(&right)?;
    let kernel_basis = matrix_column_range(&right, outgoing_rank, ambient_dimension)?;
    let incoming_in_kernel_split_coordinates = right_inverse.matmul(incoming)?;
    let obstruction = matrix_row_range(&incoming_in_kernel_split_coordinates, 0, outgoing_rank)?;
    if obstruction != Matrix::zeros(Z, outgoing_rank, incoming.shape().1) {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "consecutive integral bar coboundaries compose to zero",
            },
        ));
    }
    let boundaries_in_kernel = matrix_row_range(
        &incoming_in_kernel_split_coordinates,
        outgoing_rank,
        ambient_dimension,
    )?;
    let quotient_smith = boundaries_in_kernel.smith_normal_form_with_transforms()?;
    let quotient_rank = quotient_smith.smith().rank();
    let left = quotient_smith.left().clone();
    let left_inverse = unimodular_inverse(&left)?;
    let all_representatives = kernel_basis.matmul(&left_inverse)?;
    let full_coordinate_map = left.matmul(&matrix_row_range(
        &right_inverse,
        outgoing_rank,
        ambient_dimension,
    )?)?;

    let mut torsion_coefficients = Vec::new();
    let mut coordinate_indices = Vec::new();
    let form = quotient_smith.smith().form();
    for index in 0..quotient_rank {
        let order = form
            .get(index, index)
            .expect("Smith diagonal index exists")
            .abs();
        if !order.is_one() {
            torsion_coefficients.push(order);
            coordinate_indices.push(index);
        }
    }
    coordinate_indices.extend(quotient_rank..kernel_dimension);
    let free_rank = kernel_dimension.saturating_sub(quotient_rank);
    let additive_group =
        FinitelyGeneratedAbelianGroup::new(free_rank, torsion_coefficients.clone())?;
    let basis_orders = torsion_coefficients
        .into_iter()
        .map(Some)
        .chain((0..free_rank).map(|_| None))
        .collect::<Vec<_>>();
    let representatives = coordinate_indices
        .iter()
        .map(|index| matrix_column_vector(&all_representatives, *index))
        .collect::<Result<Vec<_>, _>>()?;
    let coordinate_map = matrix_select_rows(&full_coordinate_map, &coordinate_indices)?;
    Ok(IntegralCohomologyDegree {
        additive_group,
        basis_orders,
        representatives,
        coordinate_map,
    })
}

fn unimodular_inverse(
    matrix: &Matrix<IntegerRing>,
) -> Result<Matrix<IntegerRing>, FiniteGroupIntegralCohomologyError> {
    let (rows, columns) = matrix.shape();
    if rows != columns {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "a requested unimodular inverse is square",
            },
        ));
    }
    let smith = matrix.smith_normal_form_with_transforms()?;
    if smith.smith().form() != &Matrix::identity(Z, rows) {
        return Err(FiniteGroupIntegralCohomologyError::Verification(
            VerificationError::InvalidWitness {
                claim: "a Smith transformation matrix is unimodular",
            },
        ));
    }
    smith.right().matmul(smith.left()).map_err(Into::into)
}

fn matrix_row_range(
    matrix: &Matrix<IntegerRing>,
    start: usize,
    end: usize,
) -> Result<Matrix<IntegerRing>, LinalgError> {
    if start > end || end > matrix.shape().0 {
        return Err(LinalgError::ShapeMismatch {
            operation: "integer matrix row range",
            left: matrix.shape(),
            right: (start, end),
        });
    }
    if start == end {
        return Ok(Matrix::zeros(Z, 0, matrix.shape().1));
    }
    Matrix::from_rows(
        Z,
        (start..end).map(|row| {
            (0..matrix.shape().1)
                .map(|column| matrix.get(row, column).expect("range was checked"))
                .collect::<Vec<_>>()
        }),
    )
}

fn matrix_column_range(
    matrix: &Matrix<IntegerRing>,
    start: usize,
    end: usize,
) -> Result<Matrix<IntegerRing>, LinalgError> {
    if start > end || end > matrix.shape().1 {
        return Err(LinalgError::ShapeMismatch {
            operation: "integer matrix column range",
            left: matrix.shape(),
            right: (start, end),
        });
    }
    if matrix.shape().0 == 0 {
        return Ok(Matrix::zeros(Z, 0, end - start));
    }
    Matrix::from_rows(
        Z,
        (0..matrix.shape().0).map(|row| {
            (start..end)
                .map(|column| matrix.get(row, column).expect("range was checked"))
                .collect::<Vec<_>>()
        }),
    )
}

fn matrix_select_rows(
    matrix: &Matrix<IntegerRing>,
    rows: &[usize],
) -> Result<Matrix<IntegerRing>, LinalgError> {
    if rows.iter().any(|row| *row >= matrix.shape().0) {
        return Err(LinalgError::ShapeMismatch {
            operation: "integer matrix row selection",
            left: matrix.shape(),
            right: (rows.len(), matrix.shape().1),
        });
    }
    if rows.is_empty() {
        return Ok(Matrix::zeros(Z, 0, matrix.shape().1));
    }
    Matrix::from_rows(
        Z,
        rows.iter().map(|row| {
            (0..matrix.shape().1)
                .map(|column| matrix.get(*row, column).expect("row was checked"))
                .collect::<Vec<_>>()
        }),
    )
}

fn matrix_column_vector(
    matrix: &Matrix<IntegerRing>,
    column: usize,
) -> Result<Vector<IntegerRing>, LinalgError> {
    Vector::from_values(
        Z,
        (0..matrix.shape().0).map(|row| matrix.get(row, column).expect("column was checked")),
    )
}

fn decode_tuple(mut index: usize, degree: usize, alphabet_size: usize) -> Vec<usize> {
    let mut tuple = vec![0; degree];
    for value in tuple.iter_mut().rev() {
        *value = index % alphabet_size;
        index /= alphabet_size;
    }
    tuple
}

fn encode_tuple(tuple: &[usize], alphabet_size: usize) -> usize {
    tuple
        .iter()
        .fold(0usize, |index, value| index * alphabet_size + value)
}

#[derive(Clone)]
struct FiniteGroupOperationCandidate {
    source_generator_index: usize,
    operation: AdmissibleCohomologyOperation,
    image: Vector<FiniteField>,
}

fn finite_group_operation_candidates(
    cohomology: &FiniteGroupCohomologyTruncation,
    generators: &[FiniteGroupCohomologyAlgebraGenerator],
    target_degree: usize,
) -> Result<Vec<FiniteGroupOperationCandidate>, FiniteGroupCohomologyError> {
    let prime = cohomology
        .field()
        .characteristic()
        .to_u64_checked()
        .ok_or(FiniteGroupCohomologyError::InvalidPrime { prime: 0 })?;
    let mut candidates = Vec::new();
    for (source_generator_index, generator) in generators.iter().enumerate() {
        let Some(shift) = target_degree.checked_sub(generator.degree) else {
            continue;
        };
        let operations = if prime == 2 {
            if shift == 0 || shift > generator.degree {
                Vec::new()
            } else {
                u32::try_from(shift)
                    .ok()
                    .map(AdmissibleCohomologyOperation::steenrod_square)
                    .into_iter()
                    .collect()
            }
        } else {
            let prime_minus_one = usize::try_from(prime.saturating_sub(1)).map_err(|_| {
                FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX }
            })?;
            let period = prime_minus_one
                .checked_mul(2)
                .ok_or(FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX })?;
            let mut operations = Vec::new();
            if shift > 0 && shift.is_multiple_of(period) {
                let power = shift / period;
                if power.saturating_mul(2) <= generator.degree {
                    if let Ok(power) = u32::try_from(power) {
                        operations
                            .push(AdmissibleCohomologyOperation::reduced_power(prime, power)?);
                    }
                }
            }
            if shift >= 1 && (shift - 1).is_multiple_of(period) {
                let power = (shift - 1) / period;
                if power.saturating_mul(2) < generator.degree {
                    if let Ok(power) = u32::try_from(power) {
                        operations.push(AdmissibleCohomologyOperation::bockstein_reduced_power(
                            prime, power,
                        )?);
                    }
                }
            }
            operations
        };
        for operation in operations {
            let image = match cohomology.apply_operation(
                &operation,
                generator.degree,
                generator.basis_index,
            ) {
                Ok(image) => image,
                Err(FiniteGroupCohomologyError::CohomologyOperationUnavailable { .. }) => {
                    continue;
                }
                Err(error) => return Err(error),
            };
            if image.values().all(|coordinate| coordinate.is_zero()) {
                continue;
            }
            candidates.push(FiniteGroupOperationCandidate {
                source_generator_index,
                operation,
                image,
            });
        }
    }
    Ok(candidates)
}

fn operation_witness_for_basis(
    field: &FiniteField,
    ambient_dimension: usize,
    basis: &Vector<FiniteField>,
    decomposable_span: &[Vector<FiniteField>],
    operations: &[FiniteGroupOperationCandidate],
) -> Result<Option<FiniteGroupCohomologyGeneratorOperationWitness>, FiniteGroupCohomologyError> {
    let decomposable_rank =
        matrix_from_columns(field, ambient_dimension, decomposable_span)?.rank()?;
    for candidate in operations {
        let mut columns = decomposable_span.to_vec();
        columns.push(candidate.image.clone());
        let matrix = matrix_from_columns(field, ambient_dimension, &columns)?;
        if matrix.rank()? == decomposable_rank {
            continue;
        }
        let mut with_basis = columns.clone();
        with_basis.push(basis.clone());
        if matrix_from_columns(field, ambient_dimension, &with_basis)?.rank()? != matrix.rank()? {
            continue;
        }
        let solution = matrix.solve(basis)?;
        let operation_coefficient = solution
            .particular()
            .get(columns.len() - 1)
            .expect("operation coefficient exists");
        if operation_coefficient.is_zero() {
            continue;
        }
        let decomposable_correction = Vector::from_values(
            field.clone(),
            (0..ambient_dimension).map(|index| {
                let scaled = field
                    .mul(
                        &operation_coefficient,
                        &candidate.image.get(index).expect("operation image exists"),
                    )
                    .expect("operation image belongs to the cohomology field");
                field
                    .sub(&basis.get(index).expect("basis coordinate exists"), &scaled)
                    .expect("operation image belongs to the cohomology field")
            }),
        )?;
        return Ok(Some(FiniteGroupCohomologyGeneratorOperationWitness {
            source_generator_index: candidate.source_generator_index,
            operation: candidate.operation.clone(),
            operation_coefficient,
            decomposable_correction,
        }));
    }
    Ok(None)
}

fn algebra_polynomial_ring(
    field: &FiniteField,
    generators: &[FiniteGroupCohomologyAlgebraGenerator],
) -> Result<GradedCommutativePolynomialRing, FiniteGroupCohomologyError> {
    GradedCommutativePolynomialRing::new(
        field.clone(),
        generators
            .iter()
            .map(|generator| {
                let degree = u32::try_from(generator.degree).map_err(|_| {
                    FiniteGroupCohomologyError::AlgebraDegreeOverflow {
                        degree: generator.degree,
                    }
                })?;
                Ok(GradedGenerator::new(generator.name.clone(), degree))
            })
            .collect::<Result<Vec<_>, FiniteGroupCohomologyError>>()?,
    )
    .map_err(Into::into)
}

fn evaluate_algebra_monomial(
    cohomology: &FiniteGroupCohomologyTruncation,
    generators: &[FiniteGroupCohomologyAlgebraGenerator],
    monomial: &GradedMonomial,
    multiplication_cache: &mut BTreeMap<(usize, usize), Matrix<FiniteField>>,
) -> Result<Vector<FiniteField>, FiniteGroupCohomologyError> {
    if monomial.exponents().len() != generators.len() {
        return Err(FiniteGroupCohomologyError::InvalidAlgebraTruncation {
            reason: "monomial arity does not match the selected generator family",
        });
    }
    let mut value = Vector::from_values(cohomology.field.clone(), [cohomology.field.one()])?;
    let mut current_degree = 0_usize;
    for (generator_index, (generator, exponent)) in
        generators.iter().zip(monomial.exponents()).enumerate()
    {
        for _ in 0..*exponent {
            let key = (generator_index, current_degree);
            let multiplication = match multiplication_cache.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(cohomology.basis_left_multiplication(
                    generator.degree,
                    generator.basis_index,
                    current_degree,
                )?),
            };
            value = multiplication.apply(&value)?;
            if generator.degree % 2 == 1 && current_degree % 2 == 1 {
                value = Vector::from_values(
                    cohomology.field.clone(),
                    value
                        .values()
                        .map(|coefficient| {
                            cohomology
                                .field
                                .neg(&coefficient)
                                .map_err(LinalgError::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
            }
            current_degree = current_degree
                .checked_add(generator.degree)
                .ok_or(FiniteGroupCohomologyError::AlgebraDegreeOverflow { degree: usize::MAX })?;
        }
    }
    Ok(value)
}

fn standard_basis_vector(
    field: &FiniteField,
    dimension: usize,
    index: usize,
) -> Result<Vector<FiniteField>, LinalgError> {
    Vector::from_values(
        field.clone(),
        (0..dimension).map(|coordinate| {
            if coordinate == index {
                field.one()
            } else {
                field.zero()
            }
        }),
    )
}

fn matrix_from_columns(
    field: &FiniteField,
    ambient_dimension: usize,
    columns: &[Vector<FiniteField>],
) -> Result<Matrix<FiniteField>, LinalgError> {
    if columns.is_empty() || ambient_dimension == 0 {
        return Ok(Matrix::zeros(
            field.clone(),
            ambient_dimension,
            columns.len(),
        ));
    }
    Matrix::from_rows(
        field.clone(),
        (0..ambient_dimension).map(|row| {
            columns
                .iter()
                .map(|column| column.get(row).expect("column dimension was checked"))
                .collect::<Vec<_>>()
        }),
    )
}

fn normalized_cup_product(
    field: &FiniteField,
    alphabet_size: usize,
    left_degree: usize,
    left: &Vector<FiniteField>,
    right_degree: usize,
    right: &Vector<FiniteField>,
) -> Result<Vector<FiniteField>, LinalgError> {
    let target_degree = left_degree + right_degree;
    let target_dimension = normalized_basis_dimension(alphabet_size, target_degree)
        .expect("target bar dimension belongs to the constructed truncation");
    Vector::from_values(
        field.clone(),
        (0..target_dimension).map(|index| {
            let tuple = decode_tuple(index, target_degree, alphabet_size);
            let left_index = encode_tuple(&tuple[..left_degree], alphabet_size);
            let right_index = encode_tuple(&tuple[left_degree..], alphabet_size);
            field
                .mul(
                    &left.get(left_index).expect("left bar coordinate exists"),
                    &right.get(right_index).expect("right bar coordinate exists"),
                )
                .expect("cup-product coefficients belong to this field")
        }),
    )
}

fn normalized_integral_cup_product(
    alphabet_size: usize,
    left_degree: usize,
    left: &Vector<IntegerRing>,
    right_degree: usize,
    right: &Vector<IntegerRing>,
) -> Result<Vector<IntegerRing>, LinalgError> {
    let target_degree = left_degree + right_degree;
    let target_dimension = normalized_basis_dimension(alphabet_size, target_degree)
        .expect("target bar dimension belongs to the constructed truncation");
    Vector::from_values(
        Z,
        (0..target_dimension).map(|index| {
            let tuple = decode_tuple(index, target_degree, alphabet_size);
            let left_index = encode_tuple(&tuple[..left_degree], alphabet_size);
            let right_index = encode_tuple(&tuple[left_degree..], alphabet_size);
            &left
                .get(left_index)
                .expect("left integral bar coordinate exists")
                * &right
                    .get(right_index)
                    .expect("right integral bar coordinate exists")
        }),
    )
}

fn push_frame(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| NoetherError::Serialization("component exceeds u64".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

fn push_usizes(output: &mut Vec<u8>, values: &[usize]) -> Result<(), NoetherError> {
    output.extend_from_slice(
        &u64::try_from(values.len())
            .map_err(|_| NoetherError::Serialization("value count exceeds u64".to_owned()))?
            .to_be_bytes(),
    );
    for value in values {
        output.extend_from_slice(
            &u64::try_from(*value)
                .map_err(|_| NoetherError::Serialization("value exceeds u64".to_owned()))?
                .to_be_bytes(),
        );
    }
    Ok(())
}

fn push_u32s(output: &mut Vec<u8>, values: &[u32]) -> Result<(), NoetherError> {
    output.extend_from_slice(
        &u64::try_from(values.len())
            .map_err(|_| NoetherError::Serialization("value count exceeds u64".to_owned()))?
            .to_be_bytes(),
    );
    for value in values {
        output.extend_from_slice(&value.to_be_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyclic(order: usize) -> PermutationGroup {
        let images = (0..order)
            .map(|point| u32::try_from((point + 1) % order).unwrap())
            .collect::<Vec<_>>();
        let generator = Permutation::from_images(&images).unwrap();
        PermutationGroup::from_generators(order, &[generator]).unwrap()
    }

    fn symmetric_three() -> PermutationGroup {
        let transposition = Permutation::from_images(&[1, 0, 2]).unwrap();
        let cycle = Permutation::from_images(&[1, 2, 0]).unwrap();
        PermutationGroup::from_generators(3, &[transposition, cycle]).unwrap()
    }

    fn symmetric_four() -> PermutationGroup {
        let transposition = Permutation::from_images(&[1, 0, 2, 3]).unwrap();
        let cycle = Permutation::from_images(&[1, 2, 3, 0]).unwrap();
        PermutationGroup::from_generators(4, &[transposition, cycle]).unwrap()
    }

    fn alternating_four() -> PermutationGroup {
        let first = Permutation::from_images(&[1, 2, 0, 3]).unwrap();
        let second = Permutation::from_images(&[0, 2, 3, 1]).unwrap();
        PermutationGroup::from_generators(4, &[first, second]).unwrap()
    }

    fn dihedral_eight() -> PermutationGroup {
        let rotation = Permutation::from_images(&[1, 2, 3, 0]).unwrap();
        let reflection = Permutation::from_images(&[0, 3, 2, 1]).unwrap();
        PermutationGroup::from_generators(4, &[rotation, reflection]).unwrap()
    }

    fn elementary_abelian_rank_two(prime: usize) -> PermutationGroup {
        let order = prime * prime;
        let translations = [(1usize, 0usize), (0usize, 1usize)]
            .into_iter()
            .map(|(dx, dy)| {
                let images = (0..order)
                    .map(|point| {
                        let x = point / prime;
                        let y = point % prime;
                        u32::try_from(((x + dx) % prime) * prime + (y + dy) % prime).unwrap()
                    })
                    .collect::<Vec<_>>();
                Permutation::from_images(&images).unwrap()
            })
            .collect::<Vec<_>>();
        PermutationGroup::from_generators(order, &translations).unwrap()
    }

    fn elementary_abelian_two(rank: usize) -> PermutationGroup {
        let order = 1usize << rank;
        let translations = (0..rank)
            .map(|bit| {
                let images = (0..order)
                    .map(|point| u32::try_from(point ^ (1usize << bit)).unwrap())
                    .collect::<Vec<_>>();
                Permutation::from_images(&images).unwrap()
            })
            .collect::<Vec<_>>();
        PermutationGroup::from_generators(order, &translations).unwrap()
    }

    #[test]
    fn finite_abelian_family_presentations_reuse_the_classifying_space_formula() {
        let cyclic_four = cyclic(4);
        let result = cyclic_four
            .complete_abelian_modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(result.invariant_factors(), &[Integer::from(4_u8)]);
        assert_eq!(
            result.presentation().hilbert_function(8).unwrap(),
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(result.generators()[0].display_name(), "iota_1");
        result.verify().unwrap();
        assert!(!result.canonical_bytes().unwrap().is_empty());

        let klein = elementary_abelian_two(2);
        let elementary = klein
            .complete_abelian_modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            elementary.invariant_factors(),
            &[Integer::from(2_u8), Integer::from(2_u8)]
        );
        assert_eq!(
            elementary.presentation().hilbert_function(5).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn finite_abelian_integral_family_reuses_rank_two_classifying_space_formula() {
        let klein_four = elementary_abelian_two(2);
        let result = klein_four.complete_abelian_integral_cohomology().unwrap();
        assert_eq!(
            result.invariant_factors(),
            &[Integer::from(2), Integer::from(2)]
        );
        assert_eq!(
            result.completion_strategy(),
            "finite_abelian_classifying_space"
        );
        assert_eq!(result.classifying_space_cohomology().generators().len(), 3);
        assert_eq!(result.classifying_space_cohomology().relations().len(), 4);
        assert_eq!(
            result.additive_group(6).unwrap(),
            FinitelyGeneratedAbelianGroup::from_invariants(0, vec![Integer::from(2); 4])
        );
        result.verify().unwrap();
        assert!(!result.canonical_bytes().unwrap().is_empty());

        assert!(matches!(
            elementary_abelian_two(3).complete_abelian_integral_cohomology(),
            Err(FiniteAbelianGroupCohomologyError::EilenbergMacLane(
                EilenbergMacLaneError::UnsupportedOperation(_)
            ))
        ));
        assert!(matches!(
            symmetric_three().complete_abelian_integral_cohomology(),
            Err(FiniteAbelianGroupCohomologyError::AbelianInvariants(
                FiniteAbelianInvariantError::NonAbelian
            ))
        ));
    }

    #[test]
    fn finite_abelian_family_certificate_rejects_nonabelian_groups() {
        assert!(matches!(
            symmetric_three()
                .complete_abelian_modular_cohomology(FiniteField::prime(2_u8).unwrap()),
            Err(FiniteAbelianGroupCohomologyError::AbelianInvariants(
                FiniteAbelianInvariantError::NonAbelian
            ))
        ));
    }

    #[test]
    fn cyclic_periodic_resolution_certifies_the_integral_ring() {
        let result = cyclic(4).complete_cyclic_integral_cohomology().unwrap();
        assert_eq!(result.order(), &Integer::from(4_u8));
        assert_eq!(result.generator_id(), Some("u2_1"));
        assert_eq!(result.generator_degree(), Some(2));
        assert_eq!(
            result.generator_display_name().as_deref(),
            Some("beta_4(iota_1)")
        );
        assert_eq!(result.relation_coefficient(), Some(&Integer::from(4_u8)));
        assert_eq!(
            (0..=6)
                .map(|degree| result.additive_group(degree))
                .collect::<Vec<_>>(),
            vec![
                FinitelyGeneratedAbelianGroup::free(1),
                FinitelyGeneratedAbelianGroup::trivial(),
                FinitelyGeneratedAbelianGroup::cyclic(Integer::from(4_u8)).unwrap(),
                FinitelyGeneratedAbelianGroup::trivial(),
                FinitelyGeneratedAbelianGroup::cyclic(Integer::from(4_u8)).unwrap(),
                FinitelyGeneratedAbelianGroup::trivial(),
                FinitelyGeneratedAbelianGroup::cyclic(Integer::from(4_u8)).unwrap(),
            ]
        );
        assert_eq!(result.completion_strategy(), "cyclic_periodic_resolution");
        result.verify().unwrap();
        assert!(!result.canonical_bytes().unwrap().is_empty());
    }

    #[test]
    fn cyclic_integral_ring_handles_the_trivial_group_and_rejects_klein_four() {
        let trivial = PermutationGroup::from_generators(0, &[]).unwrap();
        let result = trivial.complete_cyclic_integral_cohomology().unwrap();
        assert!(result.order().is_one());
        assert_eq!(result.generator_id(), None);
        assert_eq!(result.relation_coefficient(), None);
        assert_eq!(
            result.additive_group(8),
            FinitelyGeneratedAbelianGroup::trivial()
        );
        result.verify().unwrap();

        assert!(matches!(
            elementary_abelian_two(2).complete_cyclic_integral_cohomology(),
            Err(FiniteCyclicGroupIntegralCohomologyError::NotCyclic {
                invariant_factors
            }) if invariant_factors == vec![Integer::from(2_u8), Integer::from(2_u8)]
        ));
    }

    #[test]
    fn universal_integral_low_degrees_use_the_abelianization() {
        let result = symmetric_three().integral_cohomology(2).unwrap();
        assert_eq!(
            result.resolution_strategy(),
            FiniteGroupIntegralResolutionStrategy::UniversalLowDegree
        );
        assert_eq!(
            result.additive_group(0).unwrap(),
            &FinitelyGeneratedAbelianGroup::free(1)
        );
        assert_eq!(
            result.additive_group(1).unwrap(),
            &FinitelyGeneratedAbelianGroup::trivial()
        );
        assert_eq!(
            result.additive_group(2).unwrap(),
            &FinitelyGeneratedAbelianGroup::cyclic(Integer::from(2)).unwrap()
        );
        assert_eq!(
            result.basis_cup_product(0, 0, 2, 0).unwrap(),
            vec![Integer::one()]
        );
        assert!(result.basis_cocycles(2).unwrap().is_none());
        result.verify().unwrap();
    }

    #[test]
    fn normalized_integral_bar_computes_cyclic_products() {
        let result = cyclic(2).integral_cohomology(4).unwrap();
        assert_eq!(
            result.resolution_strategy(),
            FiniteGroupIntegralResolutionStrategy::NormalizedBar
        );
        assert_eq!(
            (0..=4)
                .map(|degree| result.additive_group(degree).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![
                FinitelyGeneratedAbelianGroup::free(1),
                FinitelyGeneratedAbelianGroup::trivial(),
                FinitelyGeneratedAbelianGroup::cyclic(Integer::from(2)).unwrap(),
                FinitelyGeneratedAbelianGroup::trivial(),
                FinitelyGeneratedAbelianGroup::cyclic(Integer::from(2)).unwrap(),
            ]
        );
        assert_eq!(
            result.basis_cup_product(2, 0, 2, 0).unwrap(),
            vec![Integer::one()]
        );
        let degree_two = &result.basis_cocycles(2).unwrap().unwrap()[0];
        let coboundary = result.boundaries[3].transpose().apply(degree_two).unwrap();
        assert!(coboundary.values().all(|value| value.is_zero()));
        result.verify().unwrap();
        assert!(!result.canonical_bytes().unwrap().is_empty());
    }

    #[test]
    fn normalized_integral_bar_stays_sparse_above_the_legacy_dense_ceiling() {
        let group = symmetric_three();
        let (_, multiplication) = normalized_alphabet(&group).unwrap();
        let alphabet_size = multiplication.len();
        let fifth = normalized_integral_boundary(alphabet_size, &multiplication, 5);
        let sixth = normalized_integral_boundary(alphabet_size, &multiplication, 6);

        assert_eq!(sixth.shape(), (3_125, 15_625));
        assert!(sixth.shape().0 * sixth.shape().1 > 25_000_000);
        assert!(sixth.nnz() <= 7 * sixth.shape().1);
        assert_eq!(fifth.matmul(&sixth).unwrap().nnz(), 0);
    }

    #[test]
    fn normalized_modular_bar_stays_sparse_above_the_legacy_dense_ceiling() {
        let group = symmetric_three();
        let (_, multiplication) = normalized_alphabet(&group).unwrap();
        let alphabet_size = multiplication.len();
        let field = FiniteField::prime(2_u8).unwrap();
        let fifth = normalized_boundary(&field, alphabet_size, &multiplication, 5);
        let sixth = normalized_boundary(&field, alphabet_size, &multiplication, 6);

        assert_eq!(sixth.shape(), (3_125, 15_625));
        assert!(sixth.shape().0 * sixth.shape().1 > MODULAR_BAR_ORACLE_WORK_LIMIT);
        assert!(sixth.nnz() <= 7 * sixth.shape().1);
        assert_eq!(fifth.matmul(&sixth).unwrap().nnz(), 0);
    }

    #[test]
    fn integral_bar_rejects_oversized_sparse_basis_before_construction() {
        let error = elementary_abelian_two(5)
            .integral_cohomology(4)
            .unwrap_err();
        assert_eq!(
            error,
            FiniteGroupIntegralCohomologyError::BarComplexTooLarge {
                degree: 5,
                rows: 923_521,
                columns: 28_629_151,
                estimated_entries: 171_774_906,
                entry_limit: NORMALIZED_BAR_SPARSE_ENTRY_LIMIT,
            }
        );
    }

    #[test]
    fn cyclic_two_mod_two_is_polynomial_on_degree_one() {
        let result = cyclic(2).modular_cohomology(2, 4).unwrap();
        assert_eq!(
            result.resolution_strategy(),
            FiniteGroupResolutionStrategy::MinimalPGroup
        );
        assert_eq!(
            (0..=4)
                .map(|degree| result.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1]
        );
        let square = result.basis_cup_product(1, 0, 1, 0).unwrap();
        assert_eq!(square.dimension(), 1);
        assert!(!square.get(0).unwrap().is_zero());
        result.verify().unwrap();
    }

    #[test]
    fn normalized_group_nerve_supports_lower_steenrod_operations() {
        let result = symmetric_three().modular_cohomology(2, 3).unwrap();
        assert_eq!(
            result.resolution_strategy(),
            FiniteGroupResolutionStrategy::NormalizedBar
        );
        let square = result
            .apply_operation(&AdmissibleCohomologyOperation::steenrod_square(1), 2, 0)
            .unwrap();
        assert!(square.values().all(|coordinate| coordinate.is_zero()));
    }

    #[test]
    fn generators_record_operation_names_modulo_decomposables() {
        let result = symmetric_three().modular_cohomology(3, 8).unwrap();
        assert_eq!(
            result.resolution_strategy(),
            FiniteGroupResolutionStrategy::SylowStableElements
        );
        let algebra = result.algebra_truncation().unwrap();
        let degree_four = algebra
            .generators()
            .iter()
            .find(|generator| generator.degree() == 4)
            .expect("S3 mod 3 has a degree-four algebra generator");
        let witness = degree_four
            .operation_witness()
            .expect("the degree-four generator is the Bockstein of degree three");
        assert_eq!(witness.operation().display_name(), "beta");
        assert_eq!(
            algebra.generators()[witness.source_generator_index()].degree(),
            3
        );
        assert!(degree_four.display_name().starts_with("beta("));
        algebra.verify().unwrap();
    }

    #[test]
    fn cyclic_three_mod_three_has_exterior_degree_one_class() {
        let result = cyclic(3).modular_cohomology(3, 4).unwrap();
        assert_eq!(
            (0..=4)
                .map(|degree| result.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1]
        );
        assert!(result
            .basis_cup_product(1, 0, 1, 0)
            .unwrap()
            .get(0)
            .unwrap()
            .is_zero());
        assert!(!result
            .basis_cup_product(1, 0, 2, 0)
            .unwrap()
            .get(0)
            .unwrap()
            .is_zero());
    }

    #[test]
    fn algebra_truncation_finds_minimal_generators_and_visible_relations() {
        let cyclic_two = cyclic(2).modular_cohomology(2, 4).unwrap();
        let polynomial = cyclic_two.algebra_truncation().unwrap();
        assert_eq!(
            polynomial
                .generators()
                .iter()
                .map(FiniteGroupCohomologyAlgebraGenerator::degree)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert!(polynomial.presentation().relations().is_empty());
        assert_eq!(
            polynomial.presentation().hilbert_function(4).unwrap(),
            vec![1, 1, 1, 1, 1]
        );
        polynomial.verify().unwrap();

        let cyclic_four = cyclic(4).modular_cohomology(2, 4).unwrap();
        let truncated = cyclic_four.algebra_truncation().unwrap();
        assert_eq!(
            truncated
                .generators()
                .iter()
                .map(|generator| (generator.degree(), generator.basis_index()))
                .collect::<Vec<_>>(),
            vec![(1, 0), (2, 0)]
        );
        assert_eq!(truncated.presentation().relations().len(), 1);
        assert_eq!(
            truncated.presentation().relations()[0]
                .homogeneous_degree()
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            truncated.presentation().hilbert_function(4).unwrap(),
            vec![1, 1, 1, 1, 1]
        );
        truncated.verify().unwrap();
    }

    #[test]
    fn symonds_universal_bound_promotes_only_sufficient_exact_ranges() {
        let cyclic_two = cyclic(2)
            .modular_cohomology(2, 2)
            .unwrap()
            .algebra_truncation()
            .unwrap();
        let complete = cyclic_two.certify_symonds_universal_bound().unwrap();
        assert_eq!(
            complete.completion_strategy(),
            FiniteGroupCohomologyCompletionStrategy::SymondsUniversalBound
        );
        assert_eq!(complete.group_order(), &Integer::from(2));
        assert_eq!(complete.generator_degree_bound(), 1);
        assert_eq!(complete.relation_degree_bound(), 2);
        complete.verify().unwrap();

        let cyclic_three = cyclic(3)
            .modular_cohomology(3, 4)
            .unwrap()
            .algebra_truncation()
            .unwrap();
        let complete = cyclic_three.certify_symonds_universal_bound().unwrap();
        assert_eq!(complete.group_order(), &Integer::from(3));
        assert_eq!(complete.generator_degree_bound(), 2);
        assert_eq!(complete.relation_degree_bound(), 4);
        complete.verify().unwrap();

        let cyclic_four = cyclic(4)
            .modular_cohomology(2, 4)
            .unwrap()
            .algebra_truncation()
            .unwrap();
        assert_eq!(
            cyclic_four.certify_symonds_universal_bound().unwrap_err(),
            FiniteGroupCohomologyError::CompletionBoundNotReached {
                computed_through_degree: 4,
                required_through_degree: 6,
            }
        );
    }

    #[test]
    fn symonds_parameter_criterion_certifies_both_symmetric_three_primes() {
        for prime in [2_u64, 3_u64] {
            let algebra = symmetric_three()
                .modular_cohomology(prime, 5)
                .unwrap()
                .algebra_truncation()
                .unwrap();
            let complete = algebra
                .certify_symonds_parameter_criterion()
                .unwrap_or_else(|error| panic!("prime {prime}: {error:?}"));
            assert_eq!(
                complete.completion_strategy(),
                FiniteGroupCohomologyCompletionStrategy::SymondsParameterCriterion
            );
            let certificate = complete.symonds_parameter_certificate().unwrap();
            assert!(!certificate.parameter_generator_indices().is_empty());
            assert_eq!(certificate.elementary_abelian_restrictions().len(), 1);
            assert!(certificate.completion_degree_bound() <= 5);
            complete.verify().unwrap();
        }
    }

    #[test]
    fn symonds_parameter_criterion_certifies_alternating_four_mod_two() {
        let complete = alternating_four()
            .modular_cohomology(2, 8)
            .unwrap()
            .algebra_truncation()
            .unwrap()
            .certify_symonds_parameter_criterion()
            .unwrap();
        let certificate = complete.symonds_parameter_certificate().unwrap();
        assert_eq!(certificate.elementary_abelian_restrictions().len(), 1);
        assert_eq!(certificate.elementary_abelian_restrictions()[0].rank(), 2);
        assert!(certificate.completion_degree_bound() <= 8);
        complete.verify().unwrap();
    }

    #[test]
    fn symonds_parameter_criterion_checks_every_dihedral_maximal_elementary_subgroup() {
        let complete = dihedral_eight()
            .modular_cohomology(2, 8)
            .unwrap()
            .algebra_truncation()
            .unwrap()
            .certify_symonds_parameter_criterion()
            .unwrap();
        let certificate = complete.symonds_parameter_certificate().unwrap();
        assert_eq!(certificate.elementary_abelian_restrictions().len(), 2);
        assert!(certificate
            .elementary_abelian_restrictions()
            .iter()
            .all(|restriction| restriction.rank() == 2));
        complete.verify().unwrap();
    }

    #[test]
    fn characteristic_away_from_order_has_no_positive_cohomology() {
        let result = cyclic(2).modular_cohomology(3, 3).unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| result.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 0, 0, 0]
        );
    }

    #[test]
    fn symmetric_three_matches_stable_element_dimensions() {
        let group = symmetric_three();
        let mod_two = group.modular_cohomology(2, 3).unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| mod_two.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1]
        );
        let mod_three = group.modular_cohomology(3, 3).unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| mod_three.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 0, 0, 1]
        );
    }

    #[test]
    fn stable_elements_scale_past_the_bar_and_recover_standard_examples() {
        let symmetric = symmetric_three().modular_cohomology(2, 5).unwrap();
        assert_eq!(
            symmetric.resolution_strategy(),
            FiniteGroupResolutionStrategy::SylowStableElements
        );
        assert_eq!(
            (0..=5)
                .map(|degree| symmetric.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1, 1]
        );
        assert!(!symmetric
            .basis_cup_product(2, 0, 3, 0)
            .unwrap()
            .get(0)
            .unwrap()
            .is_zero());

        let alternating = alternating_four().modular_cohomology(2, 4).unwrap();
        assert_eq!(
            alternating.resolution_strategy(),
            FiniteGroupResolutionStrategy::SylowStableElements
        );
        assert_eq!(
            (0..=4)
                .map(|degree| alternating.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 0, 1, 2, 1]
        );
        assert!(!alternating
            .basis_cup_product(2, 0, 2, 0)
            .unwrap()
            .get(0)
            .unwrap()
            .is_zero());
        let algebra = alternating.algebra_truncation().unwrap();
        assert_eq!(
            algebra
                .generators()
                .iter()
                .map(FiniteGroupCohomologyAlgebraGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3, 3]
        );
        assert!(algebra.presentation().relations().is_empty());
        assert_eq!(
            algebra.presentation().hilbert_function(4).unwrap(),
            vec![1, 0, 1, 2, 1]
        );
        alternating.verify().unwrap();

        let symmetric_four = symmetric_four().modular_cohomology(2, 5).unwrap();
        assert_eq!(
            (0..=5)
                .map(|degree| symmetric_four.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 2, 3, 3, 4]
        );
    }

    #[test]
    fn elementary_abelian_minimal_resolutions_recover_rank_two_products() {
        let mod_two = elementary_abelian_rank_two(2)
            .modular_cohomology(2, 3)
            .unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| mod_two.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let degree_two_products = (0..2)
            .flat_map(|left| {
                (0..2).map({
                    let value = mod_two.clone();
                    move |right| value.basis_cup_product(1, left, 1, right).unwrap()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matrix_from_columns(mod_two.field(), 3, &degree_two_products)
                .unwrap()
                .rank()
                .unwrap(),
            3
        );
        let left_multiplication = mod_two.basis_left_multiplication(1, 0, 1).unwrap();
        assert_eq!(left_multiplication.shape(), (3, 2));
        for right in 0..2 {
            let scalar = mod_two.basis_cup_product(1, 0, 1, right).unwrap();
            assert!((0..3).all(|row| {
                left_multiplication.get(row, right).unwrap() == scalar.get(row).unwrap()
            }));
        }

        let mod_three = elementary_abelian_rank_two(3)
            .modular_cohomology(3, 3)
            .unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| mod_three.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        for index in 0..2 {
            assert!(mod_three
                .basis_cup_product(1, index, 1, index)
                .unwrap()
                .values()
                .all(|value| value.is_zero()));
        }
        assert!(mod_three
            .basis_cup_product(1, 0, 1, 1)
            .unwrap()
            .values()
            .any(|value| !value.is_zero()));
    }

    #[test]
    #[ignore = "rank-four scaling is exercised by the finite_group_cohomology benchmark"]
    fn rank_four_elementary_abelian_resolution_has_expected_ranks() {
        let result = elementary_abelian_two(4).modular_cohomology(2, 4).unwrap();
        assert_eq!(
            (0..=4)
                .map(|degree| result.dimension(degree).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 4, 10, 20, 35]
        );
        let algebra = result.algebra_truncation().unwrap();
        assert_eq!(algebra.generators().len(), 4);
        assert!(algebra
            .generators()
            .iter()
            .all(|generator| generator.degree() == 1));
        assert!(algebra.presentation().relations().is_empty());
        assert_eq!(
            algebra.presentation().hilbert_function(4).unwrap(),
            vec![1, 4, 10, 20, 35]
        );
    }

    #[test]
    fn rejects_nonprime_characteristic_and_out_of_range_queries() {
        let group = cyclic(2);
        assert_eq!(
            group.modular_cohomology(4, 2).unwrap_err(),
            FiniteGroupCohomologyError::InvalidPrime { prime: 4 }
        );
        let result = group.modular_cohomology(2, 2).unwrap();
        assert!(matches!(
            result.dimension(3),
            Err(FiniteGroupCohomologyError::DegreeOutOfRange { .. })
        ));
        assert!(matches!(
            result.basis_cup_product(1, 9, 1, 0),
            Err(FiniteGroupCohomologyError::BasisIndexOutOfRange { .. })
        ));
    }
}
