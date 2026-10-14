// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Symbolic Eilenberg--Mac Lane spaces and exact bounded invariants.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    GradedCommutativeError, GradedCommutativePolynomial, GradedCommutativePolynomialRing,
    GradedCommutativePresentation, GradedGenerator,
};
use crate::homalg::{AbelianGroupError, FinitelyGeneratedAbelianGroup};
use crate::numthy::FiniteField;
use crate::{
    EuclideanDomain, Integer, NoetherError, Ring, SemanticType, TypeDescription,
    UnsupportedOperationError, VerificationError, Z,
};

use super::AdmissibleCohomologyOperation;

static SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneSpace@1",
    rust_path: "noether::topology::EilenbergMacLaneSpace",
    python_path: Some("noether.topology.EilenbergMacLaneSpace"),
};

static INTEGRAL_HOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneIntegralHomology@1",
    rust_path: "noether::topology::EilenbergMacLaneIntegralHomology",
    python_path: Some("noether.topology.EilenbergMacLaneIntegralHomology"),
};

static INTEGRAL_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneIntegralCohomology@1",
    rust_path: "noether::topology::EilenbergMacLaneIntegralCohomology",
    python_path: Some("noether.topology.EilenbergMacLaneIntegralCohomology"),
};

static INTEGRAL_COHOMOLOGY_PRESENTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneIntegralCohomologyPresentation@1",
    rust_path: "noether::topology::EilenbergMacLaneIntegralCohomologyPresentation",
    python_path: Some("noether.topology.EilenbergMacLaneIntegralCohomologyPresentation"),
};

static MODULAR_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneModularCohomology@1",
    rust_path: "noether::topology::EilenbergMacLaneModularCohomology",
    python_path: Some("noether.topology.EilenbergMacLaneModularCohomology"),
};

static MODULAR_COHOMOLOGY_TRUNCATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/EilenbergMacLaneModularCohomologyTruncation@1",
    rust_path: "noether::topology::EilenbergMacLaneModularCohomologyTruncation",
    python_path: Some("noether.topology.EilenbergMacLaneModularCohomologyTruncation"),
};

const INTEGRAL_HOMOLOGY_OPERATION: &str =
    "noether/topology/EilenbergMacLaneSpace/integral_homology@1";
const INTEGRAL_COHOMOLOGY_PRESENTATION_OPERATION: &str =
    "noether/topology/EilenbergMacLaneSpace/integral_cohomology_presentation@1";

/// Failure to construct an Eilenberg--Mac Lane space or compute an invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EilenbergMacLaneError {
    /// `K(A,n)` requires a positive delooping degree.
    ZeroDeloopingDegree,
    /// An inclusive degree range cannot be represented as a vector length.
    DegreeBoundTooLarge {
        /// Rejected inclusive bound.
        max_degree: usize,
    },
    /// A prime characteristic cannot be represented by graded monomial exponents.
    CharacteristicTooLarge {
        /// Rejected prime characteristic.
        characteristic: Integer,
    },
    /// Canonical abelian-group arithmetic failed.
    AbelianGroup(AbelianGroupError),
    /// Exact graded-commutative presentation construction failed.
    GradedCommutative(GradedCommutativeError),
    /// The exact native playbook does not yet cover this input regime.
    UnsupportedOperation(UnsupportedOperationError),
    /// A computed result failed exact deterministic replay.
    Verification(VerificationError),
}

impl fmt::Display for EilenbergMacLaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDeloopingDegree => {
                f.write_str("an Eilenberg--Mac Lane delooping degree must be positive")
            }
            Self::DegreeBoundTooLarge { max_degree } => write!(
                f,
                "inclusive Eilenberg--Mac Lane degree bound {max_degree} exceeds supported indexing"
            ),
            Self::CharacteristicTooLarge { characteristic } => write!(
                f,
                "finite-field characteristic {characteristic} exceeds graded exponent storage"
            ),
            Self::AbelianGroup(error) => error.fmt(f),
            Self::GradedCommutative(error) => error.fmt(f),
            Self::UnsupportedOperation(error) => error.fmt(f),
            Self::Verification(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for EilenbergMacLaneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ZeroDeloopingDegree
            | Self::DegreeBoundTooLarge { .. }
            | Self::CharacteristicTooLarge { .. } => None,
            Self::AbelianGroup(error) => Some(error),
            Self::GradedCommutative(error) => Some(error),
            Self::UnsupportedOperation(error) => Some(error),
            Self::Verification(error) => Some(error),
        }
    }
}

impl From<AbelianGroupError> for EilenbergMacLaneError {
    fn from(value: AbelianGroupError) -> Self {
        Self::AbelianGroup(value)
    }
}

impl From<GradedCommutativeError> for EilenbergMacLaneError {
    fn from(value: GradedCommutativeError) -> Self {
        Self::GradedCommutative(value)
    }
}

impl From<UnsupportedOperationError> for EilenbergMacLaneError {
    fn from(value: UnsupportedOperationError) -> Self {
        Self::UnsupportedOperation(value)
    }
}

impl From<VerificationError> for EilenbergMacLaneError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

/// A symbolic Eilenberg--Mac Lane space `K(A,n)`.
///
/// This is intentionally not represented by a finite simplicial set: a
/// nontrivial Eilenberg--Mac Lane space generally has infinitely many cells.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneSpace {
    group: FinitelyGeneratedAbelianGroup,
    delooping_degree: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpaceWire {
    #[serde(rename = "type")]
    type_id: String,
    group: FinitelyGeneratedAbelianGroup,
    delooping_degree: usize,
}

impl EilenbergMacLaneSpace {
    /// Construct `K(group, delooping_degree)`.
    pub fn new(
        group: FinitelyGeneratedAbelianGroup,
        delooping_degree: usize,
    ) -> Result<Self, EilenbergMacLaneError> {
        if delooping_degree == 0 {
            return Err(EilenbergMacLaneError::ZeroDeloopingDegree);
        }
        group.verify()?;
        Ok(Self {
            group,
            delooping_degree,
        })
    }

    /// The sole nonzero homotopy group.
    pub fn group(&self) -> &FinitelyGeneratedAbelianGroup {
        &self.group
    }

    /// Degree containing the sole nonzero homotopy group.
    pub fn delooping_degree(&self) -> usize {
        self.delooping_degree
    }

    /// Compute integral homology exactly through `max_degree`.
    ///
    /// The native first edition covers `K(A,1)` for every finitely generated
    /// abelian group, `K(Z^r,2)`, and the point `K(0,n)` in every positive
    /// degree. Other regimes return a typed unsupported-operation error.
    pub fn integral_homology(
        &self,
        max_degree: usize,
    ) -> Result<EilenbergMacLaneIntegralHomology, EilenbergMacLaneError> {
        if max_degree == usize::MAX {
            return Err(EilenbergMacLaneError::DegreeBoundTooLarge { max_degree });
        }
        let (groups, strategy) = compute_integral_homology(self, max_degree)?;
        let result = EilenbergMacLaneIntegralHomology {
            space: self.clone(),
            max_degree,
            groups,
            strategy,
        };
        result.verify()?;
        Ok(result)
    }

    /// Compute additive integral cohomology exactly through `max_degree`.
    ///
    /// This applies the integral universal coefficient theorem to the exact
    /// homology result retained inside the returned evidence. It computes
    /// additive groups only; it makes no cup-product or ring-presentation
    /// claim.
    pub fn integral_cohomology(
        &self,
        max_degree: usize,
    ) -> Result<EilenbergMacLaneIntegralCohomology, EilenbergMacLaneError> {
        let homology = self.integral_homology(max_degree)?;
        let groups = universal_coefficient_groups(homology.groups())?;
        let result = EilenbergMacLaneIntegralCohomology { homology, groups };
        result.verify()?;
        Ok(result)
    }

    /// Construct a complete integral cohomology-ring presentation when a
    /// closed-form native playbook is available.
    ///
    /// The presentation is a quotient of the free graded-commutative
    /// `Z`-algebra on its ordered generators. The current exact lanes are a
    /// point, `K(Z^r,1)`, `K(Z^r + C_m,1)`,
    /// `K(Z^r + C_m + C_n,1)` for `m | n`, and `K(Z^r,2)`. In particular,
    /// square-zero relations for the odd torus classes are explicit: graded
    /// commutativity over `Z` alone would imply only that twice an odd square
    /// vanishes. The rank-two torsion formula is Theorem 6.3.1 of C. D.
    /// Roberts, *The cohomology ring of a finite abelian group* (2013).
    pub fn integral_cohomology_presentation(
        &self,
    ) -> Result<EilenbergMacLaneIntegralCohomologyPresentation, EilenbergMacLaneError> {
        let result = compute_integral_cohomology_presentation(self)?;
        result.verify()?;
        Ok(result)
    }

    /// Construct a complete prime-field cohomology presentation when a
    /// closed-form native playbook is available.
    ///
    /// The current exact lanes are `K(A,1)`, `K(Z^r,2)`, and `K(0,n)`. The
    /// result includes fundamental/derived generator names and the primary
    /// operations on those generators.
    pub fn modular_cohomology(
        &self,
        field: FiniteField,
    ) -> Result<EilenbergMacLaneModularCohomology, EilenbergMacLaneError> {
        let result = compute_modular_cohomology(self, field)?;
        result.verify()?;
        Ok(result)
    }

    /// Compute a prime-field cohomology presentation exactly through an
    /// inclusive degree bound when a bounded native playbook is available.
    ///
    /// The bounded Cartan--Serre lanes enumerate admissible squares over
    /// `F_2` and admissible Bockstein/reduced-power words at odd primes for
    /// `n >= 2`, free abelian summands, and primary cyclic torsion. The
    /// finite presentation contains every algebra generator that can affect
    /// the requested range and makes no assertion above it.
    pub fn modular_cohomology_through_degree(
        &self,
        field: FiniteField,
        max_degree: usize,
    ) -> Result<EilenbergMacLaneModularCohomologyTruncation, EilenbergMacLaneError> {
        let result = compute_modular_cohomology_truncation(self, field, max_degree)?;
        result.verify()?;
        Ok(result)
    }

    /// Recheck the positive delooping degree and canonical group.
    pub fn verify(&self) -> Result<(), EilenbergMacLaneError> {
        if self.delooping_degree == 0 {
            return Err(EilenbergMacLaneError::ZeroDeloopingDegree);
        }
        self.group.verify()?;
        Ok(())
    }

    /// Serialize the symbolic space as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reverify a symbolic space.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EilenbergMacLaneSpace")
            .field("group", &self.group)
            .field("delooping_degree", &self.delooping_degree)
            .finish()
    }
}

impl fmt::Display for EilenbergMacLaneSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "K({}, {})", self.group, self.delooping_degree)
    }
}

impl SemanticType for EilenbergMacLaneSpace {
    const TYPE_ID: &'static str = "noether/topology/EilenbergMacLaneSpace@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.group.canonical_bytes()?;
        output.extend_from_slice(&(self.delooping_degree as u64).to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SPACE_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneSpace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SpaceWire {
            type_id: Self::TYPE_ID.to_owned(),
            group: self.group.clone(),
            delooping_degree: self.delooping_degree,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneSpace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpaceWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.group, wire.delooping_degree).map_err(serde::de::Error::custom)
    }
}

/// Exact mathematical route used for a bounded integral-homology result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EilenbergMacLaneHomologyStrategy {
    /// `K(0,n)` is a point.
    TrivialSpace,
    /// Product decomposition of `K(A,1)` and the integral Künneth theorem.
    ClassifyingSpaceKunneth,
    /// Product decomposition `K(Z^r,2) = (CP^infinity)^r` and Künneth.
    FreeAbelianSecondDelooping,
}

/// Integral homology of one symbolic Eilenberg--Mac Lane space through a bound.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneIntegralHomology {
    space: EilenbergMacLaneSpace,
    max_degree: usize,
    groups: Vec<FinitelyGeneratedAbelianGroup>,
    strategy: EilenbergMacLaneHomologyStrategy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegralHomologyWire {
    #[serde(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpace,
    max_degree: usize,
    groups: Vec<FinitelyGeneratedAbelianGroup>,
    strategy: EilenbergMacLaneHomologyStrategy,
}

impl EilenbergMacLaneIntegralHomology {
    /// Symbolic source space.
    pub fn space(&self) -> &EilenbergMacLaneSpace {
        &self.space
    }

    /// Largest degree included, inclusively.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Homology groups indexed by degree `0, ..., max_degree`.
    pub fn groups(&self) -> &[FinitelyGeneratedAbelianGroup] {
        &self.groups
    }

    /// Homology in one computed degree.
    pub fn group(&self, degree: usize) -> Option<&FinitelyGeneratedAbelianGroup> {
        self.groups.get(degree)
    }

    /// Exact mathematical route used by this result.
    pub fn strategy(&self) -> EilenbergMacLaneHomologyStrategy {
        self.strategy
    }

    /// Deterministically replay the exact formula through the recorded bound.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.groups.len() != self.max_degree.saturating_add(1)
            || self.groups.iter().any(|group| group.verify().is_err())
        {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane homology contains one canonical group in every recorded degree",
            });
        }
        let (groups, strategy) =
            compute_integral_homology(&self.space, self.max_degree).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim:
                        "Eilenberg--Mac Lane homology strategy remains supported by exact replay",
                }
            })?;
        if groups != self.groups || strategy != self.strategy {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane homology agrees with exact deterministic replay",
            });
        }
        Ok(())
    }

    /// Serialize the groups and exact strategy as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay a complete bounded result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneIntegralHomology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EilenbergMacLaneIntegralHomology")
            .field("space", &self.space)
            .field("max_degree", &self.max_degree)
            .field("groups", &self.groups)
            .field("strategy", &self.strategy)
            .finish()
    }
}

impl SemanticType for EilenbergMacLaneIntegralHomology {
    const TYPE_ID: &'static str = "noether/topology/EilenbergMacLaneIntegralHomology@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.space.canonical_bytes()?;
        output.extend_from_slice(&(self.max_degree as u64).to_be_bytes());
        output.push(match self.strategy {
            EilenbergMacLaneHomologyStrategy::TrivialSpace => 0,
            EilenbergMacLaneHomologyStrategy::ClassifyingSpaceKunneth => 1,
            EilenbergMacLaneHomologyStrategy::FreeAbelianSecondDelooping => 2,
        });
        output.extend_from_slice(&(self.groups.len() as u64).to_be_bytes());
        for group in &self.groups {
            let bytes = group.canonical_bytes()?;
            output.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
            output.extend_from_slice(&bytes);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_HOMOLOGY_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneIntegralHomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegralHomologyWire {
            type_id: Self::TYPE_ID.to_owned(),
            space: self.space.clone(),
            max_degree: self.max_degree,
            groups: self.groups.clone(),
            strategy: self.strategy,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneIntegralHomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegralHomologyWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            space: wire.space,
            max_degree: wire.max_degree,
            groups: wire.groups,
            strategy: wire.strategy,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

/// Additive integral cohomology obtained from exact homology by universal
/// coefficients.
///
/// The retained homology value is the evidence for the computation. This type
/// deliberately does not claim a cup product or ring presentation.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneIntegralCohomology {
    homology: EilenbergMacLaneIntegralHomology,
    groups: Vec<FinitelyGeneratedAbelianGroup>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegralCohomologyWire {
    #[serde(rename = "type")]
    type_id: String,
    homology: EilenbergMacLaneIntegralHomology,
    groups: Vec<FinitelyGeneratedAbelianGroup>,
}

impl EilenbergMacLaneIntegralCohomology {
    /// Symbolic source space.
    pub fn space(&self) -> &EilenbergMacLaneSpace {
        self.homology.space()
    }

    /// Inclusive exact degree bound.
    pub fn max_degree(&self) -> usize {
        self.homology.max_degree()
    }

    /// Exact integral homology retained as universal-coefficient evidence.
    pub fn homology(&self) -> &EilenbergMacLaneIntegralHomology {
        &self.homology
    }

    /// Additive cohomology groups indexed by degree.
    pub fn groups(&self) -> &[FinitelyGeneratedAbelianGroup] {
        &self.groups
    }

    /// Additive cohomology in one computed degree.
    pub fn group(&self, degree: usize) -> Option<&FinitelyGeneratedAbelianGroup> {
        self.groups.get(degree)
    }

    /// Replay homology and the integral universal coefficient calculation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.homology.verify()?;
        let groups = universal_coefficient_groups(self.homology.groups()).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane integral cohomology universal-coefficient sum is representable",
            }
        })?;
        if groups != self.groups {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane integral cohomology agrees with universal coefficients",
            });
        }
        Ok(())
    }

    /// Serialize the additive groups and retained homology evidence as typed
    /// JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay an additive integral-cohomology result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneIntegralCohomology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EilenbergMacLaneIntegralCohomology")
            .field("homology", &self.homology)
            .field("groups", &self.groups)
            .finish()
    }
}

impl SemanticType for EilenbergMacLaneIntegralCohomology {
    const TYPE_ID: &'static str = "noether/topology/EilenbergMacLaneIntegralCohomology@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.homology.canonical_bytes()?;
        output.extend_from_slice(&(self.groups.len() as u64).to_be_bytes());
        for group in &self.groups {
            let bytes = group.canonical_bytes()?;
            output.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
            output.extend_from_slice(&bytes);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_COHOMOLOGY_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneIntegralCohomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegralCohomologyWire {
            type_id: Self::TYPE_ID.to_owned(),
            homology: self.homology.clone(),
            groups: self.groups.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneIntegralCohomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegralCohomologyWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            homology: wire.homology,
            groups: wire.groups,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

fn universal_coefficient_groups(
    homology: &[FinitelyGeneratedAbelianGroup],
) -> Result<Vec<FinitelyGeneratedAbelianGroup>, AbelianGroupError> {
    let mut groups = Vec::with_capacity(homology.len());
    for degree in 0..homology.len() {
        let hom = homology[degree].hom_to_integers();
        let ext = if degree == 0 {
            FinitelyGeneratedAbelianGroup::trivial()
        } else {
            homology[degree - 1].ext_1_to_integers()
        };
        groups.push(hom.direct_sum(&ext)?);
    }
    Ok(groups)
}

/// Closed-form strategy certifying a complete integral presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EilenbergMacLaneIntegralCohomologyPresentationStrategy {
    /// `K(0,n)` is a point and has coefficient ring `Z`.
    TrivialSpace,
    /// Exterior torus classes tensored with one cyclic periodic class.
    FreeAndCyclicClassifyingSpace,
    /// Exterior torus classes tensored with the rank-two finite-abelian ring.
    FreeAndRankTwoAbelianClassifyingSpace,
    /// Polynomial cohomology of a product of infinite projective spaces.
    FreeAbelianSecondDelooping,
}

/// Mathematical origin of one integral cohomology generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EilenbergMacLaneIntegralGeneratorOrigin {
    /// A degree-one coordinate of `H^1(K(Z^r,1); Z)`.
    FreeFundamentalClass {
        /// One-based free coordinate.
        component: usize,
    },
    /// Integral Bockstein of a tautological cyclic degree-one class.
    IntegralBockstein {
        /// One-based coordinate after the free coordinates.
        component: usize,
        /// Order of the cyclic coefficient group.
        modulus: Integer,
    },
    /// The integral Tor class coupling two cyclic degree-one coordinates.
    CyclicTorClass {
        /// One-based coordinate of the first cyclic summand.
        left_component: usize,
        /// One-based coordinate of the second cyclic summand.
        right_component: usize,
        /// Order of the Tor class, the gcd of the two cyclic orders.
        order: Integer,
    },
    /// A degree-two coordinate of `H^2(K(Z^r,2); Z)`.
    SecondDeloopingFundamentalClass {
        /// One-based free coordinate.
        component: usize,
    },
}

/// One ordered generator in a complete integral cohomology presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EilenbergMacLaneIntegralCohomologyGenerator {
    id: String,
    degree: u32,
    display_name: String,
    origin: EilenbergMacLaneIntegralGeneratorOrigin,
}

impl EilenbergMacLaneIntegralCohomologyGenerator {
    /// Stable algebra-generator id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Cohomological degree.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// Reader-facing mathematical name.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Evidence-backed origin of the generator.
    pub fn origin(&self) -> &EilenbergMacLaneIntegralGeneratorOrigin {
        &self.origin
    }
}

/// One integer coefficient times one monomial in a defining relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EilenbergMacLaneIntegralCohomologyRelationTerm {
    coefficient: Integer,
    exponents: Vec<u32>,
}

impl EilenbergMacLaneIntegralCohomologyRelationTerm {
    /// Integer coefficient.
    pub fn coefficient(&self) -> &Integer {
        &self.coefficient
    }

    /// Exponents in the presentation's generator order.
    pub fn exponents(&self) -> &[u32] {
        &self.exponents
    }
}

/// One homogeneous defining relation over `Z`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EilenbergMacLaneIntegralCohomologyRelation {
    terms: Vec<EilenbergMacLaneIntegralCohomologyRelationTerm>,
}

impl EilenbergMacLaneIntegralCohomologyRelation {
    /// Sparse terms in canonical monomial order.
    pub fn terms(&self) -> &[EilenbergMacLaneIntegralCohomologyRelationTerm] {
        &self.terms
    }
}

/// A complete integral cohomology-ring presentation for a symbolic `K(A,n)`.
///
/// This denotes the quotient of the free graded-commutative `Z`-algebra on
/// `generators` by the homogeneous integer-polynomial `relations`. Generator
/// order fixes every exponent vector and is part of the semantic value.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneIntegralCohomologyPresentation {
    space: EilenbergMacLaneSpace,
    generators: Vec<EilenbergMacLaneIntegralCohomologyGenerator>,
    relations: Vec<EilenbergMacLaneIntegralCohomologyRelation>,
    strategy: EilenbergMacLaneIntegralCohomologyPresentationStrategy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegralCohomologyPresentationWire {
    #[serde(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpace,
    coefficient_ring: String,
    graded_commutative: bool,
    generators: Vec<EilenbergMacLaneIntegralCohomologyGenerator>,
    relations: Vec<EilenbergMacLaneIntegralCohomologyRelation>,
    strategy: EilenbergMacLaneIntegralCohomologyPresentationStrategy,
}

impl EilenbergMacLaneIntegralCohomologyPresentation {
    /// Symbolic source space.
    pub fn space(&self) -> &EilenbergMacLaneSpace {
        &self.space
    }

    /// Coefficient-ring display name.
    pub fn coefficient_ring(&self) -> &'static str {
        "Z"
    }

    /// Ordered homogeneous algebra generators.
    pub fn generators(&self) -> &[EilenbergMacLaneIntegralCohomologyGenerator] {
        &self.generators
    }

    /// Homogeneous defining relations.
    pub fn relations(&self) -> &[EilenbergMacLaneIntegralCohomologyRelation] {
        &self.relations
    }

    /// Closed-form completeness strategy.
    pub fn strategy(&self) -> EilenbergMacLaneIntegralCohomologyPresentationStrategy {
        self.strategy
    }

    /// Additive integral cohomology in one degree, derived from the same
    /// closed-form family and checked by the bounded homology/UCT lane.
    pub fn additive_group(
        &self,
        degree: usize,
    ) -> Result<FinitelyGeneratedAbelianGroup, EilenbergMacLaneError> {
        let result = self.space.integral_cohomology(degree)?;
        result
            .group(degree)
            .cloned()
            .ok_or(EilenbergMacLaneError::DegreeBoundTooLarge { max_degree: degree })
    }

    /// Reconstruct the entire presentation from the retained symbolic space.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let expected = compute_integral_cohomology_presentation(&self.space).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane integral presentation remains exactly reconstructible",
            }
        })?;
        if &expected != self {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane integral generators and relations agree with closed-form replay",
            });
        }
        Ok(())
    }

    /// Serialize the complete integral presentation as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly replay a complete integral presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneIntegralCohomologyPresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EilenbergMacLaneIntegralCohomologyPresentation")
            .field("space", &self.space)
            .field("generators", &self.generators)
            .field("relations", &self.relations)
            .field("strategy", &self.strategy)
            .finish()
    }
}

impl SemanticType for EilenbergMacLaneIntegralCohomologyPresentation {
    const TYPE_ID: &'static str =
        "noether/topology/EilenbergMacLaneIntegralCohomologyPresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_COHOMOLOGY_PRESENTATION_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneIntegralCohomologyPresentation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegralCohomologyPresentationWire {
            type_id: Self::TYPE_ID.to_owned(),
            space: self.space.clone(),
            coefficient_ring: self.coefficient_ring().to_owned(),
            graded_commutative: true,
            generators: self.generators.clone(),
            relations: self.relations.clone(),
            strategy: self.strategy,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneIntegralCohomologyPresentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegralCohomologyPresentationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire.coefficient_ring != "Z" || !wire.graded_commutative {
            return Err(serde::de::Error::custom(
                "integral cohomology presentation must be a graded-commutative Z-algebra",
            ));
        }
        let result = Self {
            space: wire.space,
            generators: wire.generators,
            relations: wire.relations,
            strategy: wire.strategy,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

/// Mathematical origin of a selected modular cohomology generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EilenbergMacLaneGeneratorOrigin {
    /// A coordinate of the fundamental class in `H^n(K(A,n); F_p)`.
    FundamentalClass {
        /// One-based coordinate among the visible fundamental classes.
        component: usize,
    },
    /// The primary Bockstein of an earlier generator.
    Bockstein {
        /// Zero-based source generator index.
        source_generator: usize,
    },
    /// A degree-two periodic cyclic class not detected by the primary
    /// Bockstein.
    PeriodicCyclicClass {
        /// One-based cyclic component.
        component: usize,
        /// Exponent of the characteristic prime in that cyclic order.
        primary_exponent: usize,
    },
    /// An admissible Steenrod-square word applied to a fundamental class.
    AdmissibleSteenrodSquare {
        /// One-based coordinate among the mod-two-visible fundamental classes.
        component: usize,
        /// Checked admissible operation `Sq^I`.
        operation: AdmissibleCohomologyOperation,
        /// Whether the source fundamental class is the reduction of an
        /// integral class. Such words never end in `Sq^1`.
        fundamental_class_lifts_integrally: bool,
    },
    /// An admissible odd-primary Steenrod word applied to a fundamental class.
    AdmissibleOddPrimaryOperation {
        /// One-based coordinate among the mod-`p`-visible fundamental classes.
        component: usize,
        /// Checked admissible Bockstein/reduced-power word.
        operation: AdmissibleCohomologyOperation,
        /// Whether the source fundamental class is the reduction of an
        /// integral class. Such words have no terminal Bockstein.
        fundamental_class_lifts_integrally: bool,
    },
    /// The higher Bockstein of a fundamental class for a cyclic summand whose
    /// order is divisible by `p^r` with `r > 1`.
    HigherBockstein {
        /// One-based coordinate among the mod-`p`-visible fundamental classes.
        component: usize,
        /// Exponent `r` of the characteristic prime in the cyclic order.
        primary_exponent: usize,
    },
    /// An admissible square word on one of the two higher-primary seeds.
    AdmissibleSteenrodSquareOnHigherPrimaryClass {
        /// One-based coordinate among the mod-two-visible fundamental classes.
        component: usize,
        /// Exponent `r > 1` of two in the cyclic order.
        primary_exponent: usize,
        /// Whether the source is the higher Bockstein rather than the
        /// fundamental class.
        source_is_higher_bockstein: bool,
        /// Checked admissible operation `Sq^I`.
        operation: AdmissibleCohomologyOperation,
    },
    /// An admissible odd-primary word on one of the two higher-primary seeds.
    AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
        /// One-based coordinate among the mod-`p`-visible fundamental classes.
        component: usize,
        /// Exponent `r > 1` of `p` in the cyclic order.
        primary_exponent: usize,
        /// Whether the source is the higher Bockstein rather than the
        /// fundamental class.
        source_is_higher_bockstein: bool,
        /// Checked admissible Bockstein/reduced-power word.
        operation: AdmissibleCohomologyOperation,
    },
}

/// One stable generator in a closed-form modular cohomology presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EilenbergMacLaneCohomologyGenerator {
    id: String,
    degree: u32,
    display_name: String,
    origin: EilenbergMacLaneGeneratorOrigin,
}

impl EilenbergMacLaneCohomologyGenerator {
    /// Stable algebra-generator id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Cohomological degree.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// Mathematical display name, such as `iota_2` or `beta(iota_1)`.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Evidence-backed mathematical origin of the name.
    pub fn origin(&self) -> &EilenbergMacLaneGeneratorOrigin {
        &self.origin
    }
}

/// One primary cohomology operation evaluated on a presentation generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EilenbergMacLaneCohomologyOperation {
    operation: AdmissibleCohomologyOperation,
    source_generator: usize,
    target: GradedCommutativePolynomial,
}

impl EilenbergMacLaneCohomologyOperation {
    /// Typed admissible primary operation.
    pub fn operation(&self) -> &AdmissibleCohomologyOperation {
        &self.operation
    }

    /// Zero-based source generator index.
    pub fn source_generator(&self) -> usize {
        self.source_generator
    }

    /// Canonical target normal form in the recorded presentation.
    pub fn target(&self) -> &GradedCommutativePolynomial {
        &self.target
    }
}

/// Closed-form strategy certifying a complete modular presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EilenbergMacLaneModularCohomologyStrategy {
    /// Cohomology of a point.
    TrivialSpace,
    /// Tensor product of cyclic/free classifying-space formulas.
    ClassifyingSpaceDecomposition,
    /// Polynomial cohomology of a product of infinite projective spaces.
    FreeAbelianSecondDelooping,
}

/// A complete modular cohomology presentation for a symbolic `K(A,n)`.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneModularCohomology {
    space: EilenbergMacLaneSpace,
    field: FiniteField,
    presentation: GradedCommutativePresentation,
    generators: Vec<EilenbergMacLaneCohomologyGenerator>,
    operations: Vec<EilenbergMacLaneCohomologyOperation>,
    strategy: EilenbergMacLaneModularCohomologyStrategy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModularCohomologyWire {
    #[serde(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpace,
    field: FiniteField,
    presentation: GradedCommutativePresentation,
    generators: Vec<EilenbergMacLaneCohomologyGenerator>,
    operations: Vec<EilenbergMacLaneCohomologyOperation>,
    strategy: EilenbergMacLaneModularCohomologyStrategy,
}

impl EilenbergMacLaneModularCohomology {
    /// Symbolic source space.
    pub fn space(&self) -> &EilenbergMacLaneSpace {
        &self.space
    }

    /// Exact coefficient field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Complete graded-commutative quotient presentation.
    pub fn presentation(&self) -> &GradedCommutativePresentation {
        &self.presentation
    }

    /// Stable generator metadata in the presentation's generator order.
    pub fn generators(&self) -> &[EilenbergMacLaneCohomologyGenerator] {
        &self.generators
    }

    /// Primary operations computed on presentation generators.
    pub fn operations(&self) -> &[EilenbergMacLaneCohomologyOperation] {
        &self.operations
    }

    /// Closed-form completeness strategy.
    pub fn strategy(&self) -> EilenbergMacLaneModularCohomologyStrategy {
        self.strategy
    }

    /// Reconstruct the complete presentation and every recorded operation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.presentation
            .verify()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane modular presentation is canonical and valid",
            })?;
        if self.presentation.ring().field() != &self.field
            || self.presentation.ring().generators().len() != self.generators.len()
            || self
                .generators
                .iter()
                .zip(self.presentation.ring().generators())
                .any(|(metadata, generator)| {
                    metadata.id != generator.name() || metadata.degree != generator.degree()
                })
            || self.operations.iter().any(|operation| {
                operation.source_generator >= self.generators.len()
                    || operation.target.ring() != self.presentation.ring()
            })
        {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane modular metadata matches its presentation context",
            });
        }
        let expected =
            compute_modular_cohomology(&self.space, self.field.clone()).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim:
                        "Eilenberg--Mac Lane modular closed form remains exactly reconstructible",
                }
            })?;
        if &expected != self {
            return Err(VerificationError::InvalidWitness {
                claim: "Eilenberg--Mac Lane modular presentation and operations agree with closed-form replay",
            });
        }
        Ok(())
    }

    /// Serialize the complete presentation and operation data as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reconstruct a complete modular presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneModularCohomology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EilenbergMacLaneModularCohomology")
            .field("space", &self.space)
            .field("field", &self.field)
            .field("generators", &self.generators)
            .field("operations", &self.operations)
            .field("strategy", &self.strategy)
            .finish_non_exhaustive()
    }
}

impl SemanticType for EilenbergMacLaneModularCohomology {
    const TYPE_ID: &'static str = "noether/topology/EilenbergMacLaneModularCohomology@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MODULAR_COHOMOLOGY_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneModularCohomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModularCohomologyWire {
            type_id: Self::TYPE_ID.to_owned(),
            space: self.space.clone(),
            field: self.field.clone(),
            presentation: self.presentation.clone(),
            generators: self.generators.clone(),
            operations: self.operations.clone(),
            strategy: self.strategy,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneModularCohomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModularCohomologyWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            space: wire.space,
            field: wire.field,
            presentation: wire.presentation,
            generators: wire.generators,
            operations: wire.operations,
            strategy: wire.strategy,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

/// Exact bounded strategy used for modular cohomology of `K(A,n)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EilenbergMacLaneModularCohomologyTruncationStrategy {
    /// Cohomology of a point.
    TrivialSpace,
    /// Cartan--Serre admissible-square generators over `F_2`.
    CartanSerreAdmissibleSquares,
    /// Cartan--Serre admissible Bockstein/reduced-power generators at odd `p`.
    CartanSerreAdmissibleOddPrimaryOperations,
}

/// A modular cohomology presentation valid through one inclusive degree.
///
/// The finite polynomial presentation contains precisely the algebra
/// generators that can contribute through [`Self::exact_through_degree`]. It
/// is not a finite presentation of the generally infinitely generated full
/// cohomology ring and makes no assertion in larger degrees.
#[derive(Clone, PartialEq, Eq)]
pub struct EilenbergMacLaneModularCohomologyTruncation {
    space: EilenbergMacLaneSpace,
    field: FiniteField,
    exact_through_degree: usize,
    presentation: GradedCommutativePresentation,
    generators: Vec<EilenbergMacLaneCohomologyGenerator>,
    strategy: EilenbergMacLaneModularCohomologyTruncationStrategy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModularCohomologyTruncationWire {
    #[serde(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpace,
    field: FiniteField,
    exact_through_degree: usize,
    presentation: GradedCommutativePresentation,
    generators: Vec<EilenbergMacLaneCohomologyGenerator>,
    strategy: EilenbergMacLaneModularCohomologyTruncationStrategy,
}

impl EilenbergMacLaneModularCohomologyTruncation {
    /// Symbolic source space.
    pub fn space(&self) -> &EilenbergMacLaneSpace {
        &self.space
    }

    /// Exact coefficient field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Largest cohomological degree certified by this result.
    pub fn exact_through_degree(&self) -> usize {
        self.exact_through_degree
    }

    /// Finite presentation whose graded pieces are exact through the bound.
    pub fn presentation(&self) -> &GradedCommutativePresentation {
        &self.presentation
    }

    /// Stable metadata for every algebra generator through the bound.
    pub fn generators(&self) -> &[EilenbergMacLaneCohomologyGenerator] {
        &self.generators
    }

    /// Exact mathematical strategy used by the bounded result.
    pub fn strategy(&self) -> EilenbergMacLaneModularCohomologyTruncationStrategy {
        self.strategy
    }

    /// Replay the theorem-specific generator enumeration and presentation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.presentation
            .verify()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "bounded Eilenberg--Mac Lane modular presentation is canonical and valid",
            })?;
        if self.presentation.ring().field() != &self.field
            || self.presentation.ring().generators().len() != self.generators.len()
            || self
                .generators
                .iter()
                .zip(self.presentation.ring().generators())
                .any(|(metadata, generator)| {
                    metadata.id != generator.name() || metadata.degree != generator.degree()
                })
        {
            return Err(VerificationError::InvalidWitness {
                claim: "bounded Eilenberg--Mac Lane generator metadata matches its presentation",
            });
        }
        let expected = compute_modular_cohomology_truncation(
            &self.space,
            self.field.clone(),
            self.exact_through_degree,
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "bounded Eilenberg--Mac Lane modular result remains exactly reconstructible",
        })?;
        if &expected != self {
            return Err(VerificationError::InvalidWitness {
                claim:
                    "bounded Eilenberg--Mac Lane modular result agrees with deterministic replay",
            });
        }
        Ok(())
    }

    /// Serialize the bounded presentation and theorem evidence as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay a bounded modular-cohomology result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for EilenbergMacLaneModularCohomologyTruncation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EilenbergMacLaneModularCohomologyTruncation")
            .field("space", &self.space)
            .field("field", &self.field)
            .field("exact_through_degree", &self.exact_through_degree)
            .field("generators", &self.generators)
            .field("strategy", &self.strategy)
            .finish_non_exhaustive()
    }
}

impl SemanticType for EilenbergMacLaneModularCohomologyTruncation {
    const TYPE_ID: &'static str = "noether/topology/EilenbergMacLaneModularCohomologyTruncation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MODULAR_COHOMOLOGY_TRUNCATION_DESCRIPTION
    }
}

impl Serialize for EilenbergMacLaneModularCohomologyTruncation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModularCohomologyTruncationWire {
            type_id: Self::TYPE_ID.to_owned(),
            space: self.space.clone(),
            field: self.field.clone(),
            exact_through_degree: self.exact_through_degree,
            presentation: self.presentation.clone(),
            generators: self.generators.clone(),
            strategy: self.strategy,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EilenbergMacLaneModularCohomologyTruncation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModularCohomologyTruncationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            space: wire.space,
            field: wire.field,
            exact_through_degree: wire.exact_through_degree,
            presentation: wire.presentation,
            generators: wire.generators,
            strategy: wire.strategy,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[derive(Clone)]
struct ModularGeneratorBuild {
    metadata: EilenbergMacLaneCohomologyGenerator,
    square_zero: bool,
}

enum OperationTargetBuild {
    Zero,
    Power { generator: usize, exponent: u32 },
}

struct ModularOperationBuild {
    operation: AdmissibleCohomologyOperation,
    source_generator: usize,
    target: OperationTargetBuild,
}

fn compute_modular_cohomology(
    space: &EilenbergMacLaneSpace,
    field: FiniteField,
) -> Result<EilenbergMacLaneModularCohomology, EilenbergMacLaneError> {
    if field.degree() != 1 {
        return Err(UnsupportedOperationError::new(
            "noether/topology/EilenbergMacLaneSpace/modular_cohomology@1",
            format!("coefficients in F_{}", field.order()),
        )
        .with_supported_contexts(["prime-field coefficients F_p"])
        .with_detail(
            "this presentation and its named primary operations are defined over the prime field",
        )
        .into());
    }
    let characteristic = field.characteristic().clone();
    let prime = characteristic.to_u64_checked().ok_or_else(|| {
        EilenbergMacLaneError::CharacteristicTooLarge {
            characteristic: characteristic.clone(),
        }
    })?;
    let prime_exponent =
        u32::try_from(prime).map_err(|_| EilenbergMacLaneError::CharacteristicTooLarge {
            characteristic: characteristic.clone(),
        })?;
    let characteristic_two = prime == 2;

    let (generator_builds, operation_builds, strategy) = if space.group.is_trivial() {
        (
            Vec::new(),
            Vec::new(),
            EilenbergMacLaneModularCohomologyStrategy::TrivialSpace,
        )
    } else if space.delooping_degree == 1 {
        let (generators, operations) = classifying_space_modular_generators(
            &space.group,
            &characteristic,
            prime,
            characteristic_two,
            prime_exponent,
        )?;
        (
            generators,
            operations,
            EilenbergMacLaneModularCohomologyStrategy::ClassifyingSpaceDecomposition,
        )
    } else if space.delooping_degree == 2 && space.group.torsion_coefficients().is_empty() {
        let mut generators = Vec::new();
        let mut operations = Vec::new();
        for component in 1..=space.group.free_rank() {
            let index = generators.len();
            generators.push(ModularGeneratorBuild {
                metadata: EilenbergMacLaneCohomologyGenerator {
                    id: format!("g{}", index + 1),
                    degree: 2,
                    display_name: fundamental_name(2, component, space.group.free_rank()),
                    origin: EilenbergMacLaneGeneratorOrigin::FundamentalClass { component },
                },
                square_zero: false,
            });
            operations.push(ModularOperationBuild {
                operation: if characteristic_two {
                    AdmissibleCohomologyOperation::steenrod_square(2)
                } else {
                    AdmissibleCohomologyOperation::reduced_power(prime, 1)
                        .expect("a checked odd prime defines P^1")
                },
                source_generator: index,
                target: OperationTargetBuild::Power {
                    generator: index,
                    exponent: if characteristic_two {
                        2
                    } else {
                        prime_exponent
                    },
                },
            });
        }
        (
            generators,
            operations,
            EilenbergMacLaneModularCohomologyStrategy::FreeAbelianSecondDelooping,
        )
    } else {
        return Err(UnsupportedOperationError::new(
            "noether/topology/EilenbergMacLaneSpace/modular_cohomology@1",
            space.to_string(),
        )
        .with_supported_contexts([
            "K(A,1) for finitely generated abelian A",
            "K(Z^r,2)",
            "K(0,n)",
        ])
        .with_detail("higher deloopings and finite torsion in degree 2 are not yet in the closed-form modular playbook")
        .into());
    };

    let ring = GradedCommutativePolynomialRing::new(
        field.clone(),
        generator_builds
            .iter()
            .map(|generator| {
                GradedGenerator::new(generator.metadata.id.clone(), generator.metadata.degree)
            })
            .collect(),
    )?;
    let relations = generator_builds
        .iter()
        .enumerate()
        .filter(|(_, generator)| generator.square_zero)
        .map(|(index, _)| ring.term(field.one(), generator_power(ring.arity(), index, 2)))
        .collect::<Result<Vec<_>, _>>()?;
    let presentation = GradedCommutativePresentation::new(ring.clone(), relations)?;
    let operations = operation_builds
        .into_iter()
        .map(|operation| {
            let target = match operation.target {
                OperationTargetBuild::Zero => ring.zero(),
                OperationTargetBuild::Power {
                    generator,
                    exponent,
                } => ring.term(
                    field.one(),
                    generator_power(ring.arity(), generator, exponent),
                )?,
            };
            Ok(EilenbergMacLaneCohomologyOperation {
                operation: operation.operation,
                source_generator: operation.source_generator,
                target: presentation.normal_form(&target)?,
            })
        })
        .collect::<Result<Vec<_>, GradedCommutativeError>>()?;

    Ok(EilenbergMacLaneModularCohomology {
        space: space.clone(),
        field,
        presentation,
        generators: generator_builds
            .into_iter()
            .map(|generator| generator.metadata)
            .collect(),
        operations,
        strategy,
    })
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum BoundedGeneratorSource {
    FundamentalClass { lifts_integrally: bool },
    HigherPrimaryFundamental { primary_exponent: usize },
    HigherBockstein { primary_exponent: usize },
}

impl BoundedGeneratorSource {
    fn terminal_primary_bockstein_vanishes(&self) -> bool {
        match self {
            Self::FundamentalClass { lifts_integrally } => *lifts_integrally,
            Self::HigherPrimaryFundamental { .. } | Self::HigherBockstein { .. } => true,
        }
    }

    fn display_name(&self, fundamental: &str) -> String {
        match self {
            Self::FundamentalClass { .. } | Self::HigherPrimaryFundamental { .. } => {
                fundamental.to_owned()
            }
            Self::HigherBockstein { primary_exponent } => {
                format!("beta_{primary_exponent}({fundamental})")
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BoundedModTwoGenerator {
    degree: u32,
    component: usize,
    source: BoundedGeneratorSource,
    sequence: Vec<u32>,
}

fn compute_modular_cohomology_truncation(
    space: &EilenbergMacLaneSpace,
    field: FiniteField,
    max_degree: usize,
) -> Result<EilenbergMacLaneModularCohomologyTruncation, EilenbergMacLaneError> {
    let max_degree_u32 = u32::try_from(max_degree)
        .map_err(|_| EilenbergMacLaneError::DegreeBoundTooLarge { max_degree })?;
    if field.degree() != 1 {
        return Err(UnsupportedOperationError::new(
            "noether/topology/EilenbergMacLaneSpace/modular_cohomology_through_degree@1",
            format!("coefficients in F_{}", field.order()),
        )
        .with_supported_contexts(["prime-field coefficients F_p"])
        .with_detail("the bounded Cartan--Serre lanes use primary Steenrod operations")
        .into());
    }
    let prime = field.characteristic().to_u64_checked().ok_or_else(|| {
        EilenbergMacLaneError::CharacteristicTooLarge {
            characteristic: field.characteristic().clone(),
        }
    })?;
    if prime != 2 {
        return compute_odd_primary_modular_cohomology_truncation(
            space,
            field,
            prime,
            max_degree,
            max_degree_u32,
        );
    }

    let (generator_specs, strategy) = if space.group.is_trivial() {
        (
            Vec::new(),
            EilenbergMacLaneModularCohomologyTruncationStrategy::TrivialSpace,
        )
    } else {
        if space.delooping_degree < 2 {
            return Err(UnsupportedOperationError::new(
                "noether/topology/EilenbergMacLaneSpace/modular_cohomology_through_degree@1",
                space.to_string(),
            )
            .with_supported_contexts(["K(A,n) with n >= 2", "K(0,n)"])
            .with_detail("K(A,1) has a separate complete closed-form modular presentation")
            .into());
        }
        let primary_components = space
            .group
            .torsion_coefficients()
            .iter()
            .filter_map(|factor| {
                let exponent = p_adic_exponent(factor, &Integer::from(2_u8));
                (exponent > 0).then_some(exponent)
            })
            .collect::<Vec<_>>();
        let mut generators = Vec::new();
        if space.delooping_degree <= max_degree {
            let delooping_degree = u32::try_from(space.delooping_degree)
                .expect("a delooping degree below a u32 degree bound fits u32");
            for component in 1..=space.group.free_rank() {
                append_cartan_serre_generators(
                    &mut generators,
                    component,
                    true,
                    delooping_degree,
                    max_degree_u32,
                );
            }
            for (offset, primary_exponent) in primary_components.iter().copied().enumerate() {
                let component = space.group.free_rank() + offset + 1;
                if primary_exponent == 1 {
                    append_cartan_serre_generators(
                        &mut generators,
                        component,
                        false,
                        delooping_degree,
                        max_degree_u32,
                    );
                } else {
                    append_cartan_serre_generators_from_source(
                        &mut generators,
                        component,
                        BoundedGeneratorSource::HigherPrimaryFundamental { primary_exponent },
                        delooping_degree,
                        max_degree_u32,
                    );
                    if let Some(higher_degree) = delooping_degree.checked_add(1) {
                        append_cartan_serre_generators_from_source(
                            &mut generators,
                            component,
                            BoundedGeneratorSource::HigherBockstein { primary_exponent },
                            higher_degree,
                            max_degree_u32,
                        );
                    }
                }
            }
        }
        generators.sort();
        (
            generators,
            EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleSquares,
        )
    };

    let component_count = generator_specs
        .iter()
        .map(|generator| generator.component)
        .max()
        .unwrap_or(0);
    let generators = generator_specs
        .into_iter()
        .enumerate()
        .map(|(index, generator)| {
            let fundamental = fundamental_name(
                u32::try_from(space.delooping_degree)
                    .expect("the bounded delooping degree fits u32"),
                generator.component,
                component_count,
            );
            let source_name = generator.source.display_name(&fundamental);
            let display_name = if generator.sequence.is_empty() {
                source_name
            } else {
                let operation = AdmissibleCohomologyOperation::mod_two(
                    generator.sequence.clone(),
                )
                .expect("the Cartan--Serre generator enumerator emits admissible words");
                format!("{}({source_name})", operation.display_name())
            };
            EilenbergMacLaneCohomologyGenerator {
                id: format!("g{}", index + 1),
                degree: generator.degree,
                display_name,
                origin: match (generator.source, generator.sequence.is_empty()) {
                    (
                        BoundedGeneratorSource::FundamentalClass { .. }
                        | BoundedGeneratorSource::HigherPrimaryFundamental { .. },
                        true,
                    ) => EilenbergMacLaneGeneratorOrigin::FundamentalClass {
                        component: generator.component,
                    },
                    (
                        BoundedGeneratorSource::HigherBockstein { primary_exponent },
                        true,
                    ) => EilenbergMacLaneGeneratorOrigin::HigherBockstein {
                        component: generator.component,
                        primary_exponent,
                    },
                    (
                        BoundedGeneratorSource::FundamentalClass { lifts_integrally },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquare {
                        component: generator.component,
                        operation: AdmissibleCohomologyOperation::mod_two(generator.sequence)
                            .expect(
                                "the Cartan--Serre generator enumerator emits admissible words",
                            ),
                        fundamental_class_lifts_integrally: lifts_integrally,
                    },
                    (
                        BoundedGeneratorSource::HigherPrimaryFundamental { primary_exponent },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquareOnHigherPrimaryClass {
                        component: generator.component,
                        primary_exponent,
                        source_is_higher_bockstein: false,
                        operation: AdmissibleCohomologyOperation::mod_two(generator.sequence)
                            .expect(
                                "the Cartan--Serre generator enumerator emits admissible words",
                            ),
                    },
                    (
                        BoundedGeneratorSource::HigherBockstein { primary_exponent },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquareOnHigherPrimaryClass {
                        component: generator.component,
                        primary_exponent,
                        source_is_higher_bockstein: true,
                        operation: AdmissibleCohomologyOperation::mod_two(generator.sequence)
                            .expect(
                                "the Cartan--Serre generator enumerator emits admissible words",
                            ),
                    },
                },
            }
        })
        .collect::<Vec<_>>();
    let ring = GradedCommutativePolynomialRing::new(
        field.clone(),
        generators
            .iter()
            .map(|generator| GradedGenerator::new(generator.id.clone(), generator.degree))
            .collect(),
    )?;
    let presentation = GradedCommutativePresentation::new(ring, Vec::new())?;
    Ok(EilenbergMacLaneModularCohomologyTruncation {
        space: space.clone(),
        field,
        exact_through_degree: max_degree,
        presentation,
        generators,
        strategy,
    })
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BoundedOddPrimaryGenerator {
    degree: u32,
    component: usize,
    source: BoundedGeneratorSource,
    bocksteins: Vec<bool>,
    reduced_powers: Vec<u32>,
}

fn compute_odd_primary_modular_cohomology_truncation(
    space: &EilenbergMacLaneSpace,
    field: FiniteField,
    prime: u64,
    max_degree: usize,
    max_degree_u32: u32,
) -> Result<EilenbergMacLaneModularCohomologyTruncation, EilenbergMacLaneError> {
    if space.group.is_trivial() {
        let ring = GradedCommutativePolynomialRing::new(field.clone(), Vec::new())?;
        return Ok(EilenbergMacLaneModularCohomologyTruncation {
            space: space.clone(),
            field,
            exact_through_degree: max_degree,
            presentation: GradedCommutativePresentation::new(ring, Vec::new())?,
            generators: Vec::new(),
            strategy: EilenbergMacLaneModularCohomologyTruncationStrategy::TrivialSpace,
        });
    }
    if space.delooping_degree < 2 {
        return Err(UnsupportedOperationError::new(
            "noether/topology/EilenbergMacLaneSpace/modular_cohomology_through_degree@1",
            space.to_string(),
        )
        .with_supported_contexts(["K(A,n) with n >= 2", "K(0,n)"])
        .with_detail("K(A,1) has a separate complete closed-form modular presentation")
        .into());
    }
    let characteristic = Integer::from(prime);
    let primary_components = space
        .group
        .torsion_coefficients()
        .iter()
        .filter_map(|factor| {
            let exponent = p_adic_exponent(factor, &characteristic);
            (exponent > 0).then_some(exponent)
        })
        .collect::<Vec<_>>();
    let component_count = space.group.free_rank() + primary_components.len();
    let mut generator_specs = Vec::new();
    if space.delooping_degree <= max_degree {
        let delooping_degree = u32::try_from(space.delooping_degree)
            .expect("a delooping degree below a u32 degree bound fits u32");
        for component in 1..=space.group.free_rank() {
            append_cartan_serre_odd_primary_generators(
                &mut generator_specs,
                component,
                true,
                prime,
                delooping_degree,
                max_degree_u32,
            );
        }
        for (offset, primary_exponent) in primary_components.iter().copied().enumerate() {
            let component = space.group.free_rank() + offset + 1;
            if primary_exponent == 1 {
                append_cartan_serre_odd_primary_generators(
                    &mut generator_specs,
                    component,
                    false,
                    prime,
                    delooping_degree,
                    max_degree_u32,
                );
            } else {
                append_cartan_serre_odd_primary_generators_from_source(
                    &mut generator_specs,
                    component,
                    BoundedGeneratorSource::HigherPrimaryFundamental { primary_exponent },
                    prime,
                    delooping_degree,
                    max_degree_u32,
                );
                if let Some(higher_degree) = delooping_degree.checked_add(1) {
                    append_cartan_serre_odd_primary_generators_from_source(
                        &mut generator_specs,
                        component,
                        BoundedGeneratorSource::HigherBockstein { primary_exponent },
                        prime,
                        higher_degree,
                        max_degree_u32,
                    );
                }
            }
        }
    }
    generator_specs.sort();
    let generators = generator_specs
        .into_iter()
        .enumerate()
        .map(|(index, generator)| {
            let fundamental = fundamental_name(
                u32::try_from(space.delooping_degree)
                    .expect("the bounded delooping degree fits u32"),
                generator.component,
                component_count,
            );
            let source_name = generator.source.display_name(&fundamental);
            let display_name = if generator.reduced_powers.is_empty()
                && generator.bocksteins == [false]
            {
                source_name
            } else {
                let operation = AdmissibleCohomologyOperation::odd_primary(
                    prime,
                    generator.bocksteins.clone(),
                    generator.reduced_powers.clone(),
                )
                .expect("the Cartan--Serre generator enumerator emits admissible words");
                format!("{}({source_name})", operation.display_name())
            };
            EilenbergMacLaneCohomologyGenerator {
                id: format!("g{}", index + 1),
                degree: generator.degree,
                display_name,
                origin: match (
                    generator.source,
                    generator.reduced_powers.is_empty() && generator.bocksteins == [false],
                ) {
                    (
                        BoundedGeneratorSource::FundamentalClass { .. }
                        | BoundedGeneratorSource::HigherPrimaryFundamental { .. },
                        true,
                    ) => EilenbergMacLaneGeneratorOrigin::FundamentalClass {
                        component: generator.component,
                    },
                    (
                        BoundedGeneratorSource::HigherBockstein { primary_exponent },
                        true,
                    ) => EilenbergMacLaneGeneratorOrigin::HigherBockstein {
                        component: generator.component,
                        primary_exponent,
                    },
                    (
                        BoundedGeneratorSource::FundamentalClass { lifts_integrally },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperation {
                        component: generator.component,
                        operation: AdmissibleCohomologyOperation::odd_primary(
                            prime,
                            generator.bocksteins,
                            generator.reduced_powers,
                        )
                        .expect(
                            "the Cartan--Serre generator enumerator emits admissible words",
                        ),
                        fundamental_class_lifts_integrally: lifts_integrally,
                    },
                    (
                        BoundedGeneratorSource::HigherPrimaryFundamental { primary_exponent },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
                        component: generator.component,
                        primary_exponent,
                        source_is_higher_bockstein: false,
                        operation: AdmissibleCohomologyOperation::odd_primary(
                            prime,
                            generator.bocksteins,
                            generator.reduced_powers,
                        )
                        .expect(
                            "the Cartan--Serre generator enumerator emits admissible words",
                        ),
                    },
                    (
                        BoundedGeneratorSource::HigherBockstein { primary_exponent },
                        false,
                    ) => EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
                        component: generator.component,
                        primary_exponent,
                        source_is_higher_bockstein: true,
                        operation: AdmissibleCohomologyOperation::odd_primary(
                            prime,
                            generator.bocksteins,
                            generator.reduced_powers,
                        )
                        .expect(
                            "the Cartan--Serre generator enumerator emits admissible words",
                        ),
                    },
                },
            }
        })
        .collect::<Vec<_>>();
    let ring = GradedCommutativePolynomialRing::new(
        field.clone(),
        generators
            .iter()
            .map(|generator| GradedGenerator::new(generator.id.clone(), generator.degree))
            .collect(),
    )?;
    Ok(EilenbergMacLaneModularCohomologyTruncation {
        space: space.clone(),
        field,
        exact_through_degree: max_degree,
        presentation: GradedCommutativePresentation::new(ring, Vec::new())?,
        generators,
        strategy:
            EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleOddPrimaryOperations,
    })
}

fn append_cartan_serre_odd_primary_generators(
    generators: &mut Vec<BoundedOddPrimaryGenerator>,
    component: usize,
    fundamental_class_lifts_integrally: bool,
    prime: u64,
    delooping_degree: u32,
    max_degree: u32,
) {
    append_cartan_serre_odd_primary_generators_from_source(
        generators,
        component,
        BoundedGeneratorSource::FundamentalClass {
            lifts_integrally: fundamental_class_lifts_integrally,
        },
        prime,
        delooping_degree,
        max_degree,
    );
}

fn append_cartan_serre_odd_primary_generators_from_source(
    generators: &mut Vec<BoundedOddPrimaryGenerator>,
    component: usize,
    source: BoundedGeneratorSource,
    prime: u64,
    source_degree: u32,
    max_degree: u32,
) {
    if source_degree > max_degree {
        return;
    }
    generators.push(BoundedOddPrimaryGenerator {
        degree: source_degree,
        component,
        source: source.clone(),
        bocksteins: vec![false],
        reduced_powers: Vec::new(),
    });
    let max_weight = max_degree - source_degree;
    if !source.terminal_primary_bockstein_vanishes() && max_weight >= 1 {
        generators.push(BoundedOddPrimaryGenerator {
            degree: source_degree + 1,
            component,
            source: source.clone(),
            bocksteins: vec![true],
            reduced_powers: Vec::new(),
        });
    }
    let Ok(prime) = u32::try_from(prime) else {
        return;
    };
    let reduced_power_weight = 2_u32.saturating_mul(prime.saturating_sub(1));
    if reduced_power_weight == 0 || reduced_power_weight > max_weight {
        return;
    }
    let maximum_power_sum = max_weight / reduced_power_weight;
    for leading_bockstein in [false, true] {
        for first_power in 1..=maximum_power_sum {
            for trailing_bockstein in [false, true] {
                let bocksteins = vec![leading_bockstein, trailing_bockstein];
                let reduced_powers = vec![first_power];
                append_odd_primary_candidate_and_extensions(
                    generators,
                    component,
                    &source,
                    prime,
                    source_degree,
                    max_degree,
                    bocksteins,
                    reduced_powers,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_odd_primary_candidate_and_extensions(
    generators: &mut Vec<BoundedOddPrimaryGenerator>,
    component: usize,
    source: &BoundedGeneratorSource,
    prime: u32,
    source_degree: u32,
    max_degree: u32,
    bocksteins: Vec<bool>,
    reduced_powers: Vec<u32>,
) {
    let operation_degree = odd_primary_operation_degree(prime, &bocksteins, &reduced_powers);
    if operation_degree > max_degree - source_degree {
        return;
    }
    let excess = odd_primary_excess(prime, &bocksteins, &reduced_powers);
    let leading_bockstein = bocksteins[0];
    let terminal_bockstein = *bocksteins.last().unwrap_or(&false);
    if (!source.terminal_primary_bockstein_vanishes() || !terminal_bockstein)
        && (excess < source_degree || (excess == source_degree && leading_bockstein))
    {
        generators.push(BoundedOddPrimaryGenerator {
            degree: source_degree + operation_degree,
            component,
            source: source.clone(),
            bocksteins: bocksteins.clone(),
            reduced_powers: reduced_powers.clone(),
        });
    }

    let current_power = *reduced_powers
        .last()
        .expect("an odd-primary extension has a reduced power");
    let current_bockstein = u32::from(*bocksteins.last().unwrap_or(&false));
    let maximum_next = current_power.saturating_sub(current_bockstein) / prime;
    for next_power in 1..=maximum_next {
        for next_bockstein in [false, true] {
            let mut extended_bocksteins = bocksteins.clone();
            let mut extended_powers = reduced_powers.clone();
            extended_bocksteins.push(next_bockstein);
            extended_powers.push(next_power);
            append_odd_primary_candidate_and_extensions(
                generators,
                component,
                source,
                prime,
                source_degree,
                max_degree,
                extended_bocksteins,
                extended_powers,
            );
        }
    }
}

fn odd_primary_operation_degree(prime: u32, bocksteins: &[bool], reduced_powers: &[u32]) -> u32 {
    bocksteins.iter().filter(|value| **value).count() as u32
        + 2_u32
            .saturating_mul(prime.saturating_sub(1))
            .saturating_mul(reduced_powers.iter().copied().sum())
}

fn odd_primary_excess(prime: u32, bocksteins: &[bool], reduced_powers: &[u32]) -> u32 {
    let mut excess = u32::from(bocksteins.first().copied().unwrap_or(false));
    for (index, power) in reduced_powers.iter().copied().enumerate() {
        let next_power = reduced_powers.get(index + 1).copied().unwrap_or(0);
        let trailing_bockstein = u32::from(bocksteins.get(index + 1).copied().unwrap_or(false));
        excess = excess.saturating_add(
            2_u32
                .saturating_mul(power)
                .saturating_sub(2_u32.saturating_mul(prime).saturating_mul(next_power))
                .saturating_sub(trailing_bockstein),
        );
    }
    excess
}

fn append_cartan_serre_generators(
    generators: &mut Vec<BoundedModTwoGenerator>,
    component: usize,
    fundamental_class_lifts_integrally: bool,
    delooping_degree: u32,
    max_degree: u32,
) {
    append_cartan_serre_generators_from_source(
        generators,
        component,
        BoundedGeneratorSource::FundamentalClass {
            lifts_integrally: fundamental_class_lifts_integrally,
        },
        delooping_degree,
        max_degree,
    );
}

fn append_cartan_serre_generators_from_source(
    generators: &mut Vec<BoundedModTwoGenerator>,
    component: usize,
    source: BoundedGeneratorSource,
    source_degree: u32,
    max_degree: u32,
) {
    if source_degree > max_degree {
        return;
    }
    generators.push(BoundedModTwoGenerator {
        degree: source_degree,
        component,
        source: source.clone(),
        sequence: Vec::new(),
    });
    let max_weight = max_degree - source_degree;
    let mut sequence = Vec::new();
    for first in 1..=max_weight {
        sequence.push(first);
        append_admissible_square_extensions(
            generators,
            component,
            &source,
            source_degree,
            max_weight,
            first,
            first,
            &mut sequence,
        );
        sequence.pop();
    }
}

#[allow(clippy::too_many_arguments)]
fn append_admissible_square_extensions(
    generators: &mut Vec<BoundedModTwoGenerator>,
    component: usize,
    source: &BoundedGeneratorSource,
    source_degree: u32,
    max_weight: u32,
    weight: u32,
    first: u32,
    sequence: &mut Vec<u32>,
) {
    let tail_sum = weight - first;
    let excess = first.saturating_sub(tail_sum);
    let allowed_terminal =
        !source.terminal_primary_bockstein_vanishes() || sequence.last() != Some(&1);
    if excess < source_degree && allowed_terminal {
        generators.push(BoundedModTwoGenerator {
            degree: source_degree + weight,
            component,
            source: source.clone(),
            sequence: sequence.clone(),
        });
    }
    let remaining = max_weight - weight;
    let largest_next = (sequence.last().copied().unwrap_or(0) / 2).min(remaining);
    for next in 1..=largest_next {
        sequence.push(next);
        append_admissible_square_extensions(
            generators,
            component,
            source,
            source_degree,
            max_weight,
            weight + next,
            first,
            sequence,
        );
        sequence.pop();
    }
}

fn classifying_space_modular_generators(
    group: &FinitelyGeneratedAbelianGroup,
    characteristic: &Integer,
    prime: u64,
    characteristic_two: bool,
    prime_exponent: u32,
) -> Result<(Vec<ModularGeneratorBuild>, Vec<ModularOperationBuild>), EilenbergMacLaneError> {
    let relevant_torsion = group
        .torsion_coefficients()
        .iter()
        .map(|factor| p_adic_exponent(factor, characteristic))
        .filter(|exponent| *exponent > 0)
        .collect::<Vec<_>>();
    let fundamental_count = group.free_rank() + relevant_torsion.len();
    let mut generators = Vec::new();
    let mut operations = Vec::new();

    for component in 1..=group.free_rank() {
        let index = push_fundamental_generator(
            &mut generators,
            1,
            component,
            fundamental_count,
            characteristic_two,
        );
        operations.push(ModularOperationBuild {
            operation: if characteristic_two {
                AdmissibleCohomologyOperation::steenrod_square(1)
            } else {
                AdmissibleCohomologyOperation::bockstein_reduced_power(prime, 0)
                    .expect("a checked odd prime defines beta")
            },
            source_generator: index,
            target: OperationTargetBuild::Zero,
        });
    }

    for (torsion_component, primary_exponent) in relevant_torsion.into_iter().enumerate() {
        let component = group.free_rank() + torsion_component + 1;
        let fundamental = push_fundamental_generator(
            &mut generators,
            1,
            component,
            fundamental_count,
            characteristic_two && primary_exponent > 1,
        );
        if characteristic_two && primary_exponent == 1 {
            operations.push(ModularOperationBuild {
                operation: AdmissibleCohomologyOperation::steenrod_square(1),
                source_generator: fundamental,
                target: OperationTargetBuild::Power {
                    generator: fundamental,
                    exponent: 2,
                },
            });
            continue;
        }

        let periodic = generators.len();
        let fundamental_display = generators[fundamental].metadata.display_name.clone();
        generators.push(ModularGeneratorBuild {
            metadata: EilenbergMacLaneCohomologyGenerator {
                id: format!("g{}", periodic + 1),
                degree: 2,
                display_name: if !characteristic_two && primary_exponent == 1 {
                    format!("beta({fundamental_display})")
                } else if fundamental_count == 1 {
                    "x_2".to_owned()
                } else {
                    format!("x_2.{component}")
                },
                origin: if !characteristic_two && primary_exponent == 1 {
                    EilenbergMacLaneGeneratorOrigin::Bockstein {
                        source_generator: fundamental,
                    }
                } else {
                    EilenbergMacLaneGeneratorOrigin::PeriodicCyclicClass {
                        component,
                        primary_exponent,
                    }
                },
            },
            square_zero: false,
        });
        operations.push(ModularOperationBuild {
            operation: if characteristic_two {
                AdmissibleCohomologyOperation::steenrod_square(1)
            } else {
                AdmissibleCohomologyOperation::bockstein_reduced_power(prime, 0)
                    .expect("a checked odd prime defines beta")
            },
            source_generator: fundamental,
            target: if !characteristic_two && primary_exponent == 1 {
                OperationTargetBuild::Power {
                    generator: periodic,
                    exponent: 1,
                }
            } else {
                OperationTargetBuild::Zero
            },
        });
        operations.push(ModularOperationBuild {
            operation: if characteristic_two {
                AdmissibleCohomologyOperation::steenrod_square(2)
            } else {
                AdmissibleCohomologyOperation::reduced_power(prime, 1)
                    .expect("a checked odd prime defines P^1")
            },
            source_generator: periodic,
            target: OperationTargetBuild::Power {
                generator: periodic,
                exponent: if characteristic_two {
                    2
                } else {
                    prime_exponent
                },
            },
        });
    }
    Ok((generators, operations))
}

fn push_fundamental_generator(
    generators: &mut Vec<ModularGeneratorBuild>,
    degree: u32,
    component: usize,
    component_count: usize,
    square_zero: bool,
) -> usize {
    let index = generators.len();
    generators.push(ModularGeneratorBuild {
        metadata: EilenbergMacLaneCohomologyGenerator {
            id: format!("g{}", index + 1),
            degree,
            display_name: fundamental_name(degree, component, component_count),
            origin: EilenbergMacLaneGeneratorOrigin::FundamentalClass { component },
        },
        square_zero,
    });
    index
}

fn fundamental_name(degree: u32, component: usize, component_count: usize) -> String {
    if component_count == 1 {
        format!("iota_{degree}")
    } else {
        format!("iota_{degree}.{component}")
    }
}

fn p_adic_exponent(value: &Integer, prime: &Integer) -> usize {
    let mut quotient = value.clone();
    let mut exponent = 0usize;
    loop {
        let Ok((next, remainder)) = Z.div_rem(&quotient, prime) else {
            break;
        };
        if !remainder.is_zero() {
            break;
        }
        quotient = next;
        exponent += 1;
    }
    exponent
}

fn generator_power(arity: usize, generator: usize, exponent: u32) -> Vec<u32> {
    let mut exponents = vec![0; arity];
    exponents[generator] = exponent;
    exponents
}

fn compute_integral_cohomology_presentation(
    space: &EilenbergMacLaneSpace,
) -> Result<EilenbergMacLaneIntegralCohomologyPresentation, EilenbergMacLaneError> {
    if space.group.is_trivial() {
        return Ok(EilenbergMacLaneIntegralCohomologyPresentation {
            space: space.clone(),
            generators: Vec::new(),
            relations: Vec::new(),
            strategy: EilenbergMacLaneIntegralCohomologyPresentationStrategy::TrivialSpace,
        });
    }

    if space.delooping_degree == 1 && space.group.torsion_coefficients().len() <= 2 {
        let free_rank = space.group.free_rank();
        let torsion = space.group.torsion_coefficients();
        let mut generators =
            Vec::with_capacity(free_rank + torsion.len() + usize::from(torsion.len() == 2));
        for component in 1..=free_rank {
            generators.push(EilenbergMacLaneIntegralCohomologyGenerator {
                id: format!("x1_{component}"),
                degree: 1,
                display_name: if free_rank == 1 && space.group.torsion_coefficients().is_empty() {
                    "iota_1".to_owned()
                } else {
                    format!("iota_1^({component})")
                },
                origin: EilenbergMacLaneIntegralGeneratorOrigin::FreeFundamentalClass { component },
            });
        }
        for (torsion_index, modulus) in torsion.iter().enumerate() {
            let component = free_rank + torsion_index + 1;
            generators.push(EilenbergMacLaneIntegralCohomologyGenerator {
                id: format!("u2_{}", torsion_index + 1),
                degree: 2,
                display_name: if free_rank == 0 && torsion.len() == 1 {
                    format!("beta_{modulus}(iota_1)")
                } else {
                    format!("beta_{modulus}(iota_1^({component}))")
                },
                origin: EilenbergMacLaneIntegralGeneratorOrigin::IntegralBockstein {
                    component,
                    modulus: modulus.clone(),
                },
            });
        }
        if torsion.len() == 2 {
            let left_component = free_rank + 1;
            let right_component = free_rank + 2;
            generators.push(EilenbergMacLaneIntegralCohomologyGenerator {
                id: "t3_1".to_owned(),
                degree: 3,
                display_name: format!(
                    "Tor_Z(iota_1^({left_component}),iota_1^({right_component}))"
                ),
                origin: EilenbergMacLaneIntegralGeneratorOrigin::CyclicTorClass {
                    left_component,
                    right_component,
                    order: torsion[0].clone(),
                },
            });
        }

        let arity = generators.len();
        let mut relations =
            Vec::with_capacity(free_rank + torsion.len() + 2 * usize::from(torsion.len() == 2));
        for generator in 0..free_rank {
            let mut exponents = vec![0; arity];
            exponents[generator] = 2;
            relations.push(EilenbergMacLaneIntegralCohomologyRelation {
                terms: vec![EilenbergMacLaneIntegralCohomologyRelationTerm {
                    coefficient: Integer::one(),
                    exponents,
                }],
            });
        }
        for (torsion_index, modulus) in torsion.iter().enumerate() {
            let mut exponents = vec![0; arity];
            exponents[free_rank + torsion_index] = 1;
            relations.push(EilenbergMacLaneIntegralCohomologyRelation {
                terms: vec![EilenbergMacLaneIntegralCohomologyRelationTerm {
                    coefficient: modulus.clone(),
                    exponents,
                }],
            });
        }
        if torsion.len() == 2 {
            let tor_generator = arity - 1;
            relations.push(EilenbergMacLaneIntegralCohomologyRelation {
                terms: vec![EilenbergMacLaneIntegralCohomologyRelationTerm {
                    coefficient: torsion[0].clone(),
                    exponents: generator_power(arity, tor_generator, 1),
                }],
            });

            let two = Integer::from(2);
            let (quotient, quotient_remainder) = Z
                .div_rem(&torsion[1], &torsion[0])
                .expect("a canonical torsion factor is nonzero");
            debug_assert!(quotient_remainder.is_zero());
            let (half_left, left_remainder) = Z.div_rem(&torsion[0], &two).expect("two is nonzero");
            let (half_right, right_remainder) =
                Z.div_rem(&torsion[1], &two).expect("two is nonzero");
            let (_, quotient_parity) = Z.div_rem(&quotient, &two).expect("two is nonzero");

            let mut terms = vec![EilenbergMacLaneIntegralCohomologyRelationTerm {
                coefficient: Integer::one(),
                exponents: generator_power(arity, tor_generator, 2),
            }];
            if left_remainder.is_zero()
                && right_remainder.is_zero()
                && quotient_parity == Integer::one()
            {
                let mut left_squared_right = vec![0; arity];
                left_squared_right[free_rank] = 2;
                left_squared_right[free_rank + 1] = 1;
                terms.push(EilenbergMacLaneIntegralCohomologyRelationTerm {
                    coefficient: -half_left,
                    exponents: left_squared_right,
                });

                let mut left_right_squared = vec![0; arity];
                left_right_squared[free_rank] = 1;
                left_right_squared[free_rank + 1] = 2;
                terms.push(EilenbergMacLaneIntegralCohomologyRelationTerm {
                    coefficient: -half_right,
                    exponents: left_right_squared,
                });
            }
            relations.push(EilenbergMacLaneIntegralCohomologyRelation { terms });
        }

        return Ok(EilenbergMacLaneIntegralCohomologyPresentation {
            space: space.clone(),
            generators,
            relations,
            strategy: if torsion.len() == 2 {
                EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndRankTwoAbelianClassifyingSpace
            } else {
                EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndCyclicClassifyingSpace
            },
        });
    }

    if space.delooping_degree == 2 && space.group.torsion_coefficients().is_empty() {
        let generators = (1..=space.group.free_rank())
            .map(|component| EilenbergMacLaneIntegralCohomologyGenerator {
                id: format!("u2_{component}"),
                degree: 2,
                display_name: if space.group.free_rank() == 1 {
                    "iota_2".to_owned()
                } else {
                    format!("iota_2^({component})")
                },
                origin: EilenbergMacLaneIntegralGeneratorOrigin::SecondDeloopingFundamentalClass {
                    component,
                },
            })
            .collect();
        return Ok(EilenbergMacLaneIntegralCohomologyPresentation {
            space: space.clone(),
            generators,
            relations: Vec::new(),
            strategy:
                EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAbelianSecondDelooping,
        });
    }

    Err(UnsupportedOperationError::new(
        INTEGRAL_COHOMOLOGY_PRESENTATION_OPERATION,
        space.to_string(),
    )
    .with_supported_contexts([
        "K(Z^r + C_m + C_n,1) with at most two nontrivial cyclic summands and m dividing n",
        "K(Z^r,2)",
        "K(0,n)",
    ])
    .with_detail(
        "three or more torsion factors have additional interacting Tor classes, while higher deloopings require a separate integral-ring playbook",
    )
    .into())
}

fn compute_integral_homology(
    space: &EilenbergMacLaneSpace,
    max_degree: usize,
) -> Result<
    (
        Vec<FinitelyGeneratedAbelianGroup>,
        EilenbergMacLaneHomologyStrategy,
    ),
    EilenbergMacLaneError,
> {
    if space.group.is_trivial() {
        let mut groups = vec![FinitelyGeneratedAbelianGroup::trivial(); max_degree + 1];
        groups[0] = FinitelyGeneratedAbelianGroup::free(1);
        return Ok((groups, EilenbergMacLaneHomologyStrategy::TrivialSpace));
    }
    if space.delooping_degree == 1 {
        return Ok((
            classifying_space_homology(&space.group, max_degree)?,
            EilenbergMacLaneHomologyStrategy::ClassifyingSpaceKunneth,
        ));
    }
    if space.delooping_degree == 2 && space.group.torsion_coefficients().is_empty() {
        return Ok((
            free_second_delooping_homology(space.group.free_rank(), max_degree)?,
            EilenbergMacLaneHomologyStrategy::FreeAbelianSecondDelooping,
        ));
    }
    Err(UnsupportedOperationError::new(INTEGRAL_HOMOLOGY_OPERATION, space.to_string())
        .with_supported_contexts([
            "K(A,1) for finitely generated abelian A",
            "K(Z^r,2)",
            "K(0,n)",
        ])
        .with_detail("higher deloopings and finite torsion in degree 2 are not yet in the native exact playbook")
        .into())
}

fn classifying_space_homology(
    group: &FinitelyGeneratedAbelianGroup,
    max_degree: usize,
) -> Result<Vec<FinitelyGeneratedAbelianGroup>, AbelianGroupError> {
    let mut result = point_homology(max_degree);
    let circle = circle_homology(max_degree);
    for _ in 0..group.free_rank() {
        result = kunneth_product(&result, &circle)?;
    }
    for order in group.torsion_coefficients() {
        let cyclic = cyclic_classifying_space_homology(order.clone(), max_degree)?;
        result = kunneth_product(&result, &cyclic)?;
    }
    Ok(result)
}

fn free_second_delooping_homology(
    rank: usize,
    max_degree: usize,
) -> Result<Vec<FinitelyGeneratedAbelianGroup>, AbelianGroupError> {
    let mut result = point_homology(max_degree);
    let projective_space = (0..=max_degree)
        .map(|degree| {
            if degree % 2 == 0 {
                FinitelyGeneratedAbelianGroup::free(1)
            } else {
                FinitelyGeneratedAbelianGroup::trivial()
            }
        })
        .collect::<Vec<_>>();
    for _ in 0..rank {
        result = kunneth_product(&result, &projective_space)?;
    }
    Ok(result)
}

fn point_homology(max_degree: usize) -> Vec<FinitelyGeneratedAbelianGroup> {
    let mut result = vec![FinitelyGeneratedAbelianGroup::trivial(); max_degree + 1];
    result[0] = FinitelyGeneratedAbelianGroup::free(1);
    result
}

fn circle_homology(max_degree: usize) -> Vec<FinitelyGeneratedAbelianGroup> {
    let mut result = point_homology(max_degree);
    if max_degree >= 1 {
        result[1] = FinitelyGeneratedAbelianGroup::free(1);
    }
    result
}

fn cyclic_classifying_space_homology(
    order: crate::Integer,
    max_degree: usize,
) -> Result<Vec<FinitelyGeneratedAbelianGroup>, AbelianGroupError> {
    (0..=max_degree)
        .map(|degree| {
            if degree == 0 {
                Ok(FinitelyGeneratedAbelianGroup::free(1))
            } else if degree % 2 == 1 {
                FinitelyGeneratedAbelianGroup::cyclic(order.clone())
            } else {
                Ok(FinitelyGeneratedAbelianGroup::trivial())
            }
        })
        .collect()
}

fn kunneth_product(
    left: &[FinitelyGeneratedAbelianGroup],
    right: &[FinitelyGeneratedAbelianGroup],
) -> Result<Vec<FinitelyGeneratedAbelianGroup>, AbelianGroupError> {
    let max_degree = left.len().min(right.len()).saturating_sub(1);
    let mut result = Vec::with_capacity(max_degree + 1);
    for degree in 0..=max_degree {
        let mut group = FinitelyGeneratedAbelianGroup::trivial();
        for left_degree in 0..=degree {
            let summand = left[left_degree].tensor_product(&right[degree - left_degree])?;
            group = group.direct_sum(&summand)?;
        }
        if degree > 0 {
            for left_degree in 0..degree {
                let summand = left[left_degree].tor_1(&right[degree - 1 - left_degree]);
                group = group.direct_sum(&summand)?;
            }
        }
        result.push(group);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::{Integer, SemanticType};

    use super::*;

    fn group(free_rank: usize, torsion: &[i64]) -> FinitelyGeneratedAbelianGroup {
        FinitelyGeneratedAbelianGroup::from_cyclic_summands(
            free_rank,
            torsion.iter().copied().map(Integer::from).collect(),
        )
        .unwrap()
    }

    #[test]
    fn cyclic_classifying_space_has_odd_periodic_homology() {
        let space = EilenbergMacLaneSpace::new(group(0, &[6]), 1).unwrap();
        let homology = space.integral_homology(6).unwrap();
        assert_eq!(
            homology.strategy(),
            EilenbergMacLaneHomologyStrategy::ClassifyingSpaceKunneth
        );
        assert_eq!(
            homology.groups(),
            &[
                group(1, &[]),
                group(0, &[6]),
                group(0, &[]),
                group(0, &[6]),
                group(0, &[]),
                group(0, &[6]),
                group(0, &[])
            ]
        );
        homology.verify().unwrap();

        let cohomology = space.integral_cohomology(6).unwrap();
        assert_eq!(
            cohomology.groups(),
            &[
                group(1, &[]),
                group(0, &[]),
                group(0, &[6]),
                group(0, &[]),
                group(0, &[6]),
                group(0, &[]),
                group(0, &[6])
            ]
        );
        cohomology.verify().unwrap();
    }

    #[test]
    fn products_use_integral_kunneth_tensor_and_tor_terms() {
        let elementary = EilenbergMacLaneSpace::new(group(0, &[2, 2]), 1)
            .unwrap()
            .integral_homology(3)
            .unwrap();
        assert_eq!(elementary.group(1), Some(&group(0, &[2, 2])));
        assert_eq!(elementary.group(2), Some(&group(0, &[2])));
        assert_eq!(elementary.group(3), Some(&group(0, &[2, 2, 2])));

        let torus = EilenbergMacLaneSpace::new(group(2, &[]), 1)
            .unwrap()
            .integral_homology(3)
            .unwrap();
        assert_eq!(
            torus.groups(),
            &[group(1, &[]), group(2, &[]), group(1, &[]), group(0, &[])]
        );
    }

    #[test]
    fn free_second_delooping_is_a_product_of_infinite_projective_spaces() {
        let result = EilenbergMacLaneSpace::new(group(2, &[]), 2)
            .unwrap()
            .integral_homology(6)
            .unwrap();
        assert_eq!(
            result.strategy(),
            EilenbergMacLaneHomologyStrategy::FreeAbelianSecondDelooping
        );
        assert_eq!(
            result.groups(),
            &[
                group(1, &[]),
                group(0, &[]),
                group(2, &[]),
                group(0, &[]),
                group(3, &[]),
                group(0, &[]),
                group(4, &[])
            ]
        );
    }

    #[test]
    fn unsupported_regimes_fail_closed_and_json_replays_results() {
        let unsupported = EilenbergMacLaneSpace::new(group(0, &[2]), 2).unwrap();
        assert!(matches!(
            unsupported.integral_homology(8),
            Err(EilenbergMacLaneError::UnsupportedOperation(_))
        ));

        let space = EilenbergMacLaneSpace::new(group(1, &[4]), 1).unwrap();
        assert_eq!(
            EilenbergMacLaneSpace::from_json(&space.to_json().unwrap()).unwrap(),
            space
        );
        let result = space.integral_homology(8).unwrap();
        assert_eq!(
            EilenbergMacLaneIntegralHomology::from_json(&result.to_json().unwrap()).unwrap(),
            result
        );
        let cohomology = space.integral_cohomology(8).unwrap();
        assert_eq!(
            EilenbergMacLaneIntegralCohomology::from_json(&cohomology.to_json().unwrap()).unwrap(),
            cohomology
        );
    }

    #[test]
    fn integral_classifying_space_presentation_names_bockstein_and_odd_relations() {
        let result = EilenbergMacLaneSpace::new(group(1, &[6]), 1)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(
            result.strategy(),
            EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndCyclicClassifyingSpace
        );
        assert_eq!(result.coefficient_ring(), "Z");
        assert_eq!(result.generators().len(), 2);
        assert_eq!(result.generators()[0].id(), "x1_1");
        assert_eq!(result.generators()[0].degree(), 1);
        assert_eq!(result.generators()[1].display_name(), "beta_6(iota_1^(2))");
        assert_eq!(result.relations().len(), 2);
        assert_eq!(
            result.relations()[0].terms()[0].coefficient(),
            &Integer::one()
        );
        assert_eq!(result.relations()[0].terms()[0].exponents(), &[2, 0]);
        assert_eq!(
            result.relations()[1].terms()[0].coefficient(),
            &Integer::from(6)
        );
        assert_eq!(result.relations()[1].terms()[0].exponents(), &[0, 1]);
        assert_eq!(result.additive_group(3).unwrap(), group(0, &[6]));
        result.verify().unwrap();
        assert_eq!(
            EilenbergMacLaneIntegralCohomologyPresentation::from_json(&result.to_json().unwrap())
                .unwrap(),
            result
        );
    }

    #[test]
    fn integral_free_second_delooping_presentation_is_polynomial() {
        let result = EilenbergMacLaneSpace::new(group(2, &[]), 2)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(
            result.strategy(),
            EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAbelianSecondDelooping
        );
        assert_eq!(result.generators().len(), 2);
        assert_eq!(result.generators()[0].display_name(), "iota_2^(1)");
        assert!(result.relations().is_empty());
        assert_eq!(result.additive_group(4).unwrap(), group(3, &[]));
    }

    #[test]
    fn rank_two_integral_ring_presentation_records_tor_and_exceptional_square() {
        let elementary = EilenbergMacLaneSpace::new(group(0, &[2, 2]), 1)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(
            elementary.strategy(),
            EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndRankTwoAbelianClassifyingSpace
        );
        assert_eq!(elementary.generators().len(), 3);
        assert_eq!(elementary.generators()[2].id(), "t3_1");
        assert_eq!(elementary.generators()[2].degree(), 3);
        assert!(matches!(
            elementary.generators()[2].origin(),
            EilenbergMacLaneIntegralGeneratorOrigin::CyclicTorClass {
                left_component: 1,
                right_component: 2,
                order
            } if order == &Integer::from(2)
        ));
        assert_eq!(elementary.relations().len(), 4);
        assert_eq!(
            elementary.relations()[3]
                .terms()
                .iter()
                .map(|term| (term.coefficient().clone(), term.exponents().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (Integer::one(), vec![0, 0, 2]),
                (Integer::from(-1), vec![2, 1, 0]),
                (Integer::from(-1), vec![1, 2, 0]),
            ]
        );
        assert_eq!(elementary.additive_group(3).unwrap(), group(0, &[2]));
        assert_eq!(
            elementary.additive_group(6).unwrap(),
            group(0, &[2, 2, 2, 2])
        );
        assert_eq!(
            EilenbergMacLaneIntegralCohomologyPresentation::from_json(
                &elementary.to_json().unwrap()
            )
            .unwrap(),
            elementary
        );

        let c2_c4 = EilenbergMacLaneSpace::new(group(0, &[2, 4]), 1)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(c2_c4.relations()[3].terms().len(), 1);
        assert_eq!(c2_c4.relations()[3].terms()[0].exponents(), &[0, 0, 2]);

        let c4_c12 = EilenbergMacLaneSpace::new(group(0, &[4, 12]), 1)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(
            c4_c12.relations()[3]
                .terms()
                .iter()
                .map(|term| term.coefficient().clone())
                .collect::<Vec<_>>(),
            vec![Integer::one(), Integer::from(-2), Integer::from(-6)]
        );
    }

    #[test]
    fn integral_ring_presentation_fails_closed_for_three_torsion_factors() {
        let unsupported = EilenbergMacLaneSpace::new(group(0, &[2, 2, 2]), 1).unwrap();
        assert!(matches!(
            unsupported.integral_cohomology_presentation(),
            Err(EilenbergMacLaneError::UnsupportedOperation(_))
        ));

        let point = EilenbergMacLaneSpace::new(group(0, &[]), 4)
            .unwrap()
            .integral_cohomology_presentation()
            .unwrap();
        assert_eq!(
            point.strategy(),
            EilenbergMacLaneIntegralCohomologyPresentationStrategy::TrivialSpace
        );
        assert!(point.generators().is_empty());
        assert_eq!(point.additive_group(0).unwrap(), group(1, &[]));
    }

    #[test]
    fn elementary_binary_classifying_space_is_polynomial_on_degree_one() {
        let result = EilenbergMacLaneSpace::new(group(0, &[2]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(result.generators().len(), 1);
        assert_eq!(result.generators()[0].degree(), 1);
        assert_eq!(result.generators()[0].display_name(), "iota_1");
        assert!(result.presentation().relations().is_empty());
        assert_eq!(
            result.presentation().hilbert_function(6).unwrap(),
            vec![1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(result.operations().len(), 1);
        assert_eq!(result.operations()[0].operation().display_name(), "Sq^1");
        assert!(!result.operations()[0].target().is_zero());
    }

    #[test]
    fn higher_binary_cyclic_class_has_exterior_and_periodic_generators() {
        let result = EilenbergMacLaneSpace::new(group(0, &[4]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(result.presentation().relations().len(), 1);
        assert_eq!(
            result.presentation().hilbert_function(6).unwrap(),
            vec![1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(result.operations()[0].operation().display_name(), "Sq^1");
        assert!(result.operations()[0].target().is_zero());
        assert_eq!(result.operations()[1].operation().display_name(), "Sq^2");
        assert!(!result.operations()[1].target().is_zero());
    }

    #[test]
    fn odd_primary_bockstein_detects_only_the_first_primary_power() {
        let c3 = EilenbergMacLaneSpace::new(group(0, &[3]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert_eq!(c3.generators().len(), 2);
        assert!(matches!(
            c3.generators()[1].origin(),
            EilenbergMacLaneGeneratorOrigin::Bockstein {
                source_generator: 0
            }
        ));
        assert_eq!(c3.operations()[0].operation().display_name(), "beta");
        assert!(!c3.operations()[0].target().is_zero());
        assert_eq!(c3.operations()[1].operation().display_name(), "P^1");

        let c9 = EilenbergMacLaneSpace::new(group(0, &[9]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert!(matches!(
            c9.generators()[1].origin(),
            EilenbergMacLaneGeneratorOrigin::PeriodicCyclicClass {
                primary_exponent: 2,
                ..
            }
        ));
        assert!(c9.operations()[0].target().is_zero());
    }

    #[test]
    fn products_and_free_deloopings_have_closed_form_hilbert_functions() {
        let elementary = EilenbergMacLaneSpace::new(group(0, &[2, 2]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            elementary.presentation().hilbert_function(5).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );

        let circle = EilenbergMacLaneSpace::new(group(1, &[]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            circle.presentation().hilbert_function(4).unwrap(),
            vec![1, 1, 0, 0, 0]
        );

        let projective = EilenbergMacLaneSpace::new(group(1, &[]), 2)
            .unwrap()
            .modular_cohomology(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert_eq!(
            projective.presentation().hilbert_function(6).unwrap(),
            vec![1, 0, 1, 0, 1, 0, 1]
        );
        assert_eq!(projective.operations()[0].operation().display_name(), "P^1");
    }

    #[test]
    fn modular_playbook_ignores_prime_to_p_torsion_and_fails_closed() {
        let irrelevant = EilenbergMacLaneSpace::new(group(0, &[2]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert!(irrelevant.generators().is_empty());
        assert_eq!(
            irrelevant.presentation().hilbert_function(3).unwrap(),
            vec![1, 0, 0, 0]
        );

        let unsupported = EilenbergMacLaneSpace::new(group(0, &[2]), 2).unwrap();
        assert!(matches!(
            unsupported.modular_cohomology(FiniteField::prime(2_u8).unwrap()),
            Err(EilenbergMacLaneError::UnsupportedOperation(_))
        ));
        let extension = FiniteField::extension(2, 2).unwrap();
        assert!(matches!(
            EilenbergMacLaneSpace::new(group(1, &[]), 1)
                .unwrap()
                .modular_cohomology(extension),
            Err(EilenbergMacLaneError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn modular_json_replays_the_complete_presentation() {
        let result = EilenbergMacLaneSpace::new(group(1, &[4]), 1)
            .unwrap()
            .modular_cohomology(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        let json = result.to_json().unwrap();
        assert_eq!(
            EilenbergMacLaneModularCohomology::from_json(&json).unwrap(),
            result
        );
        assert!(!result.canonical_bytes().unwrap().is_empty());
    }

    #[test]
    fn integral_fundamental_classes_use_cartan_serre_words_without_sq_one() {
        let field = FiniteField::prime(2_u8).unwrap();
        let k_z_3 = EilenbergMacLaneSpace::new(group(1, &[]), 3)
            .unwrap()
            .modular_cohomology_through_degree(field.clone(), 16)
            .unwrap();
        assert_eq!(
            k_z_3
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![3, 5, 9]
        );
        assert_eq!(
            k_z_3
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::display_name)
                .collect::<Vec<_>>(),
            vec!["iota_3", "Sq^2(iota_3)", "Sq^4 Sq^2(iota_3)"]
        );
        assert_eq!(
            k_z_3.strategy(),
            EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleSquares
        );

        let k_z_4 = EilenbergMacLaneSpace::new(group(1, &[]), 4)
            .unwrap()
            .modular_cohomology_through_degree(field, 16)
            .unwrap();
        assert_eq!(
            k_z_4
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![4, 6, 7, 10, 11, 13]
        );
    }

    #[test]
    fn elementary_binary_classes_include_sq_one_towers() {
        let result = EilenbergMacLaneSpace::new(group(0, &[2]), 2)
            .unwrap()
            .modular_cohomology_through_degree(FiniteField::prime(2_u8).unwrap(), 16)
            .unwrap();
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3, 5, 9]
        );
        assert!(matches!(
            result.generators()[1].origin(),
            EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquare {
                operation,
                fundamental_class_lifts_integrally: false,
                ..
            } if operation.square_sequence() == Some([1].as_slice())
        ));
        assert_eq!(result.exact_through_degree(), 16);
        assert!(result.presentation().relations().is_empty());
    }

    #[test]
    fn higher_binary_primary_classes_use_two_sq_one_quotient_seeds() {
        let field = FiniteField::prime(2_u8).unwrap();
        let higher = EilenbergMacLaneSpace::new(group(0, &[4]), 2)
            .unwrap()
            .modular_cohomology_through_degree(field.clone(), 16)
            .unwrap();
        assert_eq!(
            higher
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3, 5, 9]
        );
        assert_eq!(
            higher
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::display_name)
                .collect::<Vec<_>>(),
            vec![
                "iota_2",
                "beta_2(iota_2)",
                "Sq^2(beta_2(iota_2))",
                "Sq^4 Sq^2(beta_2(iota_2))"
            ]
        );
        assert!(matches!(
            higher.generators()[1].origin(),
            EilenbergMacLaneGeneratorOrigin::HigherBockstein {
                component: 1,
                primary_exponent: 2
            }
        ));
        assert!(matches!(
            higher.generators()[2].origin(),
            EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquareOnHigherPrimaryClass {
                source_is_higher_bockstein: true,
                operation,
                ..
            } if operation.square_sequence() == Some([2].as_slice())
        ));
        let elementary = EilenbergMacLaneSpace::new(group(0, &[2]), 2)
            .unwrap()
            .modular_cohomology_through_degree(field, 16)
            .unwrap();
        assert_eq!(
            higher.presentation().hilbert_function(16).unwrap(),
            elementary.presentation().hilbert_function(16).unwrap()
        );
        assert_eq!(
            EilenbergMacLaneModularCohomologyTruncation::from_json(&higher.to_json().unwrap())
                .unwrap(),
            higher
        );
    }

    #[test]
    fn elementary_odd_primary_classes_use_admissible_power_bockstein_words() {
        let result = EilenbergMacLaneSpace::new(group(0, &[3]), 2)
            .unwrap()
            .modular_cohomology_through_degree(FiniteField::prime(3_u8).unwrap(), 16)
            .unwrap();
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3, 7, 8]
        );
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::display_name)
                .collect::<Vec<_>>(),
            vec![
                "iota_2",
                "beta(iota_2)",
                "P^1 beta(iota_2)",
                "beta P^1 beta(iota_2)"
            ]
        );
        assert!(matches!(
            result.generators()[2].origin(),
            EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperation {
                operation,
                fundamental_class_lifts_integrally: false,
                ..
            } if operation.bocksteins() == Some([false, true].as_slice())
                && operation.reduced_powers() == Some([1].as_slice())
        ));
        assert_eq!(
            result.strategy(),
            EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleOddPrimaryOperations
        );
        assert_eq!(
            EilenbergMacLaneModularCohomologyTruncation::from_json(&result.to_json().unwrap())
                .unwrap(),
            result
        );
    }

    #[test]
    fn integral_odd_primary_fundamental_classes_have_no_terminal_bockstein() {
        let result = EilenbergMacLaneSpace::new(group(1, &[]), 3)
            .unwrap()
            .modular_cohomology_through_degree(FiniteField::prime(3_u8).unwrap(), 16)
            .unwrap();
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![3, 7, 8]
        );
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::display_name)
                .collect::<Vec<_>>(),
            vec!["iota_3", "P^1(iota_3)", "beta P^1(iota_3)"]
        );
        assert!(result.generators().iter().all(|generator| {
            !matches!(
                generator.origin(),
                EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperation {
                    operation,
                    ..
                } if operation.bocksteins().and_then(|values| values.last()) == Some(&true)
            )
        }));
    }

    #[test]
    fn odd_primary_bounded_lane_respects_degree_and_higher_bockstein_seeds() {
        let result = EilenbergMacLaneSpace::new(group(0, &[17]), 2)
            .unwrap()
            .modular_cohomology_through_degree(FiniteField::prime(17_u8).unwrap(), 16)
            .unwrap();
        assert_eq!(
            result
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        let field = FiniteField::prime(3_u8).unwrap();
        let higher = EilenbergMacLaneSpace::new(group(0, &[9]), 2)
            .unwrap()
            .modular_cohomology_through_degree(field.clone(), 16)
            .unwrap();
        assert_eq!(
            higher
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::degree)
                .collect::<Vec<_>>(),
            vec![2, 3, 7, 8]
        );
        assert_eq!(
            higher
                .generators()
                .iter()
                .map(EilenbergMacLaneCohomologyGenerator::display_name)
                .collect::<Vec<_>>(),
            vec![
                "iota_2",
                "beta_2(iota_2)",
                "P^1(beta_2(iota_2))",
                "beta P^1(beta_2(iota_2))"
            ]
        );
        assert!(matches!(
            higher.generators()[3].origin(),
            EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
                primary_exponent: 2,
                source_is_higher_bockstein: true,
                operation,
                ..
            } if operation.bocksteins() == Some([true, false].as_slice())
                && operation.reduced_powers() == Some([1].as_slice())
        ));
        let elementary = EilenbergMacLaneSpace::new(group(0, &[3]), 2)
            .unwrap()
            .modular_cohomology_through_degree(field, 16)
            .unwrap();
        assert_eq!(
            higher.presentation().hilbert_function(16).unwrap(),
            elementary.presentation().hilbert_function(16).unwrap()
        );
    }

    #[test]
    fn bounded_modular_product_metadata_and_json_are_canonical() {
        let result = EilenbergMacLaneSpace::new(group(1, &[6]), 3)
            .unwrap()
            .modular_cohomology_through_degree(FiniteField::prime(2_u8).unwrap(), 9)
            .unwrap();
        assert_eq!(result.generators()[0].display_name(), "iota_3.1");
        assert_eq!(result.generators()[1].display_name(), "iota_3.2");
        assert!(result
            .generators()
            .iter()
            .any(|generator| generator.display_name() == "Sq^1(iota_3.2)"));
        let json = result.to_json().unwrap();
        assert_eq!(
            EilenbergMacLaneModularCohomologyTruncation::from_json(&json).unwrap(),
            result
        );
        assert!(!result.canonical_bytes().unwrap().is_empty());
    }

    #[test]
    fn bounded_modular_playbook_fails_closed_outside_its_theorem() {
        let field = FiniteField::prime(2_u8).unwrap();
        let below_connectivity = EilenbergMacLaneSpace::new(group(1, &[]), 4)
            .unwrap()
            .modular_cohomology_through_degree(field, 3)
            .unwrap();
        assert!(below_connectivity.generators().is_empty());
        assert_eq!(
            below_connectivity
                .presentation()
                .hilbert_function(3)
                .unwrap(),
            vec![1, 0, 0, 0]
        );
    }
}
