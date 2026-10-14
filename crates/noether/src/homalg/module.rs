// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Based free modules, module maps, submodules, and presentations.

use core::fmt;
use core::ops::{Add, Neg, Sub};
use std::collections::HashSet;
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{Degree, GradingError, MonomialOrder};
use crate::linalg::{LinalgError, LinearOperator, Matrix, SparseMatrix, Vector};
use crate::{
    CoefficientMap, CoefficientMappingError, CommutativeRing, ContextMismatchError, NoetherError,
    Ring, SemanticType, TypeDescription,
};

use super::module_groebner::ModuleMonomial;

mod exact_submodule;

pub use exact_submodule::{ImageSubmodule, KernelSubmodule, SubmoduleMembership};

static FREE_MODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FreeModule@1",
    rust_path: "noether::homalg::FreeModule",
    python_path: Some("noether.homalg.FreeModule"),
};

static MODULE_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleElement@1",
    rust_path: "noether::homalg::ModuleElement",
    python_path: Some("noether.homalg.ModuleElement"),
};

static MODULE_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleHomomorphism@1",
    rust_path: "noether::homalg::ModuleHomomorphism",
    python_path: Some("noether.homalg.ModuleHomomorphism"),
};

static SUBMODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/Submodule@1",
    rust_path: "noether::homalg::Submodule",
    python_path: Some("noether.homalg.Submodule"),
};

static PRESENTED_MODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PresentedModule@1",
    rust_path: "noether::homalg::PresentedModule",
    python_path: Some("noether.homalg.PresentedModule"),
};

static PRESENTED_MODULE_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PresentedModuleHomomorphism@1",
    rust_path: "noether::homalg::PresentedModuleHomomorphism",
    python_path: Some("noether.homalg.PresentedModuleHomomorphism"),
};

static PRESENTED_MODULE_HOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PresentedModuleHomology@1",
    rust_path: "noether::homalg::PresentedModuleHomology",
    python_path: Some("noether.homalg.PresentedModuleHomology"),
};

static MODULE_MONOMIAL_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleMonomialOrder@1",
    rust_path: "noether::homalg::ModuleMonomialOrder",
    python_path: Some("noether.homalg.ModuleMonomialOrder"),
};

/// Failure to construct or apply a module value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleError {
    /// Two values carried unequal coefficient contexts.
    ContextMismatch(ContextMismatchError),
    /// A list of basis labels had the wrong length.
    BasisLabelCount {
        /// Rank of the free module.
        rank: usize,
        /// Number of labels supplied.
        actual: usize,
    },
    /// One basis label was empty.
    EmptyBasisLabel {
        /// Zero-based basis position.
        position: usize,
    },
    /// One basis label occurred more than once.
    DuplicateBasisLabel {
        /// Repeated label.
        label: String,
    },
    /// A list of degree shifts had the wrong length.
    DegreeShiftCount {
        /// Rank of the free module.
        rank: usize,
        /// Number of shifts supplied.
        actual: usize,
    },
    /// One shift had the wrong free-abelian rank.
    DegreeRank {
        /// Zero-based basis position.
        position: usize,
        /// Declared grading rank.
        expected: usize,
        /// Rank of the supplied degree.
        actual: usize,
    },
    /// An operation mixed graded and ungraded modules, or unequal grading
    /// ranks.
    GradingMismatch {
        /// Stable operation name.
        operation: &'static str,
    },
    /// A coordinate vector had the wrong dimension.
    CoordinateDimension {
        /// Rank of the ambient free module.
        expected: usize,
        /// Received coordinate count.
        actual: usize,
    },
    /// A requested basis position was outside the module rank.
    BasisPosition {
        /// Rejected zero-based position.
        position: usize,
        /// Module rank.
        rank: usize,
    },
    /// A presentation matrix did not have target-rank by source-rank shape.
    MatrixShape {
        /// Required matrix shape.
        expected: (usize, usize),
        /// Received matrix shape.
        actual: (usize, usize),
    },
    /// A free-module rank operation overflowed `usize`.
    RankOverflow {
        /// Stable operation name.
        operation: &'static str,
    },
    /// Composition or application used incompatible based modules.
    ModuleMismatch {
        /// Stable operation name.
        operation: &'static str,
    },
    /// A lattice has infinite index in its ambient free abelian group.
    InfiniteIndex {
        /// Stable operation name.
        operation: &'static str,
    },
    /// A proposed map of presented modules did not carry the retained source
    /// relations into the retained target relations through its certificate.
    RelationDiagramDoesNotCommute,
    /// Underlying exact linear algebra failed.
    LinearAlgebra(LinalgError),
    /// A checked coefficient map failed.
    CoefficientMapping(CoefficientMappingError),
    /// Degree arithmetic failed.
    Grading(GradingError),
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::BasisLabelCount { rank, actual } => {
                write!(
                    f,
                    "free module of rank {rank} received {actual} basis labels"
                )
            }
            Self::EmptyBasisLabel { position } => {
                write!(f, "basis label at position {position} is empty")
            }
            Self::DuplicateBasisLabel { label } => {
                write!(f, "basis label {label:?} occurs more than once")
            }
            Self::DegreeShiftCount { rank, actual } => {
                write!(
                    f,
                    "free module of rank {rank} received {actual} degree shifts"
                )
            }
            Self::DegreeRank {
                position,
                expected,
                actual,
            } => write!(
                f,
                "degree shift at position {position} has rank {actual}; expected {expected}"
            ),
            Self::GradingMismatch { operation } => {
                write!(f, "{operation} requires compatible module gradings")
            }
            Self::CoordinateDimension { expected, actual } => write!(
                f,
                "module element has {actual} coordinates; expected {expected}"
            ),
            Self::BasisPosition { position, rank } => {
                write!(f, "basis position {position} is outside module rank {rank}")
            }
            Self::MatrixShape { expected, actual } => write!(
                f,
                "module map matrix has shape {}x{}; expected {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::RankOverflow { operation } => {
                write!(f, "free-module rank overflow in {operation}")
            }
            Self::ModuleMismatch { operation } => {
                write!(f, "{operation} requires equal based modules")
            }
            Self::InfiniteIndex { operation } => {
                write!(
                    f,
                    "{operation} is undefined because the quotient has positive free rank"
                )
            }
            Self::RelationDiagramDoesNotCommute => write!(
                f,
                "presented-module generator and relation lifts do not form a commuting square"
            ),
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::CoefficientMapping(error) => error.fmt(f),
            Self::Grading(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ModuleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::CoefficientMapping(error) => Some(error),
            Self::Grading(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinalgError> for ModuleError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<ContextMismatchError> for ModuleError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<CoefficientMappingError> for ModuleError {
    fn from(value: CoefficientMappingError) -> Self {
        Self::CoefficientMapping(value)
    }
}

impl From<GradingError> for ModuleError {
    fn from(value: GradingError) -> Self {
        Self::Grading(value)
    }
}

/// A finite based free module over one explicit coefficient ring.
///
/// Basis labels and degree shifts are optional presentation data. When shifts
/// are present, their common free-abelian rank is explicit even for the zero
/// module.
#[derive(Clone)]
pub struct FreeModule<R: Ring> {
    ring: Arc<R>,
    rank: usize,
    basis_labels: Option<Vec<String>>,
    grading_rank: Option<usize>,
    degree_shifts: Option<Vec<Degree>>,
}

impl<R: Ring> PartialEq for FreeModule<R> {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.ring, &other.ring) || self.ring == other.ring)
            && self.rank == other.rank
            && self.basis_labels == other.basis_labels
            && self.grading_rank == other.grading_rank
            && self.degree_shifts == other.degree_shifts
    }
}

impl<R: Ring> Eq for FreeModule<R> {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "R: Ring + Deserialize<'de>"))]
struct FreeModuleWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    ring: R,
    rank: usize,
    basis_labels: Option<Vec<String>>,
    grading_rank: Option<usize>,
    degree_shifts: Option<Vec<Degree>>,
}

impl<R: Ring> FreeModule<R> {
    /// Construct an ungraded rank-`rank` free module with its standard ordered
    /// basis.
    pub fn new(ring: R, rank: usize) -> Self {
        Self {
            ring: Arc::new(ring),
            rank,
            basis_labels: None,
            grading_rank: None,
            degree_shifts: None,
        }
    }

    /// Construct a free module with checked optional basis metadata.
    pub fn with_basis_data(
        ring: R,
        rank: usize,
        basis_labels: Option<Vec<String>>,
        grading_rank: Option<usize>,
        degree_shifts: Option<Vec<Degree>>,
    ) -> Result<Self, ModuleError> {
        validate_labels(rank, basis_labels.as_deref())?;
        validate_shifts(rank, grading_rank, degree_shifts.as_deref())?;
        Ok(Self {
            ring: Arc::new(ring),
            rank,
            basis_labels,
            grading_rank,
            degree_shifts,
        })
    }

    /// Exact coefficient context.
    pub fn ring(&self) -> &R {
        self.ring.as_ref()
    }

    /// Number of ordered basis vectors.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Optional ordered basis labels.
    pub fn basis_labels(&self) -> Option<&[String]> {
        self.basis_labels.as_deref()
    }

    /// Rank of the grading group when degree shifts are present.
    pub fn grading_rank(&self) -> Option<usize> {
        self.grading_rank
    }

    /// Optional degree shift of every ordered basis vector.
    pub fn degree_shifts(&self) -> Option<&[Degree]> {
        self.degree_shifts.as_deref()
    }

    /// Whether this based module carries degree shifts.
    pub fn is_graded(&self) -> bool {
        self.grading_rank.is_some()
    }

    /// Attach checked unique basis labels.
    pub fn with_basis_labels(mut self, labels: Vec<String>) -> Result<Self, ModuleError> {
        validate_labels(self.rank, Some(&labels))?;
        self.basis_labels = Some(labels);
        Ok(self)
    }

    /// Attach one checked degree shift to every basis vector.
    pub fn with_degree_shifts(
        mut self,
        grading_rank: usize,
        shifts: Vec<Degree>,
    ) -> Result<Self, ModuleError> {
        validate_shifts(self.rank, Some(grading_rank), Some(&shifts))?;
        self.grading_rank = Some(grading_rank);
        self.degree_shifts = Some(shifts);
        Ok(self)
    }

    /// Construct the requested standard basis element.
    pub fn basis_element(&self, position: usize) -> Result<ModuleElement<R>, ModuleError> {
        if position >= self.rank {
            return Err(ModuleError::BasisPosition {
                position,
                rank: self.rank,
            });
        }
        let mut coordinates = vec![self.ring.zero(); self.rank];
        coordinates[position] = self.ring.one();
        ModuleElement::from_coordinates(
            self.clone(),
            Vector::from_values(self.ring().clone(), coordinates)?,
        )
    }

    /// Direct sum of two based free modules over the same exact ring context.
    pub fn direct_sum(&self, rhs: &Self) -> Result<Self, ModuleError> {
        require_same_ring(self.ring(), rhs.ring())?;
        let rank = self
            .rank
            .checked_add(rhs.rank)
            .ok_or(ModuleError::RankOverflow {
                operation: "module direct sum",
            })?;
        let basis_labels = match (&self.basis_labels, &rhs.basis_labels) {
            (Some(left), Some(right)) => {
                let mut labels = left.clone();
                labels.extend(right.iter().cloned());
                validate_labels(rank, Some(&labels))?;
                Some(labels)
            }
            _ => None,
        };
        let (grading_rank, degree_shifts) =
            combine_gradings(self, rhs, "module direct sum", |a, b| {
                let mut shifts = a.to_vec();
                shifts.extend_from_slice(b);
                Ok(shifts)
            })?;
        Self::with_basis_data(
            self.ring().clone(),
            rank,
            basis_labels,
            grading_rank,
            degree_shifts,
        )
    }

    /// Tensor product of two based free modules over the same commutative
    /// context, with lexicographically ordered tensor basis.
    pub fn tensor(&self, rhs: &Self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        require_same_ring(self.ring(), rhs.ring())?;
        let rank = self
            .rank
            .checked_mul(rhs.rank)
            .ok_or(ModuleError::RankOverflow {
                operation: "module tensor product",
            })?;
        let basis_labels = match (&self.basis_labels, &rhs.basis_labels) {
            (Some(left), Some(right)) => Some(
                left.iter()
                    .flat_map(|left| {
                        right
                            .iter()
                            .map(move |right| format!("{left} tensor {right}"))
                    })
                    .collect(),
            ),
            _ => None,
        };
        let (grading_rank, degree_shifts) =
            combine_gradings(self, rhs, "module tensor product", |a, b| {
                a.iter()
                    .flat_map(|left| b.iter().map(move |right| left.add(right)))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ModuleError::from)
            })?;
        Self::with_basis_data(
            self.ring().clone(),
            rank,
            basis_labels,
            grading_rank,
            degree_shifts,
        )
    }

    /// Dual based free module. Degree shifts are negated.
    pub fn dual(&self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        Self::with_basis_data(
            self.ring().clone(),
            self.rank,
            self.basis_labels.clone(),
            self.grading_rank,
            self.degree_shifts
                .as_ref()
                .map(|shifts| shifts.iter().map(Degree::neg).collect()),
        )
    }

    /// Apply a checked scalar map to the coefficient context.
    pub fn map_coefficients<M>(&self, map: &M) -> Result<FreeModule<M::Target>, ModuleError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        if self.ring() != map.source() {
            return Err(context_mismatch("module coefficient map").into());
        }
        FreeModule::with_basis_data(
            map.target().clone(),
            self.rank,
            self.basis_labels.clone(),
            self.grading_rank,
            self.degree_shifts.clone(),
        )
    }

    /// Serialize the complete based module and coefficient context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a based free module.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for FreeModule<R>
where
    R: Ring + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FreeModule", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", self.ring())?;
        state.serialize_field("rank", &self.rank)?;
        state.serialize_field("basis_labels", &self.basis_labels)?;
        state.serialize_field("grading_rank", &self.grading_rank)?;
        state.serialize_field("degree_shifts", &self.degree_shifts)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for FreeModule<R>
where
    R: Ring + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FreeModuleWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::with_basis_data(
            wire.ring,
            wire.rank,
            wire.basis_labels,
            wire.grading_rank,
            wire.degree_shifts,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for FreeModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeModule")
            .field("ring_type", &R::TYPE_ID)
            .field("rank", &self.rank)
            .field("graded", &self.is_graded())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for FreeModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FreeModule<{}>({})", R::TYPE_ID, self.rank)
    }
}

impl<R: Ring> SemanticType for FreeModule<R> {
    const TYPE_ID: &'static str = "noether/homalg/FreeModule@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.ring.canonical_bytes()?)?;
        append_usize(&mut output, self.rank)?;
        append_optional_strings(&mut output, self.basis_labels.as_deref())?;
        match (self.grading_rank, self.degree_shifts.as_deref()) {
            (Some(rank), Some(shifts)) => {
                output.push(1);
                append_usize(&mut output, rank)?;
                append_usize(&mut output, shifts.len())?;
                for shift in shifts {
                    append_component(&mut output, &shift.canonical_bytes()?)?;
                }
            }
            (None, None) => output.push(0),
            _ => {
                return Err(NoetherError::Serialization(
                    "free module has inconsistent grading data".to_owned(),
                ));
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FREE_MODULE_DESCRIPTION
    }
}

/// One coordinate vector in a fixed based free module.
#[derive(Clone, PartialEq, Eq)]
pub struct ModuleElement<R: Ring> {
    module: FreeModule<R>,
    coordinates: Vector<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct ModuleElementWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    module: FreeModule<R>,
    coordinates: Vector<R>,
}

impl<R: Ring> ModuleElement<R> {
    /// Construct an element after checking context and coordinate dimension.
    pub fn from_coordinates(
        module: FreeModule<R>,
        coordinates: Vector<R>,
    ) -> Result<Self, ModuleError> {
        require_same_ring(module.ring(), coordinates.ring())?;
        if coordinates.dimension() != module.rank() {
            return Err(ModuleError::CoordinateDimension {
                expected: module.rank(),
                actual: coordinates.dimension(),
            });
        }
        Ok(Self {
            module,
            coordinates,
        })
    }

    /// Zero element of one based free module.
    pub fn zero(module: FreeModule<R>) -> Self {
        let coordinates = Vector::zeros(module.ring().clone(), module.rank());
        Self {
            module,
            coordinates,
        }
    }

    /// Ambient based free module.
    pub fn module(&self) -> &FreeModule<R> {
        &self.module
    }

    /// Exact coordinate vector in basis order.
    pub fn coordinates(&self) -> &Vector<R> {
        &self.coordinates
    }

    pub(super) fn into_coordinates(self) -> Vec<R::Element> {
        self.coordinates.into_values()
    }

    /// Add two elements of the same based module.
    pub fn add(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_module(rhs, "module element addition")?;
        Self::from_coordinates(self.module.clone(), self.coordinates.add(&rhs.coordinates)?)
    }

    /// Subtract two elements of the same based module.
    pub fn sub(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_module(rhs, "module element subtraction")?;
        Self::from_coordinates(self.module.clone(), self.coordinates.sub(&rhs.coordinates)?)
    }

    /// Negate every coordinate.
    pub fn neg(&self) -> Result<Self, ModuleError> {
        Self::from_coordinates(self.module.clone(), self.coordinates.neg()?)
    }

    /// Multiply every coordinate by a scalar in the base ring.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, ModuleError> {
        Self::from_coordinates(self.module.clone(), self.coordinates.scalar_mul(scalar)?)
    }

    /// Apply a checked scalar map coefficientwise.
    pub fn map_coefficients<M>(&self, map: &M) -> Result<ModuleElement<M::Target>, ModuleError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        ModuleElement::from_coordinates(
            self.module.map_coefficients(map)?,
            self.coordinates.map_coefficients(map)?,
        )
    }

    /// Serialize the ambient module and exact coordinates.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a module element.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_module(&self, rhs: &Self, operation: &'static str) -> Result<(), ModuleError> {
        if self.module == rhs.module {
            Ok(())
        } else {
            Err(ModuleError::ModuleMismatch { operation })
        }
    }
}

impl<R> Serialize for ModuleElement<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ModuleElement", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("module", &self.module)?;
        state.serialize_field("coordinates", &self.coordinates)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for ModuleElement<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleElementWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_coordinates(wire.module, wire.coordinates).map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for ModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleElement")
            .field("module", &self.module)
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for ModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModuleElement<{}>({} coordinates)",
            R::TYPE_ID,
            self.module.rank()
        )
    }
}

impl<R: Ring> SemanticType for ModuleElement<R> {
    const TYPE_ID: &'static str = "noether/homalg/ModuleElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.module.canonical_bytes()?)?;
        append_component(&mut output, &self.coordinates.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_ELEMENT_DESCRIPTION
    }
}

impl<R: Ring> Add<&ModuleElement<R>> for &ModuleElement<R> {
    type Output = Result<ModuleElement<R>, ModuleError>;

    fn add(self, rhs: &ModuleElement<R>) -> Self::Output {
        ModuleElement::add(self, rhs)
    }
}

impl<R: Ring> Sub<&ModuleElement<R>> for &ModuleElement<R> {
    type Output = Result<ModuleElement<R>, ModuleError>;

    fn sub(self, rhs: &ModuleElement<R>) -> Self::Output {
        ModuleElement::sub(self, rhs)
    }
}

impl<R: Ring> Neg for &ModuleElement<R> {
    type Output = Result<ModuleElement<R>, ModuleError>;

    fn neg(self) -> Self::Output {
        ModuleElement::neg(self)
    }
}

#[derive(Clone)]
enum ModuleMatrix<R: Ring> {
    Dense(Matrix<R>),
    Sparse(SparseMatrix<R>),
}

#[derive(Serialize, Deserialize)]
#[serde(
    tag = "layout",
    content = "matrix",
    rename_all = "snake_case",
    bound(
        serialize = "R: Ring + Serialize, R::Element: Serialize",
        deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>"
    )
)]
enum ModuleMatrixWire<R: Ring> {
    Dense(Matrix<R>),
    Sparse(SparseMatrix<R>),
}

impl<R: Ring> ModuleMatrix<R> {
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

    fn to_dense(&self) -> Matrix<R> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix.to_dense(),
        }
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_dense().canonical_bytes()
    }
}

impl<R: Ring> PartialEq for ModuleMatrix<R> {
    fn eq(&self, other: &Self) -> bool {
        self.to_dense() == other.to_dense()
    }
}

impl<R: Ring> Eq for ModuleMatrix<R> {}

/// An exact homomorphism between finite based free modules.
///
/// Matrices act on column coordinates and therefore have shape
/// `target.rank() x source.rank()`. Dense and sparse storage are private
/// representations of the same mathematical map.
#[derive(Clone, PartialEq, Eq)]
pub struct ModuleHomomorphism<R: Ring> {
    source: FreeModule<R>,
    target: FreeModule<R>,
    matrix: ModuleMatrix<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct ModuleHomomorphismWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: FreeModule<R>,
    target: FreeModule<R>,
    presentation: ModuleMatrixWire<R>,
}

impl<R: Ring> ModuleHomomorphism<R> {
    /// Construct a map from a dense matrix.
    pub fn from_matrix(
        source: FreeModule<R>,
        target: FreeModule<R>,
        matrix: Matrix<R>,
    ) -> Result<Self, ModuleError> {
        Self::from_storage(source, target, ModuleMatrix::Dense(matrix))
    }

    /// Construct a map from a canonical sparse matrix.
    pub fn from_sparse_matrix(
        source: FreeModule<R>,
        target: FreeModule<R>,
        matrix: SparseMatrix<R>,
    ) -> Result<Self, ModuleError> {
        Self::from_storage(source, target, ModuleMatrix::Sparse(matrix))
    }

    pub(super) fn from_coordinate_columns(
        source: FreeModule<R>,
        target: FreeModule<R>,
        columns: Vec<Vec<R::Element>>,
    ) -> Result<Self, ModuleError> {
        if columns.len() != source.rank()
            || columns.iter().any(|column| column.len() != target.rank())
        {
            return Err(ModuleError::MatrixShape {
                expected: (target.rank(), source.rank()),
                actual: (
                    columns.first().map_or(target.rank(), Vec::len),
                    columns.len(),
                ),
            });
        }
        let cell_count = target.rank().saturating_mul(source.rank());
        let nonzero_count = columns
            .iter()
            .flat_map(|column| column.iter())
            .filter(|value| !target.ring().is_zero(value))
            .count();
        if cell_count == 0 || nonzero_count.saturating_mul(2) <= cell_count {
            let column_count = columns.len();
            let ring = target.ring().clone();
            let entries = columns
                .into_iter()
                .enumerate()
                .flat_map(|(column, values)| {
                    let ring = ring.clone();
                    values
                        .into_iter()
                        .enumerate()
                        .filter(move |(_, value)| !ring.is_zero(value))
                        .map(move |(row, value)| (row, column, value))
                });
            return Self::from_sparse_matrix(
                source,
                target.clone(),
                SparseMatrix::from_triplets(
                    target.ring().clone(),
                    target.rank(),
                    column_count,
                    entries,
                )?,
            );
        }
        Self::from_matrix(
            source,
            target.clone(),
            Matrix::from_validated_columns(target.ring().clone(), target.rank(), columns),
        )
    }

    /// Sparse zero map between two modules over the same exact ring context.
    pub fn zero(source: FreeModule<R>, target: FreeModule<R>) -> Result<Self, ModuleError> {
        require_same_ring(source.ring(), target.ring())?;
        let matrix = SparseMatrix::zeros(target.ring().clone(), target.rank(), source.rank());
        Self::from_sparse_matrix(source, target, matrix)
    }

    /// Sparse identity map of one based free module.
    pub fn identity(module: FreeModule<R>) -> Result<Self, ModuleError> {
        let entries = (0..module.rank()).map(|index| (index, index, module.ring().one()));
        let matrix = SparseMatrix::from_triplets(
            module.ring().clone(),
            module.rank(),
            module.rank(),
            entries,
        )?;
        Self::from_sparse_matrix(module.clone(), module, matrix)
    }

    /// Source based free module.
    pub fn source(&self) -> &FreeModule<R> {
        &self.source
    }

    /// Target based free module.
    pub fn target(&self) -> &FreeModule<R> {
        &self.target
    }

    /// Losslessly materialize the presentation matrix.
    pub fn matrix(&self) -> Matrix<R> {
        self.matrix.to_dense()
    }

    /// Borrow the sparse presentation when this map retains sparse storage.
    pub fn sparse_matrix(&self) -> Option<&SparseMatrix<R>> {
        match &self.matrix {
            ModuleMatrix::Dense(_) => None,
            ModuleMatrix::Sparse(matrix) => Some(matrix),
        }
    }

    /// Whether the map retains canonical sparse storage.
    pub fn is_sparse(&self) -> bool {
        matches!(self.matrix, ModuleMatrix::Sparse(_))
    }

    /// Apply this map to an element of its exact source module.
    pub fn apply(&self, element: &ModuleElement<R>) -> Result<ModuleElement<R>, ModuleError> {
        if element.module() != &self.source {
            return Err(ModuleError::ModuleMismatch {
                operation: "module homomorphism application",
            });
        }
        let coordinates = match &self.matrix {
            ModuleMatrix::Dense(matrix) => {
                let product = matrix.matmul(&element.coordinates().to_column_matrix())?;
                Vector::from_values(
                    self.target.ring().clone(),
                    product.rows().filter_map(|row| row.into_iter().next()),
                )?
            }
            ModuleMatrix::Sparse(matrix) => matrix.apply(element.coordinates())?,
        };
        ModuleElement::from_coordinates(self.target.clone(), coordinates)
    }

    /// Compose `self` after `inner`.
    pub fn compose(&self, inner: &Self) -> Result<Self, ModuleError> {
        if inner.target != self.source {
            return Err(ModuleError::ModuleMismatch {
                operation: "module homomorphism composition",
            });
        }
        let matrix = match (&self.matrix, &inner.matrix) {
            (ModuleMatrix::Sparse(outer), ModuleMatrix::Sparse(inner)) => {
                ModuleMatrix::Sparse(outer.matmul(inner)?)
            }
            _ => ModuleMatrix::Dense(self.matrix.to_dense().matmul(&inner.matrix.to_dense())?),
        };
        Self::from_storage(inner.source.clone(), self.target.clone(), matrix)
    }

    /// Add maps with identical source and target based modules.
    pub fn add(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_endpoints(rhs, "module homomorphism addition")?;
        let matrix = match (&self.matrix, &rhs.matrix) {
            (ModuleMatrix::Sparse(left), ModuleMatrix::Sparse(right)) => {
                ModuleMatrix::Sparse(left.add(right)?)
            }
            _ => ModuleMatrix::Dense(self.matrix.to_dense().add(&rhs.matrix.to_dense())?),
        };
        Self::from_storage(self.source.clone(), self.target.clone(), matrix)
    }

    /// Subtract maps with identical source and target based modules.
    pub fn sub(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_endpoints(rhs, "module homomorphism subtraction")?;
        let matrix = match (&self.matrix, &rhs.matrix) {
            (ModuleMatrix::Sparse(left), ModuleMatrix::Sparse(right)) => {
                ModuleMatrix::Sparse(left.sub(right)?)
            }
            _ => ModuleMatrix::Dense(self.matrix.to_dense().sub(&rhs.matrix.to_dense())?),
        };
        Self::from_storage(self.source.clone(), self.target.clone(), matrix)
    }

    /// Copair maps with one common target into a map from the direct sum of
    /// their sources. Its matrix is the horizontal block `[self rhs]`.
    pub fn copair(&self, rhs: &Self) -> Result<Self, ModuleError> {
        if self.target != rhs.target {
            return Err(ModuleError::ModuleMismatch {
                operation: "module homomorphism copairing",
            });
        }
        let source = self.source.direct_sum(&rhs.source)?;
        let mut columns = self.coordinate_columns();
        columns.extend(rhs.coordinate_columns());
        Self::from_coordinate_columns(source, self.target.clone(), columns)
    }

    /// Pair maps with one common source into a map to the direct sum of their
    /// targets. Its matrix is the vertical block `[self; rhs]`.
    pub fn pair(&self, rhs: &Self) -> Result<Self, ModuleError> {
        if self.source != rhs.source {
            return Err(ModuleError::ModuleMismatch {
                operation: "module homomorphism pairing",
            });
        }
        let target = self.target.direct_sum(&rhs.target)?;
        let right = rhs.coordinate_columns();
        let columns = self
            .coordinate_columns()
            .into_iter()
            .zip(right)
            .map(|(mut left, right)| {
                left.extend(right);
                left
            })
            .collect();
        Self::from_coordinate_columns(self.source.clone(), target, columns)
    }

    /// Block-diagonal direct sum of two maps.
    pub fn direct_sum(&self, rhs: &Self) -> Result<Self, ModuleError> {
        let source = self.source.direct_sum(&rhs.source)?;
        let target = self.target.direct_sum(&rhs.target)?;
        let left_zero = vec![self.source.ring().zero(); self.target.rank()];
        let right_zero = vec![self.source.ring().zero(); rhs.target.rank()];
        let columns = self
            .coordinate_columns()
            .into_iter()
            .map(|mut column| {
                column.extend(right_zero.iter().cloned());
                column
            })
            .chain(rhs.coordinate_columns().into_iter().map(|column| {
                let mut output = left_zero.clone();
                output.extend(column);
                output
            }))
            .collect();
        Self::from_coordinate_columns(source, target, columns)
    }

    /// Tensor product of maps between finite based free modules.
    ///
    /// The tensor bases use the same lexicographic order as
    /// [`FreeModule::tensor`]: the left basis position is the outer index and
    /// the right basis position is the inner index. Sparse inputs retain a
    /// sparse Kronecker product without materializing zero coefficients.
    pub fn tensor(&self, rhs: &Self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        require_same_ring(self.source.ring(), rhs.source.ring())?;
        let source = self.source.tensor(&rhs.source)?;
        let target = self.target.tensor(&rhs.target)?;
        if let (ModuleMatrix::Sparse(left), ModuleMatrix::Sparse(right)) =
            (&self.matrix, &rhs.matrix)
        {
            let mut entries = Vec::new();
            for (left_row, left_offsets) in left.row_offsets().windows(2).enumerate() {
                for left_index in left_offsets[0]..left_offsets[1] {
                    let left_column = left.column_indices()[left_index];
                    for (right_row, right_offsets) in right.row_offsets().windows(2).enumerate() {
                        for right_index in right_offsets[0]..right_offsets[1] {
                            let right_column = right.column_indices()[right_index];
                            let value = self
                                .source
                                .ring()
                                .mul(&left.values()[left_index], &right.values()[right_index])?;
                            if !self.source.ring().is_zero(&value) {
                                entries.push((
                                    left_row * rhs.target.rank() + right_row,
                                    left_column * rhs.source.rank() + right_column,
                                    value,
                                ));
                            }
                        }
                    }
                }
            }
            let matrix = SparseMatrix::from_triplets(
                target.ring().clone(),
                target.rank(),
                source.rank(),
                entries,
            )?;
            return Self::from_sparse_matrix(source, target, matrix);
        }

        let left_columns = self.coordinate_columns();
        let right_columns = rhs.coordinate_columns();
        let mut columns = Vec::with_capacity(source.rank());
        for left in left_columns {
            for right in &right_columns {
                let mut column = Vec::with_capacity(target.rank());
                for left_value in &left {
                    for right_value in right {
                        column.push(
                            if self.source.ring().is_zero(left_value)
                                || self.source.ring().is_zero(right_value)
                            {
                                self.source.ring().zero()
                            } else {
                                self.source.ring().mul(left_value, right_value)?
                            },
                        );
                    }
                }
                columns.push(column);
            }
        }
        Self::from_coordinate_columns(source, target, columns)
    }

    /// Dual map between the dual based free modules.
    pub fn dual(&self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        let matrix = match &self.matrix {
            ModuleMatrix::Dense(matrix) => ModuleMatrix::Dense(matrix.transpose()),
            ModuleMatrix::Sparse(matrix) => ModuleMatrix::Sparse(matrix.transpose()),
        };
        Self::from_storage(self.target.dual()?, self.source.dual()?, matrix)
    }

    /// Apply a checked coefficient map to source, target, and every matrix
    /// entry.
    pub fn map_coefficients<M>(&self, map: &M) -> Result<ModuleHomomorphism<M::Target>, ModuleError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        let source = self.source.map_coefficients(map)?;
        let target = self.target.map_coefficients(map)?;
        match &self.matrix {
            ModuleMatrix::Dense(matrix) => {
                ModuleHomomorphism::from_matrix(source, target, matrix.map_coefficients(map)?)
            }
            ModuleMatrix::Sparse(matrix) => ModuleHomomorphism::from_sparse_matrix(
                source,
                target,
                matrix.map_coefficients(map)?,
            ),
        }
    }

    /// Serialize both endpoint modules and the retained matrix layout.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an exact module homomorphism.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn from_storage(
        source: FreeModule<R>,
        target: FreeModule<R>,
        matrix: ModuleMatrix<R>,
    ) -> Result<Self, ModuleError> {
        require_same_ring(source.ring(), target.ring())?;
        require_same_ring(source.ring(), matrix.ring())?;
        let expected = (target.rank(), source.rank());
        if matrix.shape() != expected {
            return Err(ModuleError::MatrixShape {
                expected,
                actual: matrix.shape(),
            });
        }
        Ok(Self {
            source,
            target,
            matrix,
        })
    }

    fn require_same_endpoints(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<(), ModuleError> {
        if self.source == rhs.source && self.target == rhs.target {
            Ok(())
        } else {
            Err(ModuleError::ModuleMismatch { operation })
        }
    }

    pub(super) fn reframe(
        self,
        source: FreeModule<R>,
        target: FreeModule<R>,
    ) -> Result<Self, ModuleError> {
        Self::from_storage(source, target, self.matrix)
    }

    pub(super) fn coordinate_columns(&self) -> Vec<Vec<R::Element>> {
        if let ModuleMatrix::Sparse(matrix) = &self.matrix {
            let mut columns =
                vec![vec![self.target.ring().zero(); self.target.rank()]; self.source.rank()];
            for (row, offsets) in matrix.row_offsets().windows(2).enumerate() {
                for index in offsets[0]..offsets[1] {
                    columns[matrix.column_indices()[index]][row] = matrix.values()[index].clone();
                }
            }
            return columns;
        }
        let rows = self.matrix.to_dense().rows().collect::<Vec<_>>();
        (0..self.source.rank())
            .map(|column| rows.iter().map(|row| row[column].clone()).collect())
            .collect()
    }
}

impl<R> Serialize for ModuleHomomorphism<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let presentation = match &self.matrix {
            ModuleMatrix::Dense(matrix) => ModuleMatrixWire::Dense(matrix.clone()),
            ModuleMatrix::Sparse(matrix) => ModuleMatrixWire::Sparse(matrix.clone()),
        };
        let mut state = serializer.serialize_struct("ModuleHomomorphism", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("presentation", &presentation)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for ModuleHomomorphism<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleHomomorphismWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        match wire.presentation {
            ModuleMatrixWire::Dense(matrix) => Self::from_matrix(wire.source, wire.target, matrix),
            ModuleMatrixWire::Sparse(matrix) => {
                Self::from_sparse_matrix(wire.source, wire.target, matrix)
            }
        }
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for ModuleHomomorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleHomomorphism")
            .field("source_rank", &self.source.rank())
            .field("target_rank", &self.target.rank())
            .field("sparse", &self.is_sparse())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for ModuleHomomorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModuleHomomorphism<{}>({} -> {})",
            R::TYPE_ID,
            self.source.rank(),
            self.target.rank()
        )
    }
}

impl<R: Ring> SemanticType for ModuleHomomorphism<R> {
    const TYPE_ID: &'static str = "noether/homalg/ModuleHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.source.canonical_bytes()?)?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        append_component(&mut output, &self.matrix.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_HOMOMORPHISM_DESCRIPTION
    }
}

impl<R: Ring> Add<&ModuleHomomorphism<R>> for &ModuleHomomorphism<R> {
    type Output = Result<ModuleHomomorphism<R>, ModuleError>;

    fn add(self, rhs: &ModuleHomomorphism<R>) -> Self::Output {
        ModuleHomomorphism::add(self, rhs)
    }
}

impl<R: Ring> Sub<&ModuleHomomorphism<R>> for &ModuleHomomorphism<R> {
    type Output = Result<ModuleHomomorphism<R>, ModuleError>;

    fn sub(self, rhs: &ModuleHomomorphism<R>) -> Self::Output {
        ModuleHomomorphism::sub(self, rhs)
    }
}

/// A finitely generated submodule represented as the image of a generator
/// map into one ambient based free module.
#[derive(Clone)]
pub struct Submodule<R: Ring> {
    ambient: FreeModule<R>,
    generator_map: ModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct SubmoduleWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    generator_map: ModuleHomomorphism<R>,
}

impl<R: Ring> Submodule<R> {
    /// Construct a submodule from a map whose target is the ambient module.
    pub fn from_generator_map(
        ambient: FreeModule<R>,
        generator_map: ModuleHomomorphism<R>,
    ) -> Result<Self, ModuleError> {
        if generator_map.target() != &ambient {
            return Err(ModuleError::ModuleMismatch {
                operation: "submodule generator map",
            });
        }
        exact_submodule::canonicalize_supported(Self {
            ambient,
            generator_map,
        })
    }

    /// Construct a submodule from ordered ambient elements.
    pub fn from_generators(
        ambient: FreeModule<R>,
        generators: Vec<ModuleElement<R>>,
    ) -> Result<Self, ModuleError> {
        if generators
            .iter()
            .any(|generator| generator.module() != &ambient)
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "submodule generators",
            });
        }
        let source = FreeModule::new(ambient.ring().clone(), generators.len());
        let columns = generators
            .into_iter()
            .map(ModuleElement::into_coordinates)
            .collect::<Vec<_>>();
        let generator_map =
            ModuleHomomorphism::from_coordinate_columns(source, ambient.clone(), columns)?;
        Self::from_generator_map(ambient, generator_map)
    }

    /// Ambient based free module.
    pub fn ambient(&self) -> &FreeModule<R> {
        &self.ambient
    }

    /// Ordered generator map whose image is this submodule.
    pub fn generator_map(&self) -> &ModuleHomomorphism<R> {
        &self.generator_map
    }

    /// Number of retained ordered generators.
    pub fn generator_count(&self) -> usize {
        self.generator_map.source().rank()
    }

    /// Sum with another submodule of the same free module.
    pub fn sum(&self, rhs: &Self) -> Result<Self, ModuleError> {
        if self.ambient() != rhs.ambient() {
            return Err(ModuleError::ModuleMismatch {
                operation: "submodule sum",
            });
        }
        Self::from_generator_map(
            self.ambient().clone(),
            self.generator_map().copair(rhs.generator_map())?,
        )
    }

    /// Serialize the exact generator map defining this image.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a generated submodule.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for Submodule<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Submodule", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("generator_map", &self.generator_map)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for Submodule<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SubmoduleWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = wire.generator_map;
        let ambient = supplied.target().clone();
        let canonical = Self::from_generator_map(ambient, supplied.clone())
            .map_err(serde::de::Error::custom)?;
        if canonical.generator_map != supplied {
            return Err(serde::de::Error::custom(
                "submodule generators are not in canonical form",
            ));
        }
        Ok(canonical)
    }
}

impl<R: Ring> fmt::Debug for Submodule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submodule")
            .field("ambient_rank", &self.ambient.rank())
            .field("generators", &self.generator_count())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for Submodule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Submodule<{}>({} generators in rank {})",
            R::TYPE_ID,
            self.generator_count(),
            self.ambient.rank()
        )
    }
}

impl<R: Ring> SemanticType for Submodule<R> {
    const TYPE_ID: &'static str = "noether/homalg/Submodule@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.ambient.canonical_bytes()?)?;
        append_component(&mut output, &self.generator_map.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SUBMODULE_DESCRIPTION
    }
}

/// A finitely presented module retained as the cokernel of one exact map
/// between based free modules.
#[derive(Clone, PartialEq, Eq)]
pub struct PresentedModule<R: Ring> {
    presentation: ModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PresentedModuleWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    presentation: ModuleHomomorphism<R>,
}

impl<R: Ring> PresentedModule<R> {
    /// Regard a finite based free module as a presentation with no relations.
    ///
    /// A graded free module receives a graded rank-zero relation module, so
    /// subsequent tensor and dual constructions preserve the exact grading
    /// context even though there are no relation generators.
    pub fn free(module: FreeModule<R>) -> Result<Self, ModuleError> {
        let relations = FreeModule::with_basis_data(
            module.ring().clone(),
            0,
            None,
            module.grading_rank(),
            module.degree_shifts().map(|_| Vec::new()),
        )?;
        Ok(Self::cokernel(ModuleHomomorphism::zero(relations, module)?))
    }

    /// Construct the cokernel of `presentation`.
    pub fn cokernel(presentation: ModuleHomomorphism<R>) -> Self {
        Self { presentation }
    }

    /// Relation map whose cokernel defines this module.
    pub fn presentation(&self) -> &ModuleHomomorphism<R> {
        &self.presentation
    }

    /// Based free module of retained quotient generators.
    pub fn generator_module(&self) -> &FreeModule<R> {
        self.presentation.target()
    }

    /// Based free module indexing retained relations.
    pub fn relation_module(&self) -> &FreeModule<R> {
        self.presentation.source()
    }

    /// Number of presentation generators.
    pub fn generator_count(&self) -> usize {
        self.generator_module().rank()
    }

    /// Number of retained presentation relations.
    pub fn relation_count(&self) -> usize {
        self.relation_module().rank()
    }

    /// Tensor product of two finite presentations over one commutative ring.
    ///
    /// If `self = coker(P)` and `rhs = coker(Q)`, the retained presentation is
    /// `coker([P tensor id, id tensor Q])`. Thus the construction is
    /// coefficient-generic and records both families of bilinearity relations
    /// explicitly; it does not require a Groebner basis or choose new quotient
    /// generators.
    pub fn tensor(&self, rhs: &Self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        require_same_ring(
            self.generator_module().ring(),
            rhs.generator_module().ring(),
        )?;
        let left_relations = self.presentation.tensor(&ModuleHomomorphism::identity(
            rhs.generator_module().clone(),
        )?)?;
        let right_relations = ModuleHomomorphism::identity(self.generator_module().clone())?
            .tensor(&rhs.presentation)?;
        Ok(Self::cokernel(left_relations.copair(&right_relations)?))
    }

    /// The precomposition map whose kernel presents `Hom(self, target)`.
    ///
    /// For `self = coker(P: F_1 -> F_0)`, this is the natural map
    /// `Hom(F_0, target) -> Hom(F_1, target)`. Finite free duality identifies
    /// it with `P^* tensor id_target`, and the returned checked map retains
    /// that complete construction square. Coefficient-specific module
    /// algorithms compute its kernel to obtain the actual `Hom` module.
    pub fn hom_presentation_map(
        &self,
        target: &Self,
    ) -> Result<PresentedModuleHomomorphism<R>, ModuleError>
    where
        R: CommutativeRing,
    {
        PresentedModuleHomomorphism::from_free_map(self.presentation.dual()?)?
            .tensor(&PresentedModuleHomomorphism::identity(target.clone())?)
    }

    /// Serialize the exact retained cokernel presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a presented module.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for PresentedModule<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PresentedModule", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("presentation", &self.presentation)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PresentedModule<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PresentedModuleWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::cokernel(wire.presentation))
    }
}

impl<R: Ring> fmt::Debug for PresentedModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentedModule")
            .field("generators", &self.generator_count())
            .field("relations", &self.relation_count())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for PresentedModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PresentedModule<{}>({} generators, {} relations)",
            R::TYPE_ID,
            self.generator_count(),
            self.relation_count()
        )
    }
}

impl<R: Ring> SemanticType for PresentedModule<R> {
    const TYPE_ID: &'static str = "noether/homalg/PresentedModule@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.presentation.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &PRESENTED_MODULE_DESCRIPTION
    }
}

/// An exact homomorphism between two retained cokernel presentations.
///
/// If `source` is the cokernel of `P` and `target` is the cokernel of `Q`, a
/// value retains maps `A` on generators and `B` on relations satisfying
/// `A P = Q B`. The relation lift `B` is a replayable certificate that `A`
/// descends to the cokernels. It is deliberately not part of structural
/// equality: two certificates for the same retained generator lift describe
/// the same value. Conversely, [`PartialEq`] compares retained generator
/// lifts, not equality modulo the target relations; coefficient-specific
/// algorithms provide that stronger mathematical question.
#[derive(Clone)]
pub struct PresentedModuleHomomorphism<R: Ring> {
    source: PresentedModule<R>,
    target: PresentedModule<R>,
    generator_lift: ModuleHomomorphism<R>,
    relation_lift: ModuleHomomorphism<R>,
}

impl<R: Ring> PartialEq for PresentedModuleHomomorphism<R> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.target == other.target
            && self.generator_lift == other.generator_lift
    }
}

impl<R: Ring> Eq for PresentedModuleHomomorphism<R> {}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PresentedModuleHomomorphismWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: PresentedModule<R>,
    target: PresentedModule<R>,
    generator_lift: ModuleHomomorphism<R>,
    relation_lift: ModuleHomomorphism<R>,
}

impl<R: Ring> PresentedModuleHomomorphism<R> {
    /// Regard a map of finite based free modules as a checked map between the
    /// corresponding relation-free presentations.
    pub fn from_free_map(map: ModuleHomomorphism<R>) -> Result<Self, ModuleError> {
        let source = PresentedModule::free(map.source().clone())?;
        let target = PresentedModule::free(map.target().clone())?;
        let relation_lift = ModuleHomomorphism::zero(
            source.relation_module().clone(),
            target.relation_module().clone(),
        )?;
        Self::from_lifts(source, target, map, relation_lift)
    }

    /// Construct and verify the commuting square that makes a generator lift
    /// descend to the two cokernels.
    pub fn from_lifts(
        source: PresentedModule<R>,
        target: PresentedModule<R>,
        generator_lift: ModuleHomomorphism<R>,
        relation_lift: ModuleHomomorphism<R>,
    ) -> Result<Self, ModuleError> {
        if generator_lift.source() != source.generator_module()
            || generator_lift.target() != target.generator_module()
            || relation_lift.source() != source.relation_module()
            || relation_lift.target() != target.relation_module()
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homomorphism construction",
            });
        }
        let left = generator_lift.compose(source.presentation())?;
        let right = target.presentation().compose(&relation_lift)?;
        if left != right {
            return Err(ModuleError::RelationDiagramDoesNotCommute);
        }
        Ok(Self {
            source,
            target,
            generator_lift,
            relation_lift,
        })
    }

    /// Identity map, with identity lifts on generators and relations.
    pub fn identity(module: PresentedModule<R>) -> Result<Self, ModuleError> {
        Self::from_lifts(
            module.clone(),
            module.clone(),
            ModuleHomomorphism::identity(module.generator_module().clone())?,
            ModuleHomomorphism::identity(module.relation_module().clone())?,
        )
    }

    /// Zero map, with sparse zero lifts on generators and relations.
    pub fn zero(
        source: PresentedModule<R>,
        target: PresentedModule<R>,
    ) -> Result<Self, ModuleError> {
        Self::from_lifts(
            source.clone(),
            target.clone(),
            ModuleHomomorphism::zero(
                source.generator_module().clone(),
                target.generator_module().clone(),
            )?,
            ModuleHomomorphism::zero(
                source.relation_module().clone(),
                target.relation_module().clone(),
            )?,
        )
    }

    /// Retained source cokernel presentation.
    pub fn source(&self) -> &PresentedModule<R> {
        &self.source
    }

    /// Retained target cokernel presentation.
    pub fn target(&self) -> &PresentedModule<R> {
        &self.target
    }

    /// Map between generator modules inducing this presented-module map.
    pub fn generator_lift(&self) -> &ModuleHomomorphism<R> {
        &self.generator_lift
    }

    /// Retained certificate mapping source relations to target relations.
    pub fn relation_lift(&self) -> &ModuleHomomorphism<R> {
        &self.relation_lift
    }

    /// Recheck the exact commuting-square certificate.
    pub fn verify(&self) -> Result<(), ModuleError> {
        Self::from_lifts(
            self.source.clone(),
            self.target.clone(),
            self.generator_lift.clone(),
            self.relation_lift.clone(),
        )
        .map(|_| ())
    }

    /// Compose `self` after `inner`, composing both parts of the checked
    /// commuting squares.
    pub fn compose(&self, inner: &Self) -> Result<Self, ModuleError> {
        if inner.target != self.source {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homomorphism composition",
            });
        }
        Self::from_lifts(
            inner.source.clone(),
            self.target.clone(),
            self.generator_lift.compose(&inner.generator_lift)?,
            self.relation_lift.compose(&inner.relation_lift)?,
        )
    }

    /// Add maps with identical retained source and target presentations.
    pub fn add(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_endpoints(rhs, "presented-module homomorphism addition")?;
        Self::from_lifts(
            self.source.clone(),
            self.target.clone(),
            self.generator_lift.add(&rhs.generator_lift)?,
            self.relation_lift.add(&rhs.relation_lift)?,
        )
    }

    /// Subtract maps with identical retained source and target presentations.
    pub fn sub(&self, rhs: &Self) -> Result<Self, ModuleError> {
        self.require_same_endpoints(rhs, "presented-module homomorphism subtraction")?;
        Self::from_lifts(
            self.source.clone(),
            self.target.clone(),
            self.generator_lift.sub(&rhs.generator_lift)?,
            self.relation_lift.sub(&rhs.relation_lift)?,
        )
    }

    /// Tensor product of two checked maps between retained presentations.
    ///
    /// The generator lift is the tensor product of generator lifts. The two
    /// relation blocks retain respectively the tensor products of the left
    /// relation lift with the right generator lift and the left generator
    /// lift with the right relation lift. [`Self::from_lifts`] then rechecks
    /// the complete tensor-presentation square.
    pub fn tensor(&self, rhs: &Self) -> Result<Self, ModuleError>
    where
        R: CommutativeRing,
    {
        let source = self.source.tensor(&rhs.source)?;
        let target = self.target.tensor(&rhs.target)?;
        let generator_lift = self.generator_lift.tensor(&rhs.generator_lift)?;
        let relation_lift = self
            .relation_lift
            .tensor(&rhs.generator_lift)?
            .direct_sum(&self.generator_lift.tensor(&rhs.relation_lift)?)?;
        Self::from_lifts(source, target, generator_lift, relation_lift)
    }

    /// Cokernel with its direct block presentation `[Q A]`, where `Q` is the
    /// target presentation and `A` is the retained generator lift.
    pub fn cokernel(&self) -> Result<PresentedModule<R>, ModuleError> {
        let relation_module = self
            .target
            .relation_module()
            .direct_sum(self.source.generator_module())?;
        let mut columns = self.target.presentation().coordinate_columns();
        columns.extend(self.generator_lift.coordinate_columns());
        Ok(PresentedModule::cokernel(
            ModuleHomomorphism::from_coordinate_columns(
                relation_module,
                self.target.generator_module().clone(),
                columns,
            )?,
        ))
    }

    /// Canonical quotient map from the target onto [`Self::cokernel`].
    ///
    /// Its relation certificate is the inclusion of the target's retained
    /// relations into the first block of the cokernel presentation `[Q A]`.
    pub fn cokernel_projection(&self) -> Result<Self, ModuleError> {
        let cokernel = self.cokernel()?;
        let relation_lift = ModuleHomomorphism::identity(self.target.relation_module().clone())?
            .pair(&ModuleHomomorphism::zero(
                self.target.relation_module().clone(),
                self.source.generator_module().clone(),
            )?)?;
        Self::from_lifts(
            self.target.clone(),
            cokernel,
            ModuleHomomorphism::identity(self.target.generator_module().clone())?,
            relation_lift,
        )
    }

    /// Apply a checked coefficient map to both presentations and the checked
    /// commuting square.
    pub fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<PresentedModuleHomomorphism<M::Target>, ModuleError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        PresentedModuleHomomorphism::from_lifts(
            PresentedModule::cokernel(self.source.presentation().map_coefficients(map)?),
            PresentedModule::cokernel(self.target.presentation().map_coefficients(map)?),
            self.generator_lift.map_coefficients(map)?,
            self.relation_lift.map_coefficients(map)?,
        )
    }

    /// Serialize both endpoint presentations, the inducing lift, and its
    /// commuting-square certificate.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reverify a presented-module homomorphism.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_endpoints(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<(), ModuleError> {
        if self.source == rhs.source && self.target == rhs.target {
            Ok(())
        } else {
            Err(ModuleError::ModuleMismatch { operation })
        }
    }
}

impl<R> Serialize for PresentedModuleHomomorphism<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PresentedModuleHomomorphism", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("generator_lift", &self.generator_lift)?;
        state.serialize_field("relation_lift", &self.relation_lift)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PresentedModuleHomomorphism<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PresentedModuleHomomorphismWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_lifts(
            wire.source,
            wire.target,
            wire.generator_lift,
            wire.relation_lift,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for PresentedModuleHomomorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentedModuleHomomorphism")
            .field("source", &self.source)
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for PresentedModuleHomomorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PresentedModuleHomomorphism<{}>({} -> {} generators)",
            R::TYPE_ID,
            self.source.generator_count(),
            self.target.generator_count()
        )
    }
}

impl<R: Ring> SemanticType for PresentedModuleHomomorphism<R> {
    const TYPE_ID: &'static str = "noether/homalg/PresentedModuleHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.source.canonical_bytes()?)?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        append_component(&mut output, &self.generator_lift.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PRESENTED_MODULE_HOMOMORPHISM_DESCRIPTION
    }
}

impl<R: Ring> Add<&PresentedModuleHomomorphism<R>> for &PresentedModuleHomomorphism<R> {
    type Output = Result<PresentedModuleHomomorphism<R>, ModuleError>;

    fn add(self, rhs: &PresentedModuleHomomorphism<R>) -> Self::Output {
        PresentedModuleHomomorphism::add(self, rhs)
    }
}

impl<R: Ring> Sub<&PresentedModuleHomomorphism<R>> for &PresentedModuleHomomorphism<R> {
    type Output = Result<PresentedModuleHomomorphism<R>, ModuleError>;

    fn sub(self, rhs: &PresentedModuleHomomorphism<R>) -> Self::Output {
        PresentedModuleHomomorphism::sub(self, rhs)
    }
}

/// Exact module-valued homology at one term of a complex of presented
/// modules.
///
/// For `incoming: L -> M` and `outgoing: M -> N`, this value retains the cycle
/// inclusion `Z -> M`, the factorization `L -> Z` of the boundary map, and the
/// quotient projection `Z -> H`. Coefficient-specific verification proves
/// that the first map is the kernel of `outgoing` and the last map has the
/// boundaries as its kernel.
#[derive(Clone, PartialEq, Eq)]
pub struct PresentedModuleHomology<R: Ring> {
    incoming: PresentedModuleHomomorphism<R>,
    outgoing: PresentedModuleHomomorphism<R>,
    cycle_inclusion: PresentedModuleHomomorphism<R>,
    boundary_lift: PresentedModuleHomomorphism<R>,
    quotient_projection: PresentedModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PresentedModuleHomologyWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    incoming: PresentedModuleHomomorphism<R>,
    outgoing: PresentedModuleHomomorphism<R>,
    cycle_inclusion: PresentedModuleHomomorphism<R>,
    boundary_lift: PresentedModuleHomomorphism<R>,
    quotient_projection: PresentedModuleHomomorphism<R>,
}

impl<R: Ring> PresentedModuleHomology<R> {
    pub(super) fn from_evidence(
        incoming: PresentedModuleHomomorphism<R>,
        outgoing: PresentedModuleHomomorphism<R>,
        cycle_inclusion: PresentedModuleHomomorphism<R>,
        boundary_lift: PresentedModuleHomomorphism<R>,
        quotient_projection: PresentedModuleHomomorphism<R>,
    ) -> Result<Self, ModuleError> {
        if incoming.target() != outgoing.source()
            || cycle_inclusion.target() != outgoing.source()
            || boundary_lift.source() != incoming.source()
            || boundary_lift.target() != cycle_inclusion.source()
            || quotient_projection.source() != cycle_inclusion.source()
            || cycle_inclusion.compose(&boundary_lift)? != incoming
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homology evidence",
            });
        }
        Ok(Self {
            incoming,
            outgoing,
            cycle_inclusion,
            boundary_lift,
            quotient_projection,
        })
    }

    /// Construct the canonical homology evidence for a complex ending in a
    /// zero differential, so its homology is the cokernel of `incoming`.
    ///
    /// This is the degree-zero derived-functor path.  The cycles are the
    /// entire middle module, witnessed by its identity, and the ordinary
    /// cokernel projection already carries the exact boundary quotient
    /// certificate.  No kernel or Groebner computation is needed to discover
    /// facts that are structural in this shape of complex.
    pub(super) fn from_cokernel_complex(
        incoming: &PresentedModuleHomomorphism<R>,
        outgoing: &PresentedModuleHomomorphism<R>,
    ) -> Result<Self, ModuleError> {
        if incoming.target() != outgoing.source()
            || *outgoing
                != PresentedModuleHomomorphism::zero(
                    outgoing.source().clone(),
                    outgoing.target().clone(),
                )?
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "cokernel homology evidence",
            });
        }
        let cycle_inclusion = PresentedModuleHomomorphism::identity(outgoing.source().clone())?;
        let boundary_lift = incoming.clone();
        let quotient_projection = boundary_lift.cokernel_projection()?;
        Self::from_evidence(
            incoming.clone(),
            outgoing.clone(),
            cycle_inclusion,
            boundary_lift,
            quotient_projection,
        )
    }

    /// Incoming differential whose image supplies the boundaries.
    pub fn incoming(&self) -> &PresentedModuleHomomorphism<R> {
        &self.incoming
    }

    /// Outgoing differential whose kernel supplies the cycles.
    pub fn outgoing(&self) -> &PresentedModuleHomomorphism<R> {
        &self.outgoing
    }

    /// Exact inclusion of the cycle module into the middle term.
    pub fn cycle_inclusion(&self) -> &PresentedModuleHomomorphism<R> {
        &self.cycle_inclusion
    }

    /// Incoming differential factored through the cycle module.
    pub fn boundary_lift(&self) -> &PresentedModuleHomomorphism<R> {
        &self.boundary_lift
    }

    /// Quotient map from cycles onto the homology module.
    pub fn quotient_projection(&self) -> &PresentedModuleHomomorphism<R> {
        &self.quotient_projection
    }

    /// Presented homology module `ker(outgoing) / im(incoming)`.
    pub fn module(&self) -> &PresentedModule<R> {
        self.quotient_projection.target()
    }

    /// Recheck endpoint compatibility and the exact retained factorization of
    /// the incoming map through cycles.
    pub fn verify_structure(&self) -> Result<(), ModuleError> {
        Self::from_evidence(
            self.incoming.clone(),
            self.outgoing.clone(),
            self.cycle_inclusion.clone(),
            self.boundary_lift.clone(),
            self.quotient_projection.clone(),
        )
        .map(|_| ())
    }

    /// Serialize the complete homology diagram and all relation witnesses.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the complete structural diagram.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for PresentedModuleHomology<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PresentedModuleHomology", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("incoming", &self.incoming)?;
        state.serialize_field("outgoing", &self.outgoing)?;
        state.serialize_field("cycle_inclusion", &self.cycle_inclusion)?;
        state.serialize_field("boundary_lift", &self.boundary_lift)?;
        state.serialize_field("quotient_projection", &self.quotient_projection)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PresentedModuleHomology<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PresentedModuleHomologyWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_evidence(
            wire.incoming,
            wire.outgoing,
            wire.cycle_inclusion,
            wire.boundary_lift,
            wire.quotient_projection,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for PresentedModuleHomology<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentedModuleHomology")
            .field("cycles", self.cycle_inclusion.source())
            .field("homology", self.module())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for PresentedModuleHomology<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PresentedModuleHomology<{}>({} generators)",
            R::TYPE_ID,
            self.module().generator_count()
        )
    }
}

impl<R: Ring> SemanticType for PresentedModuleHomology<R> {
    const TYPE_ID: &'static str = "noether/homalg/PresentedModuleHomology@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.incoming.canonical_bytes()?)?;
        append_component(&mut output, &self.outgoing.canonical_bytes()?)?;
        append_component(&mut output, &self.cycle_inclusion.canonical_bytes()?)?;
        append_component(&mut output, &self.boundary_lift.canonical_bytes()?)?;
        append_component(&mut output, &self.quotient_projection.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PRESENTED_MODULE_HOMOLOGY_DESCRIPTION
    }
}

/// A module-monomial order extending one polynomial monomial order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleMonomialOrder {
    /// Compare polynomial terms before basis positions.
    TermOverPosition(MonomialOrder),
    /// Compare basis positions before polynomial terms.
    PositionOverTerm(MonomialOrder),
    /// Compare source terms by their products with an ordered target leading
    /// frame, breaking induced ties by source position.
    Schreyer {
        /// Exact module order on the target of the framed generators.
        target_order: Box<Self>,
        /// Ordered leading module monomials of the framed generators.
        leading_monomials: Vec<ModuleMonomial>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModuleMonomialOrderWire {
    #[serde(rename = "type")]
    type_id: String,
    kind: String,
    base_order: Option<MonomialOrder>,
    target_order: Option<Box<ModuleMonomialOrder>>,
    leading_monomials: Option<Vec<ModuleMonomial>>,
}

impl ModuleMonomialOrder {
    /// Underlying polynomial monomial order.
    pub fn base_order(&self) -> &MonomialOrder {
        match self {
            Self::TermOverPosition(order) | Self::PositionOverTerm(order) => order,
            Self::Schreyer { target_order, .. } => target_order.base_order(),
        }
    }

    /// Construct the exact order induced by an ordered target leading frame.
    pub fn schreyer(target_order: Self, leading_monomials: Vec<ModuleMonomial>) -> Self {
        Self::Schreyer {
            target_order: Box::new(target_order),
            leading_monomials,
        }
    }

    /// Target order and ordered leading frame for a Schreyer order.
    pub fn schreyer_data(&self) -> Option<(&Self, &[ModuleMonomial])> {
        match self {
            Self::Schreyer {
                target_order,
                leading_monomials,
            } => Some((target_order, leading_monomials)),
            _ => None,
        }
    }

    #[allow(dead_code)] // consumed by the specialized module kernel in C2b
    pub(crate) fn kernel_position_order(&self) -> Option<module_core::PositionModuleOrder> {
        match self {
            Self::TermOverPosition(_) => Some(module_core::PositionModuleOrder::TermOverPosition),
            Self::PositionOverTerm(_) => Some(module_core::PositionModuleOrder::PositionOverTerm),
            Self::Schreyer { .. } => None,
        }
    }

    /// Serialize this mathematical module order as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a standard module order.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for ModuleMonomialOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModuleMonomialOrderWire {
            type_id: Self::TYPE_ID.to_owned(),
            kind: match self {
                Self::TermOverPosition(_) => "term_over_position",
                Self::PositionOverTerm(_) => "position_over_term",
                Self::Schreyer { .. } => "schreyer",
            }
            .to_owned(),
            base_order: match self {
                Self::TermOverPosition(order) | Self::PositionOverTerm(order) => {
                    Some(order.clone())
                }
                Self::Schreyer { .. } => None,
            },
            target_order: match self {
                Self::Schreyer { target_order, .. } => Some(target_order.clone()),
                _ => None,
            },
            leading_monomials: match self {
                Self::Schreyer {
                    leading_monomials, ..
                } => Some(leading_monomials.clone()),
                _ => None,
            },
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModuleMonomialOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleMonomialOrderWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        match wire.kind.as_str() {
            "term_over_position" => {
                match (wire.base_order, wire.target_order, wire.leading_monomials) {
                    (Some(order), None, None) => Ok(Self::TermOverPosition(order)),
                    _ => Err(serde::de::Error::custom(
                        "term-over-position order has inconsistent fields",
                    )),
                }
            }
            "position_over_term" => {
                match (wire.base_order, wire.target_order, wire.leading_monomials) {
                    (Some(order), None, None) => Ok(Self::PositionOverTerm(order)),
                    _ => Err(serde::de::Error::custom(
                        "position-over-term order has inconsistent fields",
                    )),
                }
            }
            "schreyer" => match (wire.base_order, wire.target_order, wire.leading_monomials) {
                (None, Some(target_order), Some(leading_monomials)) => Ok(Self::Schreyer {
                    target_order,
                    leading_monomials,
                }),
                _ => Err(serde::de::Error::custom(
                    "Schreyer order has inconsistent fields",
                )),
            },
            kind => Err(serde::de::Error::custom(format!(
                "unknown module monomial order {kind:?}"
            ))),
        }
    }
}

impl fmt::Display for ModuleMonomialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TermOverPosition(order) => write!(f, "term-over-position ({order})"),
            Self::PositionOverTerm(order) => write!(f, "position-over-term ({order})"),
            Self::Schreyer {
                target_order,
                leading_monomials,
            } => write!(
                f,
                "Schreyer(target={target_order}, frame={})",
                leading_monomials.len()
            ),
        }
    }
}

impl SemanticType for ModuleMonomialOrder {
    const TYPE_ID: &'static str = "noether/homalg/ModuleMonomialOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        match self {
            Self::TermOverPosition(order) => {
                output.push(0);
                append_component(&mut output, &order.canonical_bytes()?)?;
            }
            Self::PositionOverTerm(order) => {
                output.push(1);
                append_component(&mut output, &order.canonical_bytes()?)?;
            }
            Self::Schreyer {
                target_order,
                leading_monomials,
            } => {
                output.push(2);
                append_component(&mut output, &target_order.canonical_bytes()?)?;
                append_usize(&mut output, leading_monomials.len())?;
                for leading in leading_monomials {
                    append_component(&mut output, &leading.canonical_bytes()?)?;
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_MONOMIAL_ORDER_DESCRIPTION
    }
}

fn validate_labels(rank: usize, labels: Option<&[String]>) -> Result<(), ModuleError> {
    let Some(labels) = labels else {
        return Ok(());
    };
    if labels.len() != rank {
        return Err(ModuleError::BasisLabelCount {
            rank,
            actual: labels.len(),
        });
    }
    if let Some(position) = labels.iter().position(String::is_empty) {
        return Err(ModuleError::EmptyBasisLabel { position });
    }
    let mut seen = HashSet::with_capacity(labels.len());
    if let Some(label) = labels.iter().find(|label| !seen.insert(label.as_str())) {
        return Err(ModuleError::DuplicateBasisLabel {
            label: label.clone(),
        });
    }
    Ok(())
}

fn validate_shifts(
    rank: usize,
    grading_rank: Option<usize>,
    shifts: Option<&[Degree]>,
) -> Result<(), ModuleError> {
    match (grading_rank, shifts) {
        (None, None) => Ok(()),
        (Some(grading_rank), Some(shifts)) => {
            if shifts.len() != rank {
                return Err(ModuleError::DegreeShiftCount {
                    rank,
                    actual: shifts.len(),
                });
            }
            if let Some((position, shift)) = shifts
                .iter()
                .enumerate()
                .find(|(_, shift)| shift.rank() != grading_rank)
            {
                return Err(ModuleError::DegreeRank {
                    position,
                    expected: grading_rank,
                    actual: shift.rank(),
                });
            }
            Ok(())
        }
        _ => Err(ModuleError::GradingMismatch {
            operation: "free-module construction",
        }),
    }
}

fn combine_gradings<R, F>(
    left: &FreeModule<R>,
    right: &FreeModule<R>,
    operation: &'static str,
    combine: F,
) -> Result<(Option<usize>, Option<Vec<Degree>>), ModuleError>
where
    R: Ring,
    F: FnOnce(&[Degree], &[Degree]) -> Result<Vec<Degree>, ModuleError>,
{
    match (
        left.grading_rank,
        left.degree_shifts.as_deref(),
        right.grading_rank,
        right.degree_shifts.as_deref(),
    ) {
        (None, None, None, None) => Ok((None, None)),
        (Some(left_rank), Some(left), Some(right_rank), Some(right)) if left_rank == right_rank => {
            Ok((Some(left_rank), Some(combine(left, right)?)))
        }
        _ => Err(ModuleError::GradingMismatch { operation }),
    }
}

fn require_same_ring<R: Ring>(left: &R, right: &R) -> Result<(), ModuleError> {
    if left == right {
        Ok(())
    } else {
        Err(context_mismatch("module operation").into())
    }
}

fn context_mismatch(operation: &'static str) -> ContextMismatchError {
    ContextMismatchError {
        expected: format!("the exact coefficient context required by {operation}"),
        actual: "an unequal coefficient context".to_owned(),
    }
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value)
        .map_err(|_| NoetherError::Serialization("length does not fit in u64".to_owned()))?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

fn append_optional_strings(
    output: &mut Vec<u8>,
    values: Option<&[String]>,
) -> Result<(), NoetherError> {
    let Some(values) = values else {
        output.push(0);
        return Ok(());
    };
    output.push(1);
    append_usize(output, values.len())?;
    for value in values {
        append_component(output, value.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Integer, Z};

    use super::*;

    fn cyclic_module(modulus: i64) -> PresentedModule<crate::IntegerRing> {
        let relations = FreeModule::new(Z, 1);
        let generators = FreeModule::new(Z, 1);
        PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                relations,
                generators,
                Matrix::from_rows(Z, [[Integer::from(modulus)]]).unwrap(),
            )
            .unwrap(),
        )
    }

    fn scalar_free_map(
        source: FreeModule<crate::IntegerRing>,
        target: FreeModule<crate::IntegerRing>,
        scalar: i64,
    ) -> ModuleHomomorphism<crate::IntegerRing> {
        ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(Z, [[Integer::from(scalar)]]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn dense_and_sparse_maps_share_application_and_composition_semantics() {
        let source = FreeModule::new(Z, 2);
        let middle = FreeModule::new(Z, 2);
        let target = FreeModule::new(Z, 1);
        let dense = ModuleHomomorphism::from_matrix(
            source.clone(),
            middle.clone(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(1), Integer::from(2)],
                    [Integer::from(0), Integer::from(1)],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let dense_as_sparse = ModuleHomomorphism::from_sparse_matrix(
            dense.source().clone(),
            dense.target().clone(),
            SparseMatrix::from_dense(&dense.matrix()).unwrap(),
        )
        .unwrap();
        assert_eq!(dense, dense_as_sparse);
        assert_eq!(
            dense.canonical_bytes().unwrap(),
            dense_as_sparse.canonical_bytes().unwrap()
        );
        let sparse = ModuleHomomorphism::from_sparse_matrix(
            middle.clone(),
            target.clone(),
            SparseMatrix::from_triplets(
                Z,
                1,
                2,
                [(0, 0, Integer::from(3)), (0, 1, Integer::from(-1))],
            )
            .unwrap(),
        )
        .unwrap();
        let element = ModuleElement::from_coordinates(
            source,
            Vector::from_values(Z, [Integer::from(2), Integer::from(5)]).unwrap(),
        )
        .unwrap();
        let composed = sparse.compose(&dense).unwrap();
        assert_eq!(
            composed
                .apply(&element)
                .unwrap()
                .coordinates()
                .values()
                .collect::<Vec<_>>(),
            vec![Integer::from(31)]
        );
        assert_eq!(
            composed.apply(&element).unwrap(),
            sparse.apply(&dense.apply(&element).unwrap()).unwrap()
        );
    }

    #[test]
    fn free_module_maps_build_exact_pair_copair_and_direct_sum_blocks() {
        let one = FreeModule::new(Z, 1);
        let two = FreeModule::new(Z, 2);
        let left = ModuleHomomorphism::from_matrix(
            one.clone(),
            two.clone(),
            Matrix::from_rows(Z, [[Integer::from(2)], [Integer::from(3)]]).unwrap(),
        )
        .unwrap();
        let right = ModuleHomomorphism::from_matrix(
            one.clone(),
            two.clone(),
            Matrix::from_rows(Z, [[Integer::from(5)], [Integer::from(7)]]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            left.copair(&right).unwrap().matrix(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(2), Integer::from(5)],
                    [Integer::from(3), Integer::from(7)]
                ]
            )
            .unwrap()
        );
        assert_eq!(
            left.pair(&right).unwrap().matrix(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(2)],
                    [Integer::from(3)],
                    [Integer::from(5)],
                    [Integer::from(7)]
                ]
            )
            .unwrap()
        );
        assert_eq!(
            left.direct_sum(&right).unwrap().matrix(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(2), Integer::zero()],
                    [Integer::from(3), Integer::zero()],
                    [Integer::zero(), Integer::from(5)],
                    [Integer::zero(), Integer::from(7)]
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn free_module_map_tensor_uses_the_lexicographic_kronecker_product() {
        let left_source = FreeModule::new(Z, 2);
        let left_target = FreeModule::new(Z, 2);
        let right_source = FreeModule::new(Z, 1);
        let right_target = FreeModule::new(Z, 2);
        let left = ModuleHomomorphism::from_matrix(
            left_source,
            left_target,
            Matrix::from_rows(
                Z,
                [
                    [Integer::one(), Integer::from(2)],
                    [Integer::from(3), Integer::from(4)],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let right = ModuleHomomorphism::from_matrix(
            right_source,
            right_target,
            Matrix::from_rows(Z, [[Integer::from(5)], [Integer::from(6)]]).unwrap(),
        )
        .unwrap();

        assert_eq!(
            left.tensor(&right).unwrap().matrix(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(5), Integer::from(10)],
                    [Integer::from(6), Integer::from(12)],
                    [Integer::from(15), Integer::from(20)],
                    [Integer::from(18), Integer::from(24)],
                ],
            )
            .unwrap()
        );

        let sparse_left = ModuleHomomorphism::from_sparse_matrix(
            FreeModule::new(Z, 2),
            FreeModule::new(Z, 2),
            SparseMatrix::from_triplets(
                Z,
                2,
                2,
                [(0, 0, Integer::from(2)), (1, 1, Integer::from(3))],
            )
            .unwrap(),
        )
        .unwrap();
        let sparse_right = ModuleHomomorphism::identity(FreeModule::new(Z, 2)).unwrap();
        let sparse_tensor = sparse_left.tensor(&sparse_right).unwrap();
        assert!(sparse_tensor.is_sparse());
        assert_eq!(sparse_tensor.sparse_matrix().unwrap().nnz(), 4);
    }

    #[test]
    fn graded_free_module_operations_preserve_exact_shifts() {
        let left = FreeModule::new(Z, 2)
            .with_basis_labels(vec!["a".to_owned(), "b".to_owned()])
            .unwrap()
            .with_degree_shifts(
                2,
                vec![
                    Degree::new(vec![Integer::from(-1), Integer::zero()]),
                    Degree::new(vec![Integer::zero(), Integer::one()]),
                ],
            )
            .unwrap();
        let right = FreeModule::new(Z, 1)
            .with_basis_labels(vec!["c".to_owned()])
            .unwrap()
            .with_degree_shifts(
                2,
                vec![Degree::new(vec![Integer::from(2), Integer::from(3)])],
            )
            .unwrap();

        let tensor = left.tensor(&right).unwrap();
        assert_eq!(tensor.rank(), 2);
        assert_eq!(
            tensor.degree_shifts().unwrap(),
            [
                Degree::new(vec![Integer::one(), Integer::from(3)]),
                Degree::new(vec![Integer::from(2), Integer::from(4)]),
            ]
        );
        assert_eq!(
            left.dual().unwrap().degree_shifts().unwrap()[0],
            Degree::new(vec![Integer::one(), Integer::zero()])
        );
    }

    #[test]
    fn submodules_and_presentations_retain_their_exact_generator_maps() {
        let ambient = FreeModule::new(Z, 2);
        let generators = vec![
            ambient.basis_element(0).unwrap(),
            ambient.basis_element(1).unwrap(),
        ];
        let submodule = Submodule::from_generators(ambient.clone(), generators).unwrap();
        assert_eq!(submodule.generator_count(), 2);
        assert_eq!(submodule.generator_map().matrix(), Matrix::identity(Z, 2));

        let presented = PresentedModule::cokernel(submodule.generator_map().clone());
        assert_eq!(presented.generator_module(), &ambient);
        assert_eq!(presented.generator_count(), 2);
        assert_eq!(presented.relation_count(), 2);
    }

    #[test]
    fn invalid_basis_metadata_and_map_shapes_fail_closed() {
        assert!(matches!(
            FreeModule::new(Z, 2).with_basis_labels(vec!["x".to_owned(), "x".to_owned()]),
            Err(ModuleError::DuplicateBasisLabel { .. })
        ));
        let source = FreeModule::new(Z, 2);
        let target = FreeModule::new(Z, 1);
        assert!(matches!(
            ModuleHomomorphism::from_matrix(source, target, Matrix::zeros(Z, 2, 1)),
            Err(ModuleError::MatrixShape {
                expected: (1, 2),
                actual: (2, 1)
            })
        ));
    }

    #[test]
    fn public_module_order_routes_to_the_kernel_order_without_exposing_dispatch() {
        let order = ModuleMonomialOrder::TermOverPosition(MonomialOrder::Lexicographic);
        assert_eq!(
            order.kernel_position_order(),
            Some(module_core::PositionModuleOrder::TermOverPosition)
        );
        assert_eq!(
            ModuleMonomialOrder::from_json(&order.to_json().unwrap()).unwrap(),
            order
        );
    }

    #[test]
    fn module_values_round_trip_through_typed_json_and_revalidate() {
        let module = FreeModule::new(Z, 2)
            .with_basis_labels(vec!["u".to_owned(), "v".to_owned()])
            .unwrap()
            .with_degree_shifts(
                1,
                vec![
                    Degree::new(vec![Integer::from(-1)]),
                    Degree::new(vec![Integer::from(2)]),
                ],
            )
            .unwrap();
        let element = module.basis_element(1).unwrap();
        let identity = ModuleHomomorphism::identity(module.clone()).unwrap();
        let submodule = Submodule::from_generator_map(module.clone(), identity.clone()).unwrap();
        let presented = PresentedModule::cokernel(identity);

        assert_eq!(
            FreeModule::<crate::IntegerRing>::from_json(&module.to_json().unwrap()).unwrap(),
            module
        );
        assert_eq!(
            ModuleElement::<crate::IntegerRing>::from_json(&element.to_json().unwrap()).unwrap(),
            element
        );
        assert_eq!(
            Submodule::<crate::IntegerRing>::from_json(&submodule.to_json().unwrap())
                .unwrap()
                .canonical_bytes()
                .unwrap(),
            submodule.canonical_bytes().unwrap()
        );
        assert_eq!(
            PresentedModule::<crate::IntegerRing>::from_json(&presented.to_json().unwrap())
                .unwrap(),
            presented
        );
    }

    #[test]
    fn presented_module_maps_verify_commuting_squares_and_compose() {
        let z8 = cyclic_module(8);
        let z4 = cyclic_module(4);
        let z2 = cyclic_module(2);
        let reduction_8_4 = PresentedModuleHomomorphism::from_lifts(
            z8.clone(),
            z4.clone(),
            scalar_free_map(
                z8.generator_module().clone(),
                z4.generator_module().clone(),
                1,
            ),
            scalar_free_map(
                z8.relation_module().clone(),
                z4.relation_module().clone(),
                2,
            ),
        )
        .unwrap();
        let reduction_4_2 = PresentedModuleHomomorphism::from_lifts(
            z4.clone(),
            z2.clone(),
            scalar_free_map(
                z4.generator_module().clone(),
                z2.generator_module().clone(),
                1,
            ),
            scalar_free_map(
                z4.relation_module().clone(),
                z2.relation_module().clone(),
                2,
            ),
        )
        .unwrap();

        let reduction_8_2 = reduction_4_2.compose(&reduction_8_4).unwrap();
        assert_eq!(
            reduction_8_2.generator_lift().matrix(),
            Matrix::from_rows(Z, [[Integer::one()]]).unwrap()
        );
        assert_eq!(
            reduction_8_2.relation_lift().matrix(),
            Matrix::from_rows(Z, [[Integer::from(4)]]).unwrap()
        );
        reduction_8_2.verify().unwrap();
        assert_eq!(
            PresentedModuleHomomorphism::<crate::IntegerRing>::from_json(
                &reduction_8_2.to_json().unwrap()
            )
            .unwrap(),
            reduction_8_2
        );

        let identity = PresentedModuleHomomorphism::identity(z4.clone()).unwrap();
        assert_eq!(identity.compose(&reduction_8_4).unwrap(), reduction_8_4);
        assert_eq!(
            PresentedModuleHomomorphism::zero(z4.clone(), z2.clone())
                .unwrap()
                .add(&reduction_4_2)
                .unwrap(),
            reduction_4_2
        );

        let invalid = PresentedModuleHomomorphism::from_lifts(
            z4.clone(),
            z2.clone(),
            scalar_free_map(
                z4.generator_module().clone(),
                z2.generator_module().clone(),
                1,
            ),
            scalar_free_map(
                z4.relation_module().clone(),
                z2.relation_module().clone(),
                1,
            ),
        );
        assert_eq!(invalid, Err(ModuleError::RelationDiagramDoesNotCommute));
    }

    #[test]
    fn presented_tensor_products_retain_both_relation_families_and_map_evidence() {
        let z4 = cyclic_module(4);
        let z6 = cyclic_module(6);
        let z3 = cyclic_module(3);

        let tensor = z4.tensor(&z6).unwrap();
        assert_eq!(tensor.generator_count(), 1);
        assert_eq!(tensor.relation_count(), 2);
        assert_eq!(
            tensor.presentation().matrix(),
            Matrix::from_rows(Z, [[Integer::from(4), Integer::from(6)]]).unwrap()
        );

        let reduction_6_3 = PresentedModuleHomomorphism::from_lifts(
            z6.clone(),
            z3.clone(),
            scalar_free_map(
                z6.generator_module().clone(),
                z3.generator_module().clone(),
                1,
            ),
            scalar_free_map(
                z6.relation_module().clone(),
                z3.relation_module().clone(),
                2,
            ),
        )
        .unwrap();
        let tensor_map = PresentedModuleHomomorphism::identity(z4.clone())
            .unwrap()
            .tensor(&reduction_6_3)
            .unwrap();
        tensor_map.verify().unwrap();
        assert_eq!(tensor_map.source(), &tensor);
        assert_eq!(tensor_map.target(), &z4.tensor(&z3).unwrap());
        assert_eq!(
            tensor_map.relation_lift().matrix(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::one(), Integer::zero()],
                    [Integer::zero(), Integer::from(2)],
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn hom_presentation_map_is_dual_precomposition_tensored_with_the_target() {
        let z4 = cyclic_module(4);
        let z6 = cyclic_module(6);

        let precomposition = z4.hom_presentation_map(&z6).unwrap();
        precomposition.verify().unwrap();
        assert_eq!(precomposition.source(), &z6);
        assert_eq!(precomposition.target(), &z6);
        assert_eq!(
            precomposition.generator_lift().matrix(),
            Matrix::from_rows(Z, [[Integer::from(4)]]).unwrap()
        );
        assert_eq!(
            precomposition.relation_lift().matrix(),
            Matrix::from_rows(Z, [[Integer::from(4)]]).unwrap()
        );
    }

    #[test]
    fn presented_module_map_cokernel_uses_the_target_relations_and_image() {
        let z4 = cyclic_module(4);
        let z2 = cyclic_module(2);
        let reduction = PresentedModuleHomomorphism::from_lifts(
            z4.clone(),
            z2.clone(),
            scalar_free_map(
                z4.generator_module().clone(),
                z2.generator_module().clone(),
                1,
            ),
            scalar_free_map(
                z4.relation_module().clone(),
                z2.relation_module().clone(),
                2,
            ),
        )
        .unwrap();
        let cokernel = reduction.cokernel().unwrap();
        assert_eq!(cokernel.generator_count(), 1);
        assert_eq!(cokernel.relation_count(), 2);
        assert_eq!(
            cokernel.presentation().matrix(),
            Matrix::from_rows(Z, [[Integer::from(2), Integer::one()]]).unwrap()
        );
    }
}
