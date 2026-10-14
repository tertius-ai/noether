// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Symmetric and exterior powers of free and presented modules.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

use crate::commalg::Degree;
use crate::linalg::{LinalgError, Matrix};
use crate::{ContextMismatchError, Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::{FreeModule, ModuleError, ModuleHomomorphism, PresentedModule};

const MAX_MODULE_POWER_RANK: usize = 1_000_000;
const MAX_MODULE_POWER_BASIS_CELLS: usize = 8_000_000;
pub(super) const MAX_MODULE_POWER_MATRIX_CELLS: usize = 4_000_000;
pub(super) const MAX_MODULE_POWER_EXPANSION_WORK: usize = 16_000_000;

static MODULE_POWER_KIND_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModulePowerKind@1",
    rust_path: "noether::homalg::ModulePowerKind",
    python_path: Some("noether.homalg.ModulePowerKind"),
};
static FREE_MODULE_POWER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FreeModulePower@1",
    rust_path: "noether::homalg::FreeModulePower",
    python_path: Some("noether.homalg.FreeModulePower"),
};
static PRESENTED_MODULE_POWER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PresentedModulePower@1",
    rust_path: "noether::homalg::PresentedModulePower",
    python_path: Some("noether.homalg.PresentedModulePower"),
};

/// Symmetric or alternating power construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModulePowerKind {
    /// Symmetric power with the standard commutative monomial basis.
    Symmetric,
    /// Exterior power with the increasing wedge basis.
    Exterior,
}

impl ModulePowerKind {
    /// Stable short name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Symmetric => "symmetric",
            Self::Exterior => "exterior",
        }
    }
    /// Serialize this exact construction kind.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
    /// Deserialize an exact construction kind.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for ModulePowerKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
impl SemanticType for ModulePowerKind {
    const TYPE_ID: &'static str = "noether/homalg/ModulePowerKind@1";
    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.name().as_bytes().to_vec())
    }
    fn describe() -> &'static TypeDescription {
        &MODULE_POWER_KIND_DESCRIPTION
    }
}

/// Failure to construct or verify a symmetric or exterior module power.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModulePowerError {
    /// Based-module construction failed.
    Module(ModuleError),
    /// Matrix construction failed.
    Linalg(LinalgError),
    /// Coefficient operations crossed exact contexts.
    ContextMismatch(ContextMismatchError),
    /// A deterministic basis-size limit was exceeded.
    RankLimit {
        /// Required rank when it fit in `usize`.
        requested: Option<usize>,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// A canonical basis table would retain too many exponent cells.
    BasisTableLimit {
        /// Required cells when the product fit in `usize`.
        requested: Option<usize>,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// A power construction would materialize too many dense matrix cells.
    MatrixCellLimit {
        /// Stable construction step.
        operation: &'static str,
        /// Required cells when the product fit in `usize`.
        requested: Option<usize>,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// Expanding an induced map would inspect too many coefficient choices.
    ExpansionWorkLimit {
        /// Stable construction step.
        operation: &'static str,
        /// Required cumulative work when the sum fit in `usize`.
        requested: Option<usize>,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// An induced-map operation received the wrong power construction.
    KindMismatch {
        /// Construction required by the operation.
        expected: ModulePowerKind,
        /// Construction carried by the rejected value.
        actual: ModulePowerKind,
    },
    /// An element was not homogeneous of the degree required by an induced map.
    DegreeMismatch {
        /// Degree required by the operation.
        expected: usize,
        /// Degree of the rejected homogeneous element, or `None` when it had
        /// nonzero terms in more than one degree.
        actual: Option<usize>,
    },
    /// Retained construction evidence did not replay.
    InvalidEvidence(String),
}

impl fmt::Display for ModulePowerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::Linalg(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::RankLimit { requested, limit } => write!(
                formatter,
                "module power rank {:?} exceeds the fixed limit {limit}",
                requested
            ),
            Self::BasisTableLimit { requested, limit } => write!(
                formatter,
                "module-power basis table requires {requested:?} cells, exceeding the fixed limit {limit}"
            ),
            Self::MatrixCellLimit {
                operation,
                requested,
                limit,
            } => write!(
                formatter,
                "{operation} requires {requested:?} dense cells, exceeding the fixed limit {limit}"
            ),
            Self::ExpansionWorkLimit {
                operation,
                requested,
                limit,
            } => write!(
                formatter,
                "{operation} requires {requested:?} coefficient inspections, exceeding the fixed limit {limit}"
            ),
            Self::KindMismatch { expected, actual } => write!(
                formatter,
                "module-power map has kind {actual}; expected {expected}"
            ),
            Self::DegreeMismatch { expected, actual } => write!(
                formatter,
                "module-power map expects homogeneous degree {expected}; received {actual:?}"
            ),
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid module-power evidence: {detail}")
            }
        }
    }
}
impl std::error::Error for ModulePowerError {}
impl From<ModuleError> for ModulePowerError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}
impl From<LinalgError> for ModulePowerError {
    fn from(value: LinalgError) -> Self {
        Self::Linalg(value)
    }
}
impl From<ContextMismatchError> for ModulePowerError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A canonical symmetric or exterior power of one based free module.
///
/// `basis_exponents[i][j]` is the multiplicity of source basis vector `j` in
/// output basis vector `i`; exterior exponents are always zero or one.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields, bound(serialize = "R: Serialize"))]
pub struct FreeModulePower<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: FreeModule<R>,
    kind: ModulePowerKind,
    degree: usize,
    basis_exponents: Vec<Vec<usize>>,
    module: FreeModule<R>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "R: Ring + Deserialize<'de>"))]
struct FreeModulePowerWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: FreeModule<R>,
    kind: ModulePowerKind,
    degree: usize,
    basis_exponents: Vec<Vec<usize>>,
    module: FreeModule<R>,
}

impl<'de, R> Deserialize<'de> for FreeModulePower<R>
where
    R: Ring + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FreeModulePowerWire::<R>::deserialize(deserializer)?;
        let value = Self {
            type_id: wire.type_id,
            source: wire.source,
            kind: wire.kind,
            degree: wire.degree,
            basis_exponents: wire.basis_exponents,
            module: wire.module,
        };
        value.verify().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl<R: Ring> FreeModulePower<R> {
    /// Construct a canonical power of a based free module.
    pub fn new(
        source: FreeModule<R>,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let basis_exponents = power_basis(source.rank(), degree, kind)?;
        let labels = source.basis_labels().map(|labels| {
            basis_exponents
                .iter()
                .map(|exponent| power_label(labels, exponent, kind))
                .collect::<Vec<_>>()
        });
        let (grading_rank, shifts) = match (source.grading_rank(), source.degree_shifts()) {
            (Some(rank), Some(source_shifts)) => (
                Some(rank),
                Some(
                    basis_exponents
                        .iter()
                        .map(|exponent| power_shift(rank, source_shifts, exponent))
                        .collect::<Vec<_>>(),
                ),
            ),
            (None, None) => (None, None),
            _ => {
                return Err(ModulePowerError::InvalidEvidence(
                    "source free module has inconsistent grading metadata".to_owned(),
                ))
            }
        };
        let module = FreeModule::with_basis_data(
            source.ring().clone(),
            basis_exponents.len(),
            labels,
            grading_rank,
            shifts,
        )?;
        Ok(Self {
            type_id: Self::TYPE_ID.to_owned(),
            source,
            kind,
            degree,
            basis_exponents,
            module,
        })
    }

    /// Source based free module.
    pub fn source(&self) -> &FreeModule<R> {
        &self.source
    }
    /// Symmetric or exterior construction.
    pub const fn kind(&self) -> ModulePowerKind {
        self.kind
    }
    /// Nonnegative power degree.
    pub const fn degree(&self) -> usize {
        self.degree
    }
    /// Canonical exponent vectors indexing the output basis.
    pub fn basis_exponents(&self) -> &[Vec<usize>] {
        &self.basis_exponents
    }
    /// Constructed based free module.
    pub fn module(&self) -> &FreeModule<R> {
        &self.module
    }
    /// Recompute basis combinatorics and inherited metadata.
    pub fn verify(&self) -> Result<(), ModulePowerError> {
        if self.type_id != Self::TYPE_ID {
            return Err(ModulePowerError::InvalidEvidence(
                "free-module power has the wrong type identity".to_owned(),
            ));
        }
        if Self::new(self.source.clone(), self.kind, self.degree)? != *self {
            return Err(ModulePowerError::InvalidEvidence(
                "recomputation changed the free-module power".to_owned(),
            ));
        }
        Ok(())
    }
    /// Serialize the complete construction.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
    /// Deserialize and revalidate the complete construction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Ring> FreeModule<R> {
    /// Construct the degree-`degree` symmetric power.
    pub fn symmetric_power(&self, degree: usize) -> Result<FreeModulePower<R>, ModulePowerError> {
        FreeModulePower::new(self.clone(), ModulePowerKind::Symmetric, degree)
    }
    /// Construct the degree-`degree` exterior power.
    pub fn exterior_power(&self, degree: usize) -> Result<FreeModulePower<R>, ModulePowerError> {
        FreeModulePower::new(self.clone(), ModulePowerKind::Exterior, degree)
    }
}

impl<R: Ring> fmt::Debug for FreeModulePower<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FreeModulePower")
            .field("kind", &self.kind)
            .field("degree", &self.degree)
            .field("rank", &self.module.rank())
            .finish()
    }
}
impl<R: Ring> fmt::Display for FreeModulePower<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FreeModulePower(kind={}, degree={}, rank={})",
            self.kind,
            self.degree,
            self.module.rank()
        )
    }
}
impl<R: Ring> SemanticType for FreeModulePower<R> {
    const TYPE_ID: &'static str = "noether/homalg/FreeModulePower@1";
    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        output.extend_from_slice(self.kind.name().as_bytes());
        output.extend_from_slice(&(self.degree as u64).to_le_bytes());
        Ok(output)
    }
    fn describe() -> &'static TypeDescription {
        &FREE_MODULE_POWER_DESCRIPTION
    }
}

/// A canonical symmetric or exterior power presentation of a presented module.
///
/// This value owns a checked cokernel presentation, not quotient-element
/// normal forms. Exact arithmetic in [`super::ExteriorElement`] is therefore
/// admitted for based free modules only; presented-module power maps descend
/// through explicit commuting-square certificates without claiming equality
/// or normal forms for quotient elements.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize, R::Element: Serialize")
)]
pub struct PresentedModulePower<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: PresentedModule<R>,
    kind: ModulePowerKind,
    degree: usize,
    module: PresentedModule<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PresentedModulePowerWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: PresentedModule<R>,
    kind: ModulePowerKind,
    degree: usize,
    module: PresentedModule<R>,
}

impl<'de, R> Deserialize<'de> for PresentedModulePower<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PresentedModulePowerWire::<R>::deserialize(deserializer)?;
        let value = Self {
            type_id: wire.type_id,
            source: wire.source,
            kind: wire.kind,
            degree: wire.degree,
            module: wire.module,
        };
        value.verify().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl<R: Ring> PresentedModulePower<R> {
    /// Construct the standard quotient presentation of a module power.
    pub fn new(
        source: PresentedModule<R>,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let target_power = FreeModulePower::new(source.generator_module().clone(), kind, degree)?;
        let previous_basis = if degree == 0 {
            Vec::new()
        } else {
            power_basis(source.generator_count(), degree - 1, kind)?
        };
        let column_count = source
            .relation_count()
            .checked_mul(previous_basis.len())
            .ok_or(ModulePowerError::RankLimit {
                requested: None,
                limit: MAX_MODULE_POWER_RANK,
            })?;
        if column_count > MAX_MODULE_POWER_RANK {
            return Err(ModulePowerError::RankLimit {
                requested: Some(column_count),
                limit: MAX_MODULE_POWER_RANK,
            });
        }
        let ring = source.generator_module().ring();
        let relation_module = FreeModule::new(ring.clone(), column_count);
        let target_module = target_power.module().clone();
        require_power_matrix_cells(
            "presented module-power relation matrix",
            target_module.rank(),
            column_count,
        )?;
        let mut rows = vec![vec![ring.zero(); column_count]; target_module.rank()];
        if degree > 0 {
            let target_index = target_power
                .basis_exponents()
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, exponent)| (exponent, index))
                .collect::<BTreeMap<_, _>>();
            let presentation = source.presentation().matrix().rows().collect::<Vec<_>>();
            for (generator, coefficients) in presentation.iter().enumerate() {
                for (relation, source_coefficient) in coefficients.iter().enumerate() {
                    for (basis_index, exponent) in previous_basis.iter().enumerate() {
                        let column = relation * previous_basis.len() + basis_index;
                        if kind == ModulePowerKind::Exterior && exponent[generator] != 0 {
                            continue;
                        }
                        let mut target_exponent = exponent.clone();
                        target_exponent[generator] += 1;
                        let row = *target_index.get(&target_exponent).ok_or_else(|| {
                            ModulePowerError::InvalidEvidence(
                                "a power relation produced no target basis vector".to_owned(),
                            )
                        })?;
                        let mut coefficient = source_coefficient.clone();
                        if kind == ModulePowerKind::Exterior
                            && exponent[..generator].iter().sum::<usize>() % 2 == 1
                        {
                            coefficient = ring.neg(&coefficient)?;
                        }
                        rows[row][column] = ring.add(&rows[row][column], &coefficient)?;
                    }
                }
            }
        }
        let matrix = if rows.is_empty() {
            Matrix::zeros(ring.clone(), target_module.rank(), column_count)
        } else {
            Matrix::from_rows(ring.clone(), rows)?
        };
        let presentation = ModuleHomomorphism::from_matrix(relation_module, target_module, matrix)?;
        let module = PresentedModule::cokernel(presentation);
        Ok(Self {
            type_id: Self::TYPE_ID.to_owned(),
            source,
            kind,
            degree,
            module,
        })
    }

    /// Source presented module.
    pub fn source(&self) -> &PresentedModule<R> {
        &self.source
    }
    /// Symmetric or exterior construction.
    pub const fn kind(&self) -> ModulePowerKind {
        self.kind
    }
    /// Nonnegative power degree.
    pub const fn degree(&self) -> usize {
        self.degree
    }
    /// Resulting finitely presented module.
    pub fn module(&self) -> &PresentedModule<R> {
        &self.module
    }
    /// Recompute the complete induced presentation.
    pub fn verify(&self) -> Result<(), ModulePowerError> {
        if self.type_id != Self::TYPE_ID {
            return Err(ModulePowerError::InvalidEvidence(
                "presented-module power has the wrong type identity".to_owned(),
            ));
        }
        if Self::new(self.source.clone(), self.kind, self.degree)? != *self {
            return Err(ModulePowerError::InvalidEvidence(
                "recomputation changed the presented-module power".to_owned(),
            ));
        }
        Ok(())
    }
    /// Serialize the complete induced presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
    /// Deserialize and independently revalidate the induced presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Ring> PresentedModule<R> {
    /// Construct the degree-`degree` symmetric power presentation.
    pub fn symmetric_power(
        &self,
        degree: usize,
    ) -> Result<PresentedModulePower<R>, ModulePowerError> {
        PresentedModulePower::new(self.clone(), ModulePowerKind::Symmetric, degree)
    }
    /// Construct the degree-`degree` exterior power presentation.
    pub fn exterior_power(
        &self,
        degree: usize,
    ) -> Result<PresentedModulePower<R>, ModulePowerError> {
        PresentedModulePower::new(self.clone(), ModulePowerKind::Exterior, degree)
    }
}

impl<R: Ring> fmt::Debug for PresentedModulePower<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PresentedModulePower")
            .field("kind", &self.kind)
            .field("degree", &self.degree)
            .field("generators", &self.module.generator_count())
            .field("relations", &self.module.relation_count())
            .finish()
    }
}
impl<R: Ring> fmt::Display for PresentedModulePower<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PresentedModulePower(kind={}, degree={}, generators={}, relations={})",
            self.kind,
            self.degree,
            self.module.generator_count(),
            self.module.relation_count()
        )
    }
}
impl<R: Ring> SemanticType for PresentedModulePower<R> {
    const TYPE_ID: &'static str = "noether/homalg/PresentedModulePower@1";
    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        output.extend_from_slice(self.kind.name().as_bytes());
        output.extend_from_slice(&(self.degree as u64).to_le_bytes());
        Ok(output)
    }
    fn describe() -> &'static TypeDescription {
        &PRESENTED_MODULE_POWER_DESCRIPTION
    }
}

pub(super) fn power_basis(
    rank: usize,
    degree: usize,
    kind: ModulePowerKind,
) -> Result<Vec<Vec<usize>>, ModulePowerError> {
    let basis_rank = power_rank(rank, degree, kind)?;
    let requested_cells = basis_rank.checked_mul(rank);
    if requested_cells.is_none_or(|cells| cells > MAX_MODULE_POWER_BASIS_CELLS) {
        return Err(ModulePowerError::BasisTableLimit {
            requested: requested_cells,
            limit: MAX_MODULE_POWER_BASIS_CELLS,
        });
    }
    if degree == 0 {
        return Ok(vec![vec![0; rank]]);
    }
    if kind == ModulePowerKind::Exterior && degree == rank {
        return Ok(vec![vec![1; rank]]);
    }
    let mut result = Vec::with_capacity(basis_rank);
    match kind {
        ModulePowerKind::Symmetric => {
            if rank == 0 {
                if degree == 0 {
                    result.push(Vec::new());
                }
            } else {
                symmetric_exponents(rank, degree, 0, &mut vec![0; rank], &mut result)?;
            }
        }
        ModulePowerKind::Exterior => {
            if degree <= rank {
                exterior_exponents(rank, degree, 0, &mut vec![0; rank], &mut result)?;
            }
        }
    }
    debug_assert_eq!(result.len(), basis_rank);
    Ok(result)
}

pub(super) fn require_power_matrix_cells(
    operation: &'static str,
    rows: usize,
    columns: usize,
) -> Result<(), ModulePowerError> {
    let requested = rows.checked_mul(columns);
    if requested.is_none_or(|cells| cells > MAX_MODULE_POWER_MATRIX_CELLS) {
        Err(ModulePowerError::MatrixCellLimit {
            operation,
            requested,
            limit: MAX_MODULE_POWER_MATRIX_CELLS,
        })
    } else {
        Ok(())
    }
}

fn power_rank(
    rank: usize,
    degree: usize,
    kind: ModulePowerKind,
) -> Result<usize, ModulePowerError> {
    let (upper, lower) = match kind {
        ModulePowerKind::Symmetric if rank == 0 => return Ok(usize::from(degree == 0)),
        ModulePowerKind::Symmetric => (
            rank.checked_add(degree)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or(ModulePowerError::RankLimit {
                    requested: None,
                    limit: MAX_MODULE_POWER_RANK,
                })?,
            degree,
        ),
        ModulePowerKind::Exterior if degree > rank => return Ok(0),
        ModulePowerKind::Exterior => (rank, degree),
    };
    checked_binomial(upper, lower)
}

fn checked_binomial(upper: usize, lower: usize) -> Result<usize, ModulePowerError> {
    let lower = lower.min(upper - lower);
    let mut value = 1usize;
    for index in 1..=lower {
        let factor = upper - lower + index;
        value = value
            .checked_mul(factor)
            .and_then(|product| product.checked_div(index))
            .ok_or(ModulePowerError::RankLimit {
                requested: None,
                limit: MAX_MODULE_POWER_RANK,
            })?;
        if value > MAX_MODULE_POWER_RANK {
            return Err(ModulePowerError::RankLimit {
                requested: (index == lower).then_some(value),
                limit: MAX_MODULE_POWER_RANK,
            });
        }
    }
    Ok(value)
}

fn symmetric_exponents(
    rank: usize,
    remaining: usize,
    position: usize,
    current: &mut [usize],
    output: &mut Vec<Vec<usize>>,
) -> Result<(), ModulePowerError> {
    if position + 1 == rank {
        current[position] = remaining;
        return push_bounded(output, current.to_vec());
    }
    for value in (0..=remaining).rev() {
        current[position] = value;
        symmetric_exponents(rank, remaining - value, position + 1, current, output)?;
    }
    Ok(())
}

fn exterior_exponents(
    rank: usize,
    remaining: usize,
    position: usize,
    current: &mut [usize],
    output: &mut Vec<Vec<usize>>,
) -> Result<(), ModulePowerError> {
    if remaining == 0 {
        current[position..].fill(0);
        return push_bounded(output, current.to_vec());
    }
    if position == rank || rank - position < remaining {
        return Ok(());
    }
    current[position] = 1;
    exterior_exponents(rank, remaining - 1, position + 1, current, output)?;
    current[position] = 0;
    exterior_exponents(rank, remaining, position + 1, current, output)
}

fn push_bounded(output: &mut Vec<Vec<usize>>, value: Vec<usize>) -> Result<(), ModulePowerError> {
    if output.len() >= MAX_MODULE_POWER_RANK {
        return Err(ModulePowerError::RankLimit {
            requested: output.len().checked_add(1),
            limit: MAX_MODULE_POWER_RANK,
        });
    }
    output.push(value);
    Ok(())
}

fn power_label(labels: &[String], exponent: &[usize], kind: ModulePowerKind) -> String {
    let separator = if kind == ModulePowerKind::Exterior {
        "^"
    } else {
        "*"
    };
    let factors = labels
        .iter()
        .zip(exponent)
        .filter_map(|(label, power)| match power {
            0 => None,
            1 => Some(label.clone()),
            power => Some(format!("{label}^{power}")),
        })
        .collect::<Vec<_>>();
    if factors.is_empty() {
        "1".to_owned()
    } else {
        factors.join(separator)
    }
}

fn power_shift(rank: usize, shifts: &[Degree], exponent: &[usize]) -> Degree {
    let mut coordinates = vec![Integer::zero(); rank];
    for (power, shift) in exponent.iter().zip(shifts) {
        let power = Integer::from(*power);
        for (target, coordinate) in coordinates.iter_mut().zip(shift.coordinates()) {
            *target = &*target + &(&power * coordinate);
        }
    }
    Degree::new(coordinates)
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialRing};
    use crate::linalg::Matrix;
    use crate::{Integer, Q, Z};

    use super::*;

    #[test]
    fn free_powers_have_canonical_basis_combinatorics() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let module = FreeModule::new(ring, 3);
        let symmetric = module.symmetric_power(2).unwrap();
        let exterior = module.exterior_power(2).unwrap();
        assert_eq!(symmetric.module().rank(), 6);
        assert_eq!(exterior.module().rank(), 3);
        assert_eq!(
            exterior.basis_exponents(),
            &[vec![1, 1, 0], vec![1, 0, 1], vec![0, 1, 1]]
        );
        symmetric.verify().unwrap();
        exterior.verify().unwrap();
    }

    #[test]
    fn presented_powers_induce_relations_functorially() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let source = FreeModule::new(ring.clone(), 1);
        let target = FreeModule::new(ring.clone(), 2);
        let matrix = Matrix::from_rows(ring, [[x.clone()], [x]]).unwrap();
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(source, target, matrix).unwrap(),
        );
        let symmetric = module.symmetric_power(2).unwrap();
        let exterior = module.exterior_power(2).unwrap();
        assert_eq!(symmetric.module().generator_count(), 3);
        assert_eq!(symmetric.module().relation_count(), 2);
        assert_eq!(exterior.module().generator_count(), 1);
        assert_eq!(exterior.module().relation_count(), 2);
        symmetric.verify().unwrap();
        exterior.verify().unwrap();
    }

    #[test]
    fn presented_powers_preserve_zero_target_matrix_shapes() {
        let relations = FreeModule::new(Z, 1);
        let generator = FreeModule::new(Z, 1);
        let times_two = Matrix::from_rows(Z, [[Integer::from(2)]]).unwrap();
        let cyclic = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(relations, generator, times_two).unwrap(),
        );
        let exterior_square = cyclic.exterior_power(2).unwrap();
        assert_eq!(exterior_square.module().generator_count(), 0);
        assert_eq!(exterior_square.module().relation_count(), 1);
        assert_eq!(
            exterior_square.module().presentation().matrix().shape(),
            (0, 1)
        );
        exterior_square.verify().unwrap();

        let relations = FreeModule::new(Z, 1);
        let zero_generators = FreeModule::new(Z, 0);
        let zero_relation = Matrix::zeros(Z, 0, 1);
        let zero_module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(relations, zero_generators, zero_relation).unwrap(),
        );
        let first_power = zero_module.exterior_power(1).unwrap();
        assert_eq!(first_power.module().presentation().matrix().shape(), (0, 1));
        first_power.verify().unwrap();
        let symmetric_first_power = zero_module.symmetric_power(1).unwrap();
        assert_eq!(
            symmetric_first_power
                .module()
                .presentation()
                .matrix()
                .shape(),
            (0, 1)
        );
        symmetric_first_power.verify().unwrap();
    }
}
