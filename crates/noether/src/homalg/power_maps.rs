// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Functorial maps on symmetric and exterior powers.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::linalg::Matrix;
use crate::{CommutativeRing, NoetherError, SemanticType, TypeDescription};

use super::exterior::{exterior_basis, wedge_basis};
use super::powers::{
    require_power_matrix_cells, FreeModulePower, ModulePowerError, ModulePowerKind,
    MAX_MODULE_POWER_EXPANSION_WORK,
};
use super::{
    ExteriorAlgebra, ExteriorElement, ModuleHomomorphism, PresentedModuleHomomorphism,
    PresentedModulePower,
};

static MODULE_POWER_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModulePowerHomomorphism@1",
    rust_path: "noether::homalg::ModulePowerHomomorphism",
    python_path: Some("noether.homalg.ModulePowerHomomorphism"),
};

static PRESENTED_MODULE_POWER_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/PresentedModulePowerHomomorphism@1",
    rust_path: "noether::homalg::PresentedModulePowerHomomorphism",
    python_path: Some("noether.homalg.PresentedModulePowerHomomorphism"),
};

/// A replayable map induced on a symmetric or exterior power of a free-module map.
///
/// The source map, construction kind, degree, power contexts, and resulting
/// exact homomorphism are retained. [`Self::verify`] reconstructs the induced
/// map; for exterior powers it additionally compares every coordinate with
/// the corresponding determinant minor, an independent exact formula.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize, R::Element: Serialize")
)]
pub struct ModulePowerHomomorphism<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source_map: ModuleHomomorphism<R>,
    kind: ModulePowerKind,
    degree: usize,
    source_power: FreeModulePower<R>,
    target_power: FreeModulePower<R>,
    induced: ModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct ModulePowerHomomorphismWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source_map: ModuleHomomorphism<R>,
    kind: ModulePowerKind,
    degree: usize,
    source_power: FreeModulePower<R>,
    target_power: FreeModulePower<R>,
    induced: ModuleHomomorphism<R>,
}

impl<R: CommutativeRing> ModulePowerHomomorphism<R> {
    /// Construct the exact map induced in one nonnegative power degree.
    pub fn new(
        source_map: ModuleHomomorphism<R>,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let source_power = FreeModulePower::new(source_map.source().clone(), kind, degree)?;
        let target_power = FreeModulePower::new(source_map.target().clone(), kind, degree)?;
        let induced = if degree == 0 {
            ModuleHomomorphism::identity(source_power.module().clone())?
                .reframe(source_power.module().clone(), target_power.module().clone())?
        } else if degree == 1 {
            // Both power constructions are definitionally the source module
            // in degree one. Preserve the original dense/native or sparse
            // storage rather than enumerate and rebuild its coefficients.
            debug_assert_eq!(source_power.module(), source_map.source());
            debug_assert_eq!(target_power.module(), source_map.target());
            source_map.clone()
        } else {
            let columns = match kind {
                ModulePowerKind::Symmetric => {
                    symmetric_power_columns(&source_map, &source_power, &target_power)?
                }
                ModulePowerKind::Exterior => {
                    exterior_power_columns(&source_map, &source_power, &target_power)?
                }
            };
            ModuleHomomorphism::from_coordinate_columns(
                source_power.module().clone(),
                target_power.module().clone(),
                columns,
            )?
        };
        Ok(Self {
            type_id: Self::TYPE_ID.to_owned(),
            source_map,
            kind,
            degree,
            source_power,
            target_power,
            induced,
        })
    }

    /// Original based free-module homomorphism.
    pub fn source_map(&self) -> &ModuleHomomorphism<R> {
        &self.source_map
    }

    /// Symmetric or exterior construction.
    pub const fn kind(&self) -> ModulePowerKind {
        self.kind
    }

    /// Nonnegative power degree.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Canonical power of the source based module.
    pub fn source_power(&self) -> &FreeModulePower<R> {
        &self.source_power
    }

    /// Canonical power of the target based module.
    pub fn target_power(&self) -> &FreeModulePower<R> {
        &self.target_power
    }

    /// Exact induced homomorphism between the two power modules.
    pub fn induced(&self) -> &ModuleHomomorphism<R> {
        &self.induced
    }

    /// Compose two induced maps of the same construction and degree.
    ///
    /// The original free-module maps are composed and the power map is
    /// reconstructed. The result is checked against composition of the two
    /// retained induced maps, making functoriality an executable invariant.
    pub fn compose(&self, inner: &Self) -> Result<Self, ModulePowerError> {
        if inner.kind != self.kind {
            return Err(ModulePowerError::KindMismatch {
                expected: self.kind,
                actual: inner.kind,
            });
        }
        if inner.degree != self.degree {
            return Err(ModulePowerError::DegreeMismatch {
                expected: self.degree,
                actual: Some(inner.degree),
            });
        }
        let result = Self::new(
            self.source_map.compose(&inner.source_map)?,
            self.kind,
            self.degree,
        )?;
        if result.induced != self.induced.compose(&inner.induced)? {
            return Err(ModulePowerError::InvalidEvidence(
                "induced map composition violated functoriality".to_owned(),
            ));
        }
        Ok(result)
    }

    /// Apply an exterior-power map directly to a homogeneous exterior element.
    ///
    /// The zero element is admitted in every degree. Nonzero elements must be
    /// homogeneous of the map's exact degree and belong to the source algebra.
    pub fn apply_exterior(
        &self,
        element: &ExteriorElement<R>,
    ) -> Result<ExteriorElement<R>, ModulePowerError> {
        if self.kind != ModulePowerKind::Exterior {
            return Err(ModulePowerError::KindMismatch {
                expected: ModulePowerKind::Exterior,
                actual: self.kind,
            });
        }
        let source_algebra = ExteriorAlgebra::new(self.source_map.source().clone());
        if element.algebra() != &source_algebra {
            return Err(ModulePowerError::Module(
                super::ModuleError::ModuleMismatch {
                    operation: "exterior-power map application",
                },
            ));
        }
        if !element.is_zero() && element.homogeneous_degree() != Some(self.degree) {
            return Err(ModulePowerError::DegreeMismatch {
                expected: self.degree,
                actual: element.homogeneous_degree(),
            });
        }
        let coordinates = element
            .to_power_element(self.degree)
            .map_err(|error| ModulePowerError::InvalidEvidence(error.to_string()))?;
        let image = self.induced.apply(&coordinates)?;
        ExteriorAlgebra::new(self.source_map.target().clone())
            .from_power_element(self.degree, &image)
            .map_err(|error| ModulePowerError::InvalidEvidence(error.to_string()))
    }

    /// Reconstruct the complete result and independently check exterior minors.
    pub fn verify(&self) -> Result<(), ModulePowerError> {
        if self.type_id != Self::TYPE_ID {
            return Err(ModulePowerError::InvalidEvidence(
                "module-power homomorphism has the wrong type identity".to_owned(),
            ));
        }
        let rebuilt = Self::new(self.source_map.clone(), self.kind, self.degree)?;
        if rebuilt != *self {
            return Err(ModulePowerError::InvalidEvidence(
                "recomputation changed the induced module-power map".to_owned(),
            ));
        }
        if self.kind == ModulePowerKind::Exterior {
            self.verify_exterior_minors()?;
        }
        Ok(())
    }

    /// Serialize the source map, power contexts, and exact induced map.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay the complete induced-map evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn verify_exterior_minors(&self) -> Result<(), ModulePowerError> {
        let source_basis = exterior_basis(self.source_map.source().rank(), self.degree)?;
        let target_basis = exterior_basis(self.source_map.target().rank(), self.degree)?;
        let matrix = self.source_map.matrix();
        let induced = self.induced.matrix();
        for (row, target_indices) in target_basis.iter().enumerate() {
            for (column, source_indices) in source_basis.iter().enumerate() {
                let minor_rows = target_indices.iter().map(|target| {
                    source_indices
                        .iter()
                        .map(|source| {
                            matrix
                                .get(*target, *source)
                                .expect("power basis positions fit the source matrix")
                        })
                        .collect::<Vec<_>>()
                });
                let minor = Matrix::from_rows(self.source_map.source().ring().clone(), minor_rows)?;
                let expected = minor.det()?;
                let actual = induced
                    .get(row, column)
                    .expect("power basis positions fit the induced matrix");
                if actual != expected {
                    return Err(ModulePowerError::InvalidEvidence(format!(
                        "exterior coordinate ({row}, {column}) is not its determinant minor"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<'de, R> Deserialize<'de> for ModulePowerHomomorphism<R>
where
    R: CommutativeRing + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModulePowerHomomorphismWire::<R>::deserialize(deserializer)?;
        let result = Self {
            type_id: wire.type_id,
            source_map: wire.source_map,
            kind: wire.kind,
            degree: wire.degree,
            source_power: wire.source_power,
            target_power: wire.target_power,
            induced: wire.induced,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R: CommutativeRing> ModuleHomomorphism<R> {
    /// Map induced on the degree-`degree` symmetric powers.
    pub fn symmetric_power(
        &self,
        degree: usize,
    ) -> Result<ModulePowerHomomorphism<R>, ModulePowerError> {
        ModulePowerHomomorphism::new(self.clone(), ModulePowerKind::Symmetric, degree)
    }

    /// Map induced on the degree-`degree` exterior powers.
    pub fn exterior_power(
        &self,
        degree: usize,
    ) -> Result<ModulePowerHomomorphism<R>, ModulePowerError> {
        ModulePowerHomomorphism::new(self.clone(), ModulePowerKind::Exterior, degree)
    }
}

impl<R: CommutativeRing> fmt::Debug for ModulePowerHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModulePowerHomomorphism")
            .field("kind", &self.kind)
            .field("degree", &self.degree)
            .field(
                "shape",
                &(self.induced.target().rank(), self.induced.source().rank()),
            )
            .finish_non_exhaustive()
    }
}

impl<R: CommutativeRing> fmt::Display for ModulePowerHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ModulePowerHomomorphism(kind={}, degree={}, {} -> {})",
            self.kind,
            self.degree,
            self.source_power.module().rank(),
            self.target_power.module().rank()
        )
    }
}

impl<R: CommutativeRing> SemanticType for ModulePowerHomomorphism<R> {
    const TYPE_ID: &'static str = "noether/homalg/ModulePowerHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source_map.canonical_bytes()?;
        output.extend_from_slice(self.kind.name().as_bytes());
        output.extend_from_slice(
            &u64::try_from(self.degree)
                .map_err(|_| {
                    NoetherError::Serialization(
                        "module-power degree exceeds canonical u64 framing".to_owned(),
                    )
                })?
                .to_be_bytes(),
        );
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_POWER_HOMOMORPHISM_DESCRIPTION
    }
}

/// A checked map induced on symmetric or exterior powers of presented modules.
///
/// The relation lift is constructed as the tensor product of the original
/// relation lift with the preceding power of the generator lift. The returned
/// [`PresentedModuleHomomorphism`] rechecks the resulting commuting square, so
/// the map is admitted only when it provably descends to both cokernels.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize, R::Element: Serialize")
)]
pub struct PresentedModulePowerHomomorphism<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source_map: PresentedModuleHomomorphism<R>,
    kind: ModulePowerKind,
    degree: usize,
    source_power: PresentedModulePower<R>,
    target_power: PresentedModulePower<R>,
    induced: PresentedModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PresentedModulePowerHomomorphismWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source_map: PresentedModuleHomomorphism<R>,
    kind: ModulePowerKind,
    degree: usize,
    source_power: PresentedModulePower<R>,
    target_power: PresentedModulePower<R>,
    induced: PresentedModuleHomomorphism<R>,
}

impl<R: CommutativeRing> PresentedModulePowerHomomorphism<R> {
    /// Construct and verify the induced map between retained power presentations.
    pub fn new(
        source_map: PresentedModuleHomomorphism<R>,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let source_power = PresentedModulePower::new(source_map.source().clone(), kind, degree)?;
        let target_power = PresentedModulePower::new(source_map.target().clone(), kind, degree)?;
        let generator_power =
            ModulePowerHomomorphism::new(source_map.generator_lift().clone(), kind, degree)?;

        let relation_lift = if degree == 0 {
            ModuleHomomorphism::zero(
                source_power.module().relation_module().clone(),
                target_power.module().relation_module().clone(),
            )?
        } else {
            let preceding = ModulePowerHomomorphism::new(
                source_map.generator_lift().clone(),
                kind,
                degree - 1,
            )?;
            // `PresentedModulePower` uses this same left-outer tensor order
            // for relation coordinates, but its structural relation module
            // deliberately carries no inherited basis metadata. Reframe the
            // endpoints while moving the native dense/sparse storage intact.
            source_map
                .relation_lift()
                .tensor(preceding.induced())?
                .reframe(
                    source_power.module().relation_module().clone(),
                    target_power.module().relation_module().clone(),
                )?
        };
        let induced = PresentedModuleHomomorphism::from_lifts(
            source_power.module().clone(),
            target_power.module().clone(),
            generator_power.induced().clone(),
            relation_lift,
        )?;
        Ok(Self {
            type_id: Self::TYPE_ID.to_owned(),
            source_map,
            kind,
            degree,
            source_power,
            target_power,
            induced,
        })
    }

    /// Original checked map of presented modules.
    pub fn source_map(&self) -> &PresentedModuleHomomorphism<R> {
        &self.source_map
    }

    /// Symmetric or exterior construction.
    pub const fn kind(&self) -> ModulePowerKind {
        self.kind
    }

    /// Nonnegative power degree.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Canonical power presentation of the source.
    pub fn source_power(&self) -> &PresentedModulePower<R> {
        &self.source_power
    }

    /// Canonical power presentation of the target.
    pub fn target_power(&self) -> &PresentedModulePower<R> {
        &self.target_power
    }

    /// Checked induced map between the two power presentations.
    pub fn induced(&self) -> &PresentedModuleHomomorphism<R> {
        &self.induced
    }

    /// Compose two checked presented power maps of the same kind and degree.
    pub fn compose(&self, inner: &Self) -> Result<Self, ModulePowerError> {
        if inner.kind != self.kind {
            return Err(ModulePowerError::KindMismatch {
                expected: self.kind,
                actual: inner.kind,
            });
        }
        if inner.degree != self.degree {
            return Err(ModulePowerError::DegreeMismatch {
                expected: self.degree,
                actual: Some(inner.degree),
            });
        }
        let result = Self::new(
            self.source_map.compose(&inner.source_map)?,
            self.kind,
            self.degree,
        )?;
        if result.induced != self.induced.compose(&inner.induced)? {
            return Err(ModulePowerError::InvalidEvidence(
                "presented induced-map composition violated functoriality".to_owned(),
            ));
        }
        Ok(result)
    }

    /// Replay the construction and recheck the retained relation square.
    pub fn verify(&self) -> Result<(), ModulePowerError> {
        if self.type_id != Self::TYPE_ID {
            return Err(ModulePowerError::InvalidEvidence(
                "presented module-power homomorphism has the wrong type identity".to_owned(),
            ));
        }
        self.induced.verify()?;
        if Self::new(self.source_map.clone(), self.kind, self.degree)? != *self {
            return Err(ModulePowerError::InvalidEvidence(
                "recomputation changed the presented module-power map".to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize the retained presentations and commuting-square evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay the complete presented power map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<'de, R> Deserialize<'de> for PresentedModulePowerHomomorphism<R>
where
    R: CommutativeRing + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PresentedModulePowerHomomorphismWire::<R>::deserialize(deserializer)?;
        let result = Self {
            type_id: wire.type_id,
            source_map: wire.source_map,
            kind: wire.kind,
            degree: wire.degree,
            source_power: wire.source_power,
            target_power: wire.target_power,
            induced: wire.induced,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R: CommutativeRing> PresentedModuleHomomorphism<R> {
    /// Checked map induced on the degree-`degree` symmetric power presentations.
    pub fn symmetric_power(
        &self,
        degree: usize,
    ) -> Result<PresentedModulePowerHomomorphism<R>, ModulePowerError> {
        PresentedModulePowerHomomorphism::new(self.clone(), ModulePowerKind::Symmetric, degree)
    }

    /// Checked map induced on the degree-`degree` exterior power presentations.
    pub fn exterior_power(
        &self,
        degree: usize,
    ) -> Result<PresentedModulePowerHomomorphism<R>, ModulePowerError> {
        PresentedModulePowerHomomorphism::new(self.clone(), ModulePowerKind::Exterior, degree)
    }
}

impl<R: CommutativeRing> fmt::Debug for PresentedModulePowerHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PresentedModulePowerHomomorphism")
            .field("kind", &self.kind)
            .field("degree", &self.degree)
            .finish_non_exhaustive()
    }
}

impl<R: CommutativeRing> fmt::Display for PresentedModulePowerHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PresentedModulePowerHomomorphism(kind={}, degree={})",
            self.kind, self.degree
        )
    }
}

impl<R: CommutativeRing> SemanticType for PresentedModulePowerHomomorphism<R> {
    const TYPE_ID: &'static str = "noether/homalg/PresentedModulePowerHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source_map.canonical_bytes()?;
        output.extend_from_slice(self.kind.name().as_bytes());
        output.extend_from_slice(
            &u64::try_from(self.degree)
                .map_err(|_| {
                    NoetherError::Serialization(
                        "presented module-power degree exceeds canonical u64 framing".to_owned(),
                    )
                })?
                .to_be_bytes(),
        );
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PRESENTED_MODULE_POWER_HOMOMORPHISM_DESCRIPTION
    }
}

fn exterior_power_columns<R: CommutativeRing>(
    map: &ModuleHomomorphism<R>,
    source_power: &FreeModulePower<R>,
    target_power: &FreeModulePower<R>,
) -> Result<Vec<Vec<R::Element>>, ModulePowerError> {
    let source_basis = exterior_indices(source_power.basis_exponents());
    let target_basis = exterior_indices(target_power.basis_exponents());
    require_power_matrix_cells(
        "exterior-power induced matrix",
        target_basis.len(),
        source_basis.len(),
    )?;
    if source_basis.is_empty() {
        return Ok(Vec::new());
    }
    if target_basis.is_empty() {
        return Ok(vec![Vec::new(); source_basis.len()]);
    }
    require_power_matrix_cells(
        "exterior-power source coordinate table",
        map.target().rank(),
        map.source().rank(),
    )?;
    let target_positions = target_basis
        .into_iter()
        .enumerate()
        .map(|(position, basis)| (basis, position))
        .collect::<BTreeMap<_, _>>();
    let matrix_columns = map.coordinate_columns();
    let ring = map.source().ring();
    let mut output = Vec::with_capacity(source_basis.len());
    let mut expansion_work = 0;
    for source_indices in source_basis {
        let mut expansion = BTreeMap::from([(Vec::new(), ring.one())]);
        for source_index in source_indices {
            charge_expansion_work(
                &mut expansion_work,
                expansion.len(),
                matrix_columns[source_index].len(),
                "exterior-power coefficient expansion",
            )?;
            let mut next = BTreeMap::new();
            for (basis, coefficient) in expansion {
                for (target_index, scalar) in matrix_columns[source_index].iter().enumerate() {
                    if ring.is_zero(scalar) {
                        continue;
                    }
                    let Some((next_basis, odd_sign)) = wedge_basis(&basis, &[target_index])
                        .map_err(|error| ModulePowerError::InvalidEvidence(error.to_string()))?
                    else {
                        continue;
                    };
                    let mut value = ring.mul(&coefficient, scalar)?;
                    if odd_sign {
                        value = ring.neg(&value)?;
                    }
                    accumulate(ring, &mut next, next_basis, value)?;
                }
            }
            expansion = next;
        }
        let mut column = vec![ring.zero(); target_positions.len()];
        for (basis, coefficient) in expansion {
            let row = *target_positions.get(&basis).ok_or_else(|| {
                ModulePowerError::InvalidEvidence(
                    "exterior expansion produced a basis outside the target power".to_owned(),
                )
            })?;
            column[row] = coefficient;
        }
        output.push(column);
    }
    Ok(output)
}

fn symmetric_power_columns<R: CommutativeRing>(
    map: &ModuleHomomorphism<R>,
    source_power: &FreeModulePower<R>,
    target_power: &FreeModulePower<R>,
) -> Result<Vec<Vec<R::Element>>, ModulePowerError> {
    let source_basis = source_power.basis_exponents();
    let target_basis = target_power.basis_exponents();
    require_power_matrix_cells(
        "symmetric-power induced matrix",
        target_basis.len(),
        source_basis.len(),
    )?;
    if source_basis.is_empty() {
        return Ok(Vec::new());
    }
    if target_basis.is_empty() {
        return Ok(vec![Vec::new(); source_basis.len()]);
    }
    require_power_matrix_cells(
        "symmetric-power source coordinate table",
        map.target().rank(),
        map.source().rank(),
    )?;
    let target_positions = target_basis
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, exponent)| (exponent, position))
        .collect::<BTreeMap<_, _>>();
    let matrix_columns = map.coordinate_columns();
    let ring = map.source().ring();
    let mut output = Vec::with_capacity(source_basis.len());
    let mut expansion_work = 0;
    for source_exponent in source_basis {
        let mut expansion = BTreeMap::from([(vec![0usize; map.target().rank()], ring.one())]);
        for (source_index, &multiplicity) in source_exponent.iter().enumerate() {
            for _ in 0..multiplicity {
                if expansion.is_empty() {
                    break;
                }
                charge_expansion_work(
                    &mut expansion_work,
                    expansion.len(),
                    matrix_columns[source_index].len(),
                    "symmetric-power coefficient expansion",
                )?;
                let mut next = BTreeMap::new();
                for (exponent, coefficient) in expansion {
                    for (target_index, scalar) in matrix_columns[source_index].iter().enumerate() {
                        if ring.is_zero(scalar) {
                            continue;
                        }
                        let mut next_exponent = exponent.clone();
                        next_exponent[target_index] =
                            next_exponent[target_index].checked_add(1).ok_or_else(|| {
                                ModulePowerError::InvalidEvidence(
                                    "symmetric exponent overflow".to_owned(),
                                )
                            })?;
                        let value = ring.mul(&coefficient, scalar)?;
                        accumulate(ring, &mut next, next_exponent, value)?;
                    }
                }
                expansion = next;
            }
        }
        let mut column = vec![ring.zero(); target_positions.len()];
        for (exponent, coefficient) in expansion {
            let row = *target_positions.get(&exponent).ok_or_else(|| {
                ModulePowerError::InvalidEvidence(
                    "symmetric expansion produced a basis outside the target power".to_owned(),
                )
            })?;
            column[row] = coefficient;
        }
        output.push(column);
    }
    Ok(output)
}

fn exterior_indices(exponents: &[Vec<usize>]) -> Vec<Vec<usize>> {
    exponents
        .iter()
        .map(|exponent| {
            exponent
                .iter()
                .enumerate()
                .filter_map(|(position, multiplicity)| (*multiplicity == 1).then_some(position))
                .collect()
        })
        .collect()
}

fn charge_expansion_work(
    work: &mut usize,
    terms: usize,
    choices: usize,
    operation: &'static str,
) -> Result<(), ModulePowerError> {
    let requested = terms
        .checked_mul(choices)
        .and_then(|step| work.checked_add(step));
    match requested {
        Some(value) if value <= MAX_MODULE_POWER_EXPANSION_WORK => {
            *work = value;
            Ok(())
        }
        _ => Err(ModulePowerError::ExpansionWorkLimit {
            operation,
            requested,
            limit: MAX_MODULE_POWER_EXPANSION_WORK,
        }),
    }
}

fn accumulate<R: CommutativeRing>(
    ring: &R,
    output: &mut BTreeMap<Vec<usize>, R::Element>,
    basis: Vec<usize>,
    value: R::Element,
) -> Result<(), ModulePowerError> {
    if ring.is_zero(&value) {
        return Ok(());
    }
    if let Some(previous) = output.get(&basis) {
        let sum = ring.add(previous, &value)?;
        if ring.is_zero(&sum) {
            output.remove(&basis);
        } else {
            output.insert(basis, sum);
        }
    } else {
        output.insert(basis, value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::linalg::Matrix;
    use crate::{Integer, Z};

    use super::super::{FreeModule, PresentedModule};
    use super::*;

    #[test]
    fn degree_zero_map_is_identity_and_degree_two_is_the_determinant() {
        let module = FreeModule::new(Z, 2);
        let matrix = Matrix::from_rows(
            Z,
            [
                [Integer::from(2), Integer::from(3)],
                [Integer::from(5), Integer::from(7)],
            ],
        )
        .unwrap();
        let map = ModuleHomomorphism::from_matrix(module.clone(), module, matrix).unwrap();
        let degree_zero = map.exterior_power(0).unwrap();
        assert_eq!(degree_zero.induced().matrix(), Matrix::identity(Z, 1));
        let determinant = map.exterior_power(2).unwrap();
        assert_eq!(
            determinant.induced().matrix(),
            Matrix::from_rows(Z, [[Integer::from(-1)]]).unwrap()
        );
        determinant.verify().unwrap();
    }

    #[test]
    fn presented_exterior_map_retains_a_valid_relation_square() {
        let generators = FreeModule::new(Z, 2);
        let relations = FreeModule::new(Z, 1);
        let presentation = ModuleHomomorphism::from_matrix(
            relations,
            generators.clone(),
            Matrix::from_rows(Z, [[Integer::from(2)], [Integer::from(0)]]).unwrap(),
        )
        .unwrap();
        let module = PresentedModule::cokernel(presentation);
        let identity = PresentedModuleHomomorphism::identity(module).unwrap();
        let power = identity.exterior_power(2).unwrap();
        power.verify().unwrap();
        assert_eq!(power.induced().source(), power.source_power().module());
        assert_eq!(power.induced().target(), power.target_power().module());
    }
}
