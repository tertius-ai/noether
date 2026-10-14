// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact complete and homologically truncated free resolutions.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{Degree, PolynomialRing, QuotientRing};
use crate::linalg::Matrix;
use crate::{Field, Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::{
    ChainComplex, FreeModule, GradedBettiTable, GradedBettiTableError, HomologyError, ModuleError,
    ModuleGroebnerError, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule, Submodule,
};

const MAX_RESOLUTION_LENGTH: usize = 128;

static FREE_RESOLUTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/FreeResolution@1",
    rust_path: "noether::homalg::FreeResolution",
    python_path: Some("noether.homalg.FreeResolution"),
};

static RESOLUTION_COMPLETENESS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ResolutionCompleteness@1",
    rust_path: "noether::homalg::ResolutionCompleteness",
    python_path: Some("noether.homalg.ResolutionCompleteness"),
};

static RESOLUTION_MINIMALITY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ResolutionMinimality@1",
    rust_path: "noether::homalg::ResolutionMinimality",
    python_path: Some("noether.homalg.ResolutionMinimality"),
};

/// Exact extent justified by a finite free-resolution computation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResolutionCompleteness {
    /// The final represented differential has zero kernel.
    Complete,
    /// Exactness is verified strictly below the last displayed free module.
    ThroughHomologicalDegree {
        /// Greatest represented homological degree.
        degree: usize,
    },
}

impl ResolutionCompleteness {
    /// Stable completeness kind.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::ThroughHomologicalDegree { .. } => "through_homological_degree",
        }
    }

    /// Greatest requested degree for a truncated result.
    pub const fn through_homological_degree(&self) -> Option<usize> {
        match self {
            Self::Complete => None,
            Self::ThroughHomologicalDegree { degree } => Some(*degree),
        }
    }

    /// Serialize the exact completeness claim as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&ResolutionCompletenessWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize an exact completeness claim.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: ResolutionCompletenessWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        wire.try_into()
    }
}

impl fmt::Display for ResolutionCompleteness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complete => formatter.write_str("complete"),
            Self::ThroughHomologicalDegree { degree } => {
                write!(formatter, "through homological degree {degree}")
            }
        }
    }
}

impl SemanticType for ResolutionCompleteness {
    const TYPE_ID: &'static str = "noether/homalg/ResolutionCompleteness@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &RESOLUTION_COMPLETENESS_DESCRIPTION
    }
}

/// Exact minimality claim attached to a graded free resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionMinimality {
    /// No minimality assertion has been made.
    NotClaimed,
    /// Homogeneity and the absence of unit differential entries were verified.
    Minimal,
}

impl ResolutionMinimality {
    /// Stable minimality kind.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::NotClaimed => "not_claimed",
            Self::Minimal => "minimal",
        }
    }

    /// Whether this value carries a verified minimality claim.
    pub const fn is_minimal(&self) -> bool {
        matches!(self, Self::Minimal)
    }

    /// Serialize the exact minimality claim as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&ResolutionMinimalityWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize an exact minimality claim.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: ResolutionMinimalityWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        wire.try_into()
    }
}

impl fmt::Display for ResolutionMinimality {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotClaimed => formatter.write_str("minimality not claimed"),
            Self::Minimal => formatter.write_str("minimal"),
        }
    }
}

impl SemanticType for ResolutionMinimality {
    const TYPE_ID: &'static str = "noether/homalg/ResolutionMinimality@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &RESOLUTION_MINIMALITY_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionMinimalityWire {
    #[serde(rename = "type")]
    type_id: String,
    value: ResolutionMinimality,
}

impl From<&ResolutionMinimality> for ResolutionMinimalityWire {
    fn from(value: &ResolutionMinimality) -> Self {
        Self {
            type_id: ResolutionMinimality::TYPE_ID.to_owned(),
            value: value.clone(),
        }
    }
}

impl TryFrom<ResolutionMinimalityWire> for ResolutionMinimality {
    type Error = NoetherError;

    fn try_from(wire: ResolutionMinimalityWire) -> Result<Self, Self::Error> {
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(wire.value)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionCompletenessWire {
    #[serde(rename = "type")]
    type_id: String,
    value: ResolutionCompleteness,
}

impl From<&ResolutionCompleteness> for ResolutionCompletenessWire {
    fn from(value: &ResolutionCompleteness) -> Self {
        Self {
            type_id: ResolutionCompleteness::TYPE_ID.to_owned(),
            value: value.clone(),
        }
    }
}

impl TryFrom<ResolutionCompletenessWire> for ResolutionCompleteness {
    type Error = NoetherError;

    fn try_from(wire: ResolutionCompletenessWire) -> Result<Self, Self::Error> {
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if matches!(
            wire.value,
            Self::ThroughHomologicalDegree { degree } if degree > MAX_RESOLUTION_LENGTH
        ) {
            return Err(NoetherError::Serialization(format!(
                "homological degree exceeds the limit {MAX_RESOLUTION_LENGTH}"
            )));
        }
        Ok(wire.value)
    }
}

/// Failure to construct or verify a free resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError {
    /// A based-module operation failed.
    Module(ModuleError),
    /// A module Groebner or syzygy operation failed.
    Groebner(ModuleGroebnerError),
    /// The projected chain complex failed validation.
    Homology(HomologyError),
    /// A claimed exactness or endpoint invariant failed.
    InvalidEvidence(&'static str),
    /// A requested truncation exceeded the fixed public result bound.
    HomologicalDegreeLimit {
        /// Maximum admitted homological degree.
        limit: usize,
    },
    /// Schreyer iteration did not terminate within the polynomial-ring bound.
    ProjectiveDimensionBound {
        /// Number of polynomial variables giving the Hilbert-syzygy bound.
        variables: usize,
    },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Homology(error) => error.fmt(formatter),
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid free-resolution evidence: {reason}")
            }
            Self::HomologicalDegreeLimit { limit } => {
                write!(formatter, "homological degree exceeds limit {limit}")
            }
            Self::ProjectiveDimensionBound { variables } => write!(
                formatter,
                "Schreyer resolution did not terminate within {variables} polynomial variables"
            ),
        }
    }
}

impl std::error::Error for ResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Module(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Homology(error) => Some(error),
            Self::InvalidEvidence(_)
            | Self::HomologicalDegreeLimit { .. }
            | Self::ProjectiveDimensionBound { .. } => None,
        }
    }
}

impl From<ModuleError> for ResolutionError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<ModuleGroebnerError> for ResolutionError {
    fn from(value: ModuleGroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<HomologyError> for ResolutionError {
    fn from(value: HomologyError) -> Self {
        Self::Homology(value)
    }
}

/// A verified finite free resolution of one retained presentation.
///
/// `differentials[i - 1]` is `d_i: F_i -> F_(i-1)`. For a truncated value,
/// exactness is claimed through `F_(i-1)` but not at the final displayed
/// `F_i`. A complete value additionally verifies that its final differential
/// has zero kernel.
#[derive(Clone, PartialEq, Eq)]
pub struct FreeResolution<R: Ring> {
    module: PresentedModule<R>,
    differentials: Vec<ModuleHomomorphism<R>>,
    augmentation: ModuleHomomorphism<R>,
    augmentation_inverse: ModuleHomomorphism<R>,
    completeness: ResolutionCompleteness,
    minimality: ResolutionMinimality,
}

#[derive(Serialize)]
#[serde(bound(serialize = "R: Serialize, R::Element: Serialize"))]
struct FreeResolutionWireRef<'a, R: Ring> {
    #[serde(rename = "type")]
    type_id: &'static str,
    module: &'a PresentedModule<R>,
    differentials: &'a [ModuleHomomorphism<R>],
    augmentation: &'a ModuleHomomorphism<R>,
    augmentation_inverse: &'a ModuleHomomorphism<R>,
    completeness: &'a ResolutionCompleteness,
    minimality: &'a ResolutionMinimality,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct FreeResolutionWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    module: PresentedModule<R>,
    differentials: Vec<ModuleHomomorphism<R>>,
    augmentation: ModuleHomomorphism<R>,
    augmentation_inverse: ModuleHomomorphism<R>,
    completeness: ResolutionCompleteness,
    minimality: ResolutionMinimality,
}

impl<R: Ring> FreeResolution<R> {
    /// Presented module augmented by this resolution.
    pub fn module(&self) -> &PresentedModule<R> {
        &self.module
    }

    /// Exact completeness or truncation claim.
    pub fn completeness(&self) -> &ResolutionCompleteness {
        &self.completeness
    }

    /// Degree-zero map from the represented `F_0` to the original
    /// presentation generators.
    pub fn augmentation(&self) -> &ModuleHomomorphism<R> {
        &self.augmentation
    }

    /// Degree-zero inverse to the augmentation on the two presented
    /// cokernels.
    pub fn augmentation_inverse(&self) -> &ModuleHomomorphism<R> {
        &self.augmentation_inverse
    }

    /// Exact minimality claim.
    pub fn minimality(&self) -> &ResolutionMinimality {
        &self.minimality
    }

    /// Whether minimality has been independently verified.
    pub fn is_minimal(&self) -> bool {
        self.minimality.is_minimal()
    }

    /// Greatest represented homological degree.
    pub fn length(&self) -> usize {
        self.differentials.len()
    }

    /// Free module in one represented homological degree.
    pub fn free_module(&self, degree: usize) -> Option<&FreeModule<R>> {
        match degree {
            0 => Some(self.augmentation.source()),
            _ => self
                .differentials
                .get(degree - 1)
                .map(ModuleHomomorphism::source),
        }
    }

    /// Differential `d_degree: F_degree -> F_(degree-1)`.
    pub fn differential(&self, degree: usize) -> Option<&ModuleHomomorphism<R>> {
        degree
            .checked_sub(1)
            .and_then(|index| self.differentials.get(index))
    }

    /// Rebuild the represented maps through Noether's one chain-complex owner.
    pub(crate) fn chain_complex(&self) -> Result<ChainComplex<R>, ResolutionError> {
        let mut boundaries = BTreeMap::new();
        if self.differentials.is_empty() {
            boundaries.insert(
                1,
                crate::linalg::Matrix::zeros(
                    self.module.generator_module().ring().clone(),
                    self.augmentation.source().rank(),
                    0,
                ),
            );
        } else {
            for (index, differential) in self.differentials.iter().enumerate() {
                let degree = i32::try_from(index + 1).map_err(|_| {
                    ResolutionError::HomologicalDegreeLimit {
                        limit: MAX_RESOLUTION_LENGTH,
                    }
                })?;
                boundaries.insert(degree, differential.matrix());
            }
        }
        ChainComplex::from_boundaries(self.module.generator_module().ring().clone(), boundaries)
            .map_err(Into::into)
    }

    /// Serialize all exact maps and the completeness claim as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn invalid(reason: &'static str) -> ResolutionError {
        ResolutionError::InvalidEvidence(reason)
    }

    /// The canonical presentation, regarded as an honestly truncated free
    /// resolution through degree one.
    ///
    /// Exactness at degree zero is true by the definition of
    /// [`PresentedModule`]: its relation map presents precisely the kernel of
    /// the quotient onto the module.  This retained evidence is therefore
    /// available without replacing the relations by a Groebner basis.  It is
    /// the natural resolution witness for degree-zero derived functors.
    pub(super) fn presentation_through_one(
        module: &PresentedModule<R>,
    ) -> Result<Self, ResolutionError> {
        let identity = ModuleHomomorphism::identity(module.generator_module().clone())?;
        let result = Self {
            module: module.clone(),
            differentials: vec![module.presentation().clone()],
            augmentation: identity.clone(),
            augmentation_inverse: identity,
            completeness: ResolutionCompleteness::ThroughHomologicalDegree { degree: 1 },
            minimality: ResolutionMinimality::NotClaimed,
        };
        verify_structure(&result)?;
        result.chain_complex()?;
        Ok(result)
    }
}

impl<C: Field> FreeResolution<PolynomialRing<C>> {
    /// Construct a resolution from exact maps and admit it only after an
    /// independent verification of every requested claim.
    ///
    /// This is the checked import path for complexes produced by another
    /// algorithm or computer algebra system. `augmentation` and
    /// `augmentation_inverse` must exhibit mutually inverse maps on the
    /// original and represented cokernels; merely matching Betti numbers or
    /// chain identities is not sufficient.
    pub fn from_maps(
        module: PresentedModule<PolynomialRing<C>>,
        differentials: Vec<ModuleHomomorphism<PolynomialRing<C>>>,
        augmentation: ModuleHomomorphism<PolynomialRing<C>>,
        augmentation_inverse: ModuleHomomorphism<PolynomialRing<C>>,
        completeness: ResolutionCompleteness,
        minimality: ResolutionMinimality,
    ) -> Result<Self, ResolutionError> {
        let result = Self {
            module,
            differentials,
            augmentation,
            augmentation_inverse,
            completeness,
            minimality,
        };
        result.verify()?;
        Ok(result)
    }

    /// Independently recheck endpoints, chain identities, augmentation image,
    /// and image-kernel equality in every claimed position.
    pub fn verify(&self) -> Result<(), ResolutionError> {
        verify_structure(self)?;
        let ring = self.module.generator_module().ring();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        if !self.differentials.is_empty() || self.completeness == ResolutionCompleteness::Complete {
            verify_augmentation_equivalence(self, order.clone())?;
        }

        for index in 0..self.differentials.len().saturating_sub(1) {
            let differential = &self.differentials[index];
            let next = &self.differentials[index + 1];
            let kernel =
                differential.kernel(ModuleMonomialOrder::TermOverPosition(ring.term_order()))?;
            let image = Submodule::from_generator_map(next.target().clone(), next.clone())?;
            let image_basis = image.groebner_basis(order.clone())?;
            if !image_lies_in(kernel.generator_map(), &image_basis)? {
                return Err(Self::invalid(
                    "one represented image does not equal the preceding kernel",
                ));
            }
        }

        if self.completeness == ResolutionCompleteness::Complete {
            if let Some(last) = self.differentials.last() {
                let kernel =
                    last.kernel(ModuleMonomialOrder::TermOverPosition(ring.term_order()))?;
                if kernel.generator_count() != 0 {
                    return Err(Self::invalid(
                        "the final differential of a complete resolution has nonzero kernel",
                    ));
                }
            }
        }
        if self.minimality == ResolutionMinimality::Minimal {
            verify_minimality(self)?;
        }
        self.chain_complex()?;
        Ok(())
    }

    /// Cancel every homogeneous unit entry and verify the resulting minimal
    /// complex and its induced equivalence to the original presentation.
    pub fn minimalize(&self) -> Result<Self, ResolutionError> {
        minimalize_resolution(self)
    }

    /// Derive exact multigraded Betti numbers from verified minimal shifts.
    pub fn graded_betti_table(&self) -> Result<GradedBettiTable, GradedBettiTableError> {
        GradedBettiTable::from_resolution(self)
    }

    /// Deserialize and independently revalidate a resolution.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> PresentedModule<PolynomialRing<C>> {
    /// Compute exact multigraded Betti numbers without retaining a free
    /// resolution or materializing its minimal differential matrices.
    ///
    /// Certified Schreyer frames are streamed one at a time. Each homogeneous
    /// differential is reduced modulo the irrelevant ideal, and the Betti
    /// multiplicities are the exact homology dimensions of those
    /// coefficient-field blocks.
    pub fn graded_betti_table(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<GradedBettiTable, GradedBettiTableError> {
        construct_betti_table(self, order)
    }

    /// Compute a complete verified Schreyer free resolution.
    pub fn free_resolution(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
        construct_resolution(self, order, None, true)
    }

    /// Compute a verified resolution through one homological degree.
    ///
    /// Exactness is asserted only below the last displayed free module unless
    /// Schreyer iteration proves that the resolution already terminates.
    pub fn free_resolution_through(
        &self,
        order: ModuleMonomialOrder,
        degree: usize,
    ) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
        if degree > MAX_RESOLUTION_LENGTH {
            return Err(ResolutionError::HomologicalDegreeLimit {
                limit: MAX_RESOLUTION_LENGTH,
            });
        }
        construct_resolution(self, order, Some(degree), true)
    }

    /// Compute and verify a complete minimal graded free resolution.
    pub fn minimal_free_resolution(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
        verify_homogeneous_map(self.presentation())?;
        minimalize_constructed_resolution_owned(construct_resolution(self, order, None, false)?)
    }

    /// Compute and minimalize a verified resolution through one homological
    /// degree.
    ///
    /// Minimality of the last retained free module is witnessed by computing
    /// one additional differential before unit cancellation, then discarding
    /// that look-ahead map. This prevents a redundant last generator from
    /// surviving merely because its cancelling unit lies just beyond the
    /// requested truncation.
    pub fn minimal_free_resolution_through(
        &self,
        order: ModuleMonomialOrder,
        degree: usize,
    ) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
        if degree > MAX_RESOLUTION_LENGTH {
            return Err(ResolutionError::HomologicalDegreeLimit {
                limit: MAX_RESOLUTION_LENGTH,
            });
        }
        verify_homogeneous_map(self.presentation())?;
        if degree == 0 {
            return minimalize_constructed_resolution_owned(construct_resolution(
                self,
                order,
                Some(0),
                false,
            )?);
        }
        if degree == 1 {
            if let Some(result) = minimal_equal_degree_presentation_truncation(self)? {
                return Ok(result);
            }
        }
        let look_ahead = degree.checked_add(1);
        let resolution = match look_ahead.filter(|value| *value <= MAX_RESOLUTION_LENGTH) {
            Some(limit) => construct_resolution(self, order.clone(), Some(limit), false)?,
            None => construct_resolution(self, order.clone(), None, false)?,
        };
        let mut minimal = minimalize_constructed_resolution_owned(resolution)?;
        if minimal.completeness != ResolutionCompleteness::Complete && minimal.length() < degree {
            minimal = minimalize_constructed_resolution_owned(construct_resolution(
                self, order, None, false,
            )?)?;
        }
        truncate_minimal_resolution(minimal, degree)
    }
}

type PolynomialQuotient<C> = QuotientRing<PolynomialRing<C>>;

impl<C: Field> FreeResolution<PolynomialQuotient<C>> {
    /// Admit a bounded quotient-ring resolution only after replaying every
    /// represented exactness and augmentation claim.
    pub fn from_maps(
        module: PresentedModule<PolynomialQuotient<C>>,
        differentials: Vec<ModuleHomomorphism<PolynomialQuotient<C>>>,
        augmentation: ModuleHomomorphism<PolynomialQuotient<C>>,
        augmentation_inverse: ModuleHomomorphism<PolynomialQuotient<C>>,
        completeness: ResolutionCompleteness,
        minimality: ResolutionMinimality,
    ) -> Result<Self, ResolutionError> {
        let result = Self {
            module,
            differentials,
            augmentation,
            augmentation_inverse,
            completeness,
            minimality,
        };
        result.verify()?;
        Ok(result)
    }

    /// Independently verify a finite or honestly truncated free resolution
    /// over a polynomial quotient.
    pub fn verify(&self) -> Result<(), ResolutionError> {
        verify_structure(self)?;
        if self.minimality == ResolutionMinimality::Minimal {
            return Err(Self::invalid(
                "minimality over polynomial quotients is not yet admitted",
            ));
        }
        let order = ModuleMonomialOrder::TermOverPosition(
            self.module
                .generator_module()
                .ring()
                .base_ring()
                .term_order(),
        );
        if !self.differentials.is_empty() || self.completeness == ResolutionCompleteness::Complete {
            verify_quotient_augmentation_equivalence(self, order.clone())?;
        }
        for index in 0..self.differentials.len().saturating_sub(1) {
            let differential = &self.differentials[index];
            let next = &self.differentials[index + 1];
            let kernel = differential.kernel(order.clone())?;
            let image = Submodule::from_generator_map(next.target().clone(), next.clone())?;
            let image_basis = image.groebner_basis(order.clone())?;
            if !quotient_image_lies_in(kernel.generator_map(), &image_basis)? {
                return Err(Self::invalid(
                    "one represented quotient-module image does not equal the preceding kernel",
                ));
            }
        }
        if self.completeness == ResolutionCompleteness::Complete {
            if let Some(last) = self.differentials.last() {
                if last.kernel(order)?.generator_count() != 0 {
                    return Err(Self::invalid(
                        "the final differential of a complete quotient resolution has nonzero kernel",
                    ));
                }
            }
        }
        self.chain_complex()?;
        Ok(())
    }

    /// Deserialize and independently revalidate quotient-resolution evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> PresentedModule<PolynomialQuotient<C>> {
    /// Compute a verified free resolution through one homological degree.
    ///
    /// Polynomial quotients need not have finite global dimension, so this
    /// operation never promises termination merely from the ambient number of
    /// variables. It records a complete result only when the final displayed
    /// differential has zero kernel; otherwise its truncation boundary is
    /// explicit.
    pub fn free_resolution_through(
        &self,
        order: ModuleMonomialOrder,
        degree: usize,
    ) -> Result<FreeResolution<PolynomialQuotient<C>>, ResolutionError> {
        if degree > MAX_RESOLUTION_LENGTH {
            return Err(ResolutionError::HomologicalDegreeLimit {
                limit: MAX_RESOLUTION_LENGTH,
            });
        }
        construct_quotient_resolution(self, order, degree)
    }
}

fn construct_quotient_resolution<C: Field>(
    module: &PresentedModule<PolynomialQuotient<C>>,
    order: ModuleMonomialOrder,
    degree: usize,
) -> Result<FreeResolution<PolynomialQuotient<C>>, ResolutionError> {
    let identity = ModuleHomomorphism::identity(module.generator_module().clone())?;
    if degree == 0 {
        return FreeResolution::<PolynomialQuotient<C>>::from_maps(
            module.clone(),
            Vec::new(),
            identity.clone(),
            identity,
            ResolutionCompleteness::ThroughHomologicalDegree { degree: 0 },
            ResolutionMinimality::NotClaimed,
        );
    }

    let mut differentials = Vec::with_capacity(degree);
    let mut current = module.presentation().clone();
    let mut completeness = ResolutionCompleteness::ThroughHomologicalDegree { degree };
    for homological_degree in 1..=degree {
        differentials.push(current.clone());
        let kernel = current.kernel(order.clone())?;
        if kernel.generator_count() == 0 {
            completeness = ResolutionCompleteness::Complete;
            break;
        }
        if homological_degree < degree {
            current = kernel.generator_map().clone();
        }
    }
    FreeResolution::<PolynomialQuotient<C>>::from_maps(
        module.clone(),
        differentials,
        identity.clone(),
        identity,
        completeness,
        ResolutionMinimality::NotClaimed,
    )
}

fn construct_betti_table<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
) -> Result<GradedBettiTable, GradedBettiTableError> {
    let ring = module.generator_module().ring();
    let grading = ring.grading().clone();
    let input = Submodule::from_generator_map(
        module.generator_module().clone(),
        module.presentation().clone(),
    )
    .map_err(ResolutionError::from)?;
    if let Some(output) = super::module_groebner::packed_standard_graded_betti(&input, &order)
        .map_err(ResolutionError::from)?
    {
        let entries = output.entries.into_iter().map(|entry| {
            (
                entry.homological_degree,
                Degree::new(vec![Integer::from(entry.internal_degree)]),
                entry.multiplicity,
            )
        });
        return GradedBettiTable::from_exact_entries(
            grading,
            output.length,
            ResolutionCompleteness::Complete,
            entries,
        );
    }

    let mut module_degrees = vec![free_module_degree_counts(module.generator_module())?];
    // `differential_ranks[i]` is the multigraded rank of d_i. There is no d_0.
    let mut differential_ranks = vec![BTreeMap::<Degree, usize>::new()];
    let mut basis = input.groebner_basis(order).map_err(ResolutionError::from)?;

    if basis.basis().generator_count() != 0 {
        loop {
            let differential = basis.basis().generator_map();
            module_degrees.push(free_module_degree_counts(differential.source())?);
            differential_ranks.push(residue_differential_ranks(differential)?);

            let next = basis.next_schreyer_basis().map_err(ResolutionError::from)?;
            let Some(next_basis) = next else {
                break;
            };
            let homological_degree = differential_ranks.len() - 1;
            if homological_degree >= ring.variables().len() {
                return Err(ResolutionError::ProjectiveDimensionBound {
                    variables: ring.variables().len(),
                }
                .into());
            }
            basis = next_basis;
        }
    }

    let mut entries = Vec::new();
    for (homological_degree, degrees) in module_degrees.iter().enumerate() {
        for (degree, dimension) in degrees {
            let outgoing_rank = differential_ranks[homological_degree]
                .get(degree)
                .copied()
                .unwrap_or(0);
            let incoming_rank = differential_ranks
                .get(homological_degree + 1)
                .and_then(|ranks| ranks.get(degree))
                .copied()
                .unwrap_or(0);
            let used = outgoing_rank.checked_add(incoming_rank).ok_or(
                ResolutionError::InvalidEvidence(
                    "residue differential ranks overflow one graded free summand",
                ),
            )?;
            let multiplicity =
                dimension
                    .checked_sub(used)
                    .ok_or(ResolutionError::InvalidEvidence(
                        "residue differential ranks exceed one graded free summand",
                    ))?;
            if multiplicity != 0 {
                entries.push((homological_degree, degree.clone(), multiplicity));
            }
        }
    }
    let length = entries
        .iter()
        .map(|(homological_degree, _, _)| *homological_degree)
        .max()
        .unwrap_or(0);
    GradedBettiTable::from_exact_entries(grading, length, ResolutionCompleteness::Complete, entries)
}

fn free_module_degree_counts<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
) -> Result<BTreeMap<Degree, usize>, GradedBettiTableError> {
    let shifts = module
        .degree_shifts()
        .ok_or(GradedBettiTableError::GradingRequired)?;
    let grading_rank = module.ring().grading().rank();
    if module.grading_rank() != Some(grading_rank) {
        return Err(GradedBettiTableError::GradingRequired);
    }
    let mut counts = BTreeMap::<Degree, usize>::new();
    for shift in shifts {
        let degree = shift.neg();
        let count = counts.entry(degree).or_default();
        *count = count
            .checked_add(1)
            .ok_or(ResolutionError::InvalidEvidence(
                "graded free-module multiplicity overflowed usize",
            ))?;
    }
    Ok(counts)
}

fn residue_differential_ranks<C: Field>(
    differential: &ModuleHomomorphism<PolynomialRing<C>>,
) -> Result<BTreeMap<Degree, usize>, GradedBettiTableError> {
    verify_homogeneous_map(differential)?;
    let source_shifts = differential
        .source()
        .degree_shifts()
        .ok_or(GradedBettiTableError::GradingRequired)?;
    let target_shifts = differential
        .target()
        .degree_shifts()
        .ok_or(GradedBettiTableError::GradingRequired)?;
    let mut source_blocks = BTreeMap::<Degree, Vec<usize>>::new();
    let mut target_blocks = BTreeMap::<Degree, Vec<usize>>::new();
    for (column, shift) in source_shifts.iter().enumerate() {
        source_blocks.entry(shift.neg()).or_default().push(column);
    }
    for (row, shift) in target_shifts.iter().enumerate() {
        target_blocks.entry(shift.neg()).or_default().push(row);
    }

    let matrix = differential.matrix();
    let field = differential.source().ring().coefficient_ring().clone();
    let mut ranks = BTreeMap::new();
    for (degree, columns) in source_blocks {
        let Some(rows) = target_blocks.get(&degree) else {
            continue;
        };
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            let mut block_row = Vec::with_capacity(columns.len());
            for column in &columns {
                block_row.push(
                    matrix
                        .get(*row, *column)
                        .map_err(|_| {
                            ResolutionError::InvalidEvidence(
                                "a differential block index was outside its matrix shape",
                            )
                        })?
                        .constant_coefficient(),
                );
            }
            entries.push(block_row);
        }
        let block = Matrix::from_rows(field.clone(), entries)
            .map_err(ModuleError::from)
            .map_err(ResolutionError::from)?;
        let rank = block
            .rank()
            .map_err(ModuleError::from)
            .map_err(ResolutionError::from)?;
        if rank != 0 {
            ranks.insert(degree, rank);
        }
    }
    Ok(ranks)
}

fn construct_resolution<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
    truncation: Option<usize>,
    verify_result: bool,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    if truncation == Some(0) {
        let result = resolution_with_identity_augmentation(
            module,
            Vec::new(),
            ResolutionCompleteness::ThroughHomologicalDegree { degree: 0 },
        )?;
        verify_when_requested(&result, verify_result)?;
        return Ok(result);
    }
    let input = Submodule::from_generator_map(
        module.generator_module().clone(),
        module.presentation().clone(),
    )?;
    if truncation.is_none() {
        if let Some(differentials) =
            super::module_groebner::packed_standard_graded_resolution(&input, &order)?
        {
            let result = resolution_with_identity_augmentation(
                module,
                differentials,
                ResolutionCompleteness::Complete,
            )?;
            verify_when_requested(&result, verify_result)?;
            return Ok(result);
        }
    }
    let mut basis = input.groebner_basis(order)?;
    let mut differentials = Vec::new();
    if basis.basis().generator_count() == 0 {
        let result = resolution_with_identity_augmentation(
            module,
            differentials,
            ResolutionCompleteness::Complete,
        )?;
        verify_when_requested(&result, verify_result)?;
        return Ok(result);
    }

    loop {
        differentials.push(basis.basis().generator_map().clone());
        if truncation == Some(differentials.len()) {
            let degree = differentials.len();
            let result = resolution_with_identity_augmentation(
                module,
                differentials,
                ResolutionCompleteness::ThroughHomologicalDegree { degree },
            )?;
            verify_when_requested(&result, verify_result)?;
            return Ok(result);
        }
        let Some(next_basis) = basis.next_schreyer_basis()? else {
            let result = resolution_with_identity_augmentation(
                module,
                differentials,
                ResolutionCompleteness::Complete,
            )?;
            verify_when_requested(&result, verify_result)?;
            return Ok(result);
        };
        let variable_count = module.generator_module().ring().variables().len();
        if differentials.len() >= variable_count {
            return Err(ResolutionError::ProjectiveDimensionBound {
                variables: variable_count,
            });
        }
        basis = next_basis;
    }
}

fn verify_when_requested<C: Field>(
    resolution: &FreeResolution<PolynomialRing<C>>,
    requested: bool,
) -> Result<(), ResolutionError> {
    if requested {
        resolution.verify()?;
    }
    Ok(())
}

fn truncate_minimal_resolution<C: Field>(
    mut resolution: FreeResolution<PolynomialRing<C>>,
    degree: usize,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    if resolution.length() <= degree {
        return Ok(resolution);
    }
    resolution.differentials.truncate(degree);
    resolution.completeness = ResolutionCompleteness::ThroughHomologicalDegree { degree };
    resolution.verify()?;
    Ok(resolution)
}

fn minimal_equal_degree_presentation_truncation<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
) -> Result<Option<FreeResolution<PolynomialRing<C>>>, ResolutionError> {
    let presentation = module.presentation();
    let Some(shifts) = presentation.source().degree_shifts() else {
        return Ok(None);
    };
    let Some(first_shift) = shifts.first() else {
        return Ok(None);
    };
    if !presentation.source().ring().grading().is_standard()
        || shifts.iter().any(|shift| shift != first_shift)
        || presentation
            .matrix()
            .rows()
            .flatten()
            .any(|entry| constant_inverse(presentation.target().ring(), &entry).is_some())
    {
        return Ok(None);
    }

    let field = presentation.source().ring().coefficient_ring().clone();
    let columns = presentation.source().rank();
    let mut coefficient_rows = BTreeMap::<(usize, Vec<Integer>), Vec<C::Element>>::new();
    for (row, entries) in presentation.matrix().rows().enumerate() {
        for (column, polynomial) in entries.into_iter().enumerate() {
            for (exponent, coefficient) in polynomial.terms() {
                coefficient_rows
                    .entry((row, exponent.to_vec()))
                    .or_insert_with(|| vec![field.zero(); columns])[column] = coefficient.clone();
            }
        }
    }
    let coefficient_matrix =
        Matrix::from_rows(field, coefficient_rows.into_values()).map_err(ModuleError::from)?;
    if coefficient_matrix.rank().map_err(ModuleError::from)? != columns {
        return Ok(None);
    }

    let identity = ModuleHomomorphism::identity(module.generator_module().clone())?;
    let result = FreeResolution {
        module: module.clone(),
        differentials: vec![presentation.clone()],
        augmentation: identity.clone(),
        augmentation_inverse: identity,
        completeness: ResolutionCompleteness::ThroughHomologicalDegree { degree: 1 },
        minimality: ResolutionMinimality::Minimal,
    };
    result.verify()?;
    Ok(Some(result))
}

fn resolution_with_identity_augmentation<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
    differentials: Vec<ModuleHomomorphism<PolynomialRing<C>>>,
    completeness: ResolutionCompleteness,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    let augmentation = ModuleHomomorphism::identity(module.generator_module().clone())?;
    Ok(FreeResolution {
        module: module.clone(),
        differentials,
        augmentation: augmentation.clone(),
        augmentation_inverse: augmentation,
        completeness,
        minimality: ResolutionMinimality::NotClaimed,
    })
}

fn verify_structure<R: Ring>(resolution: &FreeResolution<R>) -> Result<(), ResolutionError> {
    if resolution.differentials.len() > MAX_RESOLUTION_LENGTH {
        return Err(ResolutionError::HomologicalDegreeLimit {
            limit: MAX_RESOLUTION_LENGTH,
        });
    }
    match resolution.completeness {
        ResolutionCompleteness::Complete => {}
        ResolutionCompleteness::ThroughHomologicalDegree { degree }
            if degree == resolution.differentials.len() => {}
        ResolutionCompleteness::ThroughHomologicalDegree { .. } => {
            return Err(ResolutionError::InvalidEvidence(
                "truncation degree differs from the represented resolution length",
            ));
        }
    }
    if resolution.augmentation.target() != resolution.module.generator_module()
        || resolution.augmentation_inverse.source() != resolution.module.generator_module()
        || resolution.augmentation.source() != resolution.augmentation_inverse.target()
    {
        return Err(ResolutionError::InvalidEvidence(
            "augmentation and inverse have incompatible endpoints",
        ));
    }
    if resolution.differentials.is_empty()
        && resolution.completeness != ResolutionCompleteness::Complete
    {
        let identity = ModuleHomomorphism::identity(resolution.module.generator_module().clone())?;
        if resolution.augmentation != identity || resolution.augmentation_inverse != identity {
            return Err(ResolutionError::InvalidEvidence(
                "degree-zero truncation requires the canonical presentation augmentation",
            ));
        }
        if resolution.minimality == ResolutionMinimality::Minimal {
            return Err(ResolutionError::InvalidEvidence(
                "degree-zero truncation cannot carry a minimality claim",
            ));
        }
    }
    if let Some(first) = resolution.differentials.first() {
        if first.target() != resolution.augmentation.source() {
            return Err(ResolutionError::InvalidEvidence(
                "d_1 does not target the represented degree-zero free module",
            ));
        }
    }
    for index in 1..resolution.differentials.len() {
        let previous = &resolution.differentials[index - 1];
        let current = &resolution.differentials[index];
        if current.target() != previous.source() {
            return Err(ResolutionError::InvalidEvidence(
                "adjacent resolution differentials have incompatible free modules",
            ));
        }
        let composite = previous.compose(current)?;
        if composite
            .matrix()
            .rows()
            .flatten()
            .any(|value| value != composite.target().ring().zero())
        {
            return Err(ResolutionError::InvalidEvidence(
                "consecutive resolution differentials do not compose to zero",
            ));
        }
    }
    Ok(())
}

fn verify_augmentation_equivalence<C: Field>(
    resolution: &FreeResolution<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
) -> Result<(), ResolutionError> {
    let ring = resolution.module.generator_module().ring();
    let represented_relation_map = match resolution.differentials.first() {
        Some(first) => first.clone(),
        None => ModuleHomomorphism::zero(
            graded_zero_module_like(resolution.augmentation.source())?,
            resolution.augmentation.source().clone(),
        )?,
    };
    if resolution.augmentation.target().ring() != ring {
        return Err(FreeResolution::<C>::invalid(
            "augmentation changed the polynomial-ring context",
        ));
    }
    let original_identity =
        ModuleHomomorphism::identity(resolution.module.generator_module().clone())?;
    if resolution.augmentation == original_identity
        && resolution.augmentation_inverse == original_identity
        && represented_relation_map == *resolution.module.presentation()
    {
        return Ok(());
    }

    let original_relations = Submodule::from_generator_map(
        resolution.module.generator_module().clone(),
        resolution.module.presentation().clone(),
    )?;
    let represented_relations = Submodule::from_generator_map(
        resolution.augmentation.source().clone(),
        represented_relation_map.clone(),
    )?;
    let original_basis = original_relations.groebner_basis(order.clone())?;
    let represented_basis = represented_relations.groebner_basis(order)?;

    let represented_in_original = resolution.augmentation.compose(&represented_relation_map)?;
    if !image_lies_in(&represented_in_original, &original_basis)? {
        return Err(FreeResolution::<C>::invalid(
            "augmentation does not carry represented relations into the original relations",
        ));
    }
    let original_in_represented = resolution
        .augmentation_inverse
        .compose(resolution.module.presentation())?;
    if !image_lies_in(&original_in_represented, &represented_basis)? {
        return Err(FreeResolution::<C>::invalid(
            "augmentation inverse does not carry original relations into represented relations",
        ));
    }

    let original_error = resolution
        .augmentation
        .compose(&resolution.augmentation_inverse)?
        .sub(&original_identity)?;
    if !image_lies_in(&original_error, &original_basis)? {
        return Err(FreeResolution::<C>::invalid(
            "augmentation composite is not the identity modulo original relations",
        ));
    }

    let represented_identity =
        ModuleHomomorphism::identity(resolution.augmentation.source().clone())?;
    let represented_error = resolution
        .augmentation_inverse
        .compose(&resolution.augmentation)?
        .sub(&represented_identity)?;
    if !image_lies_in(&represented_error, &represented_basis)? {
        return Err(FreeResolution::<C>::invalid(
            "inverse augmentation composite is not the identity modulo represented relations",
        ));
    }
    Ok(())
}

fn verify_quotient_augmentation_equivalence<C: Field>(
    resolution: &FreeResolution<PolynomialQuotient<C>>,
    order: ModuleMonomialOrder,
) -> Result<(), ResolutionError> {
    let ring = resolution.module.generator_module().ring();
    let represented_relation_map = match resolution.differentials.first() {
        Some(first) => first.clone(),
        None => ModuleHomomorphism::zero(
            zero_module_like(resolution.augmentation.source())?,
            resolution.augmentation.source().clone(),
        )?,
    };
    if resolution.augmentation.target().ring() != ring {
        return Err(FreeResolution::<PolynomialQuotient<C>>::invalid(
            "augmentation changed the polynomial-quotient context",
        ));
    }
    let original_identity =
        ModuleHomomorphism::identity(resolution.module.generator_module().clone())?;
    if resolution.augmentation == original_identity
        && resolution.augmentation_inverse == original_identity
        && represented_relation_map == *resolution.module.presentation()
    {
        return Ok(());
    }

    let original_relations = Submodule::from_generator_map(
        resolution.module.generator_module().clone(),
        resolution.module.presentation().clone(),
    )?;
    let represented_relations = Submodule::from_generator_map(
        resolution.augmentation.source().clone(),
        represented_relation_map.clone(),
    )?;
    let original_basis = original_relations.groebner_basis(order.clone())?;
    let represented_basis = represented_relations.groebner_basis(order)?;

    let represented_in_original = resolution.augmentation.compose(&represented_relation_map)?;
    if !quotient_image_lies_in(&represented_in_original, &original_basis)? {
        return Err(FreeResolution::<PolynomialQuotient<C>>::invalid(
            "augmentation does not carry represented relations into the original relations",
        ));
    }
    let original_in_represented = resolution
        .augmentation_inverse
        .compose(resolution.module.presentation())?;
    if !quotient_image_lies_in(&original_in_represented, &represented_basis)? {
        return Err(FreeResolution::<PolynomialQuotient<C>>::invalid(
            "augmentation inverse does not carry original relations into represented relations",
        ));
    }

    let original_error = resolution
        .augmentation
        .compose(&resolution.augmentation_inverse)?
        .sub(&original_identity)?;
    if !quotient_image_lies_in(&original_error, &original_basis)? {
        return Err(FreeResolution::<PolynomialQuotient<C>>::invalid(
            "augmentation composite is not the identity modulo original relations",
        ));
    }

    let represented_identity =
        ModuleHomomorphism::identity(resolution.augmentation.source().clone())?;
    let represented_error = resolution
        .augmentation_inverse
        .compose(&resolution.augmentation)?
        .sub(&represented_identity)?;
    if !quotient_image_lies_in(&represented_error, &represented_basis)? {
        return Err(FreeResolution::<PolynomialQuotient<C>>::invalid(
            "inverse augmentation composite is not the identity modulo represented relations",
        ));
    }
    Ok(())
}

fn image_lies_in<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    basis: &super::ModuleGroebnerBasis<PolynomialRing<C>>,
) -> Result<bool, ResolutionError> {
    if map.target() != basis.input().ambient() {
        return Ok(false);
    }
    let values = (0..map.source().rank())
        .map(|position| {
            map.source()
                .basis_element(position)
                .and_then(|generator| map.apply(&generator))
        })
        .collect::<Result<Vec<_>, _>>()?;
    basis.contains_all(&values).map_err(Into::into)
}

fn quotient_image_lies_in<C: Field>(
    map: &ModuleHomomorphism<PolynomialQuotient<C>>,
    basis: &super::ModuleGroebnerBasis<PolynomialQuotient<C>>,
) -> Result<bool, ResolutionError> {
    if map.target() != basis.input().ambient() {
        return Ok(false);
    }
    let values = (0..map.source().rank())
        .map(|position| {
            map.source()
                .basis_element(position)
                .and_then(|generator| map.apply(&generator))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for value in values {
        if !basis.contains(&value)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn zero_module_like<R: Ring>(module: &FreeModule<R>) -> Result<FreeModule<R>, ResolutionError> {
    match module.grading_rank() {
        Some(rank) => {
            Ok(FreeModule::new(module.ring().clone(), 0).with_degree_shifts(rank, Vec::new())?)
        }
        None => Ok(FreeModule::new(module.ring().clone(), 0)),
    }
}

fn graded_zero_module_like<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
) -> Result<FreeModule<PolynomialRing<C>>, ResolutionError> {
    zero_module_like(module)
}

fn verify_minimality<C: Field>(
    resolution: &FreeResolution<PolynomialRing<C>>,
) -> Result<(), ResolutionError> {
    verify_homogeneous_map(resolution.module.presentation())?;
    verify_homogeneous_map(&resolution.augmentation)?;
    verify_homogeneous_map(&resolution.augmentation_inverse)?;
    for differential in &resolution.differentials {
        verify_homogeneous_map(differential)?;
        for entry in differential.matrix().rows().flatten() {
            if constant_inverse(differential.target().ring(), &entry).is_some() {
                return Err(FreeResolution::<C>::invalid(
                    "a claimed minimal differential contains a unit entry",
                ));
            }
        }
    }
    Ok(())
}

fn verify_homogeneous_map<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
) -> Result<(), ResolutionError> {
    let Some(source_shifts) = map.source().degree_shifts() else {
        return Err(FreeResolution::<C>::invalid(
            "minimalization requires graded free modules",
        ));
    };
    let Some(target_shifts) = map.target().degree_shifts() else {
        return Err(FreeResolution::<C>::invalid(
            "minimalization requires graded free modules",
        ));
    };
    if map.source().grading_rank() != map.target().grading_rank()
        || map.source().grading_rank() != Some(map.source().ring().grading().rank())
    {
        return Err(FreeResolution::<C>::invalid(
            "free-module shifts do not match the polynomial grading",
        ));
    }
    for (row, entries) in map.matrix().rows().enumerate() {
        for (column, entry) in entries.into_iter().enumerate() {
            if entry == map.target().ring().zero() {
                continue;
            }
            let degree = entry
                .multihomogeneous_degree()
                .map_err(|_| FreeResolution::<C>::invalid("polynomial degree computation failed"))?
                .ok_or_else(|| {
                    FreeResolution::<C>::invalid(
                        "a resolution differential contains an inhomogeneous entry",
                    )
                })?;
            let mapped_degree = degree
                .add(&source_shifts[column])
                .map_err(ModuleError::from)?;
            if mapped_degree != target_shifts[row] {
                return Err(FreeResolution::<C>::invalid(
                    "a resolution differential does not preserve degree",
                ));
            }
        }
    }
    Ok(())
}

fn minimalize_resolution<C: Field>(
    resolution: &FreeResolution<PolynomialRing<C>>,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    minimalize_resolution_owned(resolution.clone())
}

fn minimalize_resolution_owned<C: Field>(
    resolution: FreeResolution<PolynomialRing<C>>,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    if resolution.differentials.is_empty()
        && resolution.completeness != ResolutionCompleteness::Complete
    {
        return Err(FreeResolution::<C>::invalid(
            "degree-zero truncation does not contain enough evidence to minimalize",
        ));
    }
    verify_homogeneous_map(resolution.module.presentation())?;
    verify_homogeneous_map(&resolution.augmentation)?;
    verify_homogeneous_map(&resolution.augmentation_inverse)?;
    for differential in &resolution.differentials {
        verify_homogeneous_map(differential)?;
    }
    minimalize_admitted_resolution_owned(resolution)
}

fn minimalize_constructed_resolution_owned<C: Field>(
    resolution: FreeResolution<PolynomialRing<C>>,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    if resolution.differentials.is_empty()
        && resolution.completeness != ResolutionCompleteness::Complete
    {
        return Err(FreeResolution::<C>::invalid(
            "degree-zero truncation does not contain enough evidence to minimalize",
        ));
    }
    minimalize_admitted_resolution_owned(resolution)
}

fn minimalize_admitted_resolution_owned<C: Field>(
    mut resolution: FreeResolution<PolynomialRing<C>>,
) -> Result<FreeResolution<PolynomialRing<C>>, ResolutionError> {
    if graded_shifts_preclude_units(&resolution) {
        resolution.minimality = ResolutionMinimality::Minimal;
        verify_structure(&resolution)?;
        return Ok(resolution);
    }

    let mut modules = Vec::with_capacity(resolution.differentials.len() + 1);
    modules.push(resolution.augmentation.source().clone());
    modules.extend(
        resolution
            .differentials
            .iter()
            .map(|differential| differential.source().clone()),
    );
    let mut matrices = resolution
        .differentials
        .iter()
        .map(ModuleHomomorphism::matrix)
        .collect::<Vec<_>>();
    let Some(first_unit) = find_unit(&matrices) else {
        // Over a graded polynomial ring over a field, absence of scalar units
        // is exactly the minimality criterion. The input resolution has
        // already been admitted through a validating constructor, and this
        // branch changes no map, module, or augmentation. Retain that exactness
        // evidence and strengthen only the independently checked minimality
        // claim; replaying every Schreyer kernel here would repeat the dominant
        // computation without adding a new invariant.
        resolution.minimality = ResolutionMinimality::Minimal;
        verify_structure(&resolution)?;
        return Ok(resolution);
    };
    let mut augmentation = resolution.augmentation.clone();
    let mut augmentation_inverse = resolution.augmentation_inverse.clone();

    let mut next_unit = Some(first_unit);
    while let Some((index, row, column, inverse)) = next_unit {
        cancel_unit(
            &mut modules,
            &mut matrices,
            &mut augmentation,
            &mut augmentation_inverse,
            index,
            row,
            column,
            &inverse,
        )?;
        next_unit = find_unit(&matrices);
    }

    while matrices
        .last()
        .is_some_and(|_| modules.last().is_some_and(|module| module.rank() == 0))
    {
        matrices.pop();
        modules.pop();
    }
    let completeness = if resolution.completeness == ResolutionCompleteness::Complete
        || modules.last().is_some_and(|module| module.rank() == 0)
    {
        ResolutionCompleteness::Complete
    } else {
        ResolutionCompleteness::ThroughHomologicalDegree {
            degree: matrices.len(),
        }
    };
    let differentials = matrices
        .into_iter()
        .enumerate()
        .map(|(index, matrix)| {
            ModuleHomomorphism::from_matrix(
                modules[index + 1].clone(),
                modules[index].clone(),
                matrix,
            )
            .map_err(ResolutionError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = FreeResolution {
        module: resolution.module.clone(),
        differentials,
        augmentation,
        augmentation_inverse,
        completeness,
        minimality: ResolutionMinimality::Minimal,
    };
    // Unit cancellation is a chain isomorphism applied to an already
    // admitted resolution. The row/column operations above construct every
    // replacement map through checked module homomorphisms, and the loop has
    // just proved that no scalar unit remains. Replaying all Schreyer kernels
    // here would repeat exactness verification without establishing a new
    // claim; callers can still invoke `verify()` to replay it independently.
    verify_structure(&result)?;
    Ok(result)
}

fn graded_shifts_preclude_units<C: Field>(resolution: &FreeResolution<PolynomialRing<C>>) -> bool {
    resolution.differentials.iter().all(|differential| {
        let Some(source_shifts) = differential.source().degree_shifts() else {
            return false;
        };
        let Some(target_shifts) = differential.target().degree_shifts() else {
            return false;
        };
        source_shifts
            .iter()
            .all(|source| target_shifts.iter().all(|target| source != target))
    })
}

fn find_unit<C: Field>(
    matrices: &[Matrix<PolynomialRing<C>>],
) -> Option<(usize, usize, usize, crate::commalg::Polynomial<C>)> {
    for (index, matrix) in matrices.iter().enumerate() {
        for (row, entries) in matrix.rows().enumerate() {
            for (column, entry) in entries.into_iter().enumerate() {
                if let Some(inverse) = constant_inverse(matrix.ring(), &entry) {
                    return Some((index, row, column, inverse));
                }
            }
        }
    }
    None
}

fn constant_inverse<C: Field>(
    ring: &PolynomialRing<C>,
    value: &crate::commalg::Polynomial<C>,
) -> Option<crate::commalg::Polynomial<C>> {
    if value.nonzero_term_count() != 1 {
        return None;
    }
    let (exponent, coefficient) = value.terms().next()?;
    if exponent.iter().any(|power| !power.is_zero()) {
        return None;
    }
    ring.constant(ring.coefficient_ring().inverse(coefficient).ok()?)
        .ok()
}

#[allow(clippy::too_many_arguments)]
fn cancel_unit<C: Field>(
    modules: &mut [FreeModule<PolynomialRing<C>>],
    matrices: &mut [Matrix<PolynomialRing<C>>],
    augmentation: &mut ModuleHomomorphism<PolynomialRing<C>>,
    augmentation_inverse: &mut ModuleHomomorphism<PolynomialRing<C>>,
    index: usize,
    pivot_row: usize,
    pivot_column: usize,
    pivot_inverse: &crate::commalg::Polynomial<C>,
) -> Result<(), ResolutionError> {
    let ring = matrices[index].ring().clone();
    let current = matrices[index]
        .rows()
        .collect::<Vec<Vec<crate::commalg::Polynomial<C>>>>();
    let retained_rows = (0..modules[index].rank())
        .filter(|row| *row != pivot_row)
        .collect::<Vec<_>>();
    let retained_columns = (0..modules[index + 1].rank())
        .filter(|column| *column != pivot_column)
        .collect::<Vec<_>>();

    if index == 0 {
        let old_augmentation = augmentation.matrix().rows().collect::<Vec<_>>();
        let new_augmentation = old_augmentation
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter_map(|(column, value)| (column != pivot_row).then_some(value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let old_inverse = augmentation_inverse.matrix().rows().collect::<Vec<_>>();
        let pivot_inverse_row = old_inverse[pivot_row].clone();
        let mut new_inverse = Vec::with_capacity(retained_rows.len());
        for &row in &retained_rows {
            let factor = ring
                .mul(&current[row][pivot_column], pivot_inverse)
                .map_err(ModuleError::from)?;
            let mut values = Vec::with_capacity(pivot_inverse_row.len());
            for (value, pivot_value) in old_inverse[row].iter().zip(&pivot_inverse_row) {
                let correction = ring.mul(&factor, pivot_value).map_err(ModuleError::from)?;
                values.push(ring.sub(value, &correction).map_err(ModuleError::from)?);
            }
            new_inverse.push(values);
        }
        let new_degree_zero = remove_basis(&modules[0], pivot_row, true)?;
        *augmentation = ModuleHomomorphism::from_matrix(
            new_degree_zero.clone(),
            augmentation.target().clone(),
            matrix_from_rows(
                ring.clone(),
                augmentation.target().rank(),
                new_degree_zero.rank(),
                new_augmentation,
            )?,
        )?;
        *augmentation_inverse = ModuleHomomorphism::from_matrix(
            augmentation_inverse.source().clone(),
            new_degree_zero,
            matrix_from_rows(
                ring.clone(),
                retained_rows.len(),
                augmentation_inverse.source().rank(),
                new_inverse,
            )?,
        )?;
    }

    if index > 0 {
        let previous = matrices[index - 1]
            .rows()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter_map(|(column, value)| (column != pivot_row).then_some(value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        matrices[index - 1] = matrix_from_rows(
            ring.clone(),
            modules[index - 1].rank(),
            retained_rows.len(),
            previous,
        )?;
    }
    if index + 1 < matrices.len() {
        let next = matrices[index + 1]
            .rows()
            .enumerate()
            .filter_map(|(row, values)| (row != pivot_column).then_some(values))
            .collect::<Vec<_>>();
        matrices[index + 1] = matrix_from_rows(
            ring.clone(),
            retained_columns.len(),
            modules[index + 2].rank(),
            next,
        )?;
    }

    let mut reduced = Vec::with_capacity(retained_rows.len());
    for &row in &retained_rows {
        let left = ring
            .mul(&current[row][pivot_column], pivot_inverse)
            .map_err(ModuleError::from)?;
        let mut values = Vec::with_capacity(retained_columns.len());
        for &column in &retained_columns {
            let correction = ring
                .mul(&left, &current[pivot_row][column])
                .map_err(ModuleError::from)?;
            values.push(
                ring.sub(&current[row][column], &correction)
                    .map_err(ModuleError::from)?,
            );
        }
        reduced.push(values);
    }
    matrices[index] = matrix_from_rows(ring, retained_rows.len(), retained_columns.len(), reduced)?;
    modules[index] = remove_basis(&modules[index], pivot_row, true)?;
    modules[index + 1] = remove_basis(&modules[index + 1], pivot_column, false)?;
    Ok(())
}

fn remove_basis<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
    removed: usize,
    preserve_labels: bool,
) -> Result<FreeModule<PolynomialRing<C>>, ResolutionError> {
    let labels = preserve_labels.then(|| {
        module.basis_labels().map(|labels| {
            labels
                .iter()
                .enumerate()
                .filter_map(|(index, label)| (index != removed).then_some(label.clone()))
                .collect::<Vec<_>>()
        })
    });
    let shifts = module.degree_shifts().map(|shifts| {
        shifts
            .iter()
            .enumerate()
            .filter_map(|(index, shift)| (index != removed).then_some(shift.clone()))
            .collect::<Vec<_>>()
    });
    FreeModule::with_basis_data(
        module.ring().clone(),
        module.rank() - 1,
        labels.flatten(),
        module.grading_rank(),
        shifts,
    )
    .map_err(Into::into)
}

fn matrix_from_rows<C: Field>(
    ring: PolynomialRing<C>,
    rows: usize,
    columns: usize,
    values: Vec<Vec<crate::commalg::Polynomial<C>>>,
) -> Result<Matrix<PolynomialRing<C>>, ResolutionError> {
    if rows == 0 || columns == 0 {
        return Ok(Matrix::zeros(ring, rows, columns));
    }
    Matrix::from_rows(ring, values)
        .map_err(ModuleError::from)
        .map_err(Into::into)
}

impl<R> Serialize for FreeResolution<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        FreeResolutionWireRef {
            type_id: Self::TYPE_ID,
            module: &self.module,
            differentials: &self.differentials,
            augmentation: &self.augmentation,
            augmentation_inverse: &self.augmentation_inverse,
            completeness: &self.completeness,
            minimality: &self.minimality,
        }
        .serialize(serializer)
    }
}

impl<'de, C> Deserialize<'de> for FreeResolution<PolynomialRing<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FreeResolutionWire::<PolynomialRing<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/FreeResolution@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/FreeResolution@1", wire.type_id
            )));
        }
        Self::from_maps(
            wire.module,
            wire.differentials,
            wire.augmentation,
            wire.augmentation_inverse,
            wire.completeness,
            wire.minimality,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<'de, C> Deserialize<'de> for FreeResolution<PolynomialQuotient<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FreeResolutionWire::<PolynomialQuotient<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/FreeResolution@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/FreeResolution@1", wire.type_id
            )));
        }
        Self::from_maps(
            wire.module,
            wire.differentials,
            wire.augmentation,
            wire.augmentation_inverse,
            wire.completeness,
            wire.minimality,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for FreeResolution<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FreeResolution")
            .field("length", &self.length())
            .field("completeness", &self.completeness)
            .field("minimality", &self.minimality)
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for FreeResolution<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FreeResolution(length={}, {}, {})",
            self.length(),
            self.completeness,
            self.minimality
        )
    }
}

impl<R> SemanticType for FreeResolution<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/FreeResolution@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &FREE_RESOLUTION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use crate::commalg::{Degree, MonomialOrder, PolynomialRing};
    use crate::linalg::Matrix;
    use crate::numthy::FiniteField;
    use crate::Q;

    use super::*;

    fn quotient_by<C: Field>(
        ring: &PolynomialRing<C>,
        generators: Vec<crate::commalg::Polynomial<C>>,
        degree: i64,
    ) -> PresentedModule<PolynomialRing<C>> {
        let target = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(1, vec![Degree::zero(1)])
            .unwrap();
        let source = FreeModule::new(ring.clone(), generators.len())
            .with_degree_shifts(
                1,
                (0..generators.len())
                    .map(|_| Degree::new(vec![(-degree).into()]))
                    .collect(),
            )
            .unwrap();
        let map = ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(ring.clone(), [generators]).unwrap(),
        )
        .unwrap();
        PresentedModule::cokernel(map)
    }

    #[test]
    fn quotient_resolutions_are_honestly_truncated_when_syzygies_are_periodic() {
        let polynomial_ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = polynomial_ring.generators().unwrap().remove(0);
        let quotient = polynomial_ring
            .quotient(&polynomial_ring.ideal([x.pow(2).unwrap()]).unwrap())
            .unwrap();
        let x_bar = quotient.element(&x).unwrap();
        let free = FreeModule::new(quotient.clone(), 1);
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                free.clone(),
                free,
                Matrix::from_rows(quotient.clone(), [[x_bar.clone()]]).unwrap(),
            )
            .unwrap(),
        );
        let order = ModuleMonomialOrder::TermOverPosition(polynomial_ring.term_order());

        let resolution = module.free_resolution_through(order, 3).unwrap();
        resolution.verify().unwrap();
        assert_eq!(resolution.length(), 3);
        assert_eq!(
            resolution.completeness(),
            &ResolutionCompleteness::ThroughHomologicalDegree { degree: 3 }
        );
        for degree in 1..=3 {
            assert_eq!(
                resolution.differential(degree).unwrap().matrix(),
                Matrix::from_rows(quotient.clone(), [[x_bar.clone()]]).unwrap()
            );
        }
        let round_trip = FreeResolution::<PolynomialQuotient<crate::RationalField>>::from_json(
            &resolution.to_json().unwrap(),
        )
        .unwrap();
        assert_eq!(round_trip, resolution);
    }

    #[test]
    fn twisted_cubic_resolution_is_complete_and_exact() {
        let ring = PolynomialRing::new(
            Q,
            ["x", "y", "z", "w"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let generators = ring.generators().unwrap();
        let [x, y, z, w]: [_; 4] = generators.try_into().unwrap();
        let module = quotient_by(
            &ring,
            vec![
                (&(&x * &z).unwrap() - &y.pow(2).unwrap()).unwrap(),
                (&(&x * &w).unwrap() - &(&y * &z).unwrap()).unwrap(),
                (&(&y * &w).unwrap() - &z.pow(2).unwrap()).unwrap(),
            ],
            2,
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let resolution = module.free_resolution(order.clone()).unwrap();
        assert_eq!(resolution.length(), 2);
        assert_eq!(resolution.completeness(), &ResolutionCompleteness::Complete);
        assert_eq!(resolution.free_module(0).unwrap().rank(), 1);
        assert_eq!(resolution.free_module(1).unwrap().rank(), 3);
        assert_eq!(resolution.free_module(2).unwrap().rank(), 2);
        resolution.verify().unwrap();
        assert_eq!(
            FreeResolution::<PolynomialRing<crate::RationalField>>::from_json(
                &resolution.to_json().unwrap(),
            )
            .unwrap(),
            resolution
        );
        let minimal = resolution.minimalize().unwrap();
        assert!(minimal.is_minimal());
        assert_eq!(minimal.differentials, resolution.differentials);
        assert_eq!(minimal.augmentation, resolution.augmentation);
        assert_eq!(
            minimal.augmentation_inverse,
            resolution.augmentation_inverse
        );
        assert_eq!(minimal.minimality(), &ResolutionMinimality::Minimal);
        assert_eq!(minimal, minimal.minimalize().unwrap());
        assert_eq!(
            ResolutionMinimality::from_json(&minimal.minimality().to_json().unwrap()).unwrap(),
            ResolutionMinimality::Minimal
        );
        let betti = minimal.graded_betti_table().unwrap();
        assert_eq!(betti.total(0), 1);
        assert_eq!(betti.total(1), 3);
        assert_eq!(betti.total(2), 2);
        assert_eq!(betti.get(1, &Degree::new(vec![2.into()])), 3);
        assert_eq!(betti.get(2, &Degree::new(vec![3.into()])), 2);
        assert_eq!(module.graded_betti_table(order.clone()).unwrap(), betti);

        let imported = FreeResolution::<PolynomialRing<crate::RationalField>>::from_maps(
            minimal.module().clone(),
            minimal.differentials.clone(),
            minimal.augmentation().clone(),
            minimal.augmentation_inverse().clone(),
            minimal.completeness().clone(),
            ResolutionMinimality::Minimal,
        )
        .unwrap();
        assert_eq!(imported, minimal);

        let mut incomplete_maps = minimal.differentials.clone();
        incomplete_maps[1] = ModuleHomomorphism::zero(
            incomplete_maps[1].source().clone(),
            incomplete_maps[1].target().clone(),
        )
        .unwrap();
        assert!(
            FreeResolution::<PolynomialRing<crate::RationalField>>::from_maps(
                minimal.module().clone(),
                incomplete_maps,
                minimal.augmentation().clone(),
                minimal.augmentation_inverse().clone(),
                ResolutionCompleteness::Complete,
                ResolutionMinimality::Minimal,
            )
            .is_err()
        );

        let truncated = module.free_resolution_through(order, 1).unwrap();
        assert_eq!(truncated.length(), 1);
        assert_eq!(
            truncated.completeness(),
            &ResolutionCompleteness::ThroughHomologicalDegree { degree: 1 }
        );
        truncated.verify().unwrap();
    }

    #[test]
    fn packed_prime_generic_two_by_three_minors_resolution_is_exact() {
        let ring = PolynomialRing::new(
            FiniteField::prime(32_003_u64).unwrap(),
            ["a", "b", "c", "d", "e", "f"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [a, b, c, d, e, f]: [_; 6] = ring.generators().unwrap().try_into().unwrap();
        let module = quotient_by(
            &ring,
            vec![
                (&(&a * &e).unwrap() - &(&b * &d).unwrap()).unwrap(),
                (&(&a * &f).unwrap() - &(&c * &d).unwrap()).unwrap(),
                (&(&b * &f).unwrap() - &(&c * &e).unwrap()).unwrap(),
            ],
            2,
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let resolution = module.minimal_free_resolution(order.clone()).unwrap();
        resolution.verify().unwrap();
        assert!(resolution.is_minimal());
        assert_eq!(resolution.length(), 2);
        assert_eq!(resolution.completeness(), &ResolutionCompleteness::Complete);
        assert_eq!(
            (0..=2)
                .map(|degree| resolution.free_module(degree).unwrap().rank())
                .collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
        let betti = resolution.graded_betti_table().unwrap();
        assert_eq!(betti.get(1, &Degree::new(vec![2.into()])), 3);
        assert_eq!(betti.get(2, &Degree::new(vec![3.into()])), 2);
        assert_eq!(module.graded_betti_table(order).unwrap(), betti);
    }

    #[test]
    fn complete_intersection_resolution_has_koszul_ranks() {
        let ring = PolynomialRing::new(
            Q,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let generators = ring.generators().unwrap();
        let module = quotient_by(
            &ring,
            generators
                .iter()
                .map(|variable| variable.pow(2).unwrap())
                .collect(),
            2,
        );
        let resolution = module
            .free_resolution(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert_eq!(resolution.length(), 3);
        assert_eq!(
            (0..=3)
                .map(|degree| resolution.free_module(degree).unwrap().rank())
                .collect::<Vec<_>>(),
            vec![1, 3, 3, 1]
        );
        assert_eq!(
            resolution.free_module(3).unwrap().degree_shifts().unwrap(),
            &[Degree::new(vec![(-6).into()])]
        );
        let minimal = resolution.minimalize().unwrap();
        minimal.verify().unwrap();
        let betti = minimal.graded_betti_table().unwrap();
        assert_eq!(
            (0..=3)
                .map(|degree| betti.total(degree))
                .collect::<Vec<_>>(),
            vec![1, 3, 3, 1]
        );
        assert_eq!(betti.get(3, &Degree::new(vec![6.into()])), 1);
        assert_eq!(
            module
                .graded_betti_table(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
                .unwrap(),
            betti
        );
    }

    #[test]
    fn minimal_truncation_retains_an_independent_equal_degree_presentation() {
        let ring = PolynomialRing::new(
            Q,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let generators = ring.generators().unwrap();
        let x = generators[0].clone();
        let y = generators[1].clone();
        let source = FreeModule::new(ring.clone(), 3)
            .with_degree_shifts(1, vec![Degree::new(vec![(-1).into()]); 3])
            .unwrap();
        let target = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(1, vec![Degree::zero(1); 2])
            .unwrap();
        let presentation = ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(
                ring.clone(),
                [[x.clone(), y.clone(), ring.zero()], [ring.zero(), x, y]],
            )
            .unwrap(),
        )
        .unwrap();
        let module = PresentedModule::cokernel(presentation);
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let complete = module.minimal_free_resolution(order.clone()).unwrap();
        let truncated = module.minimal_free_resolution_through(order, 1).unwrap();

        assert_eq!(
            truncated.completeness(),
            &ResolutionCompleteness::ThroughHomologicalDegree { degree: 1 }
        );
        assert!(truncated.is_minimal());
        assert_eq!(truncated.free_module(0), complete.free_module(0));
        assert_eq!(truncated.free_module(1), complete.free_module(1));
        assert_eq!(truncated.differential(1), Some(module.presentation()));
        assert_eq!(truncated.free_module(1).unwrap().rank(), 3);
        let table = truncated.graded_betti_table().unwrap();
        assert_eq!(table.total(0), 2);
        assert_eq!(table.total(1), 3);
        assert_eq!(table.projective_dimension(), None);
        truncated.verify().unwrap();
    }

    #[test]
    fn minimal_truncation_looks_ahead_to_remove_a_redundant_last_generator() {
        let ring = PolynomialRing::new(
            Q,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let generators = ring.generators().unwrap();
        let x = generators[0].clone();
        let y = generators[1].clone();
        let source = FreeModule::new(ring.clone(), 4)
            .with_degree_shifts(
                1,
                vec![
                    Degree::new(vec![(-1).into()]),
                    Degree::new(vec![(-1).into()]),
                    Degree::new(vec![(-1).into()]),
                    Degree::new(vec![(-2).into()]),
                ],
            )
            .unwrap();
        let target = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(1, vec![Degree::zero(1); 2])
            .unwrap();
        let presentation = ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(
                ring.clone(),
                [
                    [x.clone(), y.clone(), ring.zero(), x.pow(2).unwrap()],
                    [ring.zero(), x, y, ring.zero()],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let module = PresentedModule::cokernel(presentation);
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let complete = module.minimal_free_resolution(order.clone()).unwrap();
        let truncated = module.minimal_free_resolution_through(order, 1).unwrap();

        assert_eq!(truncated.free_module(0), complete.free_module(0));
        assert_eq!(truncated.free_module(1), complete.free_module(1));
        assert_eq!(truncated.free_module(1).unwrap().rank(), 3);
        assert_eq!(truncated.graded_betti_table().unwrap().total(1), 3);
        truncated.verify().unwrap();
    }

    #[test]
    fn finite_field_betti_stream_matches_minimal_resolution() {
        let ring = PolynomialRing::new(
            FiniteField::prime(65_521_u64).unwrap(),
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let generators = ring.generators().unwrap();
        let module = quotient_by(
            &ring,
            vec![
                (&generators[0] * &generators[1]).unwrap(),
                (&generators[0] * &generators[2]).unwrap(),
                (&generators[1] * &generators[2]).unwrap(),
            ],
            2,
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let direct = module.graded_betti_table(order.clone()).unwrap();
        let oracle = module
            .minimal_free_resolution(order)
            .unwrap()
            .graded_betti_table()
            .unwrap();

        assert_eq!(direct, oracle);
        assert_eq!(direct.total(0), 1);
        assert_eq!(direct.total(1), 3);
        assert_eq!(direct.total(2), 2);
        direct.verify().unwrap();
    }

    #[test]
    fn packed_resolution_retains_the_distinct_even_cycle_betti_strands() {
        let ring = PolynomialRing::new(
            FiniteField::prime(65_521_u64).unwrap(),
            ["a", "b", "c", "d", "e", "f"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let variables = ring.generators().unwrap();
        let edges = variables
            .iter()
            .zip(variables.iter().cycle().skip(1))
            .take(variables.len())
            .map(|(left, right)| (left * right).unwrap())
            .collect();
        let module = quotient_by(&ring, edges, 2);
        let resolution = module
            .minimal_free_resolution(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        resolution.verify().unwrap();
        let table = resolution.graded_betti_table().unwrap();

        assert_eq!(resolution.length(), 4);
        assert_eq!(table.get(1, &Degree::new(vec![2.into()])), 6);
        assert_eq!(table.get(2, &Degree::new(vec![3.into()])), 6);
        assert_eq!(table.get(2, &Degree::new(vec![4.into()])), 3);
        assert_eq!(table.get(3, &Degree::new(vec![5.into()])), 6);
        assert_eq!(table.get(4, &Degree::new(vec![6.into()])), 2);
        assert_eq!(
            (0..=4).map(|i| table.total(i)).collect::<Vec<_>>(),
            [1, 6, 9, 6, 2]
        );
    }

    #[test]
    fn a_free_module_and_zero_truncation_have_distinct_honest_claims() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let source = FreeModule::new(ring.clone(), 0);
        let target = FreeModule::new(ring.clone(), 2);
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(source, target, Matrix::zeros(ring.clone(), 2, 0))
                .unwrap(),
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let complete = module.free_resolution(order.clone()).unwrap();
        assert_eq!(complete.length(), 0);
        assert_eq!(complete.completeness(), &ResolutionCompleteness::Complete);
        assert_eq!(complete.chain_complex().unwrap().module_dimension(0), 2);

        let truncated = module.free_resolution_through(order, 0).unwrap();
        assert_eq!(
            truncated.completeness(),
            &ResolutionCompleteness::ThroughHomologicalDegree { degree: 0 }
        );
        truncated.verify().unwrap();
        assert_eq!(
            FreeResolution::<PolynomialRing<crate::RationalField>>::from_json(
                &truncated.to_json().unwrap(),
            )
            .unwrap(),
            truncated
        );
    }

    #[test]
    fn degree_one_unit_cancellation_minimalizes_the_presentation() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let target = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(1, vec![Degree::zero(1), Degree::zero(1)])
            .unwrap();
        let source = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(1, vec![Degree::zero(1), Degree::new(vec![(-1).into()])])
            .unwrap();
        let presentation = ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(
                ring.clone(),
                [[ring.one(), ring.zero()], [ring.zero(), x.clone()]],
            )
            .unwrap(),
        )
        .unwrap();
        let module = PresentedModule::cokernel(presentation);
        let minimal = module
            .minimal_free_resolution(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        minimal.verify().unwrap();
        assert_eq!(minimal.completeness(), &ResolutionCompleteness::Complete);
        assert!(minimal.is_minimal());
        assert_eq!(minimal.length(), 1);
        assert_eq!(minimal.free_module(0).unwrap().rank(), 1);
        assert_eq!(minimal.free_module(1).unwrap().rank(), 1);
        assert_eq!(
            minimal.differential(1).unwrap().matrix().get(0, 0).unwrap(),
            x
        );
        assert_eq!(minimal.augmentation().source().rank(), 1);
        assert_eq!(minimal.augmentation().target().rank(), 2);
        assert_eq!(minimal.augmentation_inverse().source().rank(), 2);
        let betti = minimal.graded_betti_table().unwrap();
        assert_eq!(betti.get(0, &Degree::zero(1)), 1);
        assert_eq!(betti.get(1, &Degree::new(vec![1.into()])), 1);
        assert_eq!(
            module
                .graded_betti_table(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
                .unwrap(),
            betti
        );
        assert_eq!(
            FreeResolution::<PolynomialRing<crate::RationalField>>::from_json(
                &minimal.to_json().unwrap(),
            )
            .unwrap(),
            minimal
        );
    }
}
