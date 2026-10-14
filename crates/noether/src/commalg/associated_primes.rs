// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Associated primes from verified Ext and annihilator evidence.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::homalg::{
    DerivedError, FreeModule, FreeResolution, ModuleError, ModuleHomomorphism,
    ModuleInvariantError, ModuleMonomialOrder, PresentedModule, ResolutionError,
};
use crate::linalg::{LinalgError, Matrix};
use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, SemanticType, TypeDescription};

use super::minimal_primes::{
    construct_minimal_primes, sorted_prime_certificates, MinimalPrimeCoefficientField,
};
use super::{
    DecompositionError, GroebnerError, Ideal, MinimalPrimeDecomposition,
    PolynomialPrimeCertificate, PolynomialRing,
};

static ASSOCIATED_PRIME_SET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/AssociatedPrimeSet@1",
    rust_path: "noether::commalg::AssociatedPrimeSet",
    python_path: Some("noether.commalg.AssociatedPrimeSet"),
};

static ASSOCIATED_PRIME_EXT_STAGE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/AssociatedPrimeExtStage@1",
    rust_path: "noether::commalg::AssociatedPrimeExtStage",
    python_path: Some("noether.commalg.AssociatedPrimeExtStage"),
};

/// Failure to construct or verify homological associated-prime evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssociatedPrimeError {
    /// Exact polynomial ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Construction of the cyclic module `R/I` failed.
    Module(ModuleError),
    /// Construction or verification of a complete free resolution failed.
    Resolution(ResolutionError),
    /// Construction of an Ext module failed.
    Derived(DerivedError),
    /// An annihilator computation failed.
    Invariant(ModuleInvariantError),
    /// A required minimal-prime computation failed closed.
    Decomposition(DecompositionError),
    /// Exact matrix construction failed.
    Linalg(LinalgError),
    /// Retained EHV evidence failed independent verification.
    Invalid(String),
}

impl fmt::Display for AssociatedPrimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Groebner(error) => error.fmt(formatter),
            Self::Module(error) => error.fmt(formatter),
            Self::Resolution(error) => error.fmt(formatter),
            Self::Derived(error) => error.fmt(formatter),
            Self::Invariant(error) => error.fmt(formatter),
            Self::Decomposition(error) => error.fmt(formatter),
            Self::Linalg(error) => error.fmt(formatter),
            Self::Invalid(reason) => {
                write!(formatter, "invalid associated-prime evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for AssociatedPrimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Groebner(error) => Some(error),
            Self::Module(error) => Some(error),
            Self::Resolution(error) => Some(error),
            Self::Derived(error) => Some(error),
            Self::Invariant(error) => Some(error),
            Self::Decomposition(error) => Some(error),
            Self::Linalg(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<GroebnerError> for AssociatedPrimeError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<ModuleError> for AssociatedPrimeError {
    fn from(error: ModuleError) -> Self {
        Self::Module(error)
    }
}

impl From<ResolutionError> for AssociatedPrimeError {
    fn from(error: ResolutionError) -> Self {
        Self::Resolution(error)
    }
}

impl From<DerivedError> for AssociatedPrimeError {
    fn from(error: DerivedError) -> Self {
        Self::Derived(error)
    }
}

impl From<ModuleInvariantError> for AssociatedPrimeError {
    fn from(error: ModuleInvariantError) -> Self {
        Self::Invariant(error)
    }
}

impl From<DecompositionError> for AssociatedPrimeError {
    fn from(error: DecompositionError) -> Self {
        Self::Decomposition(error)
    }
}

impl From<LinalgError> for AssociatedPrimeError {
    fn from(error: LinalgError) -> Self {
        Self::Linalg(error)
    }
}

/// One codimension stage in the Eisenbud--Huneke--Vasconcelos computation.
///
/// Its parent associated-prime result retains the common complete resolution.
/// This stage records the resulting `Ext^i_R(M,R)`, its annihilator, and the
/// height-`i` minimal-prime evidence when that annihilator has codimension `i`.
#[derive(Clone, PartialEq, Eq)]
pub struct AssociatedPrimeExtStage<R: Field> {
    codimension: usize,
    ext_module: PresentedModule<PolynomialRing<R>>,
    annihilator: Ideal<PolynomialRing<R>>,
    minimal_primes: Option<MinimalPrimeDecomposition<R>>,
    associated_primes: Vec<PolynomialPrimeCertificate<R>>,
}

impl<R: Field> AssociatedPrimeExtStage<R> {
    /// Ext degree and height selected by this stage.
    pub const fn codimension(&self) -> usize {
        self.codimension
    }

    /// Exact presentation of `Ext^i_R(M,R)`.
    pub fn ext_module(&self) -> &PresentedModule<PolynomialRing<R>> {
        &self.ext_module
    }

    /// Exact annihilator of the retained Ext module.
    pub fn annihilator(&self) -> &Ideal<PolynomialRing<R>> {
        &self.annihilator
    }

    /// Complete minimal-prime evidence when the annihilator has codimension `i`.
    pub fn minimal_primes(&self) -> Option<&MinimalPrimeDecomposition<R>> {
        self.minimal_primes.as_ref()
    }

    /// Height-`i` associated primes contributed by this stage.
    pub fn associated_primes(&self) -> &[PolynomialPrimeCertificate<R>] {
        &self.associated_primes
    }
}

impl<R: Field> fmt::Debug for AssociatedPrimeExtStage<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AssociatedPrimeExtStage")
            .field("codimension", &self.codimension)
            .field("associated_primes", &self.associated_primes.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for AssociatedPrimeExtStage<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "AssociatedPrimeExtStage(codimension={}, associated_primes={})",
            self.codimension,
            self.associated_primes.len()
        )
    }
}

impl<R> SemanticType for AssociatedPrimeExtStage<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/AssociatedPrimeExtStage@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &ASSOCIATED_PRIME_EXT_STAGE_DESCRIPTION
    }
}

/// Complete associated-prime evidence for one polynomial ideal.
///
/// One verified free resolution of `R/I` is reused for every retained Ext
/// stage. In the regular ring `R`, the Eisenbud--Huneke--Vasconcelos theorem
/// identifies the height-`i` associated primes of `R/I` with the height-`i`
/// minimal primes of `ann Ext^i_R(R/I,R)`.
#[derive(Clone, PartialEq, Eq)]
pub struct AssociatedPrimeSet<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    resolution: FreeResolution<PolynomialRing<R>>,
    stages: Vec<AssociatedPrimeExtStage<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

impl<R: Field> AssociatedPrimeSet<R> {
    /// Original ideal whose quotient module is examined.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// One complete verified resolution of `R/I`, shared by all Ext stages.
    pub fn resolution(&self) -> &FreeResolution<PolynomialRing<R>> {
        &self.resolution
    }

    /// Consecutive Ext stages from `codim(I)` through the ambient dimension.
    pub fn stages(&self) -> &[AssociatedPrimeExtStage<R>] {
        &self.stages
    }

    /// Canonically ordered associated-prime certificates, including embedded primes.
    pub fn prime_certificates(&self) -> &[PolynomialPrimeCertificate<R>] {
        &self.prime_certificates
    }

    /// Canonically ordered associated-prime ideals, including embedded primes.
    pub fn primes(&self) -> Vec<&Ideal<PolynomialRing<R>>> {
        self.prime_certificates
            .iter()
            .map(PolynomialPrimeCertificate::ideal)
            .collect()
    }

    /// Exact theorem proving completeness of the associated-prime set.
    pub const fn theorem(&self) -> &'static str {
        "the height-i associated primes of R/I are the height-i minimal primes of ann Ext^i_R(R/I,R)"
    }
}

impl<R: Field> fmt::Debug for AssociatedPrimeSet<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AssociatedPrimeSet")
            .field("stages", &self.stages.len())
            .field("associated_primes", &self.prime_certificates.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for AssociatedPrimeSet<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "AssociatedPrimeSet(stages={}, associated_primes={})",
            self.stages.len(),
            self.prime_certificates.len()
        )
    }
}

impl<R> SemanticType for AssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/AssociatedPrimeSet@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &ASSOCIATED_PRIME_SET_DESCRIPTION
    }
}

fn cyclic_quotient_module<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<PresentedModule<PolynomialRing<R>>, AssociatedPrimeError> {
    let ring = source.ring();
    let relation_module = FreeModule::new(ring.clone(), source.generators().len());
    let generator_module = FreeModule::new(ring.clone(), 1);
    let presentation_matrix = if source.generators().is_empty() {
        Matrix::zeros(ring.clone(), 1, 0)
    } else {
        Matrix::from_rows(ring.clone(), [source.generators().to_vec()])?
    };
    Ok(PresentedModule::cokernel(ModuleHomomorphism::from_matrix(
        relation_module,
        generator_module,
        presentation_matrix,
    )?))
}

pub(crate) fn quotient_codimension<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<usize, AssociatedPrimeError> {
    let ambient_dimension = ideal.ring().variables().len();
    let dimension = ideal.dimension()?;
    if dimension < 0 {
        return ambient_dimension.checked_add(1).ok_or_else(|| {
            AssociatedPrimeError::Invalid("ambient dimension overflowed".to_owned())
        });
    }
    let dimension = usize::try_from(dimension).map_err(|_| {
        AssociatedPrimeError::Invalid("quotient dimension does not fit usize".to_owned())
    })?;
    ambient_dimension.checked_sub(dimension).ok_or_else(|| {
        AssociatedPrimeError::Invalid(
            "quotient dimension exceeds the polynomial-ring dimension".to_owned(),
        )
    })
}

pub(crate) fn stage_from_ext<R>(
    resolution: &FreeResolution<PolynomialRing<R>>,
    coefficient_module: &PresentedModule<PolynomialRing<R>>,
    codimension: usize,
    order: ModuleMonomialOrder,
) -> Result<AssociatedPrimeExtStage<R>, AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ext = resolution.ext(coefficient_module, codimension, order.clone())?;
    let ext_module = ext.module().clone();
    let annihilator = ext_module.annihilator(order)?;
    let ambient_dimension = annihilator.ring().variables().len();
    let annihilator_codimension = quotient_codimension(&annihilator)?;
    let minimal_primes =
        if annihilator_codimension == codimension && annihilator_codimension <= ambient_dimension {
            Some(construct_minimal_primes(&annihilator)?)
        } else {
            None
        };
    let mut associated_primes = Vec::new();
    if let Some(decomposition) = &minimal_primes {
        for certificate in decomposition.prime_certificates() {
            if quotient_codimension(certificate.ideal())? == codimension {
                associated_primes.push(certificate.clone());
            }
        }
    }
    Ok(AssociatedPrimeExtStage {
        codimension,
        ext_module,
        annihilator,
        minimal_primes,
        associated_primes,
    })
}

pub(crate) fn sorted_associated_prime_certificates<R>(
    stages: &[AssociatedPrimeExtStage<R>],
) -> Result<Vec<PolynomialPrimeCertificate<R>>, AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    sorted_prime_certificates(
        stages
            .iter()
            .flat_map(|stage| stage.associated_primes.iter().cloned())
            .collect(),
    )
    .map_err(Into::into)
}

fn verify_ext_stage<R>(stage: &AssociatedPrimeExtStage<R>) -> Result<(), AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ring = stage.ext_module.generator_module().ring();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    let annihilator = stage.ext_module.annihilator(order)?;
    if stage.annihilator != annihilator {
        return Err(AssociatedPrimeError::Invalid(
            "the retained ideal is not the annihilator of the Ext-stage module".to_owned(),
        ));
    }
    let ambient_dimension = ring.variables().len();
    let annihilator_codimension = quotient_codimension(&annihilator)?;
    let minimal_primes = if annihilator_codimension == stage.codimension
        && annihilator_codimension <= ambient_dimension
    {
        Some(construct_minimal_primes(&annihilator)?)
    } else {
        None
    };
    if stage.minimal_primes != minimal_primes {
        return Err(AssociatedPrimeError::Invalid(
            "the retained minimal primes do not match the Ext-stage annihilator".to_owned(),
        ));
    }
    let mut associated_primes = Vec::new();
    if let Some(decomposition) = &minimal_primes {
        for certificate in decomposition.prime_certificates() {
            if quotient_codimension(certificate.ideal())? == stage.codimension {
                associated_primes.push(certificate.clone());
            }
        }
    }
    if stage.associated_primes != associated_primes {
        return Err(AssociatedPrimeError::Invalid(
            "the retained associated primes are not the height-filtered annihilator primes"
                .to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn construct_associated_primes<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<AssociatedPrimeSet<R>, AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let module = cyclic_quotient_module(source)?;
    let order = ModuleMonomialOrder::TermOverPosition(source.ring().term_order());
    let resolution = module.free_resolution(order.clone())?;
    let coefficient_module = PresentedModule::free(FreeModule::new(source.ring().clone(), 1))?;
    let first_codimension = quotient_codimension(source)?;
    let ambient_dimension = source.ring().variables().len();
    let mut stages = Vec::new();
    for codimension in first_codimension..=ambient_dimension {
        stages.push(stage_from_ext(
            &resolution,
            &coefficient_module,
            codimension,
            order.clone(),
        )?);
    }
    let prime_certificates = sorted_associated_prime_certificates(&stages)?;
    Ok(AssociatedPrimeSet {
        source: source.clone(),
        resolution,
        stages,
        prime_certificates,
    })
}

pub(super) fn verify_associated_primes<R>(
    value: &AssociatedPrimeSet<R>,
) -> Result<(), AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    value.source.verify()?;
    let expected_module = cyclic_quotient_module(&value.source)?;
    if value.resolution.module() != &expected_module {
        return Err(AssociatedPrimeError::Invalid(
            "the retained resolution does not resolve the canonical cyclic module R/I".to_owned(),
        ));
    }
    value.resolution.verify()?;

    let first_codimension = quotient_codimension(&value.source)?;
    let ambient_dimension = value.source.ring().variables().len();
    let expected_stage_count = ambient_dimension
        .checked_add(1)
        .and_then(|bound| bound.checked_sub(first_codimension))
        .unwrap_or(0);
    if value.stages.len() != expected_stage_count {
        return Err(AssociatedPrimeError::Invalid(
            "the retained Ext stages do not cover codim(I) through dim(R)".to_owned(),
        ));
    }

    let order = ModuleMonomialOrder::TermOverPosition(value.source.ring().term_order());
    let coefficient_module =
        PresentedModule::free(FreeModule::new(value.source.ring().clone(), 1))?;
    let mut expected_certificates = Vec::new();
    for (offset, stage) in value.stages.iter().enumerate() {
        let codimension = first_codimension.checked_add(offset).ok_or_else(|| {
            AssociatedPrimeError::Invalid("Ext-stage codimension overflowed".to_owned())
        })?;
        if stage.codimension != codimension {
            return Err(AssociatedPrimeError::Invalid(
                "Ext stages are not in consecutive codimension order".to_owned(),
            ));
        }
        let expected = stage_from_ext(
            &value.resolution,
            &coefficient_module,
            codimension,
            order.clone(),
        )?;
        if stage != &expected {
            return Err(AssociatedPrimeError::Invalid(format!(
                "the retained Ext/annihilator evidence changed in codimension {codimension}"
            )));
        }
        expected_certificates.extend(stage.associated_primes.iter().cloned());
    }
    let expected_certificates = sorted_prime_certificates(expected_certificates)?;
    if value.prime_certificates != expected_certificates {
        return Err(AssociatedPrimeError::Invalid(
            "the associated-prime set is not the union of the height-filtered Ext stages"
                .to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Compute all associated primes from verified EHV Ext-annihilator evidence.
    pub fn associated_primes(
        &self,
    ) -> Result<AssociatedPrimeSet<RationalField>, AssociatedPrimeError> {
        construct_associated_primes(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Compute all associated primes from verified EHV Ext-annihilator evidence.
    pub fn associated_primes(
        &self,
    ) -> Result<AssociatedPrimeSet<FiniteField>, AssociatedPrimeError> {
        construct_associated_primes(self)
    }
}

impl AssociatedPrimeExtStage<RationalField> {
    /// Independently replay this rational Ext-stage annihilator evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_ext_stage(self)
    }

    /// Deserialize and exactly reverify one rational Ext stage.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl AssociatedPrimeExtStage<FiniteField> {
    /// Independently replay this finite-field Ext-stage annihilator evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_ext_stage(self)
    }

    /// Deserialize and exactly reverify one finite-field Ext stage.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> AssociatedPrimeExtStage<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the Ext presentation, annihilator, and height-filtered primes.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for AssociatedPrimeExtStage<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AssociatedPrimeExtStage", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("codimension", &self.codimension)?;
        state.serialize_field("ext_module", &self.ext_module)?;
        state.serialize_field("annihilator", &self.annihilator)?;
        state.serialize_field("minimal_primes", &self.minimal_primes)?;
        state.serialize_field("associated_primes", &self.associated_primes)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, MinimalPrimeDecomposition<R>: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>"
    )
)]
struct AssociatedPrimeExtStageWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    codimension: usize,
    ext_module: PresentedModule<PolynomialRing<R>>,
    annihilator: Ideal<PolynomialRing<R>>,
    minimal_primes: Option<MinimalPrimeDecomposition<R>>,
    associated_primes: Vec<PolynomialPrimeCertificate<R>>,
}

fn deserialize_associated_prime_ext_stage<'de, R, D>(
    deserializer: D,
) -> Result<AssociatedPrimeExtStage<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    MinimalPrimeDecomposition<R>: Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = AssociatedPrimeExtStageWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/AssociatedPrimeExtStage@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/AssociatedPrimeExtStage@1", wire.type_id
        )));
    }
    let result = AssociatedPrimeExtStage {
        codimension: wire.codimension,
        ext_module: wire.ext_module,
        annihilator: wire.annihilator,
        minimal_primes: wire.minimal_primes,
        associated_primes: wire.associated_primes,
    };
    verify_ext_stage(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for AssociatedPrimeExtStage<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_associated_prime_ext_stage(deserializer)
    }
}

impl<'de> Deserialize<'de> for AssociatedPrimeExtStage<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_associated_prime_ext_stage(deserializer)
    }
}

impl AssociatedPrimeSet<RationalField> {
    /// Independently replay the rational EHV associated-prime evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_associated_primes(self)
    }

    /// Deserialize and exactly reverify rational associated-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl AssociatedPrimeSet<FiniteField> {
    /// Independently replay the finite-field EHV associated-prime evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_associated_primes(self)
    }

    /// Deserialize and exactly reverify finite-field associated-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> AssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the source, common resolution, all Ext stages, and prime evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for AssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AssociatedPrimeSet", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("resolution", &self.resolution)?;
        state.serialize_field("stages", &self.stages)?;
        state.serialize_field("prime_certificates", &self.prime_certificates)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, AssociatedPrimeExtStage<R>: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>"
    )
)]
struct AssociatedPrimeSetWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    resolution: FreeResolution<PolynomialRing<R>>,
    stages: Vec<AssociatedPrimeExtStage<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

fn deserialize_associated_prime_set<'de, R, D>(
    deserializer: D,
) -> Result<AssociatedPrimeSet<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    AssociatedPrimeExtStage<R>: Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = AssociatedPrimeSetWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/AssociatedPrimeSet@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/AssociatedPrimeSet@1", wire.type_id
        )));
    }
    let result = AssociatedPrimeSet {
        source: wire.source,
        resolution: wire.resolution,
        stages: wire.stages,
        prime_certificates: wire.prime_certificates,
    };
    verify_associated_primes(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for AssociatedPrimeSet<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_associated_prime_set(deserializer)
    }
}

impl<'de> Deserialize<'de> for AssociatedPrimeSet<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_associated_prime_set(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::{Ring, Q};

    fn embedded_source() -> Ideal<PolynomialRing<RationalField>> {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let z = variables[2].clone();
        let linear = (&x + &y).unwrap();
        ring.ideal([linear.pow(2).unwrap(), (&linear * &z).unwrap()])
            .unwrap()
    }

    #[test]
    fn ehv_finds_nonmonomial_embedded_associated_prime() {
        let source = embedded_source();
        let ring = source.ring();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let z = variables[2].clone();
        let linear = (&x + &y).unwrap();
        let expected = [
            ring.ideal([linear.clone()]).unwrap(),
            ring.ideal([linear, z]).unwrap(),
        ];

        let result = source.associated_primes().unwrap();
        result.verify().unwrap();
        assert_eq!(result.stages().len(), 3);
        assert_eq!(
            result
                .stages()
                .iter()
                .map(AssociatedPrimeExtStage::codimension)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(result.primes(), expected.iter().collect::<Vec<_>>());
        assert_eq!(result.stages()[0].associated_primes().len(), 1);
        assert_eq!(result.stages()[1].associated_primes().len(), 1);
        assert!(result.stages()[2].associated_primes().is_empty());

        let restored =
            AssociatedPrimeSet::<RationalField>::from_json(&result.to_json().unwrap()).unwrap();
        assert_eq!(restored, result);
    }

    #[test]
    fn associated_prime_json_rejects_annihilator_tampering() {
        let result = embedded_source().associated_primes().unwrap();
        let mut payload = serde_json::to_value(result).unwrap();
        payload["stages"][0]["annihilator"] = payload["source"].clone();
        assert!(serde_json::from_value::<AssociatedPrimeSet<RationalField>>(payload).is_err());
    }

    #[test]
    fn zero_and_unit_ideals_have_complete_edge_case_evidence() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let zero = ring.ideal([]).unwrap().associated_primes().unwrap();
        assert_eq!(zero.stages().len(), 3);
        assert_eq!(zero.prime_certificates().len(), 1);
        assert!(zero.prime_certificates()[0].ideal().generators().is_empty());
        zero.verify().unwrap();

        let unit = ring
            .ideal([ring.one()])
            .unwrap()
            .associated_primes()
            .unwrap();
        assert!(unit.stages().is_empty());
        assert!(unit.prime_certificates().is_empty());
        unit.verify().unwrap();
    }
}
