// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Associated primes of presented modules from verified Ext evidence.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    quotient_codimension, sorted_associated_prime_certificates, stage_from_ext,
    AssociatedPrimeError, AssociatedPrimeExtStage, Ideal, MinimalPrimeCoefficientField,
    PolynomialPrimeCertificate, PolynomialRing,
};
use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, SemanticType, TypeDescription};

use super::{FreeModule, FreeResolution, ModuleMonomialOrder, PresentedModule};

static MODULE_ASSOCIATED_PRIME_SET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleAssociatedPrimeSet@1",
    rust_path: "noether::homalg::ModuleAssociatedPrimeSet",
    python_path: Some("noether.homalg.ModuleAssociatedPrimeSet"),
};

/// Complete associated-prime evidence for one finitely presented module.
///
/// One verified free resolution of `M` is reused for every retained Ext stage.
/// In the regular polynomial ring `R`, the Eisenbud--Huneke--Vasconcelos
/// theorem identifies the height-`i` associated primes of `M` with the
/// height-`i` minimal primes of `ann Ext^i_R(M,R)`.
#[derive(Clone, PartialEq, Eq)]
pub struct ModuleAssociatedPrimeSet<R: Field> {
    source: PresentedModule<PolynomialRing<R>>,
    annihilator: Ideal<PolynomialRing<R>>,
    resolution: FreeResolution<PolynomialRing<R>>,
    stages: Vec<AssociatedPrimeExtStage<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

impl<R: Field> ModuleAssociatedPrimeSet<R> {
    /// Original retained presentation of the examined module.
    pub fn source(&self) -> &PresentedModule<PolynomialRing<R>> {
        &self.source
    }

    /// Exact annihilator defining the support and its first possible codimension.
    pub fn annihilator(&self) -> &Ideal<PolynomialRing<R>> {
        &self.annihilator
    }

    /// One complete verified resolution of the source module.
    pub fn resolution(&self) -> &FreeResolution<PolynomialRing<R>> {
        &self.resolution
    }

    /// Consecutive Ext stages from the support codimension through `dim(R)`.
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
        "the height-i associated primes of M are the height-i minimal primes of ann Ext^i_R(M,R)"
    }
}

impl<R: Field> fmt::Debug for ModuleAssociatedPrimeSet<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModuleAssociatedPrimeSet")
            .field("stages", &self.stages.len())
            .field("associated_primes", &self.prime_certificates.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for ModuleAssociatedPrimeSet<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ModuleAssociatedPrimeSet(stages={}, associated_primes={})",
            self.stages.len(),
            self.prime_certificates.len()
        )
    }
}

impl<R> SemanticType for ModuleAssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/ModuleAssociatedPrimeSet@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_ASSOCIATED_PRIME_SET_DESCRIPTION
    }
}

pub(super) fn construct_module_associated_primes<R>(
    source: &PresentedModule<PolynomialRing<R>>,
) -> Result<ModuleAssociatedPrimeSet<R>, AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ring = source.generator_module().ring();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    let annihilator = source.annihilator(order.clone())?;
    let first_codimension = quotient_codimension(&annihilator)?;
    let ambient_dimension = ring.variables().len();
    let resolution = source.free_resolution(order.clone())?;
    let coefficient_module = PresentedModule::free(FreeModule::new(ring.clone(), 1))?;
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
    Ok(ModuleAssociatedPrimeSet {
        source: source.clone(),
        annihilator,
        resolution,
        stages,
        prime_certificates,
    })
}

pub(super) fn verify_module_associated_primes<R>(
    value: &ModuleAssociatedPrimeSet<R>,
) -> Result<(), AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ring = value.source.generator_module().ring();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    let annihilator = value.source.annihilator(order.clone())?;
    if value.annihilator != annihilator {
        return Err(AssociatedPrimeError::Invalid(
            "the retained support ideal is not the source module's annihilator".to_owned(),
        ));
    }
    if value.resolution.module() != &value.source {
        return Err(AssociatedPrimeError::Invalid(
            "the retained resolution does not resolve the source module".to_owned(),
        ));
    }
    value.resolution.verify()?;

    let first_codimension = quotient_codimension(&annihilator)?;
    let ambient_dimension = ring.variables().len();
    let expected_stage_count = ambient_dimension
        .checked_add(1)
        .and_then(|bound| bound.checked_sub(first_codimension))
        .unwrap_or(0);
    if value.stages.len() != expected_stage_count {
        return Err(AssociatedPrimeError::Invalid(
            "the retained Ext stages do not cover codim(support M) through dim(R)".to_owned(),
        ));
    }

    let coefficient_module = PresentedModule::free(FreeModule::new(ring.clone(), 1))?;
    for (offset, stage) in value.stages.iter().enumerate() {
        let codimension = first_codimension.checked_add(offset).ok_or_else(|| {
            AssociatedPrimeError::Invalid("Ext-stage codimension overflowed".to_owned())
        })?;
        if stage.codimension() != codimension {
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
    }
    let expected_certificates = sorted_associated_prime_certificates(&value.stages)?;
    if value.prime_certificates != expected_certificates {
        return Err(AssociatedPrimeError::Invalid(
            "the associated-prime set is not the union of the height-filtered Ext stages"
                .to_owned(),
        ));
    }
    Ok(())
}

impl PresentedModule<PolynomialRing<RationalField>> {
    /// Compute all associated primes from verified EHV Ext-annihilator evidence.
    pub fn associated_primes(
        &self,
    ) -> Result<ModuleAssociatedPrimeSet<RationalField>, AssociatedPrimeError> {
        construct_module_associated_primes(self)
    }
}

impl PresentedModule<PolynomialRing<FiniteField>> {
    /// Compute all associated primes from verified EHV Ext-annihilator evidence.
    pub fn associated_primes(
        &self,
    ) -> Result<ModuleAssociatedPrimeSet<FiniteField>, AssociatedPrimeError> {
        construct_module_associated_primes(self)
    }
}

impl ModuleAssociatedPrimeSet<RationalField> {
    /// Independently replay rational module associated-prime evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_module_associated_primes(self)
    }

    /// Deserialize and exactly reverify rational module associated-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl ModuleAssociatedPrimeSet<FiniteField> {
    /// Independently replay finite-field module associated-prime evidence.
    pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
        verify_module_associated_primes(self)
    }

    /// Deserialize and exactly reverify finite-field module associated-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> ModuleAssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the source, support, common resolution, stages, and prime evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for ModuleAssociatedPrimeSet<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ModuleAssociatedPrimeSet", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("annihilator", &self.annihilator)?;
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
struct ModuleAssociatedPrimeSetWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: PresentedModule<PolynomialRing<R>>,
    annihilator: Ideal<PolynomialRing<R>>,
    resolution: FreeResolution<PolynomialRing<R>>,
    stages: Vec<AssociatedPrimeExtStage<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

fn deserialize_module_associated_prime_set<'de, R, D>(
    deserializer: D,
) -> Result<ModuleAssociatedPrimeSet<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    AssociatedPrimeExtStage<R>: Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = ModuleAssociatedPrimeSetWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/homalg/ModuleAssociatedPrimeSet@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/homalg/ModuleAssociatedPrimeSet@1", wire.type_id
        )));
    }
    let result = ModuleAssociatedPrimeSet {
        source: wire.source,
        annihilator: wire.annihilator,
        resolution: wire.resolution,
        stages: wire.stages,
        prime_certificates: wire.prime_certificates,
    };
    verify_module_associated_primes(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for ModuleAssociatedPrimeSet<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_module_associated_prime_set(deserializer)
    }
}

impl<'de> Deserialize<'de> for ModuleAssociatedPrimeSet<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_module_associated_prime_set(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::homalg::ModuleHomomorphism;
    use crate::linalg::Matrix;
    use crate::{Ring, Q};

    fn embedded_module() -> PresentedModule<PolynomialRing<RationalField>> {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let zero = ring.zero();
        let relations = FreeModule::new(ring.clone(), 3);
        let generators = FreeModule::new(ring.clone(), 2);
        PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                relations,
                generators,
                Matrix::from_rows(
                    ring,
                    [
                        [linear.clone(), zero.clone(), zero.clone()],
                        [zero, linear, variables[2].clone()],
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn ehv_finds_associated_primes_of_a_noncyclic_module() {
        let source = embedded_module();
        let ring = source.generator_module().ring();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let expected = [
            ring.ideal([linear.clone()]).unwrap(),
            ring.ideal([linear, variables[2].clone()]).unwrap(),
        ];

        let result = source.associated_primes().unwrap();

        assert_eq!(result.source(), &source);
        assert_eq!(result.annihilator(), &expected[0]);
        assert_eq!(result.primes(), expected.iter().collect::<Vec<_>>());
        assert_eq!(
            result
                .stages()
                .iter()
                .map(AssociatedPrimeExtStage::codimension)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        result.verify().unwrap();
        assert_eq!(
            ModuleAssociatedPrimeSet::<RationalField>::from_json(&result.to_json().unwrap())
                .unwrap(),
            result
        );
    }

    #[test]
    fn module_associated_prime_json_rejects_source_support_tampering() {
        let result = embedded_module().associated_primes().unwrap();
        let mut payload = serde_json::to_value(result).unwrap();
        payload["annihilator"] = payload["stages"][1]["annihilator"].clone();
        assert!(
            serde_json::from_value::<ModuleAssociatedPrimeSet<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn zero_and_free_modules_have_distinct_complete_evidence() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let zero = PresentedModule::free(FreeModule::new(ring.clone(), 0)).unwrap();
        let zero_result = zero.associated_primes().unwrap();
        assert!(zero_result.annihilator().generators()[0].is_one());
        assert!(zero_result.stages().is_empty());
        assert!(zero_result.prime_certificates().is_empty());
        zero_result.verify().unwrap();

        let free = PresentedModule::free(FreeModule::new(ring, 2)).unwrap();
        let free_result = free.associated_primes().unwrap();
        assert!(free_result.annihilator().generators().is_empty());
        assert_eq!(free_result.stages().len(), 3);
        assert_eq!(free_result.prime_certificates().len(), 1);
        assert!(free_result.prime_certificates()[0]
            .ideal()
            .generators()
            .is_empty());
        free_result.verify().unwrap();
    }
}
