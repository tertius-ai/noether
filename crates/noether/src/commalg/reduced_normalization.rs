// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Componentwise normalization of reduced affine algebras.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    ContextMismatchError, Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription,
    VerificationError,
};

use super::{
    affine_normalization::construct_affine_normalization, AffineNormalization,
    AffineNormalizationError, AffineNormalizationMap, DecompositionError, GroebnerError, Ideal,
    MinimalPrimeDecomposition, Polynomial, PolynomialError, PolynomialHomomorphismError,
    PolynomialRing, QuotientError, QuotientRing, QuotientRingMapError,
};
use crate::numthy::FiniteField;

static REDUCED_AFFINE_NORMALIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ReducedAffineNormalization@1",
    rust_path: "noether::commalg::ReducedAffineNormalization",
    python_path: Some("noether.commalg.ReducedAffineNormalization"),
};

/// Failure to construct or verify componentwise normalization of a reduced algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReducedNormalizationError {
    /// The source ideal does not define a reduced algebra.
    NotReduced,
    /// The source is the zero algebra and has no nonempty component product.
    ZeroAlgebra,
    /// Minimal-prime discovery or verification failed.
    Decomposition(DecompositionError),
    /// One irreducible component could not be normalized by the native floor.
    Component(AffineNormalizationError),
    /// Canonical ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// An ambient polynomial map failed.
    Homomorphism(PolynomialHomomorphismError),
    /// A quotient algebra could not be constructed.
    Quotient(QuotientError),
    /// A component quotient map could not be constructed.
    QuotientMap(QuotientRingMapError),
    /// Exact context identity was violated.
    ContextMismatch(ContextMismatchError),
    /// A retained checked map failed verification.
    Verification(VerificationError),
    /// Retained evidence failed exact replay.
    InvalidEvidence(String),
}

impl fmt::Display for ReducedNormalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReduced => {
                formatter.write_str("componentwise affine normalization requires a reduced source")
            }
            Self::ZeroAlgebra => formatter.write_str(
                "componentwise affine normalization requires at least one irreducible component",
            ),
            Self::Decomposition(error) => error.fmt(formatter),
            Self::Component(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Homomorphism(error) => error.fmt(formatter),
            Self::Quotient(error) => error.fmt(formatter),
            Self::QuotientMap(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Verification(error) => error.fmt(formatter),
            Self::InvalidEvidence(reason) => {
                write!(
                    formatter,
                    "invalid reduced-normalization evidence: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for ReducedNormalizationError {}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for ReducedNormalizationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(DecompositionError, Decomposition);
error_conversion!(AffineNormalizationError, Component);
error_conversion!(GroebnerError, Groebner);
error_conversion!(PolynomialError, Polynomial);
error_conversion!(PolynomialHomomorphismError, Homomorphism);
error_conversion!(QuotientError, Quotient);
error_conversion!(QuotientRingMapError, QuotientMap);
error_conversion!(ContextMismatchError, ContextMismatch);
error_conversion!(VerificationError, Verification);

/// Normalization of a reduced affine algebra as a product of normal domains.
///
/// If `A = k[x]/I` has minimal primes `P_i`, the normalization is retained as
/// the finite product of the normalizations of `k[x]/P_i`. `component_maps`
/// are the projections of the single finite map `A -> product_i B_i`.
/// `target_conductor_factors` are the projections of the conductor ideal to
/// the factors `B_i`.
#[derive(Clone, PartialEq, Eq)]
pub struct ReducedAffineNormalization<R: Field + fmt::Display> {
    source: Ideal<PolynomialRing<R>>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    minimal_primes: MinimalPrimeDecomposition<R>,
    components: Vec<AffineNormalization<R>>,
    component_maps: Vec<AffineNormalizationMap<R>>,
    source_conductor: Ideal<PolynomialRing<R>>,
    target_conductor_factors: Vec<Ideal<PolynomialRing<R>>>,
    nonnormal_locus: Ideal<PolynomialRing<R>>,
}

impl<R: Field + fmt::Display> ReducedAffineNormalization<R> {
    /// Radical defining ideal of the source algebra.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Source coordinate algebra `A`.
    pub fn source_algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.source_algebra
    }

    /// Complete certified minimal-prime decomposition of the source.
    pub fn minimal_primes(&self) -> &MinimalPrimeDecomposition<R> {
        &self.minimal_primes
    }

    /// Normalizations of the irreducible components, in minimal-prime order.
    pub fn components(&self) -> &[AffineNormalization<R>] {
        &self.components
    }

    /// Projections `A -> B_i` of the finite map into the product normalization.
    pub fn component_maps(&self) -> &[AffineNormalizationMap<R>] {
        &self.component_maps
    }

    /// Number of normal-domain factors in the product normalization.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Preimage in the source polynomial ring of the total conductor.
    pub fn source_conductor(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source_conductor
    }

    /// Projections of the conductor ideal to each normal target factor.
    pub fn target_conductor_factors(&self) -> &[Ideal<PolynomialRing<R>>] {
        &self.target_conductor_factors
    }

    /// Closed source subscheme outside which normalization is an isomorphism.
    pub fn nonnormal_locus(&self) -> &Ideal<PolynomialRing<R>> {
        &self.nonnormal_locus
    }

    /// Whether the source is already its product normalization.
    pub fn is_normal(&self) -> bool {
        self.source_conductor
            .generators()
            .first()
            .is_some_and(Polynomial::is_one)
    }

    /// Exact theorem certified by this value.
    pub const fn theorem(&self) -> &'static str {
        "the normalization of a reduced affine algebra is the product of the normalizations of its minimal-prime quotients"
    }

    /// Exact conductor identity certified by this value.
    pub const fn conductor_theorem(&self) -> &'static str {
        "on each component, the conductor consists of source elements carrying every retained finite-module generator into the ideal separating that component from all others; the total conductor is the intersection of these component conditions"
    }
}

impl<R: Field + fmt::Display> fmt::Debug for ReducedAffineNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReducedAffineNormalization")
            .field("components", &self.component_count())
            .field("is_normal", &self.is_normal())
            .finish_non_exhaustive()
    }
}

impl<R: Field + fmt::Display> fmt::Display for ReducedAffineNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ReducedAffineNormalization(components={}, is_normal={})",
            self.component_count(),
            self.is_normal()
        )
    }
}

fn construct_reduced_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    minimal_primes: MinimalPrimeDecomposition<R>,
) -> Result<ReducedAffineNormalization<R>, ReducedNormalizationError>
where
    R: Field + fmt::Display,
{
    if minimal_primes.source() != source {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "the minimal-prime decomposition belongs to another ideal".to_owned(),
        ));
    }
    if minimal_primes.radical().ideal() != source {
        return Err(ReducedNormalizationError::NotReduced);
    }
    if minimal_primes.components().is_empty() {
        return Err(ReducedNormalizationError::ZeroAlgebra);
    }
    if minimal_primes.components().len() != minimal_primes.prime_certificates().len() {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "minimal-prime ideals and certificates have different lengths".to_owned(),
        ));
    }

    let source_algebra = source.ring().quotient(source)?;
    let mut components = Vec::with_capacity(minimal_primes.components().len());
    let mut component_maps = Vec::with_capacity(minimal_primes.components().len());
    let mut conductor_conditions = Vec::with_capacity(minimal_primes.components().len());

    for (index, (prime, certificate)) in minimal_primes
        .components()
        .iter()
        .zip(minimal_primes.prime_certificates())
        .enumerate()
    {
        let component = construct_affine_normalization(prime, certificate.clone())?;
        let component_map = source_algebra.induced_map(
            component.normal_algebra(),
            component.structure_map().base_map().clone(),
        )?;
        let separator = component_separator(source.ring(), minimal_primes.components(), index)?;
        let condition = super::affine_normalization::finite_fraction_conductor_preimage(
            prime,
            &separator,
            component.integral_equations(),
            component.fraction_numerators(),
            component.fraction_denominators(),
        )?;

        components.push(component);
        component_maps.push(component_map);
        conductor_conditions.push(condition);
    }

    let mut conditions = conductor_conditions.into_iter();
    let mut source_conductor = conditions
        .next()
        .ok_or(ReducedNormalizationError::ZeroAlgebra)?;
    for condition in conditions {
        source_conductor = source_conductor.intersection(&condition)?;
    }
    let target_conductor_factors = components
        .iter()
        .map(|component| {
            let target_ring = component.normal_algebra().base_ring();
            let target_generators = source_conductor
                .generators()
                .iter()
                .map(|generator| component.structure_map().base_map().apply(generator))
                .collect::<Result<Vec<_>, PolynomialHomomorphismError>>()?;
            Ok(target_ring
                .ideal(target_generators)?
                .sum(component.normal_algebra().defining_ideal())?)
        })
        .collect::<Result<Vec<_>, ReducedNormalizationError>>()?;

    Ok(ReducedAffineNormalization {
        source: source.clone(),
        source_algebra,
        minimal_primes,
        components,
        component_maps,
        source_conductor: source_conductor.clone(),
        target_conductor_factors,
        nonnormal_locus: source_conductor,
    })
}

fn component_separator<R: Field>(
    ring: &PolynomialRing<R>,
    components: &[Ideal<PolynomialRing<R>>],
    omitted: usize,
) -> Result<Ideal<PolynomialRing<R>>, ReducedNormalizationError> {
    let mut others = components
        .iter()
        .enumerate()
        .filter_map(|(index, ideal)| (index != omitted).then_some(ideal.clone()));
    let Some(mut separator) = others.next() else {
        return Ok(ring.ideal([ring.one()])?);
    };
    for ideal in others {
        separator = separator.intersection(&ideal)?;
    }
    Ok(separator)
}

fn verify_common<R>(value: &ReducedAffineNormalization<R>) -> Result<(), ReducedNormalizationError>
where
    R: Field + fmt::Display,
{
    if value.source_algebra.defining_ideal() != &value.source {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "the source quotient has the wrong defining ideal".to_owned(),
        ));
    }
    if value.components.len() != value.component_maps.len()
        || value.components.len() != value.target_conductor_factors.len()
    {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "component normalization data have different lengths".to_owned(),
        ));
    }
    for ((prime, component), map) in value
        .minimal_primes
        .components()
        .iter()
        .zip(&value.components)
        .zip(&value.component_maps)
    {
        if component.source() != prime
            || map.source() != &value.source_algebra
            || map.target() != component.normal_algebra()
        {
            return Err(ReducedNormalizationError::InvalidEvidence(
                "a component or component map has the wrong endpoint".to_owned(),
            ));
        }
        map.verify()?;
    }
    if value.nonnormal_locus != value.source_conductor {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "the nonnormal locus differs from the source conductor".to_owned(),
        ));
    }
    let recomputed = construct_reduced_normalization(&value.source, value.minimal_primes.clone())?;
    if &recomputed != value {
        return Err(ReducedNormalizationError::InvalidEvidence(
            "recomputation changed the product normalization or conductor".to_owned(),
        ));
    }
    Ok(())
}

macro_rules! concrete_methods {
    ($coefficient:ty) => {
        impl Ideal<PolynomialRing<$coefficient>> {
            /// Normalize a reduced affine algebra componentwise.
            pub fn reduced_normalization(
                &self,
            ) -> Result<ReducedAffineNormalization<$coefficient>, ReducedNormalizationError> {
                let minimal_primes = self.minimal_primes()?;
                minimal_primes.verify()?;
                let result = construct_reduced_normalization(self, minimal_primes)?;
                result.verify()?;
                Ok(result)
            }
        }

        impl ReducedAffineNormalization<$coefficient> {
            /// Replay reducedness, component normalizations, product maps, and conductor evidence.
            pub fn verify(&self) -> Result<(), ReducedNormalizationError> {
                self.minimal_primes.verify()?;
                for component in &self.components {
                    component.verify()?;
                }
                verify_common(self)
            }

            /// Deserialize and exactly reverify a reduced affine normalization.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

concrete_methods!(RationalField);
concrete_methods!(FiniteField);

impl<R> ReducedAffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    MinimalPrimeDecomposition<R>: Serialize,
    AffineNormalization<R>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    /// Serialize the complete product normalization and conductor evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> SemanticType for ReducedAffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    MinimalPrimeDecomposition<R>: Serialize,
    AffineNormalization<R>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/ReducedAffineNormalization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &REDUCED_AFFINE_NORMALIZATION_DESCRIPTION
    }
}

impl<R> Serialize for ReducedAffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    MinimalPrimeDecomposition<R>: Serialize,
    AffineNormalization<R>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ReducedAffineNormalization", 9)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("source_algebra", &self.source_algebra)?;
        state.serialize_field("minimal_primes", &self.minimal_primes)?;
        state.serialize_field("components", &self.components)?;
        state.serialize_field("component_maps", &self.component_maps)?;
        state.serialize_field("source_conductor", &self.source_conductor)?;
        state.serialize_field("target_conductor_factors", &self.target_conductor_factors)?;
        state.serialize_field("nonnormal_locus", &self.nonnormal_locus)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + fmt::Display + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, MinimalPrimeDecomposition<R>: Deserialize<'de>, AffineNormalization<R>: Deserialize<'de>, AffineNormalizationMap<R>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>"
    )
)]
struct ReducedAffineNormalizationWire<R: Field + fmt::Display> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    minimal_primes: MinimalPrimeDecomposition<R>,
    components: Vec<AffineNormalization<R>>,
    component_maps: Vec<AffineNormalizationMap<R>>,
    source_conductor: Ideal<PolynomialRing<R>>,
    target_conductor_factors: Vec<Ideal<PolynomialRing<R>>>,
    nonnormal_locus: Ideal<PolynomialRing<R>>,
}

fn deserialize_reduced_normalization<'de, R, D>(
    deserializer: D,
) -> Result<ReducedAffineNormalization<R>, D::Error>
where
    R: Field + fmt::Display + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    MinimalPrimeDecomposition<R>: Serialize + Deserialize<'de>,
    AffineNormalization<R>: Serialize + Deserialize<'de>,
    AffineNormalizationMap<R>: Serialize + Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = ReducedAffineNormalizationWire::<R>::deserialize(deserializer)?;
    if wire.type_id != ReducedAffineNormalization::<R>::TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            ReducedAffineNormalization::<R>::TYPE_ID,
            wire.type_id
        )));
    }
    Ok(ReducedAffineNormalization {
        source: wire.source,
        source_algebra: wire.source_algebra,
        minimal_primes: wire.minimal_primes,
        components: wire.components,
        component_maps: wire.component_maps,
        source_conductor: wire.source_conductor,
        target_conductor_factors: wire.target_conductor_factors,
        nonnormal_locus: wire.nonnormal_locus,
    })
}

impl<'de> Deserialize<'de> for ReducedAffineNormalization<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: ReducedAffineNormalization<RationalField> =
            deserialize_reduced_normalization(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ReducedAffineNormalization<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: ReducedAffineNormalization<FiniteField> =
            deserialize_reduced_normalization(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::Q;

    #[test]
    fn node_normalization_retains_branch_maps_and_gluing_conductor() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring.mul(&variables[0], &variables[1]).unwrap()])
            .unwrap();

        let normalization = source.reduced_normalization().unwrap();

        assert_eq!(normalization.component_count(), 2);
        assert!(normalization
            .components()
            .iter()
            .all(AffineNormalization::is_normal));
        assert_eq!(
            normalization.source_conductor(),
            &ring
                .ideal([variables[0].clone(), variables[1].clone()])
                .unwrap()
        );
        assert_eq!(
            normalization.nonnormal_locus(),
            normalization.source_conductor()
        );
        assert!(!normalization.is_normal());
        normalization.verify().unwrap();
        assert_eq!(
            ReducedAffineNormalization::<RationalField>::from_json(
                &normalization.to_json().unwrap()
            )
            .unwrap(),
            normalization
        );
    }

    #[test]
    fn component_and_intersection_conductors_both_contribute() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let cusp = ring
            .sub(&variables[1].pow(2).unwrap(), &variables[0].pow(3).unwrap())
            .unwrap();
        let source = ring
            .ideal([ring.mul(&variables[1], &cusp).unwrap()])
            .unwrap();

        let normalization = source.reduced_normalization().unwrap();

        assert_eq!(normalization.component_count(), 2);
        assert_eq!(
            normalization.source_conductor(),
            &ring
                .ideal([
                    variables[1].pow(2).unwrap(),
                    ring.mul(&variables[0], &variables[1]).unwrap(),
                    variables[0].pow(3).unwrap(),
                ])
                .unwrap()
        );
        assert_eq!(normalization.target_conductor_factors().len(), 2);
        normalization.verify().unwrap();
    }

    #[test]
    fn nonreduced_sources_fail_closed() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .mul(&variables[0].pow(2).unwrap(), &variables[1])
                .unwrap()])
            .unwrap();

        assert!(matches!(
            source.reduced_normalization(),
            Err(ReducedNormalizationError::NotReduced)
        ));
    }
}
