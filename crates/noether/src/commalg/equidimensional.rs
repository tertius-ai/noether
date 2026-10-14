// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified equidimensional decomposition of polynomial-ideal radicals.

use core::fmt;
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription};

use super::minimal_primes::{
    construct_minimal_primes, sorted_prime_certificates, verify_minimal_primes,
    verify_polynomial_prime_certificate, MinimalPrimeCoefficientField,
};
use super::{
    DecompositionError, Ideal, MinimalPrimeDecomposition, PolynomialPrimeCertificate,
    PolynomialRing, RadicalIdeal,
};

static EQUIDIMENSIONAL_RADICAL_COMPONENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/EquidimensionalRadicalComponent@1",
    rust_path: "noether::commalg::EquidimensionalRadicalComponent",
    python_path: Some("noether.commalg.EquidimensionalRadicalComponent"),
};

static EQUIDIMENSIONAL_RADICAL_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/EquidimensionalRadicalDecomposition@1",
    rust_path: "noether::commalg::EquidimensionalRadicalDecomposition",
    python_path: Some("noether.commalg.EquidimensionalRadicalDecomposition"),
};

/// One verified equidimensional radical ideal, represented by all of its
/// minimal primes.
#[derive(Clone, PartialEq, Eq)]
pub struct EquidimensionalRadicalComponent<R: Field> {
    dimension: i64,
    ideal: Ideal<PolynomialRing<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

impl<R: Field> EquidimensionalRadicalComponent<R> {
    /// Common Krull dimension of every minimal prime in this component.
    pub const fn dimension(&self) -> i64 {
        self.dimension
    }

    /// Intersection of the component's minimal primes.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.ideal
    }

    /// Canonically ordered exact primality evidence for this component.
    pub fn prime_certificates(&self) -> &[PolynomialPrimeCertificate<R>] {
        &self.prime_certificates
    }

    /// Exact theorem used to certify equidimensionality.
    pub const fn theorem(&self) -> &'static str {
        "a radical ideal is equidimensional when all of its minimal primes have the same dimension"
    }
}

impl<R: Field> fmt::Debug for EquidimensionalRadicalComponent<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EquidimensionalRadicalComponent")
            .field("dimension", &self.dimension)
            .field("minimal_primes", &self.prime_certificates.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for EquidimensionalRadicalComponent<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "EquidimensionalRadicalComponent(dimension={}, minimal_primes={})",
            self.dimension,
            self.prime_certificates.len()
        )
    }
}

impl<R> SemanticType for EquidimensionalRadicalComponent<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/EquidimensionalRadicalComponent@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &EQUIDIMENSIONAL_RADICAL_COMPONENT_DESCRIPTION
    }
}

/// Verified decomposition of a polynomial-ideal radical by component
/// dimension.
///
/// This result deliberately decomposes the radical, not the source ideal with
/// its primary multiplicities. The latter belongs to primary decomposition.
#[derive(Clone, PartialEq, Eq)]
pub struct EquidimensionalRadicalDecomposition<R: Field> {
    minimal_primes: MinimalPrimeDecomposition<R>,
    components: Vec<EquidimensionalRadicalComponent<R>>,
}

impl<R: Field> EquidimensionalRadicalDecomposition<R> {
    /// Original ideal before passage to its radical.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        self.minimal_primes.source()
    }

    /// Verified radical reconstructed by the equidimensional components.
    pub fn radical(&self) -> &RadicalIdeal<R> {
        self.minimal_primes.radical()
    }

    /// Complete minimal-prime evidence underlying the grouping.
    pub fn minimal_primes(&self) -> &MinimalPrimeDecomposition<R> {
        &self.minimal_primes
    }

    /// Components in strictly decreasing dimension order.
    pub fn components(&self) -> &[EquidimensionalRadicalComponent<R>] {
        &self.components
    }

    /// Exact completeness theorem for the grouping.
    pub const fn theorem(&self) -> &'static str {
        "grouping all minimal primes by dimension and intersecting within each group reconstructs the radical"
    }
}

impl<R: Field> fmt::Debug for EquidimensionalRadicalDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EquidimensionalRadicalDecomposition")
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for EquidimensionalRadicalDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "EquidimensionalRadicalDecomposition(components={})",
            self.components.len()
        )
    }
}

impl<R> SemanticType for EquidimensionalRadicalDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/EquidimensionalRadicalDecomposition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &EQUIDIMENSIONAL_RADICAL_DECOMPOSITION_DESCRIPTION
    }
}

fn intersect_certified_primes<R: Field>(
    certificates: &[PolynomialPrimeCertificate<R>],
) -> Result<Ideal<PolynomialRing<R>>, DecompositionError> {
    let Some(first) = certificates.first() else {
        return Err(DecompositionError::Invalid(
            "an equidimensional component has no minimal primes".to_owned(),
        ));
    };
    let mut intersection = first.ideal().clone();
    for certificate in &certificates[1..] {
        intersection = intersection.intersection(certificate.ideal())?;
    }
    Ok(intersection)
}

fn construct_component<R>(
    dimension: i64,
    certificates: Vec<PolynomialPrimeCertificate<R>>,
) -> Result<EquidimensionalRadicalComponent<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let prime_certificates = sorted_prime_certificates(certificates)?;
    let ideal = intersect_certified_primes(&prime_certificates)?;
    let component = EquidimensionalRadicalComponent {
        dimension,
        ideal,
        prime_certificates,
    };
    verify_component(&component)?;
    Ok(component)
}

fn verify_component<R>(
    component: &EquidimensionalRadicalComponent<R>,
) -> Result<(), DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    component.ideal.verify()?;
    if component.dimension < 0
        || sorted_prime_certificates(component.prime_certificates.clone())?
            != component.prime_certificates
    {
        return Err(DecompositionError::Invalid(
            "an equidimensional component has noncanonical evidence".to_owned(),
        ));
    }
    for certificate in &component.prime_certificates {
        verify_polynomial_prime_certificate(certificate)?;
        if certificate.ideal().ring() != component.ideal.ring()
            || certificate.ideal().dimension()? != component.dimension
        {
            return Err(DecompositionError::Invalid(
                "an equidimensional component contains a prime of the wrong dimension".to_owned(),
            ));
        }
    }
    if intersect_certified_primes(&component.prime_certificates)? != component.ideal
        || component.ideal.dimension()? != component.dimension
    {
        return Err(DecompositionError::Invalid(
            "an equidimensional component does not reconstruct its recorded ideal".to_owned(),
        ));
    }
    Ok(())
}

fn construct_equidimensional_radical_decomposition<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<EquidimensionalRadicalDecomposition<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let minimal_primes = construct_minimal_primes(source)?;
    let mut groups = BTreeMap::<i64, Vec<PolynomialPrimeCertificate<R>>>::new();
    for certificate in minimal_primes.prime_certificates() {
        groups
            .entry(certificate.ideal().dimension()?)
            .or_default()
            .push(certificate.clone());
    }
    let components = groups
        .into_iter()
        .rev()
        .map(|(dimension, certificates)| construct_component(dimension, certificates))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EquidimensionalRadicalDecomposition {
        minimal_primes,
        components,
    })
}

fn verify_equidimensional_radical_decomposition<R>(
    value: &EquidimensionalRadicalDecomposition<R>,
) -> Result<(), DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    verify_minimal_primes(&value.minimal_primes)?;
    if value
        .components
        .windows(2)
        .any(|pair| pair[0].dimension <= pair[1].dimension)
    {
        return Err(DecompositionError::Invalid(
            "equidimensional components are not in decreasing dimension order".to_owned(),
        ));
    }
    let mut certificates = Vec::new();
    for component in &value.components {
        verify_component(component)?;
        certificates.extend(component.prime_certificates.iter().cloned());
    }
    if sorted_prime_certificates(certificates)? != value.minimal_primes.prime_certificates() {
        return Err(DecompositionError::Invalid(
            "equidimensional components do not partition the minimal primes".to_owned(),
        ));
    }
    let reconstructed = if let Some(first) = value.components.first() {
        let mut intersection = first.ideal.clone();
        for component in &value.components[1..] {
            intersection = intersection.intersection(&component.ideal)?;
        }
        intersection
    } else {
        value.source().ring().ideal([value.source().ring().one()])?
    };
    if &reconstructed != value.radical().ideal() {
        return Err(DecompositionError::Invalid(
            "equidimensional components do not reconstruct the radical".to_owned(),
        ));
    }
    if &construct_equidimensional_radical_decomposition(value.source())? != value {
        return Err(DecompositionError::Invalid(
            "recomputation changed the equidimensional radical decomposition".to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Decompose the radical into its equidimensional rational components.
    pub fn equidimensional_radical_decomposition(
        &self,
    ) -> Result<EquidimensionalRadicalDecomposition<RationalField>, DecompositionError> {
        construct_equidimensional_radical_decomposition(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Decompose the radical into its equidimensional finite-field components.
    pub fn equidimensional_radical_decomposition(
        &self,
    ) -> Result<EquidimensionalRadicalDecomposition<FiniteField>, DecompositionError> {
        construct_equidimensional_radical_decomposition(self)
    }
}

macro_rules! concrete_methods {
    ($coefficient:ty) => {
        impl EquidimensionalRadicalComponent<$coefficient> {
            /// Independently recheck the equidimensional component evidence.
            pub fn verify(&self) -> Result<(), DecompositionError> {
                verify_component(self)
            }

            /// Deserialize and exactly reverify equidimensional component evidence.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl EquidimensionalRadicalDecomposition<$coefficient> {
            /// Independently recheck the grouping and radical reconstruction.
            pub fn verify(&self) -> Result<(), DecompositionError> {
                verify_equidimensional_radical_decomposition(self)
            }

            /// Deserialize and exactly reverify the complete decomposition.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

concrete_methods!(RationalField);
concrete_methods!(FiniteField);

impl<R> EquidimensionalRadicalComponent<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the component and complete prime evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> EquidimensionalRadicalDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the minimal-prime evidence and canonical grouping.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for EquidimensionalRadicalComponent<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("EquidimensionalRadicalComponent", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("dimension", &self.dimension)?;
        state.serialize_field("ideal", &self.ideal)?;
        state.serialize_field("prime_certificates", &self.prime_certificates)?;
        state.end()
    }
}

impl<R> Serialize for EquidimensionalRadicalDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("EquidimensionalRadicalDecomposition", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("minimal_primes", &self.minimal_primes)?;
        state.serialize_field("components", &self.components)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>"
    )
)]
struct EquidimensionalRadicalComponentWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    dimension: i64,
    ideal: Ideal<PolynomialRing<R>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, MinimalPrimeDecomposition<R>: Deserialize<'de>, EquidimensionalRadicalComponent<R>: Deserialize<'de>"
    )
)]
struct EquidimensionalRadicalDecompositionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    minimal_primes: MinimalPrimeDecomposition<R>,
    components: Vec<EquidimensionalRadicalComponent<R>>,
}

fn deserialize_component<'de, R, D>(
    deserializer: D,
) -> Result<EquidimensionalRadicalComponent<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = EquidimensionalRadicalComponentWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/EquidimensionalRadicalComponent@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/EquidimensionalRadicalComponent@1", wire.type_id
        )));
    }
    let result = EquidimensionalRadicalComponent {
        dimension: wire.dimension,
        ideal: wire.ideal,
        prime_certificates: wire.prime_certificates,
    };
    verify_component(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

fn deserialize_decomposition<'de, R, D>(
    deserializer: D,
) -> Result<EquidimensionalRadicalDecomposition<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    MinimalPrimeDecomposition<R>: Deserialize<'de>,
    EquidimensionalRadicalComponent<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = EquidimensionalRadicalDecompositionWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/EquidimensionalRadicalDecomposition@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/EquidimensionalRadicalDecomposition@1", wire.type_id
        )));
    }
    let result = EquidimensionalRadicalDecomposition {
        minimal_primes: wire.minimal_primes,
        components: wire.components,
    };
    verify_equidimensional_radical_decomposition(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

macro_rules! concrete_deserialize {
    ($coefficient:ty) => {
        impl<'de> Deserialize<'de> for EquidimensionalRadicalComponent<$coefficient> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_component(deserializer)
            }
        }

        impl<'de> Deserialize<'de> for EquidimensionalRadicalDecomposition<$coefficient> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_decomposition(deserializer)
            }
        }
    };
}

concrete_deserialize!(RationalField);
concrete_deserialize!(FiniteField);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::numthy::FiniteField;
    use crate::Q;

    #[test]
    fn mixed_dimensional_monomial_radical_splits_by_dimension() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(3).unwrap() * &variables[1]).unwrap(),
                (&variables[0] * &variables[2].pow(4).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.equidimensional_radical_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(
            decomposition
                .components()
                .iter()
                .map(EquidimensionalRadicalComponent::dimension)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(decomposition.components()[0].prime_certificates().len(), 1);
        assert_eq!(decomposition.components()[1].prime_certificates().len(), 1);
        assert_eq!(
            EquidimensionalRadicalDecomposition::<RationalField>::from_json(
                &decomposition.to_json().unwrap()
            )
            .unwrap(),
            decomposition
        );
    }

    #[test]
    fn equidimensional_hypersurface_has_one_component() {
        let field = FiniteField::prime(5_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([(&(&variables[0] + &variables[1]).unwrap()
                * &(&variables[0] - &variables[2]).unwrap())
                .unwrap()])
            .unwrap();

        let decomposition = source.equidimensional_radical_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 1);
        assert_eq!(decomposition.components()[0].dimension(), 2);
        assert_eq!(decomposition.components()[0].prime_certificates().len(), 2);
    }

    #[test]
    fn unit_ideal_has_no_equidimensional_components() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let source = ring.ideal([ring.one()]).unwrap();

        let decomposition = source.equidimensional_radical_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert!(decomposition.components().is_empty());
    }
}
