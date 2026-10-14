// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Scheme-theoretic equidimensional hulls from verified Ext evidence.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription};

use super::associated_primes::{
    construct_associated_primes, quotient_codimension, verify_associated_primes,
};
use super::minimal_primes::MinimalPrimeCoefficientField;
use super::{
    AssociatedPrimeError, AssociatedPrimeSet, Ideal, MinimalPrimeDecomposition,
    PolynomialPrimeCertificate, PolynomialRing,
};

static EQUIDIMENSIONAL_HULL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/EquidimensionalHull@1",
    rust_path: "noether::commalg::EquidimensionalHull",
    python_path: Some("noether.commalg.EquidimensionalHull"),
};

/// Verified scheme-theoretic top-dimensional part of a polynomial ideal.
///
/// For a proper ideal `I` of codimension `c` in a polynomial ring `R`, the
/// retained ideal is `ann Ext^c_R(R/I,R)`. By the
/// Eisenbud--Huneke--Vasconcelos theorem, this is the intersection of the
/// primary components of `I` having maximal dimension. Nilpotent structure
/// and multiplicity on those components are retained.
#[derive(Clone, PartialEq, Eq)]
pub struct EquidimensionalHull<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    codimension: usize,
    ideal: Ideal<PolynomialRing<R>>,
    associated_primes: Box<AssociatedPrimeSet<R>>,
}

impl<R: Field> EquidimensionalHull<R> {
    /// Original polynomial ideal before lower-dimensional components are removed.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Codimension of the source and its top-dimensional hull.
    pub const fn codimension(&self) -> usize {
        self.codimension
    }

    /// Intersection of the source's primary components of maximal dimension.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.ideal
    }

    /// Complete EHV evidence from which the hull is extracted.
    pub fn associated_primes(&self) -> &AssociatedPrimeSet<R> {
        &self.associated_primes
    }

    /// Certified minimal primes of the hull, absent only for the unit ideal.
    pub fn minimal_primes(&self) -> Option<&MinimalPrimeDecomposition<R>> {
        self.associated_primes
            .stages()
            .first()
            .and_then(|stage| stage.minimal_primes())
    }

    /// Canonically ordered top-dimensional prime certificates.
    pub fn prime_certificates(&self) -> &[PolynomialPrimeCertificate<R>] {
        self.associated_primes
            .stages()
            .first()
            .map_or(&[], |stage| stage.associated_primes())
    }

    /// Exact theorem identifying the retained ideal.
    pub const fn theorem(&self) -> &'static str {
        "for c = codim(I), ann Ext^c_R(R/I,R) is the intersection of the primary components of I having maximal dimension"
    }
}

impl<R: Field> fmt::Debug for EquidimensionalHull<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EquidimensionalHull")
            .field("codimension", &self.codimension)
            .field("generators", &self.ideal.generators().len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for EquidimensionalHull<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "EquidimensionalHull(codimension={}, generators={})",
            self.codimension,
            self.ideal.generators().len()
        )
    }
}

impl<R> SemanticType for EquidimensionalHull<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/EquidimensionalHull@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &EQUIDIMENSIONAL_HULL_DESCRIPTION
    }
}

pub(super) fn construct_equidimensional_hull<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<EquidimensionalHull<R>, AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let codimension = quotient_codimension(source)?;
    let dimension = source.dimension()?;
    let associated_primes = construct_associated_primes(source)?;
    let ideal = if dimension < 0 {
        if !associated_primes.stages().is_empty() {
            return Err(AssociatedPrimeError::Invalid(
                "the unit ideal has nonempty Ext stages".to_owned(),
            ));
        }
        source.ring().ideal([source.ring().one()])?
    } else {
        let Some(stage) = associated_primes.stages().first() else {
            return Err(AssociatedPrimeError::Invalid(
                "a proper ideal has no codimension Ext stage".to_owned(),
            ));
        };
        if stage.codimension() != codimension || stage.minimal_primes().is_none() {
            return Err(AssociatedPrimeError::Invalid(
                "the first Ext annihilator does not have the source codimension".to_owned(),
            ));
        }
        stage.annihilator().clone()
    };
    Ok(EquidimensionalHull {
        source: source.clone(),
        codimension,
        ideal,
        associated_primes: Box::new(associated_primes),
    })
}

pub(super) fn verify_equidimensional_hull<R>(
    value: &EquidimensionalHull<R>,
) -> Result<(), AssociatedPrimeError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    value.source.verify()?;
    value.ideal.verify()?;
    verify_associated_primes(&value.associated_primes)?;
    if value.associated_primes.source() != &value.source
        || quotient_codimension(&value.source)? != value.codimension
    {
        return Err(AssociatedPrimeError::Invalid(
            "equidimensional-hull source or codimension evidence changed".to_owned(),
        ));
    }
    let source_dimension = value.source.dimension()?;
    if source_dimension < 0 {
        if !value.associated_primes.stages().is_empty()
            || value.ideal != value.source.ring().ideal([value.source.ring().one()])?
        {
            return Err(AssociatedPrimeError::Invalid(
                "the unit ideal has a nontrivial equidimensional hull".to_owned(),
            ));
        }
        return Ok(());
    }
    let Some(stage) = value.associated_primes.stages().first() else {
        return Err(AssociatedPrimeError::Invalid(
            "a proper equidimensional hull has no first Ext stage".to_owned(),
        ));
    };
    let Some(minimal_primes) = stage.minimal_primes() else {
        return Err(AssociatedPrimeError::Invalid(
            "the first Ext annihilator has no minimal-prime evidence".to_owned(),
        ));
    };
    if stage.codimension() != value.codimension
        || stage.annihilator() != &value.ideal
        || minimal_primes.source() != &value.ideal
        || value.ideal.dimension()? != source_dimension
        || stage.associated_primes() != minimal_primes.prime_certificates()
    {
        return Err(AssociatedPrimeError::Invalid(
            "the retained ideal is not the verified top-dimensional Ext annihilator".to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Compute the scheme-theoretic top-dimensional hull over the rationals.
    pub fn equidimensional_hull(
        &self,
    ) -> Result<EquidimensionalHull<RationalField>, AssociatedPrimeError> {
        construct_equidimensional_hull(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Compute the scheme-theoretic top-dimensional hull over a finite field.
    pub fn equidimensional_hull(
        &self,
    ) -> Result<EquidimensionalHull<FiniteField>, AssociatedPrimeError> {
        construct_equidimensional_hull(self)
    }
}

macro_rules! concrete_methods {
    ($coefficient:ty) => {
        impl EquidimensionalHull<$coefficient> {
            /// Independently replay the complete Ext-annihilator evidence.
            pub fn verify(&self) -> Result<(), AssociatedPrimeError> {
                verify_equidimensional_hull(self)
            }

            /// Deserialize and exactly reverify an equidimensional hull.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

concrete_methods!(RationalField);
concrete_methods!(FiniteField);

impl<R> EquidimensionalHull<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the source, hull, and complete Ext-annihilator evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for EquidimensionalHull<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("EquidimensionalHull", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("codimension", &self.codimension)?;
        state.serialize_field("ideal", &self.ideal)?;
        state.serialize_field("associated_primes", &self.associated_primes)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, AssociatedPrimeSet<R>: Deserialize<'de>"
    )
)]
struct EquidimensionalHullWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    codimension: usize,
    ideal: Ideal<PolynomialRing<R>>,
    associated_primes: Box<AssociatedPrimeSet<R>>,
}

fn deserialize_equidimensional_hull<'de, R, D>(
    deserializer: D,
) -> Result<EquidimensionalHull<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    AssociatedPrimeSet<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = EquidimensionalHullWire::<R>::deserialize(deserializer)?;
    if wire.type_id != EquidimensionalHull::<R>::TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            EquidimensionalHull::<R>::TYPE_ID,
            wire.type_id
        )));
    }
    let result = EquidimensionalHull {
        source: wire.source,
        codimension: wire.codimension,
        ideal: wire.ideal,
        associated_primes: wire.associated_primes,
    };
    verify_equidimensional_hull(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for EquidimensionalHull<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_equidimensional_hull(deserializer)
    }
}

impl<'de> Deserialize<'de> for EquidimensionalHull<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_equidimensional_hull(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::{Ring, Q};

    #[test]
    fn ext_annihilator_retains_the_nonradical_top_component() {
        let ring = PolynomialRing::new(Q, ["a", "b", "c"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let top = ring
            .ideal([variables[0].pow(2).unwrap(), variables[1].pow(2).unwrap()])
            .unwrap();
        let embedded = ring
            .ideal([
                variables[0].pow(3).unwrap(),
                variables[1].pow(3).unwrap(),
                variables[2].pow(3).unwrap(),
            ])
            .unwrap();
        let source = top.intersection(&embedded).unwrap();

        let hull = source.equidimensional_hull().unwrap();

        assert_eq!(hull.source(), &source);
        assert_eq!(hull.codimension(), 2);
        assert_eq!(hull.ideal(), &top);
        assert_eq!(hull.prime_certificates().len(), 1);
        assert_eq!(
            hull.prime_certificates()[0].ideal(),
            &ring
                .ideal([variables[0].clone(), variables[1].clone()])
                .unwrap()
        );
        assert_eq!(hull.minimal_primes().unwrap().source(), &top);
        hull.verify().unwrap();
        assert_eq!(
            EquidimensionalHull::<RationalField>::from_json(&hull.to_json().unwrap()).unwrap(),
            hull
        );

        let mut payload = serde_json::to_value(&hull).unwrap();
        payload["ideal"] = payload["source"].clone();
        assert!(serde_json::from_value::<EquidimensionalHull<RationalField>>(payload).is_err());
    }

    #[test]
    fn zero_and_unit_hulls_keep_distinct_exact_meanings() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let zero = ring.ideal([]).unwrap().equidimensional_hull().unwrap();
        assert_eq!(zero.codimension(), 0);
        assert!(zero.ideal().generators().is_empty());
        assert_eq!(zero.prime_certificates().len(), 1);
        zero.verify().unwrap();

        let unit_source = ring.ideal([ring.one()]).unwrap();
        let unit = unit_source.equidimensional_hull().unwrap();
        assert_eq!(unit.codimension(), 3);
        assert_eq!(unit.ideal(), &unit_source);
        assert!(unit.minimal_primes().is_none());
        assert!(unit.prime_certificates().is_empty());
        unit.verify().unwrap();
    }
}
