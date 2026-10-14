// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Localizations of polynomial rings at certified prime ideals.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CommutativeRing, ContextMismatchError, IntegralDomain, NoetherError, Ring, SemanticType,
    TypeDescription, UnitError, UnitInversion,
};

use super::{
    GroebnerError, Polynomial, PolynomialError, PolynomialPrimeCertificate, PolynomialRing,
};

const PRIME_LOCALIZATION_TYPE_ID: &str = "noether/commalg/PolynomialPrimeLocalization@1";
const PRIME_LOCAL_ELEMENT_TYPE_ID: &str = "noether/commalg/PolynomialPrimeLocalElement@1";

static PRIME_LOCALIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialPrimeLocalization@1",
    rust_path: "noether::commalg::PolynomialPrimeLocalization",
    python_path: Some("noether.commalg.PolynomialPrimeLocalization"),
};

static PRIME_LOCAL_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialPrimeLocalElement@1",
    rust_path: "noether::commalg::PolynomialPrimeLocalElement",
    python_path: Some("noether.commalg.PolynomialPrimeLocalElement"),
};

/// Failure to construct or compute in a polynomial prime localization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimeLocalizationError {
    /// A value belongs to another exact context.
    ContextMismatch(ContextMismatchError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Exact ideal membership failed.
    Groebner(GroebnerError),
    /// A proposed denominator belongs to the localization prime.
    DenominatorInPrime,
    /// A requested element belongs to the maximal ideal of the local ring.
    NonUnit,
    /// Retained canonical evidence failed to replay.
    InvalidEvidence(String),
}

impl fmt::Display for PrimeLocalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::DenominatorInPrime => formatter
                .write_str("a prime-localization denominator must lie outside the prime ideal"),
            Self::NonUnit => formatter.write_str("the prime-local element is not a unit"),
            Self::InvalidEvidence(detail) => {
                write!(
                    formatter,
                    "invalid polynomial prime-localization evidence: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for PrimeLocalizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::DenominatorInPrime | Self::NonUnit | Self::InvalidEvidence(_) => None,
        }
    }
}

impl From<ContextMismatchError> for PrimeLocalizationError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PolynomialError> for PrimeLocalizationError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<GroebnerError> for PrimeLocalizationError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

/// The local ring `k[x_1,...,x_n]_p` for one certified prime ideal `p`.
///
/// The complete primality certificate is context identity. Fractions are
/// admitted precisely when the denominator is outside `p`; consequently an
/// element is a unit precisely when its numerator is outside `p`.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialPrimeLocalization<R: crate::Field> {
    prime: PolynomialPrimeCertificate<R>,
}

impl<R: crate::Field> PolynomialPrimeLocalization<R> {
    /// Localize at the certified polynomial prime.
    pub fn new(prime: PolynomialPrimeCertificate<R>) -> Self {
        Self { prime }
    }

    /// Ambient polynomial ring.
    pub fn ambient_ring(&self) -> &PolynomialRing<R> {
        self.prime.ideal().ring()
    }

    /// Exact certificate for the localization prime.
    pub fn prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.prime
    }

    /// Embed an ambient polynomial.
    pub fn from_polynomial(
        &self,
        value: Polynomial<R>,
    ) -> Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError> {
        self.fraction(value, self.ambient_ring().one())
    }

    /// Construct and canonically reduce an admitted fraction.
    pub fn fraction(
        &self,
        numerator: Polynomial<R>,
        denominator: Polynomial<R>,
    ) -> Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError> {
        if !self.ambient_ring().contains(&numerator) || !self.ambient_ring().contains(&denominator)
        {
            return Err(self
                .context_error("fraction from another ambient polynomial context")
                .into());
        }
        if self.prime.ideal().contains(&denominator)? {
            return Err(PrimeLocalizationError::DenominatorInPrime);
        }
        if numerator.is_zero() {
            return Ok(self.zero());
        }
        let gcd = numerator.gcd(&denominator)?;
        let numerator = numerator.exact_div(&gcd)?;
        let denominator = denominator.exact_div(&gcd)?;
        let leading = denominator.leading_coefficient().ok_or_else(|| {
            PrimeLocalizationError::InvalidEvidence(
                "an admitted denominator became zero during normalization".to_owned(),
            )
        })?;
        let inverse = self
            .ambient_ring()
            .coefficient_ring()
            .inverse(&leading)
            .map_err(|_| {
                PrimeLocalizationError::InvalidEvidence(
                    "a denominator leading coefficient was not invertible".to_owned(),
                )
            })?;
        Ok(PolynomialPrimeLocalElement {
            ring: self.clone(),
            numerator: numerator.scalar_mul(&inverse)?,
            denominator: denominator.scalar_mul(&inverse)?,
        })
    }

    /// Additive identity.
    pub fn zero(&self) -> PolynomialPrimeLocalElement<R> {
        PolynomialPrimeLocalElement {
            ring: self.clone(),
            numerator: self.ambient_ring().zero(),
            denominator: self.ambient_ring().one(),
        }
    }

    /// Multiplicative identity.
    pub fn one(&self) -> PolynomialPrimeLocalElement<R> {
        PolynomialPrimeLocalElement {
            ring: self.clone(),
            numerator: self.ambient_ring().one(),
            denominator: self.ambient_ring().one(),
        }
    }

    /// Images of all ambient polynomial variables.
    pub fn generators(
        &self,
    ) -> Result<Vec<PolynomialPrimeLocalElement<R>>, PrimeLocalizationError> {
        self.ambient_ring()
            .generators()?
            .into_iter()
            .map(|value| self.from_polynomial(value))
            .collect()
    }

    fn require(
        &self,
        value: &PolynomialPrimeLocalElement<R>,
    ) -> Result<(), PrimeLocalizationError> {
        if self == &value.ring {
            Ok(())
        } else {
            Err(self
                .context_error("element from another polynomial prime localization")
                .into())
        }
    }

    fn context_error(&self, actual: &str) -> ContextMismatchError {
        ContextMismatchError {
            expected: PRIME_LOCALIZATION_TYPE_ID.to_owned(),
            actual: actual.to_owned(),
        }
    }

    fn checked_add(
        &self,
        left: &PolynomialPrimeLocalElement<R>,
        right: &PolynomialPrimeLocalElement<R>,
    ) -> Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError> {
        self.require(left)?;
        self.require(right)?;
        let left_term = self
            .ambient_ring()
            .mul(&left.numerator, &right.denominator)?;
        let right_term = self
            .ambient_ring()
            .mul(&right.numerator, &left.denominator)?;
        self.fraction(
            self.ambient_ring().add(&left_term, &right_term)?,
            self.ambient_ring()
                .mul(&left.denominator, &right.denominator)?,
        )
    }

    fn checked_mul(
        &self,
        left: &PolynomialPrimeLocalElement<R>,
        right: &PolynomialPrimeLocalElement<R>,
    ) -> Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError> {
        self.require(left)?;
        self.require(right)?;
        self.fraction(
            self.ambient_ring().mul(&left.numerator, &right.numerator)?,
            self.ambient_ring()
                .mul(&left.denominator, &right.denominator)?,
        )
    }

    fn checked_neg(
        &self,
        value: &PolynomialPrimeLocalElement<R>,
    ) -> Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError> {
        self.require(value)?;
        self.fraction(
            self.ambient_ring().neg(&value.numerator)?,
            value.denominator.clone(),
        )
    }

    /// Serialize the certified prime-local context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: crate::Field> fmt::Debug for PolynomialPrimeLocalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialPrimeLocalization")
            .field("prime_generators", &self.prime.ideal().generators().len())
            .finish()
    }
}

impl<R: crate::Field> fmt::Display for PolynomialPrimeLocalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PolynomialPrimeLocalization<{}>({} prime generators)",
            R::TYPE_ID,
            self.prime.ideal().generators().len()
        )
    }
}

impl<R> SemanticType for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = PRIME_LOCALIZATION_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.prime.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &PRIME_LOCALIZATION_DESCRIPTION
    }
}

impl<R> Ring for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    type Element = PolynomialPrimeLocalElement<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring
    }

    fn zero(&self) -> Self::Element {
        Self::zero(self)
    }

    fn one(&self) -> Self::Element {
        Self::one(self)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_add(a, b)
            .map_err(|error| self.context_error(&error.to_string()))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        let negative = self
            .checked_neg(b)
            .map_err(|error| self.context_error(&error.to_string()))?;
        self.checked_add(a, &negative)
            .map_err(|error| self.context_error(&error.to_string()))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_mul(a, b)
            .map_err(|error| self.context_error(&error.to_string()))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.checked_neg(value)
            .map_err(|error| self.context_error(&error.to_string()))
    }
}

impl<R> CommutativeRing for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
}

impl<R> IntegralDomain for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
}

impl<R> UnitInversion for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)
            .map_err(|error| UnitError::ContextMismatch(self.context_error(&error.to_string())))?;
        if self
            .prime
            .ideal()
            .contains(&value.numerator)
            .map_err(|error| UnitError::ContextMismatch(self.context_error(&error.to_string())))?
        {
            return Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            });
        }
        self.fraction(value.denominator.clone(), value.numerator.clone())
            .map_err(|error| UnitError::ContextMismatch(self.context_error(&error.to_string())))
    }
}

impl<R> Serialize for PolynomialPrimeLocalization<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialPrimeLocalization", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("prime", &self.prime)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "PolynomialPrimeCertificate<R>: Deserialize<'de>")
)]
struct PolynomialPrimeLocalizationWire<R: crate::Field> {
    #[serde(rename = "type")]
    type_id: String,
    prime: PolynomialPrimeCertificate<R>,
}

fn deserialize_prime_localization<'de, R, D>(
    deserializer: D,
) -> Result<PolynomialPrimeLocalization<R>, D::Error>
where
    R: crate::Field,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = PolynomialPrimeLocalizationWire::<R>::deserialize(deserializer)?;
    if wire.type_id != PRIME_LOCALIZATION_TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            PRIME_LOCALIZATION_TYPE_ID, wire.type_id
        )));
    }
    Ok(PolynomialPrimeLocalization::new(wire.prime))
}

impl<'de> Deserialize<'de> for PolynomialPrimeLocalization<crate::RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_prime_localization(deserializer)
    }
}

impl<'de> Deserialize<'de> for PolynomialPrimeLocalization<crate::numthy::FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_prime_localization(deserializer)
    }
}

impl PolynomialPrimeLocalization<crate::RationalField> {
    /// Deserialize and revalidate a rational prime-local context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PolynomialPrimeLocalization<crate::numthy::FiniteField> {
    /// Deserialize and revalidate a finite-field prime-local context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

/// A reduced fraction in a [`PolynomialPrimeLocalization`].
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialPrimeLocalElement<R: crate::Field> {
    ring: PolynomialPrimeLocalization<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

impl<R: crate::Field> PolynomialPrimeLocalElement<R> {
    /// Exact prime-local context.
    pub fn ring(&self) -> &PolynomialPrimeLocalization<R> {
        &self.ring
    }

    /// Canonical numerator.
    pub fn numerator(&self) -> &Polynomial<R> {
        &self.numerator
    }

    /// Canonical denominator outside the localization prime.
    pub fn denominator(&self) -> &Polynomial<R> {
        &self.denominator
    }

    /// Whether this fraction is zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Decide whether this fraction is a unit.
    pub fn is_unit(&self) -> Result<bool, PrimeLocalizationError> {
        Ok(!self.ring.prime.ideal().contains(&self.numerator)?)
    }

    /// Invert this value when its numerator lies outside the prime.
    pub fn inverse(&self) -> Result<Self, PrimeLocalizationError> {
        if !self.is_unit()? {
            return Err(PrimeLocalizationError::NonUnit);
        }
        self.ring
            .fraction(self.denominator.clone(), self.numerator.clone())
    }

    /// Serialize this canonical prime-local fraction.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: crate::Field> fmt::Debug for PolynomialPrimeLocalElement<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialPrimeLocalElement")
            .field("numerator", &self.numerator)
            .field("denominator", &self.denominator)
            .finish()
    }
}

impl<R: crate::Field> fmt::Display for PolynomialPrimeLocalElement<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            self.numerator.fmt(formatter)
        } else {
            write!(formatter, "({})/({})", self.numerator, self.denominator)
        }
    }
}

impl<R> SemanticType for PolynomialPrimeLocalElement<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = PRIME_LOCAL_ELEMENT_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        append_bytes(&mut output, &self.numerator.canonical_bytes()?);
        append_bytes(&mut output, &self.denominator.canonical_bytes()?);
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PRIME_LOCAL_ELEMENT_DESCRIPTION
    }
}

impl<R> Serialize for PolynomialPrimeLocalElement<R>
where
    R: crate::Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialPrimeLocalElement", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("numerator", &self.numerator)?;
        state.serialize_field("denominator", &self.denominator)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "PolynomialPrimeLocalization<R>: Deserialize<'de>, R: crate::Field + Deserialize<'de>, R::Element: Deserialize<'de>"
    )
)]
struct PolynomialPrimeLocalElementWire<R: crate::Field> {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialPrimeLocalization<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

fn deserialize_prime_local_element<'de, R, D>(
    deserializer: D,
) -> Result<PolynomialPrimeLocalElement<R>, D::Error>
where
    R: crate::Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    PolynomialPrimeLocalization<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = PolynomialPrimeLocalElementWire::<R>::deserialize(deserializer)?;
    if wire.type_id != PRIME_LOCAL_ELEMENT_TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            PRIME_LOCAL_ELEMENT_TYPE_ID, wire.type_id
        )));
    }
    let numerator = wire.numerator;
    let denominator = wire.denominator;
    let rebuilt = wire
        .ring
        .fraction(numerator.clone(), denominator.clone())
        .map_err(serde::de::Error::custom)?;
    if rebuilt.numerator != numerator || rebuilt.denominator != denominator {
        return Err(serde::de::Error::custom(
            "prime-local numerator and denominator were not canonical",
        ));
    }
    Ok(rebuilt)
}

impl<'de> Deserialize<'de> for PolynomialPrimeLocalElement<crate::RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_prime_local_element(deserializer)
    }
}

impl<'de> Deserialize<'de> for PolynomialPrimeLocalElement<crate::numthy::FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_prime_local_element(deserializer)
    }
}

impl PolynomialPrimeLocalElement<crate::RationalField> {
    /// Deserialize and revalidate a rational prime-local fraction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PolynomialPrimeLocalElement<crate::numthy::FiniteField> {
    /// Deserialize and revalidate a finite-field prime-local fraction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: crate::Field> Add for &PolynomialPrimeLocalElement<R> {
    type Output = Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError>;

    fn add(self, rhs: Self) -> Self::Output {
        self.ring.checked_add(self, rhs)
    }
}

impl<R: crate::Field> Sub for &PolynomialPrimeLocalElement<R> {
    type Output = Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)?
    }
}

impl<R: crate::Field> Mul for &PolynomialPrimeLocalElement<R> {
    type Output = Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.ring.checked_mul(self, rhs)
    }
}

impl<R: crate::Field> Neg for &PolynomialPrimeLocalElement<R> {
    type Output = Result<PolynomialPrimeLocalElement<R>, PrimeLocalizationError>;

    fn neg(self) -> Self::Output {
        self.ring.checked_neg(self)
    }
}

fn append_bytes(output: &mut Vec<u8>, component: &[u8]) {
    output.extend_from_slice(&(component.len() as u64).to_le_bytes());
    output.extend_from_slice(component);
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialRing};
    use crate::{Ring, Q};

    use super::*;

    #[test]
    fn coordinate_prime_localization_inverts_exactly_the_complement() {
        let ambient = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let [x, y] = ambient.generators().unwrap().try_into().unwrap();
        let prime_ideal = ambient.ideal([x.clone()]).unwrap();
        let decomposition = prime_ideal.minimal_primes().unwrap();
        let prime = decomposition.prime_certificates()[0].clone();
        let local = PolynomialPrimeLocalization::new(prime);

        let inverse_y = local.fraction(ambient.one(), y.clone()).unwrap();
        assert!(local.from_polynomial(y).unwrap().is_unit().unwrap());
        assert_eq!(
            (&inverse_y
                * &local
                    .from_polynomial(ambient.generator(1).unwrap())
                    .unwrap())
                .unwrap(),
            local.one()
        );
        assert!(matches!(
            local.fraction(ambient.one(), x),
            Err(PrimeLocalizationError::DenominatorInPrime)
        ));
        assert_eq!(
            PolynomialPrimeLocalization::<crate::RationalField>::from_json(
                &local.to_json().unwrap()
            )
            .unwrap(),
            local
        );
        assert_eq!(
            PolynomialPrimeLocalElement::<crate::RationalField>::from_json(
                &inverse_y.to_json().unwrap()
            )
            .unwrap(),
            inverse_y
        );
    }
}
