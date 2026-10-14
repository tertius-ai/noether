// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Context-carrying rational function fields.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CoefficientMap, CommutativeRing, ContextMismatchError, DivisionByZeroError, Field,
    IntegralDomain, InversionError, NoetherError, Ring, SemanticType, TypeDescription, UnitError,
    UnitInversion, VerificationError,
};

use super::{MonomialOrder, Polynomial, PolynomialContextError, PolynomialError, PolynomialRing};

static RATIONAL_FUNCTION_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalFunctionField@1",
    rust_path: "noether::commalg::RationalFunctionField",
    python_path: Some("noether.commalg.RationalFunctionField"),
};

static RATIONAL_FUNCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalFunction@1",
    rust_path: "noether::commalg::RationalFunction",
    python_path: Some("noether.commalg.RationalFunction"),
};

static RATIONAL_FUNCTION_FIELD_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalFunctionFieldMap@1",
    rust_path: "noether::commalg::RationalFunctionFieldMap",
    python_path: Some("noether.commalg.RationalFunctionFieldMap"),
};
const RATIONAL_FUNCTION_FIELD_MAP_TYPE_ID: &str = "noether/commalg/RationalFunctionFieldMap@1";

/// Failure to construct or operate on a rational function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalFunctionError {
    /// The underlying univariate polynomial context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Values belonged to unequal field or polynomial contexts.
    ContextMismatch(ContextMismatchError),
    /// A zero polynomial was supplied as a denominator.
    ZeroDenominator,
    /// Division by the zero rational function was attempted.
    DivisionByZero,
    /// Specialization made the denominator zero.
    Pole,
}

impl fmt::Display for RationalFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolynomialContext(error) => error.fmt(f),
            Self::Polynomial(error) => error.fmt(f),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::ZeroDenominator => f.write_str("rational function denominator must be nonzero"),
            Self::DivisionByZero => f.write_str("division by zero rational function"),
            Self::Pole => f.write_str("rational function is undefined at this specialization"),
        }
    }
}

impl std::error::Error for RationalFunctionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::ZeroDenominator | Self::DivisionByZero | Self::Pole => None,
        }
    }
}

impl From<PolynomialContextError> for RationalFunctionError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for RationalFunctionError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<ContextMismatchError> for RationalFunctionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to induce or apply a map between rational-function fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalFunctionFieldMapError {
    /// A declared endpoint or input value had the wrong context.
    ContextMismatch(ContextMismatchError),
    /// The underlying coefficient map rejected one coefficient.
    Coefficient(String),
    /// Reconstructing a mapped numerator or denominator failed.
    Polynomial(PolynomialError),
    /// Canonical rational-function reconstruction failed.
    RationalFunction(RationalFunctionError),
}

impl fmt::Display for RationalFunctionFieldMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Coefficient(detail) => {
                write!(
                    formatter,
                    "rational-function coefficient map failed: {detail}"
                )
            }
            Self::Polynomial(error) => error.fmt(formatter),
            Self::RationalFunction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RationalFunctionFieldMapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::RationalFunction(error) => Some(error),
            Self::Coefficient(_) => None,
        }
    }
}

impl From<ContextMismatchError> for RationalFunctionFieldMapError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PolynomialError> for RationalFunctionFieldMapError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<RationalFunctionError> for RationalFunctionFieldMapError {
    fn from(value: RationalFunctionError) -> Self {
        Self::RationalFunction(value)
    }
}

/// The rational function field `R(t)` over a coefficient field `R`.
///
/// The coefficient context and variable name are part of field identity.
/// Elements are reduced fractions in the associated univariate polynomial
/// ring, with a monic denominator.
#[derive(Clone)]
pub struct RationalFunctionField<R: Field> {
    coefficient_ring: R,
    polynomial_ring: PolynomialRing<R>,
    generator_polynomial: Polynomial<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>")
)]
struct RationalFunctionFieldWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    coefficient_ring: R,
    variable: String,
}

impl<R: Field> RationalFunctionField<R> {
    /// Construct `coefficient_ring(variable)`.
    pub fn new(
        coefficient_ring: R,
        variable: impl Into<String>,
    ) -> Result<Self, RationalFunctionError> {
        let polynomial_ring = PolynomialRing::new(
            coefficient_ring.clone(),
            [variable.into()],
            MonomialOrder::Lexicographic,
        )?;
        let generator_polynomial = polynomial_ring.generator(0)?;
        Ok(Self {
            coefficient_ring,
            polynomial_ring,
            generator_polynomial,
        })
    }

    /// Coefficient field.
    pub fn coefficient_ring(&self) -> &R {
        &self.coefficient_ring
    }

    /// Name of the transcendental generator.
    pub fn variable(&self) -> &str {
        &self.polynomial_ring.variables()[0]
    }

    /// Associated univariate polynomial ring.
    pub fn polynomial_ring(&self) -> &PolynomialRing<R> {
        &self.polynomial_ring
    }

    /// Serialize the complete coefficient-field and variable context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a rational-function field context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Transcendental generator.
    pub fn generator(&self) -> RationalFunction<R> {
        RationalFunction {
            field: self.clone(),
            numerator: self.generator_polynomial.clone(),
            denominator: self.polynomial_ring.one(),
        }
    }

    /// Embed one coefficient as a constant rational function.
    pub fn constant(
        &self,
        coefficient: R::Element,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        let polynomial = self.polynomial_ring.constant(coefficient)?;
        Ok(self.from_polynomial(polynomial)?)
    }

    /// Embed one polynomial from the associated univariate polynomial ring.
    pub fn from_polynomial(
        &self,
        polynomial: Polynomial<R>,
    ) -> Result<RationalFunction<R>, ContextMismatchError> {
        self.require_polynomial(&polynomial)?;
        Ok(RationalFunction {
            field: self.clone(),
            numerator: polynomial,
            denominator: self.polynomial_ring.one(),
        })
    }

    /// Construct and canonically reduce `numerator / denominator`.
    pub fn fraction(
        &self,
        numerator: Polynomial<R>,
        denominator: Polynomial<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        self.require_polynomial(&numerator)?;
        self.require_polynomial(&denominator)?;
        self.normalize(numerator, denominator)
    }

    fn require_polynomial(&self, value: &Polynomial<R>) -> Result<(), ContextMismatchError> {
        if self.polynomial_ring.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: PolynomialRing::<R>::TYPE_ID.to_owned(),
                actual: "unequal rational-function polynomial context".to_owned(),
            })
        }
    }

    fn require(&self, value: &RationalFunction<R>) -> Result<(), ContextMismatchError> {
        if self == &value.field {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: Self::TYPE_ID.to_owned(),
                actual: "unequal rational-function field context".to_owned(),
            })
        }
    }

    fn normalize(
        &self,
        numerator: Polynomial<R>,
        denominator: Polynomial<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        if denominator.is_zero() {
            return Err(RationalFunctionError::ZeroDenominator);
        }
        if numerator.is_zero() {
            return Ok(self.zero());
        }

        let gcd = numerator.gcd(&denominator)?;
        let numerator = numerator.exact_div(&gcd)?;
        let denominator = denominator.exact_div(&gcd)?;
        let leading = denominator
            .leading_coefficient()
            .ok_or(RationalFunctionError::ZeroDenominator)?;
        let inverse = self
            .coefficient_ring
            .inverse(&leading)
            .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        Ok(RationalFunction {
            field: self.clone(),
            numerator: numerator.scalar_mul(&inverse)?,
            denominator: denominator.scalar_mul(&inverse)?,
        })
    }

    fn checked_add(
        &self,
        left: &RationalFunction<R>,
        right: &RationalFunction<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        self.require(left)?;
        self.require(right)?;

        let common = left.denominator.gcd(&right.denominator)?;
        let left_scale = right.denominator.exact_div(&common)?;
        let right_scale = left.denominator.exact_div(&common)?;
        let left_numerator = (&left.numerator * &left_scale)?;
        let right_numerator = (&right.numerator * &right_scale)?;
        let numerator = (&left_numerator + &right_numerator)?;
        let denominator = (&left.denominator * &left_scale)?;
        self.normalize(numerator, denominator)
    }

    fn checked_sub(
        &self,
        left: &RationalFunction<R>,
        right: &RationalFunction<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        self.checked_add(left, &self.checked_neg(right)?)
    }

    fn checked_mul(
        &self,
        left: &RationalFunction<R>,
        right: &RationalFunction<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        self.require(left)?;
        self.require(right)?;

        let left_cancel = left.numerator.gcd(&right.denominator)?;
        let right_cancel = right.numerator.gcd(&left.denominator)?;
        let left_numerator = left.numerator.exact_div(&left_cancel)?;
        let right_denominator = right.denominator.exact_div(&left_cancel)?;
        let right_numerator = right.numerator.exact_div(&right_cancel)?;
        let left_denominator = left.denominator.exact_div(&right_cancel)?;
        let numerator = (&left_numerator * &right_numerator)?;
        let denominator = (&left_denominator * &right_denominator)?;
        self.normalize(numerator, denominator)
    }

    fn checked_neg(
        &self,
        value: &RationalFunction<R>,
    ) -> Result<RationalFunction<R>, RationalFunctionError> {
        self.require(value)?;
        Ok(RationalFunction {
            field: self.clone(),
            numerator: (-&value.numerator)?,
            denominator: value.denominator.clone(),
        })
    }
}

impl<R> Serialize for RationalFunctionField<R>
where
    R: Field + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RationalFunctionField", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("coefficient_ring", &self.coefficient_ring)?;
        state.serialize_field("variable", self.variable())?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for RationalFunctionField<R>
where
    R: Field + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalFunctionFieldWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.coefficient_ring, wire.variable).map_err(serde::de::Error::custom)
    }
}

impl<R: Field> PartialEq for RationalFunctionField<R> {
    fn eq(&self, other: &Self) -> bool {
        self.polynomial_ring == other.polynomial_ring
    }
}

impl<R: Field> Eq for RationalFunctionField<R> {}

impl<R: Field> fmt::Debug for RationalFunctionField<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RationalFunctionField")
            .field("coefficient_type", &R::TYPE_ID)
            .field("variable", &self.variable())
            .finish()
    }
}

impl<R: Field> fmt::Display for RationalFunctionField<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RationalFunctionField<{}>({})",
            R::TYPE_ID,
            self.variable()
        )
    }
}

impl<R: Field> SemanticType for RationalFunctionField<R> {
    const TYPE_ID: &'static str = "noether/commalg/RationalFunctionField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.polynomial_ring.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_FUNCTION_FIELD_DESCRIPTION
    }
}

impl<R: Field> Ring for RationalFunctionField<R> {
    type Element = RationalFunction<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.field
    }

    fn zero(&self) -> Self::Element {
        RationalFunction {
            field: self.clone(),
            numerator: self.polynomial_ring.zero(),
            denominator: self.polynomial_ring.one(),
        }
    }

    fn one(&self) -> Self::Element {
        RationalFunction {
            field: self.clone(),
            numerator: self.polynomial_ring.one(),
            denominator: self.polynomial_ring.one(),
        }
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_add(a, b).map_err(as_context_error::<R>)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_sub(a, b).map_err(as_context_error::<R>)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_mul(a, b).map_err(as_context_error::<R>)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.checked_neg(value).map_err(as_context_error::<R>)
    }
}

impl<R: Field> CommutativeRing for RationalFunctionField<R> {}

impl<R: Field> IntegralDomain for RationalFunctionField<R> {}

impl<R: Field> UnitInversion for RationalFunctionField<R> {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(UnitError::from)
    }
}

impl<R: Field> Field for RationalFunctionField<R> {
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError> {
        self.require(value)?;
        if value.is_zero() {
            return Err(DivisionByZeroError.into());
        }
        self.normalize(value.denominator.clone(), value.numerator.clone())
            .map_err(|error| InversionError::ContextMismatch(as_context_error::<R>(error)))
    }
}

/// The map `S(t) -> T(u)` induced by one checked field-coefficient map.
///
/// The source transcendental generator maps to the target generator. Variable
/// names may differ because they are presentation identity, while the source
/// and target fields and the inducing coefficient map are stored explicitly.
#[derive(Clone, PartialEq, Eq)]
pub struct RationalFunctionFieldMap<S, T, M>
where
    S: Field,
    T: Field,
    M: CoefficientMap<Source = S, Target = T>,
{
    source: RationalFunctionField<S>,
    target: RationalFunctionField<T>,
    coefficient_map: M,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "S: Field + Deserialize<'de>, T: Field + Deserialize<'de>, M: CoefficientMap<Source = S, Target = T> + Deserialize<'de>"
    )
)]
struct RationalFunctionFieldMapWire<S, T, M>
where
    S: Field,
    T: Field,
    M: CoefficientMap<Source = S, Target = T>,
{
    #[serde(rename = "type")]
    type_id: String,
    source: RationalFunctionField<S>,
    target: RationalFunctionField<T>,
    coefficient_map: M,
}

impl<S, T, M> RationalFunctionFieldMap<S, T, M>
where
    S: Field + fmt::Display,
    T: Field + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    /// Construct after checking both coefficient-map endpoints exactly.
    pub fn new(
        source: RationalFunctionField<S>,
        target: RationalFunctionField<T>,
        coefficient_map: M,
    ) -> Result<Self, RationalFunctionFieldMapError> {
        if source.coefficient_ring() != coefficient_map.source() {
            return Err(ContextMismatchError {
                expected: source.coefficient_ring().to_string(),
                actual: coefficient_map.source().to_string(),
            }
            .into());
        }
        if target.coefficient_ring() != coefficient_map.target() {
            return Err(ContextMismatchError {
                expected: target.coefficient_ring().to_string(),
                actual: coefficient_map.target().to_string(),
            }
            .into());
        }
        Ok(Self {
            source,
            target,
            coefficient_map,
        })
    }

    /// Explicit source rational-function field.
    pub fn source(&self) -> &RationalFunctionField<S> {
        &self.source
    }

    /// Explicit target rational-function field.
    pub fn target(&self) -> &RationalFunctionField<T> {
        &self.target
    }

    /// Field-coefficient map inducing this extension.
    pub fn coefficient_map(&self) -> &M {
        &self.coefficient_map
    }

    /// Verify exact identity of both inducing-map endpoints.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.source.coefficient_ring() != self.coefficient_map.source()
            || self.target.coefficient_ring() != self.coefficient_map.target()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "rational-function field map has exact coefficient-map endpoints",
            });
        }
        Ok(())
    }

    /// Serialize both fields and the checked inducing map.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        S: Serialize,
        T: Serialize,
        M: Serialize + Clone,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both exact endpoints.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        S: for<'de> Deserialize<'de>,
        T: for<'de> Deserialize<'de>,
        M: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn map_polynomial(
        &self,
        polynomial: &Polynomial<S>,
    ) -> Result<Polynomial<T>, RationalFunctionFieldMapError> {
        let mut terms = Vec::with_capacity(polynomial.nonzero_term_count());
        for (exponent, coefficient) in polynomial.terms() {
            terms.push((
                exponent.to_vec(),
                self.coefficient_map.apply(coefficient).map_err(|error| {
                    RationalFunctionFieldMapError::Coefficient(error.to_string())
                })?,
            ));
        }
        self.target
            .polynomial_ring()
            .from_terms(terms)
            .map_err(Into::into)
    }
}

impl<S, T, M> CoefficientMap for RationalFunctionFieldMap<S, T, M>
where
    S: Field + fmt::Display,
    T: Field + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    type Source = RationalFunctionField<S>;
    type Target = RationalFunctionField<T>;
    type Error = RationalFunctionFieldMapError;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &RationalFunction<S>) -> Result<RationalFunction<T>, Self::Error> {
        self.source.require(value)?;
        self.target
            .fraction(
                self.map_polynomial(value.numerator())?,
                self.map_polynomial(value.denominator())?,
            )
            .map_err(Into::into)
    }
}

impl<S, T, M> Serialize for RationalFunctionFieldMap<S, T, M>
where
    S: Field + Serialize,
    T: Field + Serialize,
    M: CoefficientMap<Source = S, Target = T> + Serialize + Clone,
{
    fn serialize<U>(&self, serializer: U) -> Result<U::Ok, U::Error>
    where
        U: Serializer,
    {
        let mut state = serializer.serialize_struct("RationalFunctionFieldMap", 4)?;
        state.serialize_field("type", RATIONAL_FUNCTION_FIELD_MAP_TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("coefficient_map", &self.coefficient_map)?;
        state.end()
    }
}

impl<'de, S, T, M> Deserialize<'de> for RationalFunctionFieldMap<S, T, M>
where
    S: Field + fmt::Display + Deserialize<'de>,
    T: Field + fmt::Display + Deserialize<'de>,
    M: CoefficientMap<Source = S, Target = T> + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalFunctionFieldMapWire::<S, T, M>::deserialize(deserializer)?;
        if wire.type_id != RATIONAL_FUNCTION_FIELD_MAP_TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                RATIONAL_FUNCTION_FIELD_MAP_TYPE_ID, wire.type_id
            )));
        }
        Self::new(wire.source, wire.target, wire.coefficient_map).map_err(serde::de::Error::custom)
    }
}

impl<S, T, M> SemanticType for RationalFunctionFieldMap<S, T, M>
where
    S: Field,
    T: Field,
    M: CoefficientMap<Source = S, Target = T> + SemanticType + PartialEq + Eq,
{
    const TYPE_ID: &'static str = RATIONAL_FUNCTION_FIELD_MAP_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.source.canonical_bytes()?)?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        append_component(&mut output, &self.coefficient_map.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_FUNCTION_FIELD_MAP_DESCRIPTION
    }
}

impl<S, T, M> fmt::Display for RationalFunctionFieldMap<S, T, M>
where
    S: Field,
    T: Field,
    M: CoefficientMap<Source = S, Target = T>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source, self.target)
    }
}

impl<S, T, M> fmt::Debug for RationalFunctionFieldMap<S, T, M>
where
    S: Field,
    T: Field,
    M: CoefficientMap<Source = S, Target = T> + fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RationalFunctionFieldMap")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("coefficient_map", &self.coefficient_map)
            .finish()
    }
}

/// One canonical element of a [`RationalFunctionField`].
///
/// The numerator and denominator belong to the field's associated polynomial
/// ring, are coprime, and the denominator is monic and nonzero.
#[derive(Clone)]
pub struct RationalFunction<R: Field> {
    field: RationalFunctionField<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct RationalFunctionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    field: RationalFunctionField<R>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
}

impl<R: Field> RationalFunction<R> {
    /// Owning rational function field.
    pub fn field(&self) -> &RationalFunctionField<R> {
        &self.field
    }

    /// Reduced numerator.
    pub fn numerator(&self) -> &Polynomial<R> {
        &self.numerator
    }

    /// Reduced, monic denominator.
    pub fn denominator(&self) -> &Polynomial<R> {
        &self.denominator
    }

    /// Serialize the field context and canonical reduced fraction.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a canonical reduced rational function.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Whether this value is zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Whether this value is one.
    pub fn is_one(&self) -> bool {
        self.numerator == self.denominator
    }

    /// Raise to a nonnegative machine-word power.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, RationalFunctionError> {
        let mut result = self.field.one();
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.field.checked_mul(&result, &base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = self.field.checked_mul(&base, &base)?;
            }
        }
        Ok(result)
    }

    /// Evaluate at one coefficient-field value, failing at a pole.
    pub fn specialize(&self, value: &R::Element) -> Result<R::Element, RationalFunctionError> {
        if !self.field.coefficient_ring.contains(value) {
            return Err(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "specialization value belongs to another coefficient context".to_owned(),
            }
            .into());
        }
        let numerator = self.numerator.evaluate(std::slice::from_ref(value))?;
        let denominator = self.denominator.evaluate(std::slice::from_ref(value))?;
        let inverse =
            self.field
                .coefficient_ring
                .inverse(&denominator)
                .map_err(|error| match error {
                    InversionError::DivisionByZero(_) => RationalFunctionError::Pole,
                    InversionError::ContextMismatch(error) => error.into(),
                })?;
        self.field
            .coefficient_ring
            .mul(&numerator, &inverse)
            .map_err(RationalFunctionError::from)
    }

    /// Divide by another value from the same field.
    pub fn checked_div(&self, divisor: &Self) -> Result<Self, RationalFunctionError> {
        self.field.require(divisor)?;
        let inverse = self.field.inverse(divisor).map_err(|error| match error {
            InversionError::DivisionByZero(_) => RationalFunctionError::DivisionByZero,
            InversionError::ContextMismatch(error) => error.into(),
        })?;
        self.field.checked_mul(self, &inverse)
    }
}

impl<R> Serialize for RationalFunction<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RationalFunction", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("field", &self.field)?;
        state.serialize_field("numerator", &self.numerator)?;
        state.serialize_field("denominator", &self.denominator)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for RationalFunction<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalFunctionWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let numerator = wire.numerator;
        let denominator = wire.denominator;
        let value = wire
            .field
            .fraction(numerator.clone(), denominator.clone())
            .map_err(serde::de::Error::custom)?;
        if value.numerator != numerator || value.denominator != denominator {
            return Err(serde::de::Error::custom(
                "rational-function numerator and denominator must already be canonical",
            ));
        }
        Ok(value)
    }
}

impl<R: Field> PartialEq for RationalFunction<R> {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.numerator == other.numerator
            && self.denominator == other.denominator
    }
}

impl<R: Field> Eq for RationalFunction<R> {}

impl<R: Field> fmt::Debug for RationalFunction<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RationalFunction")
            .field("coefficient_type", &R::TYPE_ID)
            .field("variable", &self.field.variable())
            .field("numerator_terms", &self.numerator.nonzero_term_count())
            .field("denominator_terms", &self.denominator.nonzero_term_count())
            .finish()
    }
}

impl<R: Field> fmt::Display for RationalFunction<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RationalFunction<{}>(numerator_terms={}, denominator_terms={})",
            R::TYPE_ID,
            self.numerator.nonzero_term_count(),
            self.denominator.nonzero_term_count()
        )
    }
}

impl<R: Field> SemanticType for RationalFunction<R> {
    const TYPE_ID: &'static str = "noether/commalg/RationalFunction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.field.canonical_bytes()?)?;
        append_component(&mut output, &self.numerator.canonical_bytes()?)?;
        append_component(&mut output, &self.denominator.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_FUNCTION_DESCRIPTION
    }
}

impl<R: Field> Add<&RationalFunction<R>> for &RationalFunction<R> {
    type Output = Result<RationalFunction<R>, RationalFunctionError>;

    fn add(self, rhs: &RationalFunction<R>) -> Self::Output {
        self.field.checked_add(self, rhs)
    }
}

impl<R: Field> Sub<&RationalFunction<R>> for &RationalFunction<R> {
    type Output = Result<RationalFunction<R>, RationalFunctionError>;

    fn sub(self, rhs: &RationalFunction<R>) -> Self::Output {
        self.field.checked_sub(self, rhs)
    }
}

impl<R: Field> Mul<&RationalFunction<R>> for &RationalFunction<R> {
    type Output = Result<RationalFunction<R>, RationalFunctionError>;

    fn mul(self, rhs: &RationalFunction<R>) -> Self::Output {
        self.field.checked_mul(self, rhs)
    }
}

impl<R: Field> Div<&RationalFunction<R>> for &RationalFunction<R> {
    type Output = Result<RationalFunction<R>, RationalFunctionError>;

    fn div(self, rhs: &RationalFunction<R>) -> Self::Output {
        self.checked_div(rhs)
    }
}

impl<R: Field> Neg for &RationalFunction<R> {
    type Output = Result<RationalFunction<R>, RationalFunctionError>;

    fn neg(self) -> Self::Output {
        self.field.checked_neg(self)
    }
}

fn as_context_error<R: Field>(error: RationalFunctionError) -> ContextMismatchError {
    match error {
        RationalFunctionError::ContextMismatch(error) => error,
        error => ContextMismatchError {
            expected: RationalFunctionField::<R>::TYPE_ID.to_owned(),
            actual: format!("internal rational-function invariant failed: {error}"),
        },
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("rational-function component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}
