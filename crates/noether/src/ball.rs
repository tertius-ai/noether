// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Fixed-context certified real and complex balls backed by FLINT Arb.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use flint_core::{Acb, Arb};
use flint_sys::{acb_struct, arb_struct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CommutativeRing, ContextMismatchError, NoetherError, PrecisionContext, Rational, Ring,
    SemanticType, TypeDescription,
};

static REAL_BALL_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/RealBallField@1",
    rust_path: "noether::RealBallField",
    python_path: Some("noether.RealBallField"),
};

static REAL_BALL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/RealBall@1",
    rust_path: "noether::RealBall",
    python_path: Some("noether.RealBall"),
};

static COMPLEX_BALL_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/ComplexBallField@1",
    rust_path: "noether::ComplexBallField",
    python_path: Some("noether.ComplexBallField"),
};

static COMPLEX_BALL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/ComplexBall@1",
    rust_path: "noether::ComplexBall",
    python_path: Some("noether.ComplexBall"),
};

/// Failure to construct or deserialize a certified-ball context or value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BallError {
    /// Arb needs at least two bits of working precision.
    PrecisionTooSmall {
        /// Rejected precision.
        precision: u64,
    },
    /// The requested precision does not fit FLINT's signed word interface.
    PrecisionTooLarge {
        /// Rejected precision.
        precision: u64,
    },
    /// An exact Arb dump was malformed.
    InvalidEncoding,
    /// The lower endpoint of a requested interval exceeded the upper endpoint.
    InvalidInterval,
}

impl fmt::Display for BallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrecisionTooSmall { precision } => write!(
                f,
                "certified-ball precision must be at least 2 bits; received {precision}"
            ),
            Self::PrecisionTooLarge { precision } => write!(
                f,
                "certified-ball precision {precision} does not fit the FLINT interface"
            ),
            Self::InvalidEncoding => f.write_str("invalid exact Arb ball encoding"),
            Self::InvalidInterval => {
                f.write_str("real-ball interval lower endpoint exceeds its upper endpoint")
            }
        }
    }
}

impl std::error::Error for BallError {}

fn checked_precision(precision: u64) -> Result<i64, BallError> {
    if precision < 2 {
        return Err(BallError::PrecisionTooSmall { precision });
    }
    i64::try_from(precision).map_err(|_| BallError::PrecisionTooLarge { precision })
}

/// A real-ball context whose working precision is part of its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RealBallField {
    precision: u64,
}

impl RealBallField {
    /// Construct a real-ball context at `precision` bits.
    pub fn new(precision: u64) -> Result<Self, BallError> {
        checked_precision(precision)?;
        Ok(Self { precision })
    }

    /// Working precision in bits.
    pub fn precision_bits(&self) -> u64 {
        self.precision
    }

    /// Enclose one exact rational at this context's precision.
    pub fn element<T: Into<Rational>>(self, value: T) -> RealBall {
        RealBall::from_rational(self, &value.into())
    }

    /// Enclose the exact rational interval `[lower, upper]`.
    pub fn interval<L: Into<Rational>, U: Into<Rational>>(
        self,
        lower: L,
        upper: U,
    ) -> Result<RealBall, BallError> {
        let lower = lower.into();
        let upper = upper.into();
        if lower > upper {
            return Err(BallError::InvalidInterval);
        }
        let lower = RealBall::from_rational(self, &lower);
        let upper = RealBall::from_rational(self, &upper);
        Ok(RealBall::from_raw(
            self,
            lower.raw.union(&upper.raw, self.precision as i64),
        ))
    }

    fn require(self, value: &RealBall) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.field.to_string(),
            })
        }
    }

    fn binary(
        self,
        left: &RealBall,
        right: &RealBall,
        operation: fn(&Arb, &Arb, i64) -> Arb,
    ) -> Result<RealBall, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        Ok(RealBall {
            field: self,
            raw: operation(&left.raw, &right.raw, self.precision as i64),
        })
    }

    /// Serialize this context to canonical JSON.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate a real-ball context from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for RealBallField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RealBallField({} bits)", self.precision)
    }
}

impl SemanticType for RealBallField {
    const TYPE_ID: &'static str = "noether/root/RealBallField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.precision.to_be_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &REAL_BALL_FIELD_DESCRIPTION
    }
}

impl Ring for RealBallField {
    type Element = RealBall;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.field
    }

    fn zero(&self) -> Self::Element {
        RealBall::initialized(*self)
    }

    fn one(&self) -> Self::Element {
        RealBall {
            field: *self,
            raw: Arb::one(),
        }
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Arb::add)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Arb::sub)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Arb::mul)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(RealBall {
            field: *self,
            raw: value.raw.neg(),
        })
    }
}

impl CommutativeRing for RealBallField {}

impl PrecisionContext for RealBallField {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealBallFieldRepr {
    #[serde(rename = "type")]
    type_id: String,
    precision: u64,
}

impl Serialize for RealBallField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        RealBallFieldRepr {
            type_id: Self::TYPE_ID.to_owned(),
            precision: self.precision,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RealBallField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = RealBallFieldRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        Self::new(repr.precision).map_err(serde::de::Error::custom)
    }
}

/// A certified real enclosure carrying its fixed-precision context.
pub struct RealBall {
    field: RealBallField,
    raw: Arb,
}

impl RealBall {
    fn initialized(field: RealBallField) -> Self {
        Self {
            field,
            raw: Arb::zero(),
        }
    }

    fn from_rational(field: RealBallField, value: &Rational) -> Self {
        Self {
            field,
            raw: Arb::from_rational(value, field.precision as i64),
        }
    }

    pub(crate) fn clone_from_ptr(field: RealBallField, value: *const arb_struct) -> Self {
        Self {
            field,
            raw: unsafe { Arb::clone_from_ptr(value) },
        }
    }

    pub(crate) fn as_ptr(&self) -> *const arb_struct {
        self.raw.as_ptr()
    }

    pub(crate) fn raw(&self) -> &Arb {
        &self.raw
    }

    pub(crate) fn from_raw(field: RealBallField, raw: Arb) -> Self {
        Self { field, raw }
    }

    fn from_dump(field: RealBallField, value: &str) -> Result<Self, BallError> {
        Ok(Self {
            field,
            raw: Arb::from_dump(value).map_err(|_| BallError::InvalidEncoding)?,
        })
    }

    /// The concrete precision context containing this ball.
    pub fn field(&self) -> RealBallField {
        self.field
    }

    /// Whether this enclosure is a point ball with radius zero.
    pub fn is_exact(&self) -> bool {
        self.raw.is_exact()
    }

    /// Whether zero is contained in this enclosure.
    pub fn contains_zero(&self) -> bool {
        self.raw.contains_zero()
    }

    /// Whether this enclosure is exactly the point zero.
    pub fn is_zero(&self) -> bool {
        self.raw.is_exact_zero()
    }

    /// Whether this enclosure is exactly the point one.
    pub fn is_one(&self) -> bool {
        self.raw.is_one()
    }

    /// Test whether two same-context enclosures intersect.
    pub fn overlaps(&self, other: &Self) -> Result<bool, ContextMismatchError> {
        self.field.require(other)?;
        Ok(self.raw.overlaps(&other.raw))
    }

    /// Raise this enclosure to a nonnegative integer power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = self.field.one();
        let mut base = self.clone();
        let mut power = exponent;
        while power != 0 {
            if power & 1 == 1 {
                result = self
                    .field
                    .mul(&result, &base)
                    .expect("both operands retain this ball context");
            }
            power >>= 1;
            if power != 0 {
                base = self
                    .field
                    .mul(&base, &base)
                    .expect("both operands retain this ball context");
            }
        }
        result
    }

    /// Serialize this exact enclosure and context to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate an exact real enclosure from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Clone for RealBall {
    fn clone(&self) -> Self {
        Self {
            field: self.field,
            raw: self.raw.clone(),
        }
    }
}

impl PartialEq for RealBall {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.raw.same_ball(&other.raw)
    }
}

impl Eq for RealBall {}

impl fmt::Display for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} in {}",
            self.raw.display_decimal(self.field.precision),
            self.field
        )
    }
}

impl fmt::Debug for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RealBall({self})")
    }
}

impl SemanticType for RealBall {
    const TYPE_ID: &'static str = "noether/root/RealBall@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &REAL_BALL_DESCRIPTION
    }
}

impl Add<&RealBall> for &RealBall {
    type Output = Result<RealBall, ContextMismatchError>;

    fn add(self, rhs: &RealBall) -> Self::Output {
        self.field.add(self, rhs)
    }
}

impl Sub<&RealBall> for &RealBall {
    type Output = Result<RealBall, ContextMismatchError>;

    fn sub(self, rhs: &RealBall) -> Self::Output {
        self.field.sub(self, rhs)
    }
}

impl Mul<&RealBall> for &RealBall {
    type Output = Result<RealBall, ContextMismatchError>;

    fn mul(self, rhs: &RealBall) -> Self::Output {
        self.field.mul(self, rhs)
    }
}

impl Neg for &RealBall {
    type Output = Result<RealBall, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.field.neg(self)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealBallRepr {
    #[serde(rename = "type")]
    type_id: String,
    precision: u64,
    value: String,
}

impl Serialize for RealBall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        RealBallRepr {
            type_id: Self::TYPE_ID.to_owned(),
            precision: self.field.precision,
            value: self.raw.dump().map_err(serde::ser::Error::custom)?,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RealBall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = RealBallRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        let field = RealBallField::new(repr.precision).map_err(serde::de::Error::custom)?;
        Self::from_dump(field, &repr.value).map_err(serde::de::Error::custom)
    }
}

/// A complex-ball context whose working precision is part of its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComplexBallField {
    precision: u64,
}

impl ComplexBallField {
    /// Construct a complex-ball context at `precision` bits.
    pub fn new(precision: u64) -> Result<Self, BallError> {
        checked_precision(precision)?;
        Ok(Self { precision })
    }

    /// Working precision in bits.
    pub fn precision_bits(&self) -> u64 {
        self.precision
    }

    /// Enclose an exact rational on the real axis.
    pub fn element<T: Into<Rational>>(self, value: T) -> ComplexBall {
        self.element_parts(value, Rational::zero())
    }

    /// Enclose exact rational real and imaginary parts.
    pub fn element_parts<R: Into<Rational>, I: Into<Rational>>(
        self,
        real: R,
        imaginary: I,
    ) -> ComplexBall {
        ComplexBall::from_rationals(self, &real.into(), &imaginary.into())
    }

    fn require(self, value: &ComplexBall) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.field.to_string(),
            })
        }
    }

    fn binary(
        self,
        left: &ComplexBall,
        right: &ComplexBall,
        operation: fn(&Acb, &Acb, i64) -> Acb,
    ) -> Result<ComplexBall, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        Ok(ComplexBall {
            field: self,
            raw: operation(&left.raw, &right.raw, self.precision as i64),
        })
    }

    /// Serialize this context to canonical JSON.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate a complex-ball context from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for ComplexBallField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexBallField({} bits)", self.precision)
    }
}

impl SemanticType for ComplexBallField {
    const TYPE_ID: &'static str = "noether/root/ComplexBallField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.precision.to_be_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &COMPLEX_BALL_FIELD_DESCRIPTION
    }
}

impl Ring for ComplexBallField {
    type Element = ComplexBall;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.field
    }

    fn zero(&self) -> Self::Element {
        ComplexBall::initialized(*self)
    }

    fn one(&self) -> Self::Element {
        ComplexBall {
            field: *self,
            raw: Acb::one(),
        }
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Acb::add)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Acb::sub)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, Acb::mul)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(ComplexBall {
            field: *self,
            raw: value.raw.neg(),
        })
    }
}

impl CommutativeRing for ComplexBallField {}

impl PrecisionContext for ComplexBallField {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexBallFieldRepr {
    #[serde(rename = "type")]
    type_id: String,
    precision: u64,
}

impl Serialize for ComplexBallField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ComplexBallFieldRepr {
            type_id: Self::TYPE_ID.to_owned(),
            precision: self.precision,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ComplexBallField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ComplexBallFieldRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        Self::new(repr.precision).map_err(serde::de::Error::custom)
    }
}

/// A certified complex enclosure carrying its fixed-precision context.
pub struct ComplexBall {
    field: ComplexBallField,
    raw: Acb,
}

impl ComplexBall {
    fn initialized(field: ComplexBallField) -> Self {
        Self {
            field,
            raw: Acb::zero(),
        }
    }

    pub(crate) fn clone_from_ptr(field: ComplexBallField, value: *const acb_struct) -> Self {
        Self {
            field,
            raw: unsafe { Acb::clone_from_ptr(value) },
        }
    }

    pub(crate) fn as_ptr(&self) -> *const acb_struct {
        self.raw.as_ptr()
    }

    pub(crate) fn raw(&self) -> &Acb {
        &self.raw
    }

    pub(crate) fn from_raw(field: ComplexBallField, raw: Acb) -> Self {
        Self { field, raw }
    }

    fn from_rationals(field: ComplexBallField, real: &Rational, imaginary: &Rational) -> Self {
        Self {
            field,
            raw: Acb::from_rationals(real, imaginary, field.precision as i64),
        }
    }

    fn from_dump(field: ComplexBallField, real: &str, imaginary: &str) -> Result<Self, BallError> {
        let real = Arb::from_dump(real).map_err(|_| BallError::InvalidEncoding)?;
        let imaginary = Arb::from_dump(imaginary).map_err(|_| BallError::InvalidEncoding)?;
        Ok(Self {
            field,
            raw: Acb::from_arb_parts(&real, &imaginary),
        })
    }

    /// The concrete precision context containing this ball.
    pub fn field(&self) -> ComplexBallField {
        self.field
    }

    /// Certified real component in the corresponding real-ball context.
    pub fn real(&self) -> RealBall {
        let field = RealBallField {
            precision: self.field.precision,
        };
        RealBall {
            field,
            raw: self.raw.real(),
        }
    }

    /// Certified imaginary component in the corresponding real-ball context.
    pub fn imaginary(&self) -> RealBall {
        let field = RealBallField {
            precision: self.field.precision,
        };
        RealBall {
            field,
            raw: self.raw.imag(),
        }
    }

    /// Whether both components are point balls with radius zero.
    pub fn is_exact(&self) -> bool {
        self.raw.is_exact()
    }

    /// Whether this enclosure is exactly the point zero.
    pub fn is_zero(&self) -> bool {
        self.raw.is_exact_zero()
    }

    /// Whether this enclosure is exactly the point one.
    pub fn is_one(&self) -> bool {
        self.raw.is_one()
    }

    /// Whether the complex enclosure contains the point zero.
    pub fn contains_zero(&self) -> bool {
        self.raw.contains_zero()
    }

    /// Test whether two same-context rectangular enclosures intersect.
    pub fn overlaps(&self, other: &Self) -> Result<bool, ContextMismatchError> {
        self.field.require(other)?;
        Ok(self.raw.overlaps(&other.raw))
    }

    /// Raise this enclosure to a nonnegative integer power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = self.field.one();
        let mut base = self.clone();
        let mut power = exponent;
        while power != 0 {
            if power & 1 == 1 {
                result = self
                    .field
                    .mul(&result, &base)
                    .expect("both operands retain this ball context");
            }
            power >>= 1;
            if power != 0 {
                base = self
                    .field
                    .mul(&base, &base)
                    .expect("both operands retain this ball context");
            }
        }
        result
    }

    /// Serialize this exact enclosure and context to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate an exact complex enclosure from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Clone for ComplexBall {
    fn clone(&self) -> Self {
        Self {
            field: self.field,
            raw: self.raw.clone(),
        }
    }
}

impl PartialEq for ComplexBall {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.raw.same_ball(&other.raw)
    }
}

impl Eq for ComplexBall {}

impl fmt::Display for ComplexBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}) + ({})i in {}",
            self.raw.real().display_decimal(self.field.precision),
            self.raw.imag().display_decimal(self.field.precision),
            self.field
        )
    }
}

impl fmt::Debug for ComplexBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexBall({self})")
    }
}

impl SemanticType for ComplexBall {
    const TYPE_ID: &'static str = "noether/root/ComplexBall@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &COMPLEX_BALL_DESCRIPTION
    }
}

impl Add<&ComplexBall> for &ComplexBall {
    type Output = Result<ComplexBall, ContextMismatchError>;

    fn add(self, rhs: &ComplexBall) -> Self::Output {
        self.field.add(self, rhs)
    }
}

impl Sub<&ComplexBall> for &ComplexBall {
    type Output = Result<ComplexBall, ContextMismatchError>;

    fn sub(self, rhs: &ComplexBall) -> Self::Output {
        self.field.sub(self, rhs)
    }
}

impl Mul<&ComplexBall> for &ComplexBall {
    type Output = Result<ComplexBall, ContextMismatchError>;

    fn mul(self, rhs: &ComplexBall) -> Self::Output {
        self.field.mul(self, rhs)
    }
}

impl Neg for &ComplexBall {
    type Output = Result<ComplexBall, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.field.neg(self)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexBallRepr {
    #[serde(rename = "type")]
    type_id: String,
    precision: u64,
    real: String,
    imaginary: String,
}

impl Serialize for ComplexBall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ComplexBallRepr {
            type_id: Self::TYPE_ID.to_owned(),
            precision: self.field.precision,
            real: self.raw.real().dump().map_err(serde::ser::Error::custom)?,
            imaginary: self.raw.imag().dump().map_err(serde::ser::Error::custom)?,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ComplexBall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ComplexBallRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        let field = ComplexBallField::new(repr.precision).map_err(serde::de::Error::custom)?;
        Self::from_dump(field, &repr.real, &repr.imaginary).map_err(serde::de::Error::custom)
    }
}
