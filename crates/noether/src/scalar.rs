// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical FLINT-backed exact scalars and their singleton contexts.

#![allow(unsafe_code)]

use core::fmt;

pub use flint_core::{Integer, Rational};
use flint_sys::{fmpz_fdiv_q, fmpz_gcd, fmpz_mod, fmpz_xgcd};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{commalg::ResidueRing, numthy::FiniteField};
use crate::{
    BezoutIdentity, CommutativeRing, ContextMismatchError, DivisionByZeroError, EuclideanDomain,
    Field, IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction, IntegralDomain,
    InversionError, NoetherError, PrincipalIdealDomain, Ring, SemanticType, TypeDescription,
    UnitError, UnitInversion,
};

static INTEGER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/Integer@1",
    rust_path: "noether::Integer",
    python_path: Some("builtins.int"),
};

static INTEGER_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/IntegerRing@1",
    rust_path: "noether::IntegerRing",
    python_path: Some("noether.IntegerRing"),
};

static RATIONAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/Rational@1",
    rust_path: "noether::Rational",
    python_path: Some("fractions.Fraction"),
};

static RATIONAL_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/RationalField@1",
    rust_path: "noether::RationalField",
    python_path: Some("noether.RationalField"),
};

impl SemanticType for Integer {
    const TYPE_ID: &'static str = "noether/root/Integer@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_string().into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_DESCRIPTION
    }
}

/// The unique context representing the ring of integers `Z`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct IntegerRing;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerRingWire {
    #[serde(rename = "type")]
    type_id: String,
}

/// The canonical integer-ring context.
pub const Z: IntegerRing = IntegerRing;

impl IntegerRing {
    /// Construct an integer from any accepted exact integer input.
    pub fn element<T: Into<Integer>>(&self, value: T) -> Integer {
        value.into()
    }

    /// The canonical unital embedding `Z -> Q`.
    pub const fn rational_embedding(&self) -> IntegerRationalEmbedding {
        IntegerRationalEmbedding::new()
    }

    /// Canonical reduction into one explicit residue ring.
    pub fn reduction(&self, target: &ResidueRing) -> IntegerReduction {
        IntegerReduction::new(target.clone())
    }

    /// Canonical unital map into one explicit finite field.
    pub fn finite_field_map(&self, target: &FiniteField) -> IntegerFiniteFieldMap {
        IntegerFiniteFieldMap::new(target.clone())
    }

    /// Serialize the singleton integer-ring context as canonical typed JSON.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate the singleton integer-ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for IntegerRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegerRingWire {
            type_id: Self::TYPE_ID.to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IntegerRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegerRingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self)
    }
}

impl fmt::Display for IntegerRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Z")
    }
}

impl SemanticType for IntegerRing {
    const TYPE_ID: &'static str = "noether/root/IntegerRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(b"Z".to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_RING_DESCRIPTION
    }
}

impl Ring for IntegerRing {
    type Element = Integer;

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }

    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    fn one(&self) -> Self::Element {
        Integer::one()
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a + b)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a - b)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a * b)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        Ok(-value)
    }
}

impl CommutativeRing for IntegerRing {}

impl UnitInversion for IntegerRing {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        if value.is_zero() {
            return Err(DivisionByZeroError.into());
        }
        let one = Integer::one();
        if value == &one {
            Ok(one)
        } else if value == &-&one {
            Ok(-&one)
        } else {
            Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            })
        }
    }
}

impl IntegralDomain for IntegerRing {}

impl PrincipalIdealDomain for IntegerRing {
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut output = Integer::initialized();
        unsafe { fmpz_gcd(output.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
        output
    }

    fn extended_gcd(&self, a: &Self::Element, b: &Self::Element) -> BezoutIdentity<Self> {
        let mut gcd = Integer::initialized();
        let mut coefficient_a = Integer::initialized();
        let mut coefficient_b = Integer::initialized();
        unsafe {
            fmpz_xgcd(
                gcd.as_mut_ptr(),
                coefficient_a.as_mut_ptr(),
                coefficient_b.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
            )
        };
        BezoutIdentity {
            gcd,
            coefficient_a,
            coefficient_b,
        }
    }
}

impl EuclideanDomain for IntegerRing {
    type Degree = Integer;

    fn div_rem(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<(Self::Element, Self::Element), DivisionByZeroError> {
        if b.is_zero() {
            return Err(DivisionByZeroError);
        }

        let modulus = b.abs();
        let mut quotient = Integer::initialized();
        let mut remainder = Integer::initialized();
        unsafe {
            fmpz_fdiv_q(quotient.as_mut_ptr(), a.as_ptr(), modulus.as_ptr());
            fmpz_mod(remainder.as_mut_ptr(), a.as_ptr(), modulus.as_ptr());
        }
        if b.sign() < 0 {
            quotient = -quotient;
        }
        Ok((quotient, remainder))
    }

    fn euclidean_degree(&self, value: &Self::Element) -> Option<Self::Degree> {
        (!value.is_zero()).then(|| value.abs())
    }
}

impl SemanticType for Rational {
    const TYPE_ID: &'static str = "noether/root/Rational@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(format!("{}/{}", self.numerator(), self.denominator()).into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_DESCRIPTION
    }
}

/// The unique context representing the rational field `Q`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RationalField;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalFieldWire {
    #[serde(rename = "type")]
    type_id: String,
}

/// The canonical rational-field context.
pub const Q: RationalField = RationalField;

impl RationalField {
    /// Construct a rational from any accepted exact rational input.
    pub fn element<T: Into<Rational>>(&self, value: T) -> Rational {
        value.into()
    }

    /// Serialize the singleton rational-field context as canonical typed JSON.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate the singleton rational-field context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Construct and canonicalize `numerator / denominator`.
    pub fn fraction(
        &self,
        numerator: Integer,
        denominator: Integer,
    ) -> Result<Rational, DivisionByZeroError> {
        Rational::new(numerator, denominator)
    }
}

impl Serialize for RationalField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        RationalFieldWire {
            type_id: Self::TYPE_ID.to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RationalField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalFieldWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self)
    }
}

impl fmt::Display for RationalField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Q")
    }
}

impl SemanticType for RationalField {
    const TYPE_ID: &'static str = "noether/root/RationalField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(b"Q".to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &RATIONAL_FIELD_DESCRIPTION
    }
}

impl Ring for RationalField {
    type Element = Rational;

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }

    fn zero(&self) -> Self::Element {
        Rational::zero()
    }

    fn one(&self) -> Self::Element {
        Rational::one()
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a + b)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a - b)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(a * b)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        Ok(-value)
    }
}

impl CommutativeRing for RationalField {}

impl UnitInversion for RationalField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(UnitError::from)
    }
}

impl IntegralDomain for RationalField {}

impl Field for RationalField {
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError> {
        value.inverse().map_err(InversionError::from)
    }
}

impl PrincipalIdealDomain for RationalField {
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.is_zero() && b.is_zero() {
            Rational::zero()
        } else {
            Rational::one()
        }
    }

    fn extended_gcd(&self, a: &Self::Element, b: &Self::Element) -> BezoutIdentity<Self> {
        if let Ok(coefficient_a) = a.inverse() {
            BezoutIdentity {
                gcd: Rational::one(),
                coefficient_a,
                coefficient_b: Rational::zero(),
            }
        } else if let Ok(coefficient_b) = b.inverse() {
            BezoutIdentity {
                gcd: Rational::one(),
                coefficient_a: Rational::zero(),
                coefficient_b,
            }
        } else {
            BezoutIdentity {
                gcd: Rational::zero(),
                coefficient_a: Rational::zero(),
                coefficient_b: Rational::zero(),
            }
        }
    }
}

impl EuclideanDomain for RationalField {
    type Degree = u8;

    fn div_rem(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<(Self::Element, Self::Element), DivisionByZeroError> {
        Ok((a.checked_div(b)?, Rational::zero()))
    }

    fn euclidean_degree(&self, value: &Self::Element) -> Option<Self::Degree> {
        (!value.is_zero()).then_some(0)
    }
}
