// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical checked maps between cross-domain scalar contexts.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{ResidueClass, ResidueRing};
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    CoefficientMap, CommutativeRing, ContextMismatchError, Integer, IntegerRing, NoetherError,
    Rational, RationalField, Ring, SemanticType, TypeDescription, VerificationError, Q, Z,
};

static IDENTITY_COEFFICIENT_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/IdentityCoefficientMap@1",
    rust_path: "noether::IdentityCoefficientMap",
    python_path: None,
};

static COMPOSITE_COEFFICIENT_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/CompositeCoefficientMap@1",
    rust_path: "noether::CompositeCoefficientMap",
    python_path: None,
};
const COMPOSITE_COEFFICIENT_MAP_TYPE_ID: &str = "noether/root/CompositeCoefficientMap@1";

static INTEGER_RATIONAL_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/IntegerRationalEmbedding@1",
    rust_path: "noether::IntegerRationalEmbedding",
    python_path: Some("noether.IntegerRationalEmbedding"),
};

static INTEGER_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/IntegerReduction@1",
    rust_path: "noether::IntegerReduction",
    python_path: Some("noether.IntegerReduction"),
};

static INTEGER_FINITE_FIELD_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/IntegerFiniteFieldMap@1",
    rust_path: "noether::IntegerFiniteFieldMap",
    python_path: Some("noether.IntegerFiniteFieldMap"),
};

static PRIME_SUBFIELD_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/PrimeSubfieldEmbedding@1",
    rust_path: "noether::PrimeSubfieldEmbedding",
    python_path: Some("noether.PrimeSubfieldEmbedding"),
};

/// Failure to construct a checked embedding from a prime field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimeSubfieldEmbeddingError {
    /// The declared source was a proper finite-field extension.
    SourceIsExtension {
        /// Degree of the rejected source over its prime field.
        degree: usize,
    },
    /// Source and target had unequal prime characteristics.
    CharacteristicMismatch {
        /// Source characteristic.
        source: Integer,
        /// Target characteristic.
        target: Integer,
    },
    /// An input value did not belong to the declared prime-field source.
    ContextMismatch(ContextMismatchError),
}

impl fmt::Display for PrimeSubfieldEmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceIsExtension { degree } => write!(
                formatter,
                "a prime-subfield embedding requires a degree-one source; received degree {degree}"
            ),
            Self::CharacteristicMismatch { source, target } => write!(
                formatter,
                "prime-subfield characteristics differ: source {source}, target {target}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PrimeSubfieldEmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::SourceIsExtension { .. } | Self::CharacteristicMismatch { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PrimeSubfieldEmbeddingError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure in one named stage of a composed coefficient map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeCoefficientMapError {
    /// The inner map rejected its input.
    Inner(String),
    /// The outer map rejected the inner image.
    Outer(String),
}

impl fmt::Display for CompositeCoefficientMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inner(detail) => write!(formatter, "inner coefficient map failed: {detail}"),
            Self::Outer(detail) => write!(formatter, "outer coefficient map failed: {detail}"),
        }
    }
}

impl std::error::Error for CompositeCoefficientMapError {}

/// The identity map on one explicit commutative-ring context.
#[derive(Clone, PartialEq, Eq)]
pub struct IdentityCoefficientMap<R: CommutativeRing> {
    context: R,
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct IdentityCoefficientMapWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    context: R,
}

impl<R: CommutativeRing> IdentityCoefficientMap<R> {
    /// Construct the identity on `context`.
    pub fn new(context: R) -> Self {
        Self { context }
    }

    /// Explicit source and target context.
    pub fn context(&self) -> &R {
        &self.context
    }

    /// Verify that this value is an identity map on one context.
    pub fn verify(&self) -> Result<(), VerificationError> {
        Ok(())
    }

    /// Serialize this map together with its complete context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(&IdentityCoefficientMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            context: self.context.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an identity map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        let wire = serde_json::from_str::<IdentityCoefficientMapWire<R>>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        Ok(Self::new(wire.context))
    }
}

impl<R> Serialize for IdentityCoefficientMap<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IdentityCoefficientMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            context: self.context.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de, R> Deserialize<'de> for IdentityCoefficientMap<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IdentityCoefficientMapWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::new(wire.context))
    }
}

impl<R> CoefficientMap for IdentityCoefficientMap<R>
where
    R: CommutativeRing + fmt::Display,
{
    type Source = R;
    type Target = R;
    type Error = ContextMismatchError;

    fn source(&self) -> &Self::Source {
        &self.context
    }

    fn target(&self) -> &Self::Target {
        &self.context
    }

    fn apply(&self, value: &R::Element) -> Result<R::Element, Self::Error> {
        if self.context.contains(value) {
            Ok(value.clone())
        } else {
            Err(ContextMismatchError {
                expected: self.context.to_string(),
                actual: "element from an unequal coefficient context".to_owned(),
            })
        }
    }
}

impl<R: CommutativeRing> SemanticType for IdentityCoefficientMap<R> {
    const TYPE_ID: &'static str = "noether/root/IdentityCoefficientMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.context.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &IDENTITY_COEFFICIENT_MAP_DESCRIPTION
    }
}

impl<R: CommutativeRing + fmt::Display> fmt::Display for IdentityCoefficientMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.context, self.context)
    }
}

impl<R: CommutativeRing + fmt::Debug> fmt::Debug for IdentityCoefficientMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdentityCoefficientMap")
            .field("context", &self.context)
            .finish()
    }
}

/// The canonical unital embedding `Z -> Q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntegerRationalEmbedding;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SingletonMapWire {
    #[serde(rename = "type")]
    type_id: String,
}

impl IntegerRationalEmbedding {
    /// Construct the unique embedding `Z -> Q`.
    pub const fn new() -> Self {
        Self
    }

    /// Verify the fixed endpoints of this canonical embedding.
    pub fn verify(self) -> Result<(), VerificationError> {
        Ok(())
    }

    /// Serialize this canonical singleton map.
    pub fn to_json(self) -> Result<String, NoetherError> {
        singleton_to_json(Self::TYPE_ID)
    }

    /// Deserialize and revalidate the canonical singleton map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        singleton_from_json(value, Self::TYPE_ID).map(|()| Self)
    }
}

impl Serialize for IntegerRationalEmbedding {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SingletonMapWire {
            type_id: Self::TYPE_ID.to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IntegerRationalEmbedding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SingletonMapWire::deserialize(deserializer)?;
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

impl CoefficientMap for IntegerRationalEmbedding {
    type Source = IntegerRing;
    type Target = RationalField;
    type Error = ContextMismatchError;

    fn source(&self) -> &Self::Source {
        &Z
    }

    fn target(&self) -> &Self::Target {
        &Q
    }

    fn apply(&self, value: &Integer) -> Result<Rational, Self::Error> {
        Ok(Q.element(value.clone()))
    }
}

impl SemanticType for IntegerRationalEmbedding {
    const TYPE_ID: &'static str = "noether/root/IntegerRationalEmbedding@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        endpoint_bytes(&Z, &Q)
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_RATIONAL_EMBEDDING_DESCRIPTION
    }
}

impl fmt::Display for IntegerRationalEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Z -> Q")
    }
}

/// Canonical reduction `Z -> Z/nZ` for one explicit modulus.
#[derive(Clone, PartialEq, Eq)]
pub struct IntegerReduction {
    target: ResidueRing,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerReductionWire {
    #[serde(rename = "type")]
    type_id: String,
    target: ResidueRing,
}

impl IntegerReduction {
    /// Construct canonical reduction into `target`.
    pub fn new(target: ResidueRing) -> Self {
        Self { target }
    }

    /// Verify that the target remains a valid residue-ring context.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.target.modulus() > &Integer::one() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "integer reduction targets Z/nZ with n > 1",
            })
        }
    }

    /// Serialize this reduction with its modulus.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&IntegerReductionWire {
            type_id: Self::TYPE_ID.to_owned(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an integer reduction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<IntegerReductionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        let result = Self::new(wire.target);
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl Serialize for IntegerReduction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegerReductionWire {
            type_id: Self::TYPE_ID.to_owned(),
            target: self.target.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IntegerReduction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegerReductionWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self::new(wire.target);
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl CoefficientMap for IntegerReduction {
    type Source = IntegerRing;
    type Target = ResidueRing;
    type Error = ContextMismatchError;

    fn source(&self) -> &Self::Source {
        &Z
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &Integer) -> Result<ResidueClass, Self::Error> {
        Ok(self.target.element(value.clone()))
    }
}

impl SemanticType for IntegerReduction {
    const TYPE_ID: &'static str = "noether/root/IntegerReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        endpoint_bytes(self.source(), self.target())
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_REDUCTION_DESCRIPTION
    }
}

impl fmt::Debug for IntegerReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegerReduction")
            .field("source", self.source())
            .field("target", self.target())
            .finish()
    }
}

impl fmt::Display for IntegerReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source(), self.target())
    }
}

/// Canonical unital map from `Z` into one explicit finite field.
///
/// For an extension field this lands in its prime subfield.
#[derive(Clone, PartialEq, Eq)]
pub struct IntegerFiniteFieldMap {
    target: FiniteField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerFiniteFieldMapWire {
    #[serde(rename = "type")]
    type_id: String,
    target: FiniteField,
}

impl IntegerFiniteFieldMap {
    /// Construct the canonical map into `target`.
    pub fn new(target: FiniteField) -> Self {
        Self { target }
    }

    /// Verify the target's proved prime characteristic.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.target.characteristic().is_prime() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "integer-to-finite-field map has prime characteristic",
            })
        }
    }

    /// Serialize this map with the full finite-field presentation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&IntegerFiniteFieldMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an integer-to-finite-field map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<IntegerFiniteFieldMapWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        let result = Self::new(wire.target);
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl Serialize for IntegerFiniteFieldMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegerFiniteFieldMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            target: self.target.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IntegerFiniteFieldMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IntegerFiniteFieldMapWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self::new(wire.target);
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl CoefficientMap for IntegerFiniteFieldMap {
    type Source = IntegerRing;
    type Target = FiniteField;
    type Error = ContextMismatchError;

    fn source(&self) -> &Self::Source {
        &Z
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &Integer) -> Result<FiniteFieldElement, Self::Error> {
        Ok(self.target.element(value.clone()))
    }
}

impl SemanticType for IntegerFiniteFieldMap {
    const TYPE_ID: &'static str = "noether/root/IntegerFiniteFieldMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        endpoint_bytes(self.source(), self.target())
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_FINITE_FIELD_MAP_DESCRIPTION
    }
}

impl fmt::Debug for IntegerFiniteFieldMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegerFiniteFieldMap")
            .field("source", self.source())
            .field("target", self.target())
            .finish()
    }
}

impl fmt::Display for IntegerFiniteFieldMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source(), self.target())
    }
}

/// Canonical embedding `F_p -> F_q` into one explicit finite-field presentation.
#[derive(Clone, PartialEq, Eq)]
pub struct PrimeSubfieldEmbedding {
    source: FiniteField,
    target: FiniteField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimeSubfieldEmbeddingWire {
    #[serde(rename = "type")]
    type_id: String,
    source: FiniteField,
    target: FiniteField,
}

impl PrimeSubfieldEmbedding {
    /// Construct the canonical embedding of `target`'s prime subfield.
    pub fn for_target(target: FiniteField) -> Self {
        let source = FiniteField::prime(target.characteristic().clone())
            .expect("a constructed finite field has proved prime characteristic");
        Self { source, target }
    }

    /// Construct the embedding after checking degree and characteristic.
    pub fn new(
        source: FiniteField,
        target: FiniteField,
    ) -> Result<Self, PrimeSubfieldEmbeddingError> {
        let result = Self { source, target };
        result.verify().map_err(|_| {
            if result.source.degree() != 1 {
                PrimeSubfieldEmbeddingError::SourceIsExtension {
                    degree: result.source.degree(),
                }
            } else {
                PrimeSubfieldEmbeddingError::CharacteristicMismatch {
                    source: result.source.characteristic().clone(),
                    target: result.target.characteristic().clone(),
                }
            }
        })?;
        Ok(result)
    }

    /// Verify that the source is prime and both characteristics agree.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.source.degree() != 1 {
            return Err(VerificationError::InvalidWitness {
                claim: "prime-subfield embedding has degree-one source",
            });
        }
        if self.source.characteristic() != self.target.characteristic() {
            return Err(VerificationError::InvalidWitness {
                claim: "prime-subfield embedding preserves characteristic",
            });
        }
        Ok(())
    }

    /// Serialize both finite-field presentations.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PrimeSubfieldEmbeddingWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly revalidate an embedding.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PrimeSubfieldEmbeddingWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        Self::new(wire.source, wire.target)
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for PrimeSubfieldEmbedding {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PrimeSubfieldEmbeddingWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PrimeSubfieldEmbedding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PrimeSubfieldEmbeddingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.source, wire.target).map_err(serde::de::Error::custom)
    }
}

impl CoefficientMap for PrimeSubfieldEmbedding {
    type Source = FiniteField;
    type Target = FiniteField;
    type Error = PrimeSubfieldEmbeddingError;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &FiniteFieldElement) -> Result<FiniteFieldElement, Self::Error> {
        if !self.source.contains(value) {
            return Err(ContextMismatchError {
                expected: self.source.to_string(),
                actual: value.field().to_string(),
            }
            .into());
        }
        Ok(self.target.element(value.representative().clone()))
    }
}

impl SemanticType for PrimeSubfieldEmbedding {
    const TYPE_ID: &'static str = "noether/root/PrimeSubfieldEmbedding@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        endpoint_bytes(self.source(), self.target())
    }

    fn describe() -> &'static TypeDescription {
        &PRIME_SUBFIELD_EMBEDDING_DESCRIPTION
    }
}

impl fmt::Debug for PrimeSubfieldEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrimeSubfieldEmbedding")
            .field("source", self.source())
            .field("target", self.target())
            .finish()
    }
}

impl fmt::Display for PrimeSubfieldEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source(), self.target())
    }
}

/// Exact composition `outer ∘ inner` of two checked coefficient maps.
#[derive(Clone, PartialEq, Eq)]
pub struct CompositeCoefficientMap<M, N>
where
    M: CoefficientMap,
    N: CoefficientMap<Source = M::Target>,
{
    inner: M,
    outer: N,
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "M: Deserialize<'de>, N: Deserialize<'de>, M: CoefficientMap, N: CoefficientMap<Source = M::Target>"
    )
)]
struct CompositeCoefficientMapWire<M, N>
where
    M: CoefficientMap,
    N: CoefficientMap<Source = M::Target>,
{
    #[serde(rename = "type")]
    type_id: String,
    inner: M,
    outer: N,
}

impl<M, N> CompositeCoefficientMap<M, N>
where
    M: CoefficientMap,
    N: CoefficientMap<Source = M::Target>,
{
    /// Compose `outer` after `inner`, checking their shared middle context.
    pub fn new(inner: M, outer: N) -> Result<Self, ContextMismatchError> {
        if inner.target() != outer.source() {
            return Err(ContextMismatchError {
                expected: outer.source().to_string(),
                actual: inner.target().to_string(),
            });
        }
        Ok(Self { inner, outer })
    }

    /// Inner map applied first.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Outer map applied second.
    pub fn outer(&self) -> &N {
        &self.outer
    }

    /// Verify exact identity of the shared middle context.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.inner.target() == self.outer.source() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "composite coefficient maps share one exact middle context",
            })
        }
    }

    /// Serialize both maps in application order.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        M: Serialize + Clone,
        N: Serialize + Clone,
    {
        serde_json::to_string(&CompositeCoefficientMapWire {
            type_id: COMPOSITE_COEFFICIENT_MAP_TYPE_ID.to_owned(),
            inner: self.inner.clone(),
            outer: self.outer.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the shared middle context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        M: for<'de> Deserialize<'de>,
        N: for<'de> Deserialize<'de>,
    {
        let wire = serde_json::from_str::<CompositeCoefficientMapWire<M, N>>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(COMPOSITE_COEFFICIENT_MAP_TYPE_ID, &wire.type_id)?;
        Self::new(wire.inner, wire.outer)
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<M, N> Serialize for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap + Serialize + Clone,
    N: CoefficientMap<Source = M::Target> + Serialize + Clone,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        CompositeCoefficientMapWire {
            type_id: COMPOSITE_COEFFICIENT_MAP_TYPE_ID.to_owned(),
            inner: self.inner.clone(),
            outer: self.outer.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de, M, N> Deserialize<'de> for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap + Deserialize<'de>,
    N: CoefficientMap<Source = M::Target> + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CompositeCoefficientMapWire::<M, N>::deserialize(deserializer)?;
        if wire.type_id != COMPOSITE_COEFFICIENT_MAP_TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                COMPOSITE_COEFFICIENT_MAP_TYPE_ID, wire.type_id
            )));
        }
        Self::new(wire.inner, wire.outer).map_err(serde::de::Error::custom)
    }
}

impl<M, N> CoefficientMap for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap,
    N: CoefficientMap<Source = M::Target>,
{
    type Source = M::Source;
    type Target = N::Target;
    type Error = CompositeCoefficientMapError;

    fn source(&self) -> &Self::Source {
        self.inner.source()
    }

    fn target(&self) -> &Self::Target {
        self.outer.target()
    }

    fn apply(
        &self,
        value: &<Self::Source as Ring>::Element,
    ) -> Result<<Self::Target as Ring>::Element, Self::Error> {
        let middle = self
            .inner
            .apply(value)
            .map_err(|error| CompositeCoefficientMapError::Inner(error.to_string()))?;
        self.outer
            .apply(&middle)
            .map_err(|error| CompositeCoefficientMapError::Outer(error.to_string()))
    }
}

impl<M, N> SemanticType for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap + SemanticType + PartialEq + Eq,
    N: CoefficientMap<Source = M::Target> + SemanticType + PartialEq + Eq,
{
    const TYPE_ID: &'static str = COMPOSITE_COEFFICIENT_MAP_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = endpoint_bytes(self.source(), self.target())?;
        append_component(&mut output, &self.inner.canonical_bytes()?)?;
        append_component(&mut output, &self.outer.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &COMPOSITE_COEFFICIENT_MAP_DESCRIPTION
    }
}

impl<M, N> fmt::Display for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap,
    N: CoefficientMap<Source = M::Target>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source(), self.target())
    }
}

impl<M, N> fmt::Debug for CompositeCoefficientMap<M, N>
where
    M: CoefficientMap + fmt::Debug,
    N: CoefficientMap<Source = M::Target> + fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompositeCoefficientMap")
            .field("inner", &self.inner)
            .field("outer", &self.outer)
            .finish()
    }
}

fn singleton_to_json(type_id: &str) -> Result<String, NoetherError> {
    serde_json::to_string(&SingletonMapWire {
        type_id: type_id.to_owned(),
    })
    .map_err(|error| NoetherError::Serialization(error.to_string()))
}

fn singleton_from_json(value: &str, type_id: &str) -> Result<(), NoetherError> {
    let wire = serde_json::from_str::<SingletonMapWire>(value)
        .map_err(|error| NoetherError::Serialization(error.to_string()))?;
    require_type(type_id, &wire.type_id)
}

fn endpoint_bytes<S: SemanticType, T: SemanticType>(
    source: &S,
    target: &T,
) -> Result<Vec<u8>, NoetherError> {
    let mut output = Vec::new();
    append_component(&mut output, &source.canonical_bytes()?)?;
    append_component(&mut output, &target.canonical_bytes()?)?;
    Ok(output)
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("coefficient-map component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

fn require_type(expected: &str, actual: &str) -> Result<(), NoetherError> {
    if expected == actual {
        Ok(())
    } else {
        Err(NoetherError::Serialization(format!(
            "expected type {expected}, received {actual}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;

    #[test]
    fn canonical_integer_maps_apply_and_round_trip() {
        let rational = IntegerRationalEmbedding::new();
        assert_eq!(rational.apply(&Integer::from(-7)).unwrap(), Q.element(-7));
        assert_eq!(
            IntegerRationalEmbedding::from_json(&rational.to_json().unwrap()).unwrap(),
            rational
        );

        let residues = IntegerReduction::new(ResidueRing::new(12_u8).unwrap());
        assert_eq!(
            residues.apply(&Integer::from(-7)).unwrap().representative(),
            &Integer::from(5)
        );
        assert_eq!(
            IntegerReduction::from_json(&residues.to_json().unwrap()).unwrap(),
            residues
        );

        let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
        let finite = IntegerFiniteFieldMap::new(extension.clone());
        assert_eq!(
            finite.apply(&Integer::from(12)).unwrap(),
            extension.element(2)
        );
        assert_eq!(
            IntegerFiniteFieldMap::from_json(&finite.to_json().unwrap()).unwrap(),
            finite
        );
    }

    #[test]
    fn prime_subfield_embedding_checks_both_endpoints() {
        let prime = FiniteField::prime(5_u8).unwrap();
        let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
        let embedding = PrimeSubfieldEmbedding::new(prime.clone(), extension.clone()).unwrap();
        assert_eq!(
            embedding.apply(&prime.element(3)).unwrap(),
            extension.element(3)
        );
        assert_eq!(
            PrimeSubfieldEmbedding::from_json(&embedding.to_json().unwrap()).unwrap(),
            embedding
        );

        assert!(matches!(
            PrimeSubfieldEmbedding::new(extension, prime),
            Err(PrimeSubfieldEmbeddingError::SourceIsExtension { degree: 2 })
        ));
        assert!(matches!(
            PrimeSubfieldEmbedding::new(
                FiniteField::prime(5_u8).unwrap(),
                FiniteField::prime(7_u8).unwrap()
            ),
            Err(PrimeSubfieldEmbeddingError::CharacteristicMismatch { .. })
        ));
    }

    #[test]
    fn identity_and_composition_preserve_checked_contexts() {
        let residues = ResidueRing::new(11_u8).unwrap();
        let reduction = IntegerReduction::new(residues.clone());
        let identity = IdentityCoefficientMap::new(residues.clone());
        let composite = CompositeCoefficientMap::new(reduction, identity).unwrap();
        assert_eq!(
            composite.apply(&Integer::from(25)).unwrap(),
            residues.element(3)
        );
        composite.verify().unwrap();
        type Composite =
            CompositeCoefficientMap<IntegerReduction, IdentityCoefficientMap<ResidueRing>>;
        assert_eq!(
            Composite::from_json(&composite.to_json().unwrap()).unwrap(),
            composite
        );

        let other = IdentityCoefficientMap::new(ResidueRing::new(13_u8).unwrap());
        assert!(CompositeCoefficientMap::new(IntegerReduction::new(residues), other).is_err());
    }

    #[test]
    fn contexts_construct_their_canonical_maps() {
        let residues = ResidueRing::new(17_u8).unwrap();
        assert_eq!(Z.rational_embedding(), IntegerRationalEmbedding::new());
        assert_eq!(Z.reduction(&residues).target(), &residues);

        let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
        assert_eq!(Z.finite_field_map(&extension).target(), &extension);
        let embedding = extension.prime_subfield_embedding();
        assert_eq!(embedding.target(), &extension);
        assert_eq!(embedding.source(), &FiniteField::prime(5_u8).unwrap());
    }
}
