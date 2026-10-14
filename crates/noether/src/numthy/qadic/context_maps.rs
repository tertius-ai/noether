// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked maps between fixed-cap q-adic contexts.

use core::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::*;
use crate::commalg::{Polynomial, PowerSeries};
use crate::linalg::{Matrix, Vector};
use crate::{CoefficientMap, CoefficientMappingError, Ring, VerificationError};

static QADIC_RING_RESTRICTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicRingRestriction@1",
    rust_path: "noether::numthy::QadicRingRestriction",
    python_path: Some("noether.numthy.QadicRingRestriction"),
};

static QADIC_FIELD_RESTRICTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicFieldRestriction@1",
    rust_path: "noether::numthy::QadicFieldRestriction",
    python_path: Some("noether.numthy.QadicFieldRestriction"),
};

static QADIC_INTEGRAL_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicIntegralEmbedding@1",
    rust_path: "noether::numthy::QadicIntegralEmbedding",
    python_path: Some("noether.numthy.QadicIntegralEmbedding"),
};

/// Failure to embed or retract between a fixed-cap Witt ring and its field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QadicEmbeddingError {
    /// The value belonged to a context other than the declared source or target.
    ContextMismatch(ContextMismatchError),
    /// Retraction was requested for a value of negative valuation.
    NonIntegral {
        /// Exact negative valuation of the rejected field value.
        valuation: i64,
    },
}

impl fmt::Display for QadicEmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::NonIntegral { valuation } => write!(
                formatter,
                "cannot retract a q-adic field value of valuation {valuation} to its integral ring"
            ),
        }
    }
}

impl std::error::Error for QadicEmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::NonIntegral { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for QadicEmbeddingError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to divide by a q-adic uniformizer with an honest target precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QadicUniformizerDivisionError {
    /// The value did not belong to the context performing the division.
    ContextMismatch(ContextMismatchError),
    /// Division would leave no positive absolute precision.
    PrecisionExhausted {
        /// Requested uniformizer exponent.
        exponent: u64,
        /// Source absolute precision.
        available: u64,
    },
    /// Integral-ring division was not certified to remain integral.
    InsufficientValuation {
        /// Minimum valuation required for integral division.
        required: u64,
        /// Available exact valuation or finite-precision lower bound.
        actual: PadicValuation,
    },
}

impl fmt::Display for QadicUniformizerDivisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::PrecisionExhausted {
                exponent,
                available,
            } => write!(
                formatter,
                "cannot divide by p^{exponent} at absolute precision {available}; the target precision must remain positive"
            ),
            Self::InsufficientValuation { required, actual } => write!(
                formatter,
                "cannot certify integral division by p^{required}; the q-adic value has valuation {actual}"
            ),
        }
    }
}

impl std::error::Error for QadicUniformizerDivisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::PrecisionExhausted { .. } | Self::InsufficientValuation { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for QadicUniformizerDivisionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Explicit precision restriction between two fixed-cap Witt rings.
#[derive(Clone, PartialEq, Eq)]
pub struct QadicRingRestriction {
    source: QadicRing,
    target: QadicRing,
}

/// Explicit precision restriction between two fixed-cap unramified fields.
#[derive(Clone, PartialEq, Eq)]
pub struct QadicFieldRestriction {
    source: QadicField,
    target: QadicField,
}

/// Canonical fixed-cap embedding `W(F_q) -> Frac(W(F_q))` and retraction.
#[derive(Clone, PartialEq, Eq)]
pub struct QadicIntegralEmbedding {
    source: QadicRing,
    target: QadicField,
}

struct QadicIntegralRetraction<'a>(&'a QadicIntegralEmbedding);

impl CoefficientMap for QadicRingRestriction {
    type Source = QadicRing;
    type Target = QadicRing;
    type Error = QadicRestrictionError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &QadicElement) -> Result<QadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for QadicFieldRestriction {
    type Source = QadicField;
    type Target = QadicField;
    type Error = QadicRestrictionError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &QadicElement) -> Result<QadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for QadicIntegralEmbedding {
    type Source = QadicRing;
    type Target = QadicField;
    type Error = QadicEmbeddingError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &QadicElement) -> Result<QadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for QadicIntegralRetraction<'_> {
    type Source = QadicField;
    type Target = QadicRing;
    type Error = QadicEmbeddingError;

    fn source(&self) -> &Self::Source {
        self.0.target()
    }

    fn target(&self) -> &Self::Target {
        self.0.source()
    }

    fn apply(&self, value: &QadicElement) -> Result<QadicElement, Self::Error> {
        self.0.retract(value)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicRingRestrictionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: QadicRing,
    target: QadicRing,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicFieldRestrictionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: QadicField,
    target: QadicField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicIntegralEmbeddingWire {
    #[serde(rename = "type")]
    type_id: String,
    source: QadicRing,
    target: QadicField,
}

impl QadicRingRestriction {
    /// Construct the unique digit-discarding map to `target_absolute_precision`.
    pub fn new(
        source: QadicRing,
        target_absolute_precision: u64,
    ) -> Result<Self, QadicRestrictionError> {
        validate_precision(source.absolute_precision(), target_absolute_precision)?;
        let target = QadicRing::new(source.residue_field().clone(), target_absolute_precision)
            .map_err(QadicRestrictionError::Context)?;
        Ok(Self { source, target })
    }

    /// Source fixed-cap Witt ring.
    pub fn source(&self) -> &QadicRing {
        &self.source
    }

    /// Target fixed-cap Witt ring.
    pub fn target(&self) -> &QadicRing {
        &self.target
    }

    /// Discard the high digits of a value from the exact source context.
    pub fn apply(&self, value: &QadicElement) -> Result<QadicElement, QadicRestrictionError> {
        self.source
            .require(value)
            .map_err(QadicRestrictionError::ContextMismatch)?;
        let coefficients = value
            .value
            .integral_coefficients(self.source.residue_degree())
            .expect("Witt-ring values retain nonnegative valuation");
        Ok(self.target.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.target.context),
            &coefficients,
        )))
    }

    /// Recheck the source/target relationship defining this map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_qadic_restriction(
            self.source.residue_field(),
            self.source.absolute_precision(),
            self.target.residue_field(),
            self.target.absolute_precision(),
        )
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&QadicRingRestrictionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both endpoint contexts and their precision relation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<QadicRingRestrictionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        let result = Self {
            source: wire.source,
            target: wire.target,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl QadicFieldRestriction {
    /// Construct the unique digit-discarding map to `target_absolute_precision`.
    pub fn new(
        source: QadicField,
        target_absolute_precision: u64,
    ) -> Result<Self, QadicRestrictionError> {
        validate_precision(source.absolute_precision(), target_absolute_precision)?;
        let target = QadicField::new(source.residue_field().clone(), target_absolute_precision)
            .map_err(QadicRestrictionError::Context)?;
        Ok(Self { source, target })
    }

    /// Source fixed-cap field.
    pub fn source(&self) -> &QadicField {
        &self.source
    }

    /// Target fixed-cap field.
    pub fn target(&self) -> &QadicField {
        &self.target
    }

    /// Discard high digits while preserving the represented valuation when visible.
    pub fn apply(&self, value: &QadicElement) -> Result<QadicElement, QadicRestrictionError> {
        self.source
            .require(value)
            .map_err(QadicRestrictionError::ContextMismatch)?;
        if value.value.raw.length == 0
            || value.value.raw.val
                >= i64::try_from(self.target.absolute_precision()).unwrap_or(i64::MAX)
        {
            return Ok(self.target.zero());
        }
        let unit_precision = i64::try_from(self.target.absolute_precision())
            .expect("q-adic contexts have FLINT-sized precision")
            .checked_sub(value.value.raw.val)
            .and_then(|precision| u64::try_from(precision).ok())
            .filter(|precision| *precision > 0)
            .ok_or_else(|| {
                QadicRestrictionError::Representation(
                    "valuation is incompatible with the target absolute precision".to_owned(),
                )
            })?;
        let modulus = self.target.prime().pow(unit_precision);
        let coefficients = value
            .value
            .unit_coefficients(self.source.residue_degree())
            .into_iter()
            .map(|coefficient| &coefficient % &modulus)
            .collect::<Vec<_>>();
        let value = QadicValue::from_unit_coordinates(
            Arc::clone(&self.target.integral_ring.context),
            &coefficients,
            value.value.raw.val,
        )
        .map_err(QadicRestrictionError::Representation)?;
        Ok(self.target.wrap(value))
    }

    /// Recheck the source/target relationship defining this map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_qadic_restriction(
            self.source.residue_field(),
            self.source.absolute_precision(),
            self.target.residue_field(),
            self.target.absolute_precision(),
        )
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&QadicFieldRestrictionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both endpoint contexts and their precision relation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<QadicFieldRestrictionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        let result = Self {
            source: wire.source,
            target: wire.target,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl QadicIntegralEmbedding {
    /// Construct the canonical embedding attached to a fixed-cap field.
    pub fn new(target: QadicField) -> Self {
        Self {
            source: target.integral_ring().clone(),
            target,
        }
    }

    /// Integral source context.
    pub fn source(&self) -> &QadicRing {
        &self.source
    }

    /// Fraction-field target context.
    pub fn target(&self) -> &QadicField {
        &self.target
    }

    /// Embed an integral value after checking its exact source context.
    pub fn apply(&self, value: &QadicElement) -> Result<QadicElement, QadicEmbeddingError> {
        self.source.require(value)?;
        let coefficients = value
            .value
            .integral_coefficients(self.source.residue_degree())
            .expect("Witt-ring values retain nonnegative valuation");
        Ok(self.target.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.target.integral_ring.context),
            &coefficients,
        )))
    }

    /// Retract a field value after certifying nonnegative valuation.
    pub fn retract(&self, value: &QadicElement) -> Result<QadicElement, QadicEmbeddingError> {
        self.target.require(value)?;
        if let PadicValuation::Exact(valuation) = value.valuation() {
            if valuation < 0 {
                return Err(QadicEmbeddingError::NonIntegral { valuation });
            }
        }
        let coefficients = value
            .value
            .integral_coefficients(self.source.residue_degree())
            .expect("a certified integral q-adic field value has integral coordinates");
        Ok(self.source.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.source.context),
            &coefficients,
        )))
    }

    /// Retract every coefficient of a polynomial after certifying integrality.
    pub fn retract_polynomial(
        &self,
        value: &Polynomial<QadicField>,
    ) -> Result<Polynomial<QadicRing>, CoefficientMappingError> {
        value.map_coefficients(&QadicIntegralRetraction(self))
    }

    /// Retract every retained series coefficient after certifying integrality.
    pub fn retract_power_series(
        &self,
        value: &PowerSeries<QadicField>,
    ) -> Result<PowerSeries<QadicRing>, CoefficientMappingError> {
        value.map_coefficients(&QadicIntegralRetraction(self))
    }

    /// Retract every vector coefficient after certifying integrality.
    pub fn retract_vector(
        &self,
        value: &Vector<QadicField>,
    ) -> Result<Vector<QadicRing>, CoefficientMappingError> {
        value.map_coefficients(&QadicIntegralRetraction(self))
    }

    /// Retract every matrix coefficient after certifying integrality.
    pub fn retract_matrix(
        &self,
        value: &Matrix<QadicField>,
    ) -> Result<Matrix<QadicRing>, CoefficientMappingError> {
        value.map_coefficients(&QadicIntegralRetraction(self))
    }

    /// Recheck that source and target have identical presentation and precision identity.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if &self.source == self.target.integral_ring() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "q-adic integral embedding endpoints share one fixed-cap context",
            })
        }
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&QadicIntegralEmbeddingWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the embedding endpoints.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<QadicIntegralEmbeddingWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        require_type(Self::TYPE_ID, &wire.type_id)?;
        let result = Self {
            source: wire.source,
            target: wire.target,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl QadicRing {
    /// Construct a first-class restriction to a coarser fixed-cap Witt ring.
    pub fn precision_restriction(
        &self,
        target_absolute_precision: u64,
    ) -> Result<QadicRingRestriction, QadicRestrictionError> {
        QadicRingRestriction::new(self.clone(), target_absolute_precision)
    }

    /// Divide by `p^exponent`, requiring the quotient to remain integral.
    pub fn divide_uniformizer_exact(
        &self,
        value: &QadicElement,
        exponent: u64,
    ) -> Result<QadicElement, QadicUniformizerDivisionError> {
        self.require(value)?;
        let target_precision = remaining_precision(self.absolute_precision(), exponent)?;
        let target = QadicRing::new(self.residue_field().clone(), target_precision)
            .expect("a lower positive precision is valid for an existing q-adic ring");
        if value.value.raw.length == 0 {
            return Ok(target.zero());
        }
        let valuation = value.valuation();
        if !matches!(valuation, PadicValuation::Exact(actual) if actual >= exponent as i64) {
            return Err(QadicUniformizerDivisionError::InsufficientValuation {
                required: exponent,
                actual: valuation,
            });
        }
        let exponent = i64::try_from(exponent).map_err(|_| {
            QadicUniformizerDivisionError::PrecisionExhausted {
                exponent,
                available: self.absolute_precision(),
            }
        })?;
        let quotient = QadicValue::from_unit_coordinates(
            Arc::clone(&target.context),
            &value.value.unit_coefficients(self.residue_degree()),
            value.value.raw.val - exponent,
        )
        .expect("integral q-adic division preserves the source unit precision");
        Ok(target.wrap(quotient))
    }
}

impl QadicField {
    /// Construct a first-class restriction to a coarser fixed-cap field.
    pub fn precision_restriction(
        &self,
        target_absolute_precision: u64,
    ) -> Result<QadicFieldRestriction, QadicRestrictionError> {
        QadicFieldRestriction::new(self.clone(), target_absolute_precision)
    }

    /// Canonical embedding of this field's fixed-cap Witt ring.
    pub fn integral_embedding(&self) -> QadicIntegralEmbedding {
        QadicIntegralEmbedding::new(self.clone())
    }

    /// Divide by `p^exponent` and lower absolute precision by `exponent`.
    pub fn divide_uniformizer_exact(
        &self,
        value: &QadicElement,
        exponent: u64,
    ) -> Result<QadicElement, QadicUniformizerDivisionError> {
        self.require(value)?;
        let target_precision = remaining_precision(self.absolute_precision(), exponent)?;
        let target = QadicField::new(self.residue_field().clone(), target_precision)
            .expect("a lower positive precision is valid for an existing q-adic field");
        if value.value.raw.length == 0 {
            return Ok(target.zero());
        }
        let exponent = i64::try_from(exponent).map_err(|_| {
            QadicUniformizerDivisionError::PrecisionExhausted {
                exponent,
                available: self.absolute_precision(),
            }
        })?;
        let quotient = QadicValue::from_unit_coordinates(
            Arc::clone(&target.integral_ring.context),
            &value.value.unit_coefficients(self.residue_degree()),
            value.value.raw.val - exponent,
        )
        .expect("q-adic field division preserves the source unit precision");
        Ok(target.wrap(quotient))
    }
}

fn validate_precision(source: u64, target: u64) -> Result<(), QadicRestrictionError> {
    if target == 0 || target > source {
        Err(QadicRestrictionError::InvalidPrecision {
            requested: target,
            available: source,
        })
    } else {
        Ok(())
    }
}

fn remaining_precision(
    available: u64,
    exponent: u64,
) -> Result<u64, QadicUniformizerDivisionError> {
    available
        .checked_sub(exponent)
        .filter(|value| *value > 0)
        .ok_or(QadicUniformizerDivisionError::PrecisionExhausted {
            exponent,
            available,
        })
}

fn verify_qadic_restriction(
    source_residue_field: &FiniteField,
    source_precision: u64,
    target_residue_field: &FiniteField,
    target_precision: u64,
) -> Result<(), VerificationError> {
    if source_residue_field == target_residue_field
        && target_precision > 0
        && target_precision <= source_precision
    {
        Ok(())
    } else {
        Err(VerificationError::InvalidWitness {
            claim:
                "q-adic restriction preserves the residue presentation and only discards precision",
        })
    }
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

fn endpoint_bytes<A: SemanticType, B: SemanticType>(
    source: &A,
    target: &B,
) -> Result<Vec<u8>, NoetherError> {
    let source = source.canonical_bytes()?;
    let target = target.canonical_bytes()?;
    let mut output = Vec::with_capacity(16 + source.len() + target.len());
    output.extend_from_slice(&(source.len() as u64).to_be_bytes());
    output.extend_from_slice(&source);
    output.extend_from_slice(&(target.len() as u64).to_be_bytes());
    output.extend_from_slice(&target);
    Ok(output)
}

macro_rules! impl_map_semantics {
    ($type:ty, $id:literal, $description:ident) => {
        impl SemanticType for $type {
            const TYPE_ID: &'static str = $id;

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                endpoint_bytes(self.source(), self.target())
            }

            fn describe() -> &'static TypeDescription {
                &$description
            }
        }

        impl fmt::Debug for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($type))
                    .field("source", self.source())
                    .field("target", self.target())
                    .finish()
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{} -> {}", self.source(), self.target())
            }
        }
    };
}

impl_map_semantics!(
    QadicRingRestriction,
    "noether/numthy/QadicRingRestriction@1",
    QADIC_RING_RESTRICTION_DESCRIPTION
);
impl_map_semantics!(
    QadicFieldRestriction,
    "noether/numthy/QadicFieldRestriction@1",
    QADIC_FIELD_RESTRICTION_DESCRIPTION
);
impl_map_semantics!(
    QadicIntegralEmbedding,
    "noether/numthy/QadicIntegralEmbedding@1",
    QADIC_INTEGRAL_EMBEDDING_DESCRIPTION
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{PolynomialRing, PowerSeriesRing};

    fn residue_field() -> FiniteField {
        FiniteField::extension(3_u64, 2).unwrap()
    }

    #[test]
    fn ring_and_field_restrictions_preserve_the_residue_presentation() {
        let ring = QadicRing::new(residue_field(), 6).unwrap();
        let map = ring.precision_restriction(3).unwrap();
        let value = ring
            .add(&ring.generator(), &ring.element(3_u8.pow(4)))
            .unwrap();
        let restricted = map.apply(&value).unwrap();
        assert_eq!(restricted.ring(), Some(map.target()));
        assert_eq!(restricted.residue(), value.residue());
        assert_eq!(
            QadicRingRestriction::from_json(&map.to_json().unwrap()).unwrap(),
            map
        );

        let field = QadicField::new(residue_field(), 6).unwrap();
        let field_map = field.precision_restriction(3).unwrap();
        let restricted_generator = field_map.apply(&field.generator()).unwrap();
        assert_eq!(restricted_generator.field(), Some(field_map.target()));
        assert_eq!(restricted_generator.residue(), field.generator().residue());
        let inverse = field.inverse(&field.element(3)).unwrap();
        let restricted_inverse = field_map.apply(&inverse).unwrap();
        assert_eq!(restricted_inverse.field(), Some(field_map.target()));
        assert_eq!(restricted_inverse.valuation(), PadicValuation::Exact(-1));
    }

    #[test]
    fn integral_embedding_retracts_exactly_the_integral_values() {
        let field = QadicField::new(residue_field(), 5).unwrap();
        let embedding = field.integral_embedding();
        let source = embedding
            .source()
            .mul(
                &embedding.source().generator(),
                &embedding.source().element(3),
            )
            .unwrap();
        let image = embedding.apply(&source).unwrap();
        assert_eq!(embedding.retract(&image).unwrap(), source);
        let nonintegral = field.inverse(&field.element(3)).unwrap();
        assert!(matches!(
            embedding.retract(&nonintegral),
            Err(QadicEmbeddingError::NonIntegral { valuation: -1 })
        ));
        assert_eq!(
            QadicIntegralEmbedding::from_json(&embedding.to_json().unwrap()).unwrap(),
            embedding
        );
    }

    #[test]
    fn uniformizer_division_lowers_precision_without_losing_unit_coordinates() {
        let ring = QadicRing::new(residue_field(), 6).unwrap();
        let value = ring
            .mul(&ring.generator(), &ring.element(3_u8.pow(3)))
            .unwrap();
        let quotient = ring.divide_uniformizer_exact(&value, 2).unwrap();
        assert_eq!(quotient.ring().unwrap().absolute_precision(), 4);
        assert_eq!(quotient.valuation(), PadicValuation::Exact(1));
        assert_eq!(quotient.residue(), Some(ring.residue_field().zero()));

        let field = QadicField::new(residue_field(), 6).unwrap();
        let negative = field.inverse(&field.element(3)).unwrap();
        let divided = field.divide_uniformizer_exact(&negative, 2).unwrap();
        assert_eq!(divided.field().unwrap().absolute_precision(), 4);
        assert_eq!(divided.valuation(), PadicValuation::Exact(-3));
    }

    #[test]
    fn qadic_composite_maps_preserve_presentations_shapes_and_truncations() {
        let ring = QadicRing::new(residue_field(), 5).unwrap();
        let restriction = ring.precision_restriction(3).unwrap();
        let polynomial_ring = PolynomialRing::new(
            ring.clone(),
            ["x", "y"],
            crate::commalg::MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let polynomial = polynomial_ring
            .add(
                &polynomial_ring.generator(0).unwrap(),
                &polynomial_ring.constant(ring.generator()).unwrap(),
            )
            .unwrap();
        let mapped_polynomial = polynomial.map_coefficients(&restriction).unwrap();
        assert_eq!(
            mapped_polynomial.ring().coefficient_ring(),
            restriction.target()
        );
        assert_eq!(
            mapped_polynomial.ring().term_order(),
            polynomial_ring.term_order()
        );

        let series_ring = PowerSeriesRing::with_grading(
            ring.clone(),
            ["z", "w"],
            crate::commalg::WeightedGrading::new(vec![1, 2]).unwrap(),
            6,
        )
        .unwrap();
        let series = series_ring
            .add(
                &series_ring.generator(1).unwrap(),
                &series_ring.constant(ring.generator()).unwrap(),
            )
            .unwrap();
        let mapped_series = series.map_coefficients(&restriction).unwrap();
        assert_eq!(mapped_series.ring().truncation(), series_ring.truncation());
        assert_eq!(
            mapped_series.ring().coefficient_ring(),
            restriction.target()
        );

        let matrix = Matrix::from_rows(
            ring.clone(),
            [
                [ring.generator(), ring.one()],
                [ring.zero(), ring.element(3)],
            ],
        )
        .unwrap();
        let mapped_matrix = matrix.map_coefficients(&restriction).unwrap();
        assert_eq!(mapped_matrix.ring(), restriction.target());
        assert_eq!(mapped_matrix.shape(), (2, 2));

        let field = QadicField::new(residue_field(), 5).unwrap();
        let embedding = field.integral_embedding();
        let field_matrix = matrix.map_coefficients(&embedding).unwrap();
        assert_eq!(embedding.retract_matrix(&field_matrix).unwrap(), matrix);
        let field_series = series.map_coefficients(&embedding).unwrap();
        assert_eq!(
            embedding.retract_power_series(&field_series).unwrap(),
            series
        );
    }
}
