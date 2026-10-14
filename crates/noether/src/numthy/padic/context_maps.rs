// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked maps between fixed-cap p-adic contexts.

use core::fmt;

use serde::{Deserialize, Serialize};

use super::*;
use crate::commalg::{Polynomial, PowerSeries};
use crate::linalg::{Matrix, Vector};
use crate::{CoefficientMap, CoefficientMappingError, Ring, VerificationError};

static PADIC_RING_RESTRICTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicRingRestriction@1",
    rust_path: "noether::numthy::PadicRingRestriction",
    python_path: Some("noether.numthy.PadicRingRestriction"),
};

static PADIC_FIELD_RESTRICTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicFieldRestriction@1",
    rust_path: "noether::numthy::PadicFieldRestriction",
    python_path: Some("noether.numthy.PadicFieldRestriction"),
};

static PADIC_INTEGRAL_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicIntegralEmbedding@1",
    rust_path: "noether::numthy::PadicIntegralEmbedding",
    python_path: Some("noether.numthy.PadicIntegralEmbedding"),
};

/// Failure to embed or retract between fixed-cap `Z_p` and `Q_p` contexts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicEmbeddingError {
    /// The value belonged to a context other than the declared source or target.
    ContextMismatch(ContextMismatchError),
    /// Retraction was requested for a value of negative valuation.
    NonIntegral {
        /// Exact negative valuation of the rejected field value.
        valuation: i64,
    },
}

impl fmt::Display for PadicEmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::NonIntegral { valuation } => write!(
                formatter,
                "cannot retract a p-adic field value of valuation {valuation} to its integral ring"
            ),
        }
    }
}

impl std::error::Error for PadicEmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::NonIntegral { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PadicEmbeddingError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to divide by a p-adic uniformizer with an honest target precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicUniformizerDivisionError {
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

impl fmt::Display for PadicUniformizerDivisionError {
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
                "cannot certify integral division by p^{required}; the value has valuation {actual}"
            ),
        }
    }
}

impl std::error::Error for PadicUniformizerDivisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::PrecisionExhausted { .. } | Self::InsufficientValuation { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PadicUniformizerDivisionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Explicit precision restriction between two fixed-cap p-adic rings.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicRingRestriction {
    source: PadicRing,
    target: PadicRing,
}

/// Explicit precision restriction between two fixed-cap p-adic fields.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicFieldRestriction {
    source: PadicField,
    target: PadicField,
}

/// Canonical fixed-cap embedding `Z_p -> Q_p` and checked integral retraction.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicIntegralEmbedding {
    source: PadicRing,
    target: PadicField,
}

struct PadicIntegralRetraction<'a>(&'a PadicIntegralEmbedding);

impl CoefficientMap for PadicRingRestriction {
    type Source = PadicRing;
    type Target = PadicRing;
    type Error = PadicRestrictionError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &PadicElement) -> Result<PadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for PadicFieldRestriction {
    type Source = PadicField;
    type Target = PadicField;
    type Error = PadicRestrictionError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &PadicElement) -> Result<PadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for PadicIntegralEmbedding {
    type Source = PadicRing;
    type Target = PadicField;
    type Error = PadicEmbeddingError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(&self, value: &PadicElement) -> Result<PadicElement, Self::Error> {
        self.apply(value)
    }
}

impl CoefficientMap for PadicIntegralRetraction<'_> {
    type Source = PadicField;
    type Target = PadicRing;
    type Error = PadicEmbeddingError;

    fn source(&self) -> &Self::Source {
        self.0.target()
    }

    fn target(&self) -> &Self::Target {
        self.0.source()
    }

    fn apply(&self, value: &PadicElement) -> Result<PadicElement, Self::Error> {
        self.0.retract(value)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicRingRestrictionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: PadicRing,
    target: PadicRing,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicFieldRestrictionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: PadicField,
    target: PadicField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicIntegralEmbeddingWire {
    #[serde(rename = "type")]
    type_id: String,
    source: PadicRing,
    target: PadicField,
}

impl PadicRingRestriction {
    /// Construct the unique digit-discarding map to `target_absolute_precision`.
    pub fn new(
        source: PadicRing,
        target_absolute_precision: u64,
    ) -> Result<Self, PadicRestrictionError> {
        validate_precision(source.absolute_precision(), target_absolute_precision)?;
        let target =
            PadicRing::new(source.prime().clone(), target_absolute_precision).map_err(|_| {
                PadicRestrictionError::Precision(PadicPrecisionError {
                    requested: target_absolute_precision,
                    available: source.absolute_precision(),
                })
            })?;
        Ok(Self { source, target })
    }

    /// Source fixed-cap ring.
    pub fn source(&self) -> &PadicRing {
        &self.source
    }

    /// Target fixed-cap ring.
    pub fn target(&self) -> &PadicRing {
        &self.target
    }

    /// Discard the high digits of a value from the exact source context.
    pub fn apply(&self, value: &PadicElement) -> Result<PadicElement, PadicRestrictionError> {
        self.source.require(value)?;
        Ok(self
            .target
            .element(self.source.canonicalize(&value.value.integer())))
    }

    /// Recheck the source/target relationship defining this map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_padic_restriction(
            self.source.prime(),
            self.source.absolute_precision(),
            self.target.prime(),
            self.target.absolute_precision(),
        )
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PadicRingRestrictionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both endpoint contexts and their precision relation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PadicRingRestrictionWire>(value)
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

impl PadicFieldRestriction {
    /// Construct the unique digit-discarding map to `target_absolute_precision`.
    pub fn new(
        source: PadicField,
        target_absolute_precision: u64,
    ) -> Result<Self, PadicRestrictionError> {
        validate_precision(source.absolute_precision(), target_absolute_precision)?;
        let target =
            PadicField::new(source.prime().clone(), target_absolute_precision).map_err(|_| {
                PadicRestrictionError::Precision(PadicPrecisionError {
                    requested: target_absolute_precision,
                    available: source.absolute_precision(),
                })
            })?;
        Ok(Self { source, target })
    }

    /// Source fixed-cap field.
    pub fn source(&self) -> &PadicField {
        &self.source
    }

    /// Target fixed-cap field.
    pub fn target(&self) -> &PadicField {
        &self.target
    }

    /// Discard high digits while preserving the represented valuation when visible.
    pub fn apply(&self, value: &PadicElement) -> Result<PadicElement, PadicRestrictionError> {
        self.source.require(value)?;
        if value.value.is_zero()
            || value.value.raw.v
                >= i64::try_from(self.target.absolute_precision()).unwrap_or(i64::MAX)
        {
            return Ok(self.target.zero());
        }
        Ok(self
            .target
            .element_from_unit_and_valuation(&value.value.unit(), value.value.raw.v))
    }

    /// Recheck the source/target relationship defining this map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_padic_restriction(
            self.source.prime(),
            self.source.absolute_precision(),
            self.target.prime(),
            self.target.absolute_precision(),
        )
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PadicFieldRestrictionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both endpoint contexts and their precision relation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PadicFieldRestrictionWire>(value)
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

impl PadicIntegralEmbedding {
    /// Construct the canonical embedding attached to a fixed-cap field.
    pub fn new(target: PadicField) -> Self {
        Self {
            source: target.integral_ring().clone(),
            target,
        }
    }

    /// Integral source context.
    pub fn source(&self) -> &PadicRing {
        &self.source
    }

    /// Fraction-field target context.
    pub fn target(&self) -> &PadicField {
        &self.target
    }

    /// Embed an integral value after checking its exact source context.
    pub fn apply(&self, value: &PadicElement) -> Result<PadicElement, PadicEmbeddingError> {
        self.source.require(value)?;
        Ok(self.target.element(value.value.integer()))
    }

    /// Retract a field value after certifying nonnegative valuation.
    pub fn retract(&self, value: &PadicElement) -> Result<PadicElement, PadicEmbeddingError> {
        self.target.require(value)?;
        if let PadicValuation::Exact(valuation) = value.valuation() {
            if valuation < 0 {
                return Err(PadicEmbeddingError::NonIntegral { valuation });
            }
        }
        Ok(self.source.element(value.value.integer()))
    }

    /// Retract every coefficient of a polynomial after certifying integrality.
    pub fn retract_polynomial(
        &self,
        value: &Polynomial<PadicField>,
    ) -> Result<Polynomial<PadicRing>, CoefficientMappingError> {
        value.map_coefficients(&PadicIntegralRetraction(self))
    }

    /// Retract every retained series coefficient after certifying integrality.
    pub fn retract_power_series(
        &self,
        value: &PowerSeries<PadicField>,
    ) -> Result<PowerSeries<PadicRing>, CoefficientMappingError> {
        value.map_coefficients(&PadicIntegralRetraction(self))
    }

    /// Retract every vector coefficient after certifying integrality.
    pub fn retract_vector(
        &self,
        value: &Vector<PadicField>,
    ) -> Result<Vector<PadicRing>, CoefficientMappingError> {
        value.map_coefficients(&PadicIntegralRetraction(self))
    }

    /// Retract every matrix coefficient after certifying integrality.
    pub fn retract_matrix(
        &self,
        value: &Matrix<PadicField>,
    ) -> Result<Matrix<PadicRing>, CoefficientMappingError> {
        value.map_coefficients(&PadicIntegralRetraction(self))
    }

    /// Recheck that source and target have identical prime and precision identity.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if &self.source == self.target.integral_ring() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "p-adic integral embedding endpoints share one fixed-cap context",
            })
        }
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PadicIntegralEmbeddingWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the embedding endpoints.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PadicIntegralEmbeddingWire>(value)
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

impl PadicRing {
    /// Construct a first-class restriction to a coarser fixed-cap ring.
    pub fn precision_restriction(
        &self,
        target_absolute_precision: u64,
    ) -> Result<PadicRingRestriction, PadicRestrictionError> {
        PadicRingRestriction::new(self.clone(), target_absolute_precision)
    }

    /// Divide by `p^exponent`, requiring the quotient to remain integral.
    pub fn divide_uniformizer_exact(
        &self,
        value: &PadicElement,
        exponent: u64,
    ) -> Result<PadicElement, PadicUniformizerDivisionError> {
        self.require(value)?;
        let target_precision = remaining_precision(self.absolute_precision(), exponent)?;
        let target = PadicRing::new(self.prime().clone(), target_precision)
            .expect("a lower positive precision is valid for an existing p-adic ring");
        if value.value.is_zero() {
            return Ok(target.zero());
        }
        let valuation = value.valuation();
        if !matches!(valuation, PadicValuation::Exact(actual) if actual >= exponent as i64) {
            return Err(PadicUniformizerDivisionError::InsufficientValuation {
                required: exponent,
                actual: valuation,
            });
        }
        let divisor = self.prime().pow(exponent);
        Ok(target.element(&value.value.integer() / &divisor))
    }
}

impl PadicField {
    /// Construct a first-class restriction to a coarser fixed-cap field.
    pub fn precision_restriction(
        &self,
        target_absolute_precision: u64,
    ) -> Result<PadicFieldRestriction, PadicRestrictionError> {
        PadicFieldRestriction::new(self.clone(), target_absolute_precision)
    }

    /// Canonical embedding of this field's fixed-cap integral ring.
    pub fn integral_embedding(&self) -> PadicIntegralEmbedding {
        PadicIntegralEmbedding::new(self.clone())
    }

    /// Divide by `p^exponent` and lower absolute precision by `exponent`.
    pub fn divide_uniformizer_exact(
        &self,
        value: &PadicElement,
        exponent: u64,
    ) -> Result<PadicElement, PadicUniformizerDivisionError> {
        self.require(value)?;
        let target_precision = remaining_precision(self.absolute_precision(), exponent)?;
        let target = PadicField::new(self.prime().clone(), target_precision)
            .expect("a lower positive precision is valid for an existing p-adic field");
        if value.value.is_zero() {
            return Ok(target.zero());
        }
        let exponent = i64::try_from(exponent).map_err(|_| {
            PadicUniformizerDivisionError::PrecisionExhausted {
                exponent,
                available: self.absolute_precision(),
            }
        })?;
        Ok(target
            .element_from_unit_and_valuation(&value.value.unit(), value.value.raw.v - exponent))
    }
}

fn validate_precision(source: u64, target: u64) -> Result<(), PadicRestrictionError> {
    if target == 0 || target > source {
        Err(PadicPrecisionError {
            requested: target,
            available: source,
        }
        .into())
    } else {
        Ok(())
    }
}

fn remaining_precision(
    available: u64,
    exponent: u64,
) -> Result<u64, PadicUniformizerDivisionError> {
    available
        .checked_sub(exponent)
        .filter(|value| *value > 0)
        .ok_or(PadicUniformizerDivisionError::PrecisionExhausted {
            exponent,
            available,
        })
}

fn verify_padic_restriction(
    source_prime: &Integer,
    source_precision: u64,
    target_prime: &Integer,
    target_precision: u64,
) -> Result<(), VerificationError> {
    if source_prime == target_prime && target_precision > 0 && target_precision <= source_precision
    {
        Ok(())
    } else {
        Err(VerificationError::InvalidWitness {
            claim: "p-adic restriction preserves the prime and only discards precision",
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
    PadicRingRestriction,
    "noether/numthy/PadicRingRestriction@1",
    PADIC_RING_RESTRICTION_DESCRIPTION
);
impl_map_semantics!(
    PadicFieldRestriction,
    "noether/numthy/PadicFieldRestriction@1",
    PADIC_FIELD_RESTRICTION_DESCRIPTION
);
impl_map_semantics!(
    PadicIntegralEmbedding,
    "noether/numthy/PadicIntegralEmbedding@1",
    PADIC_INTEGRAL_EMBEDDING_DESCRIPTION
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{PolynomialRing, PowerSeriesRing};

    #[test]
    fn ring_and_field_restrictions_are_explicit_and_revalidated() {
        let ring = PadicRing::new(3_u8, 6).unwrap();
        let ring_map = ring.precision_restriction(3).unwrap();
        let value = ring.element(3_u8.pow(4) + 5);
        let restricted = ring_map.apply(&value).unwrap();
        assert_eq!(restricted.ring(), Some(ring_map.target()));
        assert_eq!(restricted.representative(), Some(Integer::from(5)));
        assert_eq!(
            PadicRingRestriction::from_json(&ring_map.to_json().unwrap()).unwrap(),
            ring_map
        );

        let field = PadicField::new(3_u8, 6).unwrap();
        let field_map = field.precision_restriction(3).unwrap();
        let inverse = field.inverse(&field.element(3)).unwrap();
        let restricted_inverse = field_map.apply(&inverse).unwrap();
        assert_eq!(restricted_inverse.field(), Some(field_map.target()));
        assert_eq!(restricted_inverse.valuation(), PadicValuation::Exact(-1));
        assert_eq!(
            PadicFieldRestriction::from_json(&field_map.to_json().unwrap()).unwrap(),
            field_map
        );
    }

    #[test]
    fn integral_embedding_retracts_exactly_the_integral_values() {
        let field = PadicField::new(5_u8, 5).unwrap();
        let embedding = field.integral_embedding();
        let source = embedding.source().element(-25);
        let image = embedding.apply(&source).unwrap();
        assert_eq!(embedding.retract(&image).unwrap(), source);
        let nonintegral = field.inverse(&field.element(5)).unwrap();
        assert!(matches!(
            embedding.retract(&nonintegral),
            Err(PadicEmbeddingError::NonIntegral { valuation: -1 })
        ));
        assert_eq!(
            PadicIntegralEmbedding::from_json(&embedding.to_json().unwrap()).unwrap(),
            embedding
        );
    }

    #[test]
    fn uniformizer_division_lowers_precision_and_checks_integrality() {
        let ring = PadicRing::new(3_u8, 6).unwrap();
        let quotient = ring
            .divide_uniformizer_exact(&ring.element(3_u8.pow(3) * 7), 2)
            .unwrap();
        assert_eq!(quotient.ring().unwrap().absolute_precision(), 4);
        assert_eq!(quotient.valuation(), PadicValuation::Exact(1));
        assert!(matches!(
            ring.divide_uniformizer_exact(&ring.element(3), 2),
            Err(PadicUniformizerDivisionError::InsufficientValuation { .. })
        ));

        let field = PadicField::new(3_u8, 6).unwrap();
        let negative = field.inverse(&field.element(3)).unwrap();
        let divided = field.divide_uniformizer_exact(&negative, 2).unwrap();
        assert_eq!(divided.field().unwrap().absolute_precision(), 4);
        assert_eq!(divided.valuation(), PadicValuation::Exact(-3));
    }

    #[test]
    fn checked_maps_lift_through_every_composite_container() {
        let ring = PadicRing::new(5_u8, 5).unwrap();
        let restriction = ring.precision_restriction(3).unwrap();
        let polynomials = PolynomialRing::new(
            ring.clone(),
            ["x"],
            crate::commalg::MonomialOrder::Lexicographic,
        )
        .unwrap();
        let polynomial = polynomials
            .add(
                &polynomials.generator(0).unwrap(),
                &polynomials.constant(ring.element(-1)).unwrap(),
            )
            .unwrap();
        let mapped_polynomial = polynomial.map_coefficients(&restriction).unwrap();
        assert_eq!(
            mapped_polynomial.ring().coefficient_ring(),
            restriction.target()
        );
        assert_eq!(mapped_polynomial.ring().variables(), &["x"]);

        let series_ring = PowerSeriesRing::with_precision(ring.clone(), ["z"], 4).unwrap();
        let series = series_ring
            .add(
                &series_ring.generator(0).unwrap(),
                &series_ring.constant(ring.element(125 + 2)).unwrap(),
            )
            .unwrap();
        let mapped_series = series.map_coefficients(&restriction).unwrap();
        assert_eq!(
            mapped_series.ring().coefficient_ring(),
            restriction.target()
        );
        assert_eq!(
            mapped_series.ring().truncation(),
            series.ring().truncation()
        );

        let vector =
            Vector::from_values(ring.clone(), [ring.element(1), ring.element(-1)]).unwrap();
        let mapped_vector = vector.map_coefficients(&restriction).unwrap();
        assert_eq!(mapped_vector.ring(), restriction.target());
        assert_eq!(mapped_vector.dimension(), 2);
        let empty_matrix = Matrix::zeros(ring.clone(), 0, 3);
        let mapped_matrix = empty_matrix.map_coefficients(&restriction).unwrap();
        assert_eq!(mapped_matrix.ring(), restriction.target());
        assert_eq!(mapped_matrix.shape(), (0, 3));

        let field = PadicField::new(5_u8, 5).unwrap();
        let embedding = field.integral_embedding();
        let image = polynomial.map_coefficients(&embedding).unwrap();
        assert_eq!(embedding.retract_polynomial(&image).unwrap(), polynomial);
        let field_series = series.map_coefficients(&embedding).unwrap();
        assert_eq!(
            embedding.retract_power_series(&field_series).unwrap(),
            series
        );
        let field_vector = vector.map_coefficients(&embedding).unwrap();
        assert_eq!(embedding.retract_vector(&field_vector).unwrap(), vector);

        let bad_ring = PolynomialRing::new(
            field.clone(),
            ["x"],
            crate::commalg::MonomialOrder::Lexicographic,
        )
        .unwrap();
        let bad = bad_ring
            .constant(field.inverse(&field.element(5)).unwrap())
            .unwrap();
        assert!(matches!(
            embedding.retract_polynomial(&bad),
            Err(CoefficientMappingError::Coefficient(_))
        ));
    }
}
