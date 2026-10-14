// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite-precision mixed-characteristic local fields in Eisenstein form.

use core::fmt;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{Polynomial, PolynomialError};
use crate::linalg::{
    FrobeniusSemilinearMap, LinalgError, LinearOperator, Matrix, SemilinearMapError, Vector,
};
use crate::{
    CoefficientMap, CommutativeRing, ContextMismatchError, Integer, NoetherError, PrecisionContext,
    Ring, SemanticType, TypeDescription, UnitError, UnitInversion, ValuedRing, VerificationError,
};

use super::{
    CoefficientFrobeniusError, FiniteField, FiniteFieldElement, PadicElement, PadicField,
    PadicValuation, QadicContextError, QadicElement, QadicField, QadicRestrictionError, QadicRing,
    QadicRingFrobenius,
};

static PADIC_LOCAL_PRECISION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalPrecision@1",
    rust_path: "noether::numthy::PadicLocalPrecision",
    python_path: Some("noether.numthy.PadicLocalPrecision"),
};

static PADIC_LOCAL_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalField@1",
    rust_path: "noether::numthy::PadicLocalField",
    python_path: Some("noether.numthy.PadicLocalField"),
};

static PADIC_LOCAL_FIELD_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalFieldElement@1",
    rust_path: "noether::numthy::PadicLocalFieldElement",
    python_path: Some("noether.numthy.PadicLocalFieldElement"),
};

static PADIC_LOCAL_FIELD_RESTRICTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalFieldRestriction@1",
    rust_path: "noether::numthy::PadicLocalFieldRestriction",
    python_path: Some("noether.numthy.PadicLocalFieldRestriction"),
};

static PADIC_LOCAL_FIELD_COEFFICIENT_FROBENIUS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalFieldCoefficientFrobenius@1",
    rust_path: "noether::numthy::PadicLocalFieldCoefficientFrobenius",
    python_path: Some("noether.numthy.PadicLocalFieldCoefficientFrobenius"),
};

/// Invalid absolute uniformizer precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadicLocalPrecisionError {
    /// A fixed-precision local context must retain at least one uniformizer digit.
    Zero,
    /// The precision cannot be represented by the signed valuation model.
    TooLarge,
}

impl fmt::Display for PadicLocalPrecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str(
                "p-adic local precision must retain at least one absolute uniformizer digit",
            ),
            Self::TooLarge => formatter
                .write_str("p-adic local precision is too large for the signed valuation model"),
        }
    }
}

impl std::error::Error for PadicLocalPrecisionError {}

/// Absolute precision measured in powers of a local uniformizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PadicLocalPrecision {
    absolute_uniformizer_precision: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalPrecisionWire {
    #[serde(rename = "type")]
    type_id: String,
    absolute_uniformizer_precision: u64,
}

impl PadicLocalPrecision {
    /// Construct a positive absolute `pi`-adic precision.
    pub fn new(absolute_uniformizer_precision: u64) -> Result<Self, PadicLocalPrecisionError> {
        if absolute_uniformizer_precision == 0 {
            return Err(PadicLocalPrecisionError::Zero);
        }
        if i64::try_from(absolute_uniformizer_precision).is_err() {
            return Err(PadicLocalPrecisionError::TooLarge);
        }
        Ok(Self {
            absolute_uniformizer_precision,
        })
    }

    /// Number `M` in the precision ideal `(pi^M)`.
    pub const fn absolute_uniformizer_precision(self) -> u64 {
        self.absolute_uniformizer_precision
    }

    /// Canonical typed JSON.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a typed local precision.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(self) -> PadicLocalPrecisionWire {
        PadicLocalPrecisionWire {
            type_id: Self::TYPE_ID.to_owned(),
            absolute_uniformizer_precision: self.absolute_uniformizer_precision,
        }
    }

    fn from_wire(wire: PadicLocalPrecisionWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        Self::new(wire.absolute_uniformizer_precision).map_err(|error| error.to_string())
    }
}

impl Serialize for PadicLocalPrecision {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicLocalPrecision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicLocalPrecisionWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for PadicLocalPrecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "absolute uniformizer precision {}",
            self.absolute_uniformizer_precision
        )
    }
}

impl SemanticType for PadicLocalPrecision {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalPrecision@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.absolute_uniformizer_precision.to_be_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_PRECISION_DESCRIPTION
    }
}

/// Failure to construct or use an Eisenstein-presented p-adic local field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalFieldError {
    /// The supplied polynomial context was not univariate.
    DefiningPolynomialNotUnivariate {
        /// Number of variables in the supplied polynomial context.
        variables: usize,
    },
    /// The defining polynomial was zero or constant.
    DegreeTooSmall,
    /// A polynomial degree or absolute degree did not fit this platform.
    DegreeTooLarge,
    /// The polynomial coefficient context differed from the declared coefficient ring.
    CoefficientContextMismatch(ContextMismatchError),
    /// The defining polynomial was not monic.
    DefiningPolynomialNotMonic,
    /// A nonleading coefficient was not divisible by `p`.
    NonEisensteinCoefficient {
        /// Power of the rejected coefficient.
        degree: usize,
        /// Available valuation evidence.
        valuation: PadicValuation,
    },
    /// The constant coefficient did not have valuation exactly one.
    ConstantCoefficientNotValuationOne {
        /// Available valuation evidence.
        valuation: PadicValuation,
    },
    /// The coefficient ring did not carry the digits required by local precision.
    InsufficientCoefficientPrecision {
        /// Minimum required p-adic coefficient precision.
        required: u64,
        /// Supplied q-adic coefficient precision.
        available: u64,
    },
    /// The display name for the distinguished uniformizer was invalid.
    InvalidUniformizerName,
    /// Too many integral power-basis coordinates were supplied.
    TooManyCoordinates {
        /// Eisenstein degree and power-basis length.
        degree: usize,
        /// Supplied coordinate count.
        actual: usize,
    },
    /// An integral coordinate belonged to another q-adic ring.
    CoordinateContext {
        /// Zero-based power-basis coordinate.
        index: usize,
        /// Underlying context mismatch.
        source: ContextMismatchError,
    },
    /// A scalar belonged to another p-adic or q-adic context.
    ScalarContextMismatch(ContextMismatchError),
    /// Generic polynomial handling failed.
    Polynomial(PolynomialError),
    /// A derived q-adic context could not be constructed.
    QadicContext(QadicContextError),
    /// A staggered coordinate could not be restricted safely.
    QadicRestriction(QadicRestrictionError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for PadicLocalFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefiningPolynomialNotUnivariate { variables } => write!(
                formatter,
                "p-adic local defining polynomial must be univariate; received {variables} variables"
            ),
            Self::DegreeTooSmall => formatter
                .write_str("p-adic local defining polynomial must have positive degree"),
            Self::DegreeTooLarge => formatter
                .write_str("p-adic local field degree does not fit this platform"),
            Self::CoefficientContextMismatch(error) => error.fmt(formatter),
            Self::DefiningPolynomialNotMonic => {
                formatter.write_str("p-adic local defining polynomial must be monic")
            }
            Self::NonEisensteinCoefficient { degree, valuation } => write!(
                formatter,
                "p-adic local defining coefficient of degree {degree} must be divisible by p; valuation is {valuation}"
            ),
            Self::ConstantCoefficientNotValuationOne { valuation } => write!(
                formatter,
                "p-adic local defining constant must have valuation exactly one; valuation is {valuation}"
            ),
            Self::InsufficientCoefficientPrecision {
                required,
                available,
            } => write!(
                formatter,
                "p-adic local precision requires {required} q-adic digits; coefficient ring supplies {available}"
            ),
            Self::InvalidUniformizerName => formatter.write_str(
                "local uniformizer name must be nonempty, at most 64 characters, and contain no control characters",
            ),
            Self::TooManyCoordinates { degree, actual } => write!(
                formatter,
                "p-adic local element has {actual} integral coordinates; field degree over its unramified coefficient field is {degree}"
            ),
            Self::CoordinateContext { index, source } => {
                write!(formatter, "local coordinate {index}: {source}")
            }
            Self::ScalarContextMismatch(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::QadicContext(error) => error.fmt(formatter),
            Self::QadicRestriction(error) => error.fmt(formatter),
            Self::InvalidEncoding(message) => {
                write!(formatter, "invalid p-adic local field encoding: {message}")
            }
        }
    }
}

impl std::error::Error for PadicLocalFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CoefficientContextMismatch(error) => Some(error),
            Self::ScalarContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::QadicContext(error) => Some(error),
            Self::QadicRestriction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolynomialError> for PadicLocalFieldError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<QadicContextError> for PadicLocalFieldError {
    fn from(value: QadicContextError) -> Self {
        Self::QadicContext(value)
    }
}

impl From<QadicRestrictionError> for PadicLocalFieldError {
    fn from(value: QadicRestrictionError) -> Self {
        Self::QadicRestriction(value)
    }
}

/// Failure to restrict a ramified local field or one of its values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalFieldRestrictionError {
    /// The requested target cap was finer than the source cap.
    InvalidPrecision {
        /// Requested absolute uniformizer precision.
        requested: u64,
        /// Available source precision.
        available: u64,
    },
    /// The value did not belong to the declared source field.
    ContextMismatch(ContextMismatchError),
    /// Reconstructing the lower-precision context failed validation.
    ContextConstruction(PadicLocalFieldError),
}

impl fmt::Display for PadicLocalFieldRestrictionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrecision {
                requested,
                available,
            } => write!(
                formatter,
                "cannot restrict local precision {available} to {requested}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::ContextConstruction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PadicLocalFieldRestrictionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::ContextConstruction(error) => Some(error),
            Self::InvalidPrecision { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PadicLocalFieldRestrictionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to compute a local-field trace, norm, or characteristic polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalInvariantError {
    /// The value did not belong to the field performing the operation.
    ContextMismatch(ContextMismatchError),
    /// No positive uniform q-adic coefficient digit survives at this cap.
    InsufficientPrecision {
        /// Available absolute precision in powers of the local uniformizer.
        available: i64,
        /// Ramification index converting local digits to coefficient digits.
        ramification_index: usize,
    },
    /// The available digits did not certify that the element generates the full extension.
    MinimalPolynomialNotCertified {
        /// Absolute degree of the presented extension.
        degree: usize,
        /// Lower bound on the valuation of the power-basis determinant.
        determinant_valuation_at_least: i64,
    },
    /// Dense division-free linear algebra rejected a shape or context.
    LinearAlgebra(LinalgError),
    /// Restriction of scalars could not verify the q-adic presentation.
    RestrictionOfScalars(SemilinearMapError),
}

impl fmt::Display for PadicLocalInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InsufficientPrecision {
                available,
                ramification_index,
            } => write!(
                formatter,
                "local absolute precision {available} leaves no positive coefficient digit at ramification index {ramification_index}"
            ),
            Self::MinimalPolynomialNotCertified {
                degree,
                determinant_valuation_at_least,
            } => write!(
                formatter,
                "available precision does not certify that this element generates the degree-{degree} extension; its power-basis determinant has valuation at least {determinant_valuation_at_least}"
            ),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::RestrictionOfScalars(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PadicLocalInvariantError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::RestrictionOfScalars(error) => Some(error),
            Self::InsufficientPrecision { .. } | Self::MinimalPolynomialNotCertified { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PadicLocalInvariantError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<LinalgError> for PadicLocalInvariantError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<SemilinearMapError> for PadicLocalInvariantError {
    fn from(value: SemilinearMapError) -> Self {
        Self::RestrictionOfScalars(value)
    }
}

/// Failure to construct, apply, or compose coefficient-Frobenius conjugation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalCoefficientFrobeniusError {
    /// A value or composable map had the wrong presented field context.
    ContextMismatch(ContextMismatchError),
    /// Two Frobenius exponents could not be added in `u64`.
    ExponentOverflow {
        /// Exponent of the outer map.
        outer: u64,
        /// Exponent of the inner map.
        inner: u64,
    },
    /// Conjugating the defining Eisenstein presentation failed validation.
    TargetConstruction(PadicLocalFieldError),
    /// A q-adic coordinate Frobenius application failed its context check.
    Coefficient(CoefficientFrobeniusError),
}

impl fmt::Display for PadicLocalCoefficientFrobeniusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::ExponentOverflow { outer, inner } => write!(
                formatter,
                "cannot compose local coefficient Frobenius exponents {outer} and {inner}"
            ),
            Self::TargetConstruction(error) => error.fmt(formatter),
            Self::Coefficient(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PadicLocalCoefficientFrobeniusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::TargetConstruction(error) => Some(error),
            Self::Coefficient(error) => Some(error),
            Self::ExponentOverflow { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for PadicLocalCoefficientFrobeniusError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<CoefficientFrobeniusError> for PadicLocalCoefficientFrobeniusError {
    fn from(value: CoefficientFrobeniusError) -> Self {
        Self::Coefficient(value)
    }
}

struct PadicLocalFieldInner {
    coefficient_ring: QadicRing,
    coefficient_field: QadicField,
    defining_polynomial: Polynomial<QadicRing>,
    defining_coefficients: Vec<QadicElement>,
    precision: PadicLocalPrecision,
    coordinate_precisions: Vec<u64>,
    ramification_index: usize,
    absolute_degree: usize,
    uniformizer_name: String,
}

/// A finite-precision mixed-characteristic local field in Eisenstein form.
///
/// Context identity includes the explicit unramified coefficient presentation,
/// Eisenstein coefficients, and absolute uniformizer precision. The
/// uniformizer name is bounded display metadata and is not mathematical
/// identity. General field arithmetic is added on top of this context without
/// changing its staggered coefficient-precision contract.
#[derive(Clone)]
pub struct PadicLocalField {
    inner: Arc<PadicLocalFieldInner>,
}

/// Explicit checked restriction between two caps of one ramified presentation.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicLocalFieldRestriction {
    source: PadicLocalField,
    target: PadicLocalField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFieldRestrictionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: PadicLocalField,
    target: PadicLocalField,
}

/// Coefficient Frobenius from a presented local field to its conjugate field.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicLocalFieldCoefficientFrobenius {
    source: PadicLocalField,
    target: PadicLocalField,
    exponent: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFieldCoefficientFrobeniusWire {
    #[serde(rename = "type")]
    type_id: String,
    source: PadicLocalField,
    target: PadicLocalField,
    exponent: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFieldWire {
    #[serde(rename = "type")]
    type_id: String,
    coefficient_ring: QadicRing,
    defining_polynomial: Polynomial<QadicRing>,
    precision: PadicLocalPrecision,
    uniformizer_name: String,
}

#[derive(Serialize)]
struct PadicLocalFieldIdentity<'a> {
    coefficient_ring: &'a QadicRing,
    defining_coefficients: &'a [QadicElement],
    precision: PadicLocalPrecision,
}

impl PadicLocalField {
    /// Construct a local field from a monic Eisenstein polynomial over a Witt ring.
    pub fn from_eisenstein(
        coefficient_ring: QadicRing,
        eisenstein_polynomial: Polynomial<QadicRing>,
        precision: PadicLocalPrecision,
        uniformizer_name: Option<String>,
    ) -> Result<Self, PadicLocalFieldError> {
        let variables = eisenstein_polynomial.ring().variables().len();
        if variables != 1 {
            return Err(PadicLocalFieldError::DefiningPolynomialNotUnivariate { variables });
        }
        if eisenstein_polynomial.ring().coefficient_ring() != &coefficient_ring {
            return Err(PadicLocalFieldError::CoefficientContextMismatch(
                ContextMismatchError {
                    expected: coefficient_ring.to_string(),
                    actual: eisenstein_polynomial.ring().coefficient_ring().to_string(),
                },
            ));
        }
        let defining_coefficients = dense_coefficients(&eisenstein_polynomial)?;
        if defining_coefficients.len() < 2 {
            return Err(PadicLocalFieldError::DegreeTooSmall);
        }
        if defining_coefficients.last() != Some(&coefficient_ring.one()) {
            return Err(PadicLocalFieldError::DefiningPolynomialNotMonic);
        }
        let ramification_index = defining_coefficients.len() - 1;
        for (degree, coefficient) in defining_coefficients[..ramification_index]
            .iter()
            .enumerate()
        {
            let valuation = coefficient.valuation();
            if degree == 0 {
                if valuation != PadicValuation::Exact(1) {
                    return Err(PadicLocalFieldError::ConstantCoefficientNotValuationOne {
                        valuation,
                    });
                }
            } else if !matches!(
                valuation,
                PadicValuation::Exact(value) if value >= 1
            ) && !matches!(valuation, PadicValuation::AtLeast(value) if value >= 1)
            {
                return Err(PadicLocalFieldError::NonEisensteinCoefficient { degree, valuation });
            }
        }

        let required_precision = staggered_coefficient_precision(
            precision.absolute_uniformizer_precision(),
            0,
            ramification_index,
        )?;
        if coefficient_ring.absolute_precision() < required_precision {
            return Err(PadicLocalFieldError::InsufficientCoefficientPrecision {
                required: required_precision,
                available: coefficient_ring.absolute_precision(),
            });
        }
        let uniformizer_name =
            uniformizer_name.unwrap_or_else(|| eisenstein_polynomial.ring().variables()[0].clone());
        validate_uniformizer_name(&uniformizer_name)?;
        let residue_degree = coefficient_ring.residue_degree();
        let absolute_degree = ramification_index
            .checked_mul(residue_degree)
            .ok_or(PadicLocalFieldError::DegreeTooLarge)?;
        let coordinate_precisions = (0..ramification_index)
            .map(|index| {
                staggered_coefficient_precision(
                    precision.absolute_uniformizer_precision(),
                    index,
                    ramification_index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coefficient_field = QadicField::new(
            coefficient_ring.residue_field().clone(),
            coefficient_ring.absolute_precision(),
        )?;

        Ok(Self {
            inner: Arc::new(PadicLocalFieldInner {
                coefficient_ring,
                coefficient_field,
                defining_polynomial: eisenstein_polynomial,
                defining_coefficients,
                precision,
                coordinate_precisions,
                ramification_index,
                absolute_degree,
                uniformizer_name,
            }),
        })
    }

    /// Original fixed-cap Witt coefficient ring used by the presentation.
    pub fn coefficient_ring(&self) -> &QadicRing {
        &self.inner.coefficient_ring
    }

    /// Maximal unramified coefficient field in the declared presentation.
    pub fn coefficient_field(&self) -> &QadicField {
        &self.inner.coefficient_field
    }

    /// Explicit finite residue field.
    pub fn residue_field(&self) -> &FiniteField {
        self.coefficient_ring().residue_field()
    }

    /// Residue characteristic `p`.
    pub fn prime(&self) -> &Integer {
        self.coefficient_ring().prime()
    }

    /// Absolute uniformizer precision.
    pub fn local_precision(&self) -> PadicLocalPrecision {
        self.inner.precision
    }

    /// Ramification index over the maximal unramified coefficient field.
    pub fn ramification_index(&self) -> usize {
        self.inner.ramification_index
    }

    /// Residue degree over `Q_p`.
    pub fn residue_degree(&self) -> usize {
        self.coefficient_ring().residue_degree()
    }

    /// Absolute degree over `Q_p`.
    pub fn degree(&self) -> usize {
        self.inner.absolute_degree
    }

    /// Monic Eisenstein polynomial defining the distinguished uniformizer.
    pub fn defining_polynomial(&self) -> &Polynomial<QadicRing> {
        &self.inner.defining_polynomial
    }

    /// Ascending coefficients of the defining Eisenstein polynomial.
    pub fn defining_coefficients(&self) -> &[QadicElement] {
        &self.inner.defining_coefficients
    }

    /// Bounded display name of the distinguished uniformizer.
    pub fn uniformizer_name(&self) -> &str {
        &self.inner.uniformizer_name
    }

    /// Staggered p-adic caps for `1, pi, ..., pi^(e-1)`.
    pub fn coordinate_precisions(&self) -> &[u64] {
        &self.inner.coordinate_precisions
    }

    /// Construct an integral element from ascending coefficients in the
    /// original presentation ring, discarding only digits outside each
    /// staggered coordinate cap.
    pub fn integral_element(
        &self,
        coefficients: &[QadicElement],
    ) -> Result<PadicLocalFieldElement, PadicLocalFieldError> {
        if coefficients.len() > self.ramification_index() {
            return Err(PadicLocalFieldError::TooManyCoordinates {
                degree: self.ramification_index(),
                actual: coefficients.len(),
            });
        }
        for (index, coefficient) in coefficients.iter().enumerate() {
            if coefficient.ring() != Some(self.coefficient_ring()) {
                return Err(PadicLocalFieldError::CoordinateContext {
                    index,
                    source: ContextMismatchError {
                        expected: self.coefficient_ring().to_string(),
                        actual: coefficient.ring().map_or_else(
                            || "a q-adic field element".to_owned(),
                            ToString::to_string,
                        ),
                    },
                });
            }
        }
        let relative_precision = self.local_precision().absolute_uniformizer_precision();
        let coordinate_rings = self.coordinate_rings(relative_precision)?;
        let coordinates = coordinate_rings
            .iter()
            .enumerate()
            .map(|(index, target)| {
                let Some(target) = target else {
                    return Ok(None);
                };
                let source = coefficients
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| self.coefficient_ring().zero());
                self.coefficient_ring()
                    .precision_restriction(target.absolute_precision())?
                    .apply(&source)
                    .map(Some)
                    .map_err(PadicLocalFieldError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.normalize_integral(0, relative_precision, coordinates))
    }

    /// Embed a scalar from the exact maximal unramified coefficient field.
    ///
    /// The scalar's p-adic valuation is scaled by the ramification index. A
    /// negative scalar can consequently have relative precision larger than
    /// the field's maximum absolute cap; its normalized unit coordinates keep
    /// precisely those extra digits rather than rounding them away.
    pub fn element_from_qadic(
        &self,
        value: &QadicElement,
    ) -> Result<PadicLocalFieldElement, PadicLocalFieldError> {
        if value.field() != Some(self.coefficient_field()) {
            return Err(PadicLocalFieldError::ScalarContextMismatch(
                ContextMismatchError {
                    expected: self.coefficient_field().to_string(),
                    actual: value
                        .field()
                        .map_or_else(|| "a q-adic ring element".to_owned(), ToString::to_string),
                },
            ));
        }
        let Some((coefficient_valuation, unit)) = value.normalized_integral_unit() else {
            return Ok(self.zero());
        };
        let ramification_index = i64::try_from(self.ramification_index())
            .map_err(|_| PadicLocalFieldError::DegreeTooLarge)?;
        let valuation = ramification_index
            .checked_mul(coefficient_valuation)
            .ok_or(PadicLocalFieldError::DegreeTooLarge)?;
        let absolute_precision = self.maximum_absolute_precision();
        if valuation >= absolute_precision {
            return Ok(self.zero());
        }
        let mut relative_precision = u64::try_from(absolute_precision - valuation)
            .map_err(|_| PadicLocalFieldError::DegreeTooLarge)?;
        if coefficient_valuation != 0 {
            relative_precision = relative_precision.min(
                self.relation_relative_precision_capacity()
                    .saturating_sub(self.ramification_index() as u64),
            );
        }
        if relative_precision == 0 {
            return Ok(self.zero_at(valuation));
        }
        let mut coordinates = self.raw_zero(relative_precision);
        let target = coordinates[0]
            .as_ref()
            .and_then(QadicElement::ring)
            .expect("positive local relative precision retains a constant coordinate")
            .clone();
        coordinates[0] = Some(self.coordinate_in_ring(Some(&unit), &target));
        if coefficient_valuation != 0 {
            let ramification_unit = self.ramification_unit(relative_precision);
            let exponent = coefficient_valuation.unsigned_abs();
            let factor = if coefficient_valuation < 0 {
                self.pow_integral(&ramification_unit, exponent, relative_precision)
            } else {
                let inverse = self.inverse_integral_unit(&ramification_unit, relative_precision);
                self.pow_integral(&inverse, exponent, relative_precision)
            };
            coordinates = self.mul_integral(&coordinates, &factor, relative_precision);
        }
        Ok(PadicLocalFieldElement {
            field: self.clone(),
            valuation: Some(valuation),
            absolute_precision: valuation + relative_precision as i64,
            coordinates,
        })
    }

    /// Embed a scalar from the exact base `Q_p` context.
    pub fn element_from_padic(
        &self,
        value: &PadicElement,
    ) -> Result<PadicLocalFieldElement, PadicLocalFieldError> {
        let expected = self.coefficient_field().coefficient_field();
        if value.field() != Some(expected) {
            let actual = value.field().map_or_else(
                || {
                    value.ring().map_or_else(
                        || "an unknown p-adic parent".to_owned(),
                        ToString::to_string,
                    )
                },
                ToString::to_string,
            );
            return Err(PadicLocalFieldError::ScalarContextMismatch(
                ContextMismatchError {
                    expected: expected.to_string(),
                    actual,
                },
            ));
        }
        let coefficient = self
            .coefficient_field()
            .element_from_coefficients(std::slice::from_ref(value))
            .expect("the checked base scalar is a valid constant q-adic coordinate");
        self.element_from_qadic(&coefficient)
    }

    /// Reduce an integral value to the explicit finite residue field.
    ///
    /// `None` means that the value has negative valuation, or that a zero is
    /// known only at nonpositive absolute precision and therefore has no
    /// determined residue.
    pub fn residue(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<Option<FiniteFieldElement>, ContextMismatchError> {
        self.require(value)?;
        match value.valuation {
            None if value.absolute_precision > 0 => Ok(Some(self.residue_field().zero())),
            None => Ok(None),
            Some(valuation) if valuation < 0 => Ok(None),
            Some(valuation) if valuation > 0 => Ok(Some(self.residue_field().zero())),
            Some(_) => value.coordinates[0]
                .as_ref()
                .expect("a normalized unit has its constant coordinate")
                .ring()
                .expect("normalized coordinates are integral")
                .residue(
                    value.coordinates[0]
                        .as_ref()
                        .expect("a normalized unit has its constant coordinate"),
                )
                .map(Some),
        }
    }

    /// Lift a residue-field value through the maximal unramified coefficient ring.
    pub fn teichmuller(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<PadicLocalFieldElement, ContextMismatchError> {
        let lift = self.coefficient_ring().teichmuller(value)?;
        self.integral_element(&[lift])
            .map_err(|error| ContextMismatchError {
                expected: self.to_string(),
                actual: error.to_string(),
            })
    }

    /// Construct a first-class restriction to a coarser uniformizer cap.
    pub fn precision_restriction(
        &self,
        target_precision: PadicLocalPrecision,
    ) -> Result<PadicLocalFieldRestriction, PadicLocalFieldRestrictionError> {
        PadicLocalFieldRestriction::new(self.clone(), target_precision)
    }

    /// Explicitly discard high uniformizer digits into a distinct field context.
    pub fn restrict(
        &self,
        value: &PadicLocalFieldElement,
        target_precision: PadicLocalPrecision,
    ) -> Result<PadicLocalFieldElement, PadicLocalFieldRestrictionError> {
        self.precision_restriction(target_precision)?.apply(value)
    }

    /// Conjugate this presentation by a lifted coefficient Frobenius power.
    pub fn coefficient_frobenius_conjugate(
        &self,
        exponent: u64,
    ) -> Result<PadicLocalFieldCoefficientFrobenius, PadicLocalCoefficientFrobeniusError> {
        PadicLocalFieldCoefficientFrobenius::new(self.clone(), exponent)
    }

    fn trace_value(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<PadicElement, PadicLocalInvariantError> {
        let matrix = self.multiplication_matrix_over_unramified(value)?;
        let mut trace = matrix.ring().zero();
        for index in 0..self.ramification_index() {
            let diagonal = matrix
                .get(index, index)
                .expect("multiplication matrix has the declared square shape");
            trace = matrix
                .ring()
                .add(&trace, &diagonal)
                .expect("multiplication-matrix entries share one q-adic field");
        }
        Ok(trace.trace())
    }

    fn norm_value(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<PadicElement, PadicLocalInvariantError> {
        Ok(self
            .multiplication_matrix_over_unramified(value)?
            .det()?
            .norm())
    }

    fn characteristic_polynomial_value(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<Polynomial<PadicField>, PadicLocalInvariantError> {
        Ok(self.multiplication_matrix_over_base(value)?.charpoly()?)
    }

    fn minimal_polynomial_value(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<Polynomial<PadicField>, PadicLocalInvariantError> {
        let matrix = self.multiplication_matrix_over_base(value)?;
        let degree = self.degree();
        let base = matrix.ring().clone();
        let mut first = vec![base.zero(); degree];
        first[0] = base.one();
        let mut power = Vector::from_values(base.clone(), first)?;
        let mut columns = Vec::with_capacity(degree);
        for exponent in 0..degree {
            columns.push(power.values().collect::<Vec<_>>());
            if exponent + 1 != degree {
                power = matrix.apply(&power)?;
            }
        }
        let power_basis = Matrix::from_rows(
            base,
            (0..degree).map(|row| {
                columns
                    .iter()
                    .map(|column| column[row].clone())
                    .collect::<Vec<_>>()
            }),
        )?;
        match power_basis.det()?.valuation() {
            PadicValuation::Exact(_) => Ok(matrix.charpoly()?),
            PadicValuation::AtLeast(determinant_valuation_at_least) => {
                Err(PadicLocalInvariantError::MinimalPolynomialNotCertified {
                    degree,
                    determinant_valuation_at_least,
                })
            }
        }
    }

    fn multiplication_matrix_over_base(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<Matrix<PadicField>, PadicLocalInvariantError> {
        Ok(
            FrobeniusSemilinearMap::new(self.multiplication_matrix_over_unramified(value)?, 0)
                .restrict_scalars()?,
        )
    }

    fn multiplication_matrix_over_unramified(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<Matrix<QadicField>, PadicLocalInvariantError> {
        self.require(value)?;
        let value = self.cap_for_relation(value);
        let available = value.absolute_precision;
        let ramification_index = self.ramification_index();
        let coefficient_precision = u64::try_from(available)
            .ok()
            .map(|precision| precision / ramification_index as u64)
            .filter(|precision| *precision != 0)
            .ok_or(PadicLocalInvariantError::InsufficientPrecision {
                available,
                ramification_index,
            })?;
        let coefficient_field =
            QadicField::new(self.residue_field().clone(), coefficient_precision)
                .expect("a surviving coefficient precision is a valid q-adic context");
        let relation = self.qadic_relation(&coefficient_field);
        let uniformizer = self.qadic_uniformizer(&coefficient_field, &relation);
        let element =
            self.qadic_standard_coordinates(&value, &coefficient_field, &relation, &uniformizer);
        let one = {
            let mut value = vec![coefficient_field.zero(); ramification_index];
            value[0] = coefficient_field.one();
            value
        };
        let mut basis = one;
        let mut columns = Vec::with_capacity(ramification_index);
        for _ in 0..ramification_index {
            columns.push(self.mul_qadic_quotient(&coefficient_field, &relation, &element, &basis));
            basis = self.mul_qadic_quotient(&coefficient_field, &relation, &basis, &uniformizer);
        }
        Matrix::from_rows(
            coefficient_field,
            (0..ramification_index).map(|row| {
                columns
                    .iter()
                    .map(|column| column[row].clone())
                    .collect::<Vec<_>>()
            }),
        )
        .map_err(Into::into)
    }

    fn qadic_relation(&self, target: &QadicField) -> Vec<QadicElement> {
        self.defining_coefficients()
            .iter()
            .map(|coefficient| self.embed_qadic_integral(coefficient, target))
            .collect()
    }

    fn embed_qadic_integral(&self, value: &QadicElement, target: &QadicField) -> QadicElement {
        let integral = self.coordinate_in_ring(Some(value), target.integral_ring());
        target
            .integral_embedding()
            .apply(&integral)
            .expect("the coordinate was placed in the target integral ring")
    }

    fn qadic_uniformizer(
        &self,
        target: &QadicField,
        relation: &[QadicElement],
    ) -> Vec<QadicElement> {
        let mut uniformizer = vec![target.zero(); self.ramification_index()];
        if self.ramification_index() == 1 {
            uniformizer[0] = target
                .neg(&relation[0])
                .expect("the linear relation belongs to its q-adic field");
        } else {
            uniformizer[1] = target.one();
        }
        uniformizer
    }

    fn qadic_inverse_uniformizer(
        &self,
        target: &QadicField,
        relation: &[QadicElement],
    ) -> Vec<QadicElement> {
        let constant_inverse = target
            .inverse(&relation[0])
            .expect("an Eisenstein constant is nonzero at surviving precision");
        let mut inverse = vec![target.zero(); self.ramification_index()];
        for index in 0..self.ramification_index() {
            let numerator = if index + 1 == self.ramification_index() {
                target.one()
            } else {
                relation[index + 1].clone()
            };
            inverse[index] = target
                .neg(
                    &target
                        .mul(&numerator, &constant_inverse)
                        .expect("inverse-uniformizer coefficients share one context"),
                )
                .expect("inverse-uniformizer coefficients share one context");
        }
        inverse
    }

    fn qadic_standard_coordinates(
        &self,
        value: &PadicLocalFieldElement,
        target: &QadicField,
        relation: &[QadicElement],
        uniformizer: &[QadicElement],
    ) -> Vec<QadicElement> {
        let Some(valuation) = value.valuation else {
            return vec![target.zero(); self.ramification_index()];
        };
        let unit = value
            .coordinates
            .iter()
            .map(|coordinate| {
                coordinate.as_ref().map_or_else(
                    || target.zero(),
                    |coefficient| self.embed_qadic_integral(coefficient, target),
                )
            })
            .collect::<Vec<_>>();
        let factor = if valuation >= 0 {
            self.pow_qadic_quotient(target, relation, uniformizer, valuation.unsigned_abs())
        } else {
            self.pow_qadic_quotient(
                target,
                relation,
                &self.qadic_inverse_uniformizer(target, relation),
                valuation.unsigned_abs(),
            )
        };
        self.mul_qadic_quotient(target, relation, &unit, &factor)
    }

    /// Uniform `Z_p` coordinates in the integral basis
    /// `theta^j pi^i`, reduced modulo `p^precision`.
    ///
    /// This is crate-internal support for certified local lattices. Returning
    /// `None` means either that the value is nonintegral, belongs to another
    /// field, or does not carry all digits required at the requested uniform
    /// coefficient precision.
    pub(super) fn integral_flat_coordinates(
        &self,
        value: &PadicLocalFieldElement,
        precision: u64,
    ) -> Option<Vec<Integer>> {
        if precision == 0 || !self.contains(value) {
            return None;
        }
        match value.valuation {
            Some(valuation) if valuation < 0 => return None,
            Some(_) | None => {}
        }
        let required = precision.checked_mul(self.ramification_index() as u64)?;
        if value.absolute_precision < i64::try_from(required).ok()? {
            return None;
        }
        if value.valuation.is_none() {
            return Some(vec![Integer::zero(); self.degree()]);
        }

        let target = QadicField::new(self.residue_field().clone(), precision).ok()?;
        let relation = self.qadic_relation(&target);
        let uniformizer = self.qadic_uniformizer(&target, &relation);
        let coordinates = self.qadic_standard_coordinates(value, &target, &relation, &uniformizer);
        let embedding = target.integral_embedding();
        let mut flattened = Vec::with_capacity(self.degree());
        for coordinate in &coordinates {
            let integral = embedding.retract(coordinate).ok()?;
            for coefficient in integral.coefficients() {
                flattened.push(coefficient.representative()?);
            }
        }
        (flattened.len() == self.degree()).then_some(flattened)
    }

    fn pow_qadic_quotient(
        &self,
        target: &QadicField,
        relation: &[QadicElement],
        value: &[QadicElement],
        exponent: u64,
    ) -> Vec<QadicElement> {
        let mut result = vec![target.zero(); self.ramification_index()];
        result[0] = target.one();
        let mut base = value.to_vec();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.mul_qadic_quotient(target, relation, &result, &base);
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self.mul_qadic_quotient(target, relation, &base, &base);
            }
        }
        result
    }

    fn mul_qadic_quotient(
        &self,
        target: &QadicField,
        relation: &[QadicElement],
        left: &[QadicElement],
        right: &[QadicElement],
    ) -> Vec<QadicElement> {
        let degree = self.ramification_index();
        let mut product = vec![target.zero(); 2 * degree - 1];
        for (left_index, left_value) in left.iter().enumerate() {
            for (right_index, right_value) in right.iter().enumerate() {
                let term = target
                    .mul(left_value, right_value)
                    .expect("quotient operands share one q-adic field");
                product[left_index + right_index] = target
                    .add(&product[left_index + right_index], &term)
                    .expect("quotient operands share one q-adic field");
            }
        }
        for power in (degree..product.len()).rev() {
            let leading = product[power].clone();
            for (index, relation_coefficient) in relation[..degree].iter().enumerate() {
                let term = target
                    .mul(&leading, relation_coefficient)
                    .expect("relation and product share one q-adic field");
                let destination = power - degree + index;
                product[destination] = target
                    .sub(&product[destination], &term)
                    .expect("quotient reduction stays in one q-adic field");
            }
        }
        product.truncate(degree);
        product
    }

    /// Additive identity at the declared staggered precision.
    pub fn zero(&self) -> PadicLocalFieldElement {
        self.zero_at(self.maximum_absolute_precision())
    }

    /// Multiplicative identity at the declared staggered precision.
    pub fn one(&self) -> PadicLocalFieldElement {
        self.integral_element(&[self.coefficient_ring().one()])
            .expect("one has valid coordinates in every local field")
    }

    /// Distinguished root of the defining Eisenstein polynomial.
    pub fn uniformizer(&self) -> PadicLocalFieldElement {
        if self.ramification_index() == 1 {
            let value = self
                .coefficient_ring()
                .neg(&self.inner.defining_coefficients[0])
                .expect("a defining coefficient belongs to its q-adic ring");
            return self
                .integral_element(&[value])
                .expect("the linear Eisenstein root belongs to its local field");
        }
        let mut coordinates = vec![self.coefficient_ring().zero(); 2];
        coordinates[1] = self.coefficient_ring().one();
        self.integral_element(&coordinates)
            .expect("the distinguished basis vector belongs to its local field")
    }

    /// Invert a value that is nonzero at its available precision.
    ///
    /// If `x = pi^v u` is known with relative precision `R`, its inverse is
    /// returned as `pi^-v u^-1` with the same relative precision. Thus the
    /// absolute precision changes from `v + R` to `-v + R`; it is not reset
    /// to the context cap.
    pub fn inverse(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<PadicLocalFieldElement, UnitError> {
        self.require(value)?;
        let Some(valuation) = value.valuation else {
            return Err(crate::DivisionByZeroError.into());
        };
        let inverse_valuation = -valuation;
        if inverse_valuation >= self.maximum_absolute_precision() {
            return Ok(self.zero());
        }
        let source_relative = value
            .relative_precision()
            .expect("a nonzero local value has positive relative precision")
            .min(self.relation_relative_precision_capacity());
        let target_relative = source_relative.min(
            u64::try_from(self.maximum_absolute_precision() - inverse_valuation)
                .expect("the inverse valuation lies below the context cap"),
        );
        let source = self.restrict_integral(&value.coordinates, source_relative, target_relative);
        let coordinates = self.inverse_integral_unit(&source, target_relative);
        Ok(PadicLocalFieldElement {
            field: self.clone(),
            valuation: Some(inverse_valuation),
            absolute_precision: inverse_valuation + target_relative as i64,
            coordinates,
        })
    }

    fn maximum_absolute_precision(&self) -> i64 {
        i64::try_from(self.local_precision().absolute_uniformizer_precision())
            .expect("local precision construction checks the signed valuation range")
    }

    fn relation_relative_precision_capacity(&self) -> u64 {
        self.coefficient_ring()
            .absolute_precision()
            .saturating_mul(self.ramification_index() as u64)
            .min(i64::MAX as u64)
    }

    fn require(&self, value: &PadicLocalFieldElement) -> Result<(), ContextMismatchError> {
        if self == value.field() {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.field().to_string(),
            })
        }
    }

    fn coordinate_rings(
        &self,
        relative_precision: u64,
    ) -> Result<Vec<Option<QadicRing>>, PadicLocalFieldError> {
        (0..self.ramification_index())
            .map(|index| {
                let cap = staggered_coefficient_precision(
                    relative_precision,
                    index,
                    self.ramification_index(),
                )?;
                (cap != 0)
                    .then(|| QadicRing::new(self.residue_field().clone(), cap))
                    .transpose()
                    .map_err(PadicLocalFieldError::from)
            })
            .collect()
    }

    fn zero_at(&self, absolute_precision: i64) -> PadicLocalFieldElement {
        PadicLocalFieldElement {
            field: self.clone(),
            valuation: None,
            absolute_precision,
            coordinates: vec![None; self.ramification_index()],
        }
    }

    fn normalize_integral(
        &self,
        valuation_shift: i64,
        mut relative_precision: u64,
        mut coordinates: Vec<Option<QadicElement>>,
    ) -> PadicLocalFieldElement {
        let absolute_precision = valuation_shift
            .checked_add(relative_precision as i64)
            .expect("local absolute precision remains in the signed valuation range");
        let Some(integral_valuation) = self.integral_valuation(&coordinates) else {
            return self.zero_at(absolute_precision);
        };
        for _ in 0..integral_valuation {
            coordinates = self.divide_integral_by_uniformizer(&coordinates, relative_precision);
            relative_precision -= 1;
        }
        let valuation = valuation_shift
            .checked_add(integral_valuation as i64)
            .expect("normalized local valuation remains representable");
        debug_assert_eq!(self.integral_valuation(&coordinates), Some(0));
        PadicLocalFieldElement {
            field: self.clone(),
            valuation: Some(valuation),
            absolute_precision,
            coordinates,
        }
    }

    fn integral_valuation(&self, coordinates: &[Option<QadicElement>]) -> Option<u64> {
        let ramification_index = self.ramification_index() as u64;
        coordinates
            .iter()
            .enumerate()
            .filter_map(|(index, coordinate)| {
                let value = coordinate.as_ref()?;
                let PadicValuation::Exact(valuation) = value.valuation() else {
                    return None;
                };
                let valuation = u64::try_from(valuation)
                    .expect("normalized integral coordinates have nonnegative valuation");
                Some(
                    ramification_index
                        .checked_mul(valuation)
                        .and_then(|value| value.checked_add(index as u64))
                        .expect("local precision bounds retained coordinate valuations"),
                )
            })
            .min()
    }

    fn add_values(
        &self,
        left: &PadicLocalFieldElement,
        right: &PadicLocalFieldElement,
    ) -> PadicLocalFieldElement {
        let left = self.cap_for_relation(left);
        let right = self.cap_for_relation(right);
        let left = &left;
        let right = &right;
        match (left.valuation, right.valuation) {
            (None, None) => self.zero_at(left.absolute_precision.min(right.absolute_precision)),
            (None, Some(_)) => {
                self.restrict_value(right, left.absolute_precision.min(right.absolute_precision))
            }
            (Some(_), None) => {
                self.restrict_value(left, left.absolute_precision.min(right.absolute_precision))
            }
            (Some(left_valuation), Some(right_valuation)) => {
                if left_valuation > right_valuation {
                    return self.add_values(right, left);
                }
                let shift = u64::try_from(right_valuation - left_valuation)
                    .expect("ordered local valuations have a nonnegative difference");
                let left_relative = left
                    .relative_precision()
                    .expect("a nonzero value has relative precision");
                let right_relative = right
                    .relative_precision()
                    .expect("a nonzero value has relative precision");
                if shift >= left_relative {
                    return left.clone();
                }
                let target_relative = left_relative.min(right_relative.saturating_add(shift));
                let left_coordinates =
                    self.restrict_integral(&left.coordinates, left_relative, target_relative);
                let right_coordinates = self.multiply_integral_by_uniformizer_power(
                    &right.coordinates,
                    right_relative,
                    shift,
                    target_relative,
                );
                self.normalize_integral(
                    left_valuation,
                    target_relative,
                    self.add_integral(&left_coordinates, &right_coordinates, target_relative),
                )
            }
        }
    }

    fn mul_values(
        &self,
        left: &PadicLocalFieldElement,
        right: &PadicLocalFieldElement,
    ) -> PadicLocalFieldElement {
        let left = self.cap_for_relation(left);
        let right = self.cap_for_relation(right);
        let left = &left;
        let right = &right;
        match (left.valuation, right.valuation) {
            (None, None) => self.zero_at(
                left.absolute_precision
                    .checked_add(right.absolute_precision)
                    .unwrap_or(self.maximum_absolute_precision())
                    .min(self.maximum_absolute_precision()),
            ),
            (None, Some(valuation)) => self.zero_at(
                left.absolute_precision
                    .checked_add(valuation)
                    .unwrap_or(i64::MIN)
                    .min(self.maximum_absolute_precision()),
            ),
            (Some(valuation), None) => self.zero_at(
                right
                    .absolute_precision
                    .checked_add(valuation)
                    .unwrap_or(i64::MIN)
                    .min(self.maximum_absolute_precision()),
            ),
            (Some(left_valuation), Some(right_valuation)) => {
                let valuation = left_valuation + right_valuation;
                if valuation >= self.maximum_absolute_precision() {
                    return self.zero();
                }
                let relative_precision = left
                    .relative_precision()
                    .expect("a nonzero value has relative precision")
                    .min(
                        right
                            .relative_precision()
                            .expect("a nonzero value has relative precision"),
                    )
                    .min(self.relation_relative_precision_capacity())
                    .min(
                        u64::try_from(self.maximum_absolute_precision() - valuation)
                            .expect("the product valuation lies below the context cap"),
                    );
                let left_coordinates = self.restrict_integral(
                    &left.coordinates,
                    left.relative_precision().expect("nonzero"),
                    relative_precision,
                );
                let right_coordinates = self.restrict_integral(
                    &right.coordinates,
                    right.relative_precision().expect("nonzero"),
                    relative_precision,
                );
                PadicLocalFieldElement {
                    field: self.clone(),
                    valuation: Some(valuation),
                    absolute_precision: valuation + relative_precision as i64,
                    coordinates: self.mul_integral(
                        &left_coordinates,
                        &right_coordinates,
                        relative_precision,
                    ),
                }
            }
        }
    }

    fn neg_value(&self, value: &PadicLocalFieldElement) -> PadicLocalFieldElement {
        let Some(valuation) = value.valuation else {
            return value.clone();
        };
        let relative_precision = value
            .relative_precision()
            .expect("a nonzero value has relative precision");
        let rings = self
            .coordinate_rings(relative_precision)
            .expect("an element's relative precision has valid coordinate rings");
        PadicLocalFieldElement {
            field: self.clone(),
            valuation: Some(valuation),
            absolute_precision: value.absolute_precision,
            coordinates: value
                .coordinates
                .iter()
                .zip(rings)
                .map(|(coordinate, ring)| {
                    ring.map(|ring| {
                        ring.neg(
                            coordinate
                                .as_ref()
                                .expect("a positive-cap coordinate is present"),
                        )
                        .expect("the coordinate belongs to its ring")
                    })
                })
                .collect(),
        }
    }

    fn restrict_value(
        &self,
        value: &PadicLocalFieldElement,
        target_absolute_precision: i64,
    ) -> PadicLocalFieldElement {
        if target_absolute_precision >= value.absolute_precision {
            return value.clone();
        }
        let Some(valuation) = value.valuation else {
            return self.zero_at(target_absolute_precision);
        };
        if target_absolute_precision <= valuation {
            return self.zero_at(target_absolute_precision);
        }
        let target_relative = u64::try_from(target_absolute_precision - valuation)
            .expect("target precision lies above the exact valuation");
        PadicLocalFieldElement {
            field: self.clone(),
            valuation: Some(valuation),
            absolute_precision: target_absolute_precision,
            coordinates: self.restrict_integral(
                &value.coordinates,
                value.relative_precision().expect("nonzero"),
                target_relative,
            ),
        }
    }

    fn cap_for_relation(&self, value: &PadicLocalFieldElement) -> PadicLocalFieldElement {
        let Some(valuation) = value.valuation else {
            return value.clone();
        };
        let relation_cap = valuation
            .checked_add(self.relation_relative_precision_capacity() as i64)
            .unwrap_or(i64::MAX);
        self.restrict_value(value, value.absolute_precision.min(relation_cap))
    }

    fn raw_zero(&self, relative_precision: u64) -> Vec<Option<QadicElement>> {
        self.coordinate_rings(relative_precision)
            .expect("positive local precision derives valid q-adic rings")
            .into_iter()
            .map(|ring| ring.map(|ring| ring.zero()))
            .collect()
    }

    fn raw_one(&self, relative_precision: u64) -> Vec<Option<QadicElement>> {
        let mut result = self.raw_zero(relative_precision);
        if let Some(coordinate) = result[0].as_mut() {
            let ring = coordinate
                .ring()
                .expect("raw local coordinates are integral");
            *coordinate = ring.one();
        }
        result
    }

    /// The normalized local unit `pi^e / p`, obtained coefficientwise from
    /// the Eisenstein relation. Dividing the relation by `p` costs one q-adic
    /// digit, which is why callers cap its relative precision at `e(P-1)`.
    fn ramification_unit(&self, relative_precision: u64) -> Vec<Option<QadicElement>> {
        let targets = self
            .coordinate_rings(relative_precision)
            .expect("the relation unit precision is representable");
        targets
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| {
                    let divided = self
                        .coefficient_ring()
                        .divide_uniformizer_exact(&self.defining_coefficients()[index], 1)
                        .expect("every nonleading Eisenstein coefficient is divisible by p");
                    let value = self.coordinate_in_ring(Some(&divided), &target);
                    target
                        .neg(&value)
                        .expect("the relation coefficient belongs to its target ring")
                })
            })
            .collect()
    }

    fn pow_integral(
        &self,
        value: &[Option<QadicElement>],
        exponent: u64,
        relative_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        let mut result = self.raw_one(relative_precision);
        let mut base = value.to_vec();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.mul_integral(&result, &base, relative_precision);
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self.mul_integral(&base, &base, relative_precision);
            }
        }
        result
    }

    fn restrict_integral(
        &self,
        coordinates: &[Option<QadicElement>],
        source_precision: u64,
        target_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        debug_assert!(target_precision <= source_precision);
        let targets = self
            .coordinate_rings(target_precision)
            .expect("positive local precision derives valid q-adic rings");
        targets
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| self.coordinate_in_ring(coordinates[index].as_ref(), &target))
            })
            .collect()
    }

    fn coordinate_in_ring(&self, value: Option<&QadicElement>, target: &QadicRing) -> QadicElement {
        let Some(value) = value else {
            return target.zero();
        };
        let source = value
            .ring()
            .expect("local normalized coordinates are q-adic integers");
        if source == target {
            return value.clone();
        }
        if source.absolute_precision() > target.absolute_precision() {
            return source
                .precision_restriction(target.absolute_precision())
                .expect("coordinate precision only decreases")
                .apply(value)
                .expect("coordinate belongs to the restriction source");
        }
        let coefficients = value
            .coefficients()
            .into_iter()
            .map(|coefficient| {
                target.coefficient_ring().element(
                    coefficient
                        .representative()
                        .expect("an integral p-adic coordinate has a representative"),
                )
            })
            .collect::<Vec<_>>();
        target
            .element_from_coefficients(&coefficients)
            .expect("canonical lifting preserves the q-adic power basis")
    }

    fn defining_coefficient_in_ring(&self, index: usize, target: &QadicRing) -> QadicElement {
        self.coordinate_in_ring(Some(&self.defining_coefficients()[index]), target)
    }

    fn add_integral(
        &self,
        left: &[Option<QadicElement>],
        right: &[Option<QadicElement>],
        relative_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        self.coordinate_rings(relative_precision)
            .expect("positive local precision derives valid q-adic rings")
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| {
                    target
                        .add(
                            &self.coordinate_in_ring(left[index].as_ref(), &target),
                            &self.coordinate_in_ring(right[index].as_ref(), &target),
                        )
                        .expect("raw operands share the target coordinate ring")
                })
            })
            .collect()
    }

    fn sub_integral(
        &self,
        left: &[Option<QadicElement>],
        right: &[Option<QadicElement>],
        relative_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        self.coordinate_rings(relative_precision)
            .expect("positive local precision derives valid q-adic rings")
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| {
                    target
                        .sub(
                            &self.coordinate_in_ring(left[index].as_ref(), &target),
                            &self.coordinate_in_ring(right[index].as_ref(), &target),
                        )
                        .expect("raw operands share the target coordinate ring")
                })
            })
            .collect()
    }

    fn mul_integral(
        &self,
        left: &[Option<QadicElement>],
        right: &[Option<QadicElement>],
        relative_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        let coefficient_precision =
            staggered_coefficient_precision(relative_precision, 0, self.ramification_index())
                .expect("local degree is representable");
        let scratch = QadicRing::new(self.residue_field().clone(), coefficient_precision)
            .expect("positive relative precision requires a positive scratch cap");
        let degree = self.ramification_index();
        let left = left
            .iter()
            .map(|value| self.coordinate_in_ring(value.as_ref(), &scratch))
            .collect::<Vec<_>>();
        let right = right
            .iter()
            .map(|value| self.coordinate_in_ring(value.as_ref(), &scratch))
            .collect::<Vec<_>>();
        let mut product = vec![scratch.zero(); 2 * degree - 1];
        for (left_index, left_value) in left.iter().enumerate() {
            for (right_index, right_value) in right.iter().enumerate() {
                let term = scratch
                    .mul(left_value, right_value)
                    .expect("scratch values share one q-adic context");
                product[left_index + right_index] = scratch
                    .add(&product[left_index + right_index], &term)
                    .expect("scratch values share one q-adic context");
            }
        }
        for power in (degree..product.len()).rev() {
            let leading = product[power].clone();
            for index in 0..degree {
                let relation = self.defining_coefficient_in_ring(index, &scratch);
                let term = scratch
                    .mul(&leading, &relation)
                    .expect("relation and product use the scratch q-adic ring");
                let target = power - degree + index;
                product[target] = scratch
                    .sub(&product[target], &term)
                    .expect("reduction stays in the scratch q-adic ring");
            }
        }
        self.coordinate_rings(relative_precision)
            .expect("positive local precision derives valid q-adic rings")
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| self.coordinate_in_ring(Some(&product[index]), &target))
            })
            .collect()
    }

    fn multiply_integral_by_uniformizer_power(
        &self,
        coordinates: &[Option<QadicElement>],
        source_precision: u64,
        exponent: u64,
        target_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        if exponent >= target_precision {
            return self.raw_zero(target_precision);
        }
        let base_precision = target_precision - exponent;
        debug_assert!(base_precision <= source_precision);
        let mut result = self.restrict_integral(coordinates, source_precision, base_precision);
        let mut precision = base_precision;
        for _ in 0..exponent {
            result = self.multiply_integral_by_uniformizer(&result, precision);
            precision += 1;
        }
        result
    }

    fn multiply_integral_by_uniformizer(
        &self,
        coordinates: &[Option<QadicElement>],
        source_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        let target_precision = source_precision + 1;
        let rings = self
            .coordinate_rings(target_precision)
            .expect("local precision remains within the context cap");
        let last = coordinates.last().and_then(Option::as_ref);
        rings
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| {
                    let wrapped = self.coordinate_in_ring(last, &target);
                    let relation = self.defining_coefficient_in_ring(index, &target);
                    let relation_term = target
                        .mul(&wrapped, &relation)
                        .expect("uniformizer reduction stays in one q-adic ring");
                    let shifted = if index == 0 {
                        target.zero()
                    } else {
                        self.coordinate_in_ring(coordinates[index - 1].as_ref(), &target)
                    };
                    target
                        .sub(&shifted, &relation_term)
                        .expect("uniformizer reduction stays in one q-adic ring")
                })
            })
            .collect()
    }

    fn divide_integral_by_uniformizer(
        &self,
        coordinates: &[Option<QadicElement>],
        source_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        debug_assert!(source_precision > 1);
        let target_precision = source_precision - 1;
        let source_constant = coordinates[0]
            .as_ref()
            .expect("positive relative precision retains a constant coordinate");
        let quotient_cap = source_constant
            .ring()
            .expect("local coordinates are integral")
            .absolute_precision()
            - 1;
        let quotient = if quotient_cap == 0 {
            None
        } else {
            let numerator = source_constant
                .ring()
                .expect("constant coordinate is integral")
                .divide_uniformizer_exact(source_constant, 1)
                .expect("positive local valuation makes the constant divisible by p");
            let source_unit = self
                .coefficient_ring()
                .divide_uniformizer_exact(&self.defining_coefficients()[0], 1)
                .expect("an Eisenstein constant has valuation one");
            let quotient_ring = numerator
                .ring()
                .expect("integral division returns a q-adic ring value");
            let denominator = self.coordinate_in_ring(Some(&source_unit), quotient_ring);
            let denominator_inverse = quotient_ring
                .inverse_unit(&denominator)
                .expect("the Eisenstein constant divided by p is a unit");
            Some(
                quotient_ring
                    .mul(&numerator, &denominator_inverse)
                    .expect("quotient factors share one q-adic ring"),
            )
        };
        self.coordinate_rings(target_precision)
            .expect("positive local precision derives valid q-adic rings")
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                target.map(|target| {
                    let shifted = coordinates
                        .get(index + 1)
                        .and_then(Option::as_ref)
                        .map_or_else(
                            || target.zero(),
                            |value| self.coordinate_in_ring(Some(value), &target),
                        );
                    let quotient = self.coordinate_in_ring(quotient.as_ref(), &target);
                    let relation = self.defining_coefficient_in_ring(index + 1, &target);
                    let correction = target
                        .mul(&quotient, &relation)
                        .expect("division correction stays in one q-adic ring");
                    target
                        .sub(&shifted, &correction)
                        .expect("division correction stays in one q-adic ring")
                })
            })
            .collect()
    }

    fn inverse_integral_unit(
        &self,
        coordinates: &[Option<QadicElement>],
        relative_precision: u64,
    ) -> Vec<Option<QadicElement>> {
        debug_assert_eq!(self.integral_valuation(coordinates), Some(0));
        let constant = coordinates[0]
            .as_ref()
            .expect("a normalized local unit has a constant coordinate");
        let constant_ring = constant
            .ring()
            .expect("normalized local coordinates are integral");
        let constant_inverse = constant_ring
            .inverse_unit(constant)
            .expect("a normalized local unit has a q-adic unit constant");
        let mut inverse = self.raw_zero(relative_precision);
        inverse[0] = Some(constant_inverse);
        let one = self.raw_one(relative_precision);
        let two = self.add_integral(&one, &one, relative_precision);
        let mut correct_digits = 1_u64;
        while correct_digits < relative_precision {
            let product = self.mul_integral(coordinates, &inverse, relative_precision);
            let correction = self.sub_integral(&two, &product, relative_precision);
            inverse = self.mul_integral(&inverse, &correction, relative_precision);
            correct_digits = correct_digits.saturating_mul(2);
        }
        inverse
    }

    /// Canonical typed JSON preserving display metadata as well as identity.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an Eisenstein presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> PadicLocalFieldWire {
        PadicLocalFieldWire {
            type_id: Self::TYPE_ID.to_owned(),
            coefficient_ring: self.coefficient_ring().clone(),
            defining_polynomial: self.defining_polynomial().clone(),
            precision: self.local_precision(),
            uniformizer_name: self.uniformizer_name().to_owned(),
        }
    }

    fn from_wire(wire: PadicLocalFieldWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        Self::from_eisenstein(
            wire.coefficient_ring,
            wire.defining_polynomial,
            wire.precision,
            Some(wire.uniformizer_name),
        )
        .map_err(|error| error.to_string())
    }
}

impl PartialEq for PadicLocalField {
    fn eq(&self, other: &Self) -> bool {
        self.coefficient_ring() == other.coefficient_ring()
            && self.defining_coefficients() == other.defining_coefficients()
            && self.local_precision() == other.local_precision()
    }
}

impl Eq for PadicLocalField {}

impl fmt::Debug for PadicLocalField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalField")
            .field("prime", &self.prime())
            .field("ramification_index", &self.ramification_index())
            .field("residue_degree", &self.residue_degree())
            .field("precision", &self.local_precision())
            .field("uniformizer_name", &self.uniformizer_name())
            .finish()
    }
}

impl fmt::Display for PadicLocalField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Q_{} extension of degree {} (e={}, f={}) at {}",
            self.prime(),
            self.degree(),
            self.ramification_index(),
            self.residue_degree(),
            self.local_precision()
        )
    }
}

impl Serialize for PadicLocalField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicLocalField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicLocalFieldWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl SemanticType for PadicLocalField {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&PadicLocalFieldIdentity {
            coefficient_ring: self.coefficient_ring(),
            defining_coefficients: self.defining_coefficients(),
            precision: self.local_precision(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FIELD_DESCRIPTION
    }
}

impl PadicLocalFieldRestriction {
    /// Construct the canonical lower-cap context for one presentation.
    pub fn new(
        source: PadicLocalField,
        target_precision: PadicLocalPrecision,
    ) -> Result<Self, PadicLocalFieldRestrictionError> {
        let requested = target_precision.absolute_uniformizer_precision();
        let available = source.local_precision().absolute_uniformizer_precision();
        if requested > available {
            return Err(PadicLocalFieldRestrictionError::InvalidPrecision {
                requested,
                available,
            });
        }
        let target = PadicLocalField::from_eisenstein(
            source.coefficient_ring().clone(),
            source.defining_polynomial().clone(),
            target_precision,
            Some(source.uniformizer_name().to_owned()),
        )
        .map_err(PadicLocalFieldRestrictionError::ContextConstruction)?;
        Ok(Self { source, target })
    }

    /// Exact source context.
    pub fn source(&self) -> &PadicLocalField {
        &self.source
    }

    /// Distinct lower-cap target context.
    pub fn target(&self) -> &PadicLocalField {
        &self.target
    }

    /// Apply the restriction after checking exact source identity.
    pub fn apply(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<PadicLocalFieldElement, PadicLocalFieldRestrictionError> {
        self.source.require(value)?;
        let target_absolute = value
            .absolute_precision
            .min(self.target.maximum_absolute_precision());
        let Some(valuation) = value.valuation else {
            return Ok(self.target.zero_at(target_absolute));
        };
        if target_absolute <= valuation {
            return Ok(self.target.zero_at(target_absolute));
        }
        let target_relative = u64::try_from(target_absolute - valuation)
            .expect("a surviving restricted value has positive relative precision");
        Ok(PadicLocalFieldElement {
            field: self.target.clone(),
            valuation: Some(valuation),
            absolute_precision: target_absolute,
            coordinates: self.target.restrict_integral(
                &value.coordinates,
                value.relative_precision().expect("a nonzero value"),
                target_relative,
            ),
        })
    }

    /// Recheck the presentation identity and direction of precision loss.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let same_presentation = self.source.coefficient_ring() == self.target.coefficient_ring()
            && self.source.defining_coefficients() == self.target.defining_coefficients();
        let loses_precision = self
            .target
            .local_precision()
            .absolute_uniformizer_precision()
            <= self
                .source
                .local_precision()
                .absolute_uniformizer_precision();
        if same_presentation && loses_precision {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "ramified local restriction preserves presentation and only loses precision",
            })
        }
    }

    /// Canonical JSON containing both endpoint contexts.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PadicLocalFieldRestrictionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both endpoint contexts and their relation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PadicLocalFieldRestrictionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
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

impl CoefficientMap for PadicLocalFieldRestriction {
    type Source = PadicLocalField;
    type Target = PadicLocalField;
    type Error = PadicLocalFieldRestrictionError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(
        &self,
        value: &<Self::Source as Ring>::Element,
    ) -> Result<<Self::Target as Ring>::Element, Self::Error> {
        self.apply(value)
    }
}

impl fmt::Debug for PadicLocalFieldRestriction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalFieldRestriction")
            .field("source", &self.source)
            .field("target", &self.target)
            .finish()
    }
}

impl SemanticType for PadicLocalFieldRestriction {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalFieldRestriction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FIELD_RESTRICTION_DESCRIPTION
    }
}

impl PadicLocalFieldCoefficientFrobenius {
    /// Construct `sigma^r: K_E -> K_{sigma^r(E)}`.
    pub fn new(
        source: PadicLocalField,
        exponent: u64,
    ) -> Result<Self, PadicLocalCoefficientFrobeniusError> {
        let coefficient_frobenius =
            QadicRingFrobenius::new(source.coefficient_ring().clone(), exponent);
        let conjugated_polynomial = source
            .defining_polynomial()
            .map_coefficients(&coefficient_frobenius)
            .expect("a defining polynomial belongs to its declared coefficient ring");
        let target = PadicLocalField::from_eisenstein(
            source.coefficient_ring().clone(),
            conjugated_polynomial,
            source.local_precision(),
            Some(source.uniformizer_name().to_owned()),
        )
        .map_err(PadicLocalCoefficientFrobeniusError::TargetConstruction)?;
        Ok(Self {
            source,
            target,
            exponent,
        })
    }

    /// Exact presented source field.
    pub fn source(&self) -> &PadicLocalField {
        &self.source
    }

    /// Exact coefficient-conjugate target field.
    pub fn target(&self) -> &PadicLocalField {
        &self.target
    }

    /// Unreduced nonnegative coefficient Frobenius exponent.
    pub fn exponent(&self) -> u64 {
        self.exponent
    }

    /// Whether coefficient conjugation returns to the same presented context.
    pub fn is_endomorphism(&self) -> bool {
        self.source == self.target
    }

    /// Twist every normalized q-adic coordinate and fix the uniformizer.
    pub fn apply(
        &self,
        value: &PadicLocalFieldElement,
    ) -> Result<PadicLocalFieldElement, PadicLocalCoefficientFrobeniusError> {
        self.source.require(value)?;
        let Some(valuation) = value.valuation else {
            return Ok(self.target.zero_at(value.absolute_precision));
        };
        let coordinates = value
            .coordinates
            .iter()
            .map(|coordinate| {
                coordinate
                    .as_ref()
                    .map(|value| {
                        let ring = value
                            .ring()
                            .expect("normalized local coordinates are integral");
                        QadicRingFrobenius::new(ring.clone(), self.exponent)
                            .apply(value)
                            .map_err(PadicLocalCoefficientFrobeniusError::from)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PadicLocalFieldElement {
            field: self.target.clone(),
            valuation: Some(valuation),
            absolute_precision: value.absolute_precision,
            coordinates,
        })
    }

    /// Compose `self` after a map ending at this source presentation.
    pub fn compose(&self, inner: &Self) -> Result<Self, PadicLocalCoefficientFrobeniusError> {
        if self.source != inner.target {
            return Err(ContextMismatchError {
                expected: self.source.to_string(),
                actual: inner.target.to_string(),
            }
            .into());
        }
        let exponent = self.exponent.checked_add(inner.exponent).ok_or(
            PadicLocalCoefficientFrobeniusError::ExponentOverflow {
                outer: self.exponent,
                inner: inner.exponent,
            },
        )?;
        let composed = Self::new(inner.source.clone(), exponent)?;
        if composed.target != self.target {
            return Err(ContextMismatchError {
                expected: self.target.to_string(),
                actual: composed.target.to_string(),
            }
            .into());
        }
        Ok(composed)
    }

    /// Recompute the conjugate presentation and verify the stored target.
    pub fn verify(&self) -> Result<(), VerificationError> {
        match Self::new(self.source.clone(), self.exponent) {
            Ok(recomputed) if recomputed.target == self.target => Ok(()),
            _ => Err(VerificationError::InvalidWitness {
                claim:
                    "local coefficient Frobenius target is the conjugated Eisenstein presentation",
            }),
        }
    }

    /// Canonical JSON preserving both endpoint presentations.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PadicLocalFieldCoefficientFrobeniusWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            target: self.target.clone(),
            exponent: self.exponent,
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the conjugated target presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PadicLocalFieldCoefficientFrobeniusWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            target: wire.target,
            exponent: wire.exponent,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl CoefficientMap for PadicLocalFieldCoefficientFrobenius {
    type Source = PadicLocalField;
    type Target = PadicLocalField;
    type Error = PadicLocalCoefficientFrobeniusError;

    fn source(&self) -> &Self::Source {
        self.source()
    }

    fn target(&self) -> &Self::Target {
        self.target()
    }

    fn apply(
        &self,
        value: &<Self::Source as Ring>::Element,
    ) -> Result<<Self::Target as Ring>::Element, Self::Error> {
        self.apply(value)
    }
}

impl fmt::Debug for PadicLocalFieldCoefficientFrobenius {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalFieldCoefficientFrobenius")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("exponent", &self.exponent)
            .finish()
    }
}

impl fmt::Display for PadicLocalFieldCoefficientFrobenius {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "coefficient Frobenius^{} from {} to {}",
            self.exponent, self.source, self.target
        )
    }
}

impl SemanticType for PadicLocalFieldCoefficientFrobenius {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalFieldCoefficientFrobenius@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        output.extend_from_slice(&self.exponent.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FIELD_COEFFICIENT_FROBENIUS_DESCRIPTION
    }
}

/// One finite-precision value in an Eisenstein-presented local field.
///
/// A nonzero value has the normalized form `pi^v u`, where `u` is integral
/// of valuation zero. Its stored coordinates describe `u` at relative
/// precision `R`; the value's honest absolute precision is `v + R`. A zero
/// instead records only its absolute precision lower bound. This distinction
/// is what permits negative valuations without rounding staggered coordinate
/// caps upward or confusing the field with its ring of integers.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicLocalFieldElement {
    field: PadicLocalField,
    valuation: Option<i64>,
    absolute_precision: i64,
    coordinates: Vec<Option<QadicElement>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFieldElementWire {
    #[serde(rename = "type")]
    type_id: String,
    field: PadicLocalField,
    valuation: Option<i64>,
    absolute_precision: i64,
    coordinates: Vec<Option<QadicElement>>,
}

impl PadicLocalFieldElement {
    /// Concrete local field presentation.
    pub fn field(&self) -> &PadicLocalField {
        &self.field
    }

    /// Staggered power-basis coordinates of the normalized integral unit.
    ///
    /// Zero has no normalized unit and therefore returns only absent
    /// coordinates. For a nonzero value, a missing coordinate means that its
    /// required relative cap is zero.
    pub fn normalized_coordinates(&self) -> &[Option<QadicElement>] {
        &self.coordinates
    }

    /// Exact valuation, or a lower bound for a value zero at available precision.
    pub fn valuation(&self) -> PadicValuation {
        self.valuation.map_or(
            PadicValuation::AtLeast(self.absolute_precision),
            PadicValuation::Exact,
        )
    }

    /// Known absolute precision in powers of the distinguished uniformizer.
    pub const fn known_absolute_precision(&self) -> i64 {
        self.absolute_precision
    }

    /// Precision relative to the exact valuation of a nonzero value.
    pub fn relative_precision(&self) -> Option<u64> {
        self.valuation.map(|valuation| {
            u64::try_from(self.absolute_precision - valuation)
                .expect("a normalized local value has positive relative precision")
        })
    }

    /// Whether the value vanishes at its honest absolute precision.
    pub fn is_zero(&self) -> bool {
        self.valuation.is_none()
    }

    /// Multiplicative inverse, preserving relative rather than absolute precision.
    pub fn inverse(&self) -> Result<Self, UnitError> {
        self.field.inverse(self)
    }

    /// Field trace to `Q_p` at the largest uniformly justified coefficient cap.
    pub fn trace(&self) -> Result<PadicElement, PadicLocalInvariantError> {
        self.field.trace_value(self)
    }

    /// Field norm to `Q_p` at the largest uniformly justified coefficient cap.
    pub fn norm(&self) -> Result<PadicElement, PadicLocalInvariantError> {
        self.field.norm_value(self)
    }

    /// Characteristic polynomial over `Q_p` of multiplication by this value.
    pub fn characteristic_polynomial(
        &self,
    ) -> Result<Polynomial<PadicField>, PadicLocalInvariantError> {
        self.field.characteristic_polynomial_value(self)
    }

    /// Minimal polynomial over `Q_p` when a full power basis is certified.
    ///
    /// A nonzero determinant for `1, a, ..., a^(n-1)` proves that `a`
    /// generates the degree-`n` extension, in which case its minimal and
    /// characteristic polynomials agree. If that determinant vanishes at the
    /// available cap, this operation fails rather than infer a smaller degree
    /// from finite-precision zeros.
    pub fn minimal_polynomial(&self) -> Result<Polynomial<PadicField>, PadicLocalInvariantError> {
        self.field.minimal_polynomial_value(self)
    }

    /// Raise this value to a nonnegative word-sized power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = self.field.one();
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self
                    .field
                    .mul(&result, &base)
                    .expect("powers stay in one local field context");
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self
                    .field
                    .mul(&base, &base)
                    .expect("powers stay in one local field context");
            }
        }
        result
    }

    /// Canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate staggered coordinate contexts.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn from_wire(wire: PadicLocalFieldElementWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        if wire.coordinates.len() != wire.field.ramification_index() {
            return Err(
                "local element coordinate count does not match its field degree".to_owned(),
            );
        }
        if wire.absolute_precision > wire.field.maximum_absolute_precision() {
            return Err("local element exceeds its context precision cap".to_owned());
        }
        let Some(valuation) = wire.valuation else {
            if wire.coordinates.iter().any(Option::is_some) {
                return Err("a zero local element cannot carry a normalized unit".to_owned());
            }
            return Ok(Self {
                field: wire.field,
                valuation: None,
                absolute_precision: wire.absolute_precision,
                coordinates: wire.coordinates,
            });
        };
        let relative_precision = wire
            .absolute_precision
            .checked_sub(valuation)
            .and_then(|precision| u64::try_from(precision).ok())
            .filter(|precision| *precision != 0)
            .ok_or_else(|| {
                "a nonzero local element must have positive relative precision".to_owned()
            })?;
        let rings = wire
            .field
            .coordinate_rings(relative_precision)
            .map_err(|error| error.to_string())?;
        for (index, (coordinate, ring)) in wire.coordinates.iter().zip(&rings).enumerate() {
            match (coordinate, ring) {
                (None, None) => {}
                (Some(value), Some(expected)) if value.ring() == Some(expected) => {}
                _ => {
                    return Err(format!(
                        "local element coordinate {index} does not carry its staggered q-adic context"
                    ));
                }
            }
        }
        let result = Self {
            field: wire.field,
            valuation: Some(valuation),
            absolute_precision: wire.absolute_precision,
            coordinates: wire.coordinates,
        };
        if result.field.integral_valuation(&result.coordinates) != Some(0) {
            return Err("local element unit coordinates are not normalized".to_owned());
        }
        Ok(result)
    }
}

impl Serialize for PadicLocalFieldElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PadicLocalFieldElementWire {
            type_id: Self::TYPE_ID.to_owned(),
            field: self.field.clone(),
            valuation: self.valuation,
            absolute_precision: self.absolute_precision,
            coordinates: self.coordinates.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicLocalFieldElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicLocalFieldElementWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for PadicLocalFieldElement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalFieldElement")
            .field("valuation", &self.valuation())
            .field("absolute_precision", &self.absolute_precision)
            .field("coordinates", &self.coordinates)
            .field("field", &self.field)
            .finish()
    }
}

impl fmt::Display for PadicLocalFieldElement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.valuation {
            None => write!(
                formatter,
                "O({}^{}) in {}",
                self.field.uniformizer_name(),
                self.absolute_precision,
                self.field
            ),
            Some(valuation) => write!(
                formatter,
                "{}^{} * {:?} + O({}^{}) in {}",
                self.field.uniformizer_name(),
                valuation,
                self.coordinates,
                self.field.uniformizer_name(),
                self.absolute_precision,
                self.field
            ),
        }
    }
}

impl SemanticType for PadicLocalFieldElement {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalFieldElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        output.extend_from_slice(&self.absolute_precision.to_be_bytes());
        match self.valuation {
            None => output.push(0),
            Some(valuation) => {
                output.push(1);
                output.extend_from_slice(&valuation.to_be_bytes());
            }
        }
        for coordinate in &self.coordinates {
            match coordinate {
                None => output.push(0),
                Some(value) => {
                    output.push(1);
                    append_component(&mut output, &value.canonical_bytes()?)?;
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FIELD_ELEMENT_DESCRIPTION
    }
}

impl Ring for PadicLocalField {
    type Element = PadicLocalFieldElement;

    fn contains(&self, value: &Self::Element) -> bool {
        self == value.field()
    }

    fn zero(&self) -> Self::Element {
        PadicLocalField::zero(self)
    }

    fn one(&self) -> Self::Element {
        PadicLocalField::one(self)
    }

    fn add(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        Ok(self.add_values(left, right))
    }

    fn sub(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        Ok(self.add_values(left, &self.neg_value(right)))
    }

    fn mul(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        Ok(self.mul_values(left, right))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(self.neg_value(value))
    }
}

impl CommutativeRing for PadicLocalField {}

impl UnitInversion for PadicLocalField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value)
    }
}

impl PrecisionContext for PadicLocalField {
    type Precision = PadicLocalPrecision;

    fn precision(&self) -> &Self::Precision {
        &self.inner.precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

impl ValuedRing for PadicLocalField {
    fn uniformizer(&self) -> Self::Element {
        PadicLocalField::uniformizer(self)
    }

    fn valuation_lower_bound(&self, value: &Self::Element) -> Option<i64> {
        match value.valuation() {
            PadicValuation::Exact(valuation) => Some(valuation),
            PadicValuation::AtLeast(_) => None,
        }
    }
}

fn dense_coefficients(
    polynomial: &Polynomial<QadicRing>,
) -> Result<Vec<QadicElement>, PadicLocalFieldError> {
    let mut terms = Vec::new();
    let mut degree = 0usize;
    for (exponent, coefficient) in polynomial.terms() {
        let power = exponent[0]
            .to_string()
            .parse::<usize>()
            .map_err(|_| PadicLocalFieldError::DegreeTooLarge)?;
        degree = degree.max(power);
        terms.push((power, coefficient.clone()));
    }
    if terms.is_empty() {
        return Err(PadicLocalFieldError::DegreeTooSmall);
    }
    let mut coefficients = vec![polynomial.ring().coefficient_ring().zero(); degree + 1];
    for (power, coefficient) in terms {
        coefficients[power] = coefficient;
    }
    Ok(coefficients)
}

fn staggered_coefficient_precision(
    local_precision: u64,
    coordinate: usize,
    ramification_index: usize,
) -> Result<u64, PadicLocalFieldError> {
    let coordinate = u64::try_from(coordinate).map_err(|_| PadicLocalFieldError::DegreeTooLarge)?;
    let ramification_index =
        u64::try_from(ramification_index).map_err(|_| PadicLocalFieldError::DegreeTooLarge)?;
    if local_precision <= coordinate {
        return Ok(0);
    }
    Ok((local_precision - coordinate).div_ceil(ramification_index))
}

fn validate_uniformizer_name(name: &str) -> Result<(), PadicLocalFieldError> {
    if name.is_empty() || name.chars().count() > 64 || name.chars().any(char::is_control) {
        Err(PadicLocalFieldError::InvalidUniformizerName)
    } else {
        Ok(())
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "p-adic local component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialRing};
    use crate::numthy::FiniteField;

    use super::*;

    fn eisenstein_field(
        coefficient_precision: u64,
        local_precision: u64,
        variable: &str,
        name: Option<&str>,
    ) -> Result<PadicLocalField, PadicLocalFieldError> {
        let residue = FiniteField::extension(3_u64, 2).unwrap();
        let coefficients = QadicRing::new(residue, coefficient_precision).unwrap();
        let polynomials = PolynomialRing::new(
            coefficients.clone(),
            [variable],
            MonomialOrder::Lexicographic,
        )
        .unwrap();
        let polynomial = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![2], coefficients.one()),
            ])
            .unwrap();
        PadicLocalField::from_eisenstein(
            coefficients,
            polynomial,
            PadicLocalPrecision::new(local_precision).unwrap(),
            name.map(str::to_owned),
        )
    }

    #[test]
    fn eisenstein_context_reports_ramified_and_unramified_data() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();

        assert_eq!(field.prime(), &Integer::from(3));
        assert_eq!(field.ramification_index(), 2);
        assert_eq!(field.residue_degree(), 2);
        assert_eq!(field.degree(), 4);
        assert_eq!(field.coordinate_precisions(), [4, 3]);
        assert_eq!(field.uniformizer_name(), "pi");
        assert_eq!(
            field.coefficient_field().residue_field(),
            field.residue_field()
        );
        assert_eq!(field.local_precision().absolute_uniformizer_precision(), 7);
    }

    #[test]
    fn staggered_integral_coordinates_do_not_claim_extra_digits() {
        let field = eisenstein_field(4, 7, "x", None).unwrap();
        let one = field.one();
        let uniformizer = field.uniformizer();

        assert_eq!(
            one.normalized_coordinates()[0]
                .as_ref()
                .unwrap()
                .known_absolute_precision(),
            4
        );
        assert_eq!(
            one.normalized_coordinates()[1]
                .as_ref()
                .unwrap()
                .known_absolute_precision(),
            3
        );
        assert_eq!(
            uniformizer.normalized_coordinates()[1]
                .as_ref()
                .unwrap()
                .known_absolute_precision(),
            3
        );
        assert!(field.zero().is_zero());
        assert!(!one.is_zero());

        let coarse = eisenstein_field(4, 1, "x", None).unwrap();
        assert_eq!(coarse.coordinate_precisions(), [1, 0]);
        assert_eq!(coarse.uniformizer(), coarse.zero());
    }

    #[test]
    fn presentation_names_are_metadata_but_coefficients_are_identity() {
        let x = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let y = eisenstein_field(4, 7, "y", Some("tau")).unwrap();

        assert_eq!(x, y);
        assert_eq!(x.canonical_bytes().unwrap(), y.canonical_bytes().unwrap());
        assert_ne!(x.uniformizer_name(), y.uniformizer_name());
    }

    #[test]
    fn construction_rejects_every_eisenstein_and_precision_failure() {
        let residue = FiniteField::extension(3_u64, 2).unwrap();
        let coefficients = QadicRing::new(residue, 3).unwrap();
        let polynomials =
            PolynomialRing::new(coefficients.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
        let precision = PadicLocalPrecision::new(7).unwrap();

        let insufficient = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![2], coefficients.one()),
            ])
            .unwrap();
        assert!(matches!(
            PadicLocalField::from_eisenstein(coefficients.clone(), insufficient, precision, None),
            Err(PadicLocalFieldError::InsufficientCoefficientPrecision { .. })
        ));

        let nonmonic = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![2], coefficients.element(2)),
            ])
            .unwrap();
        assert!(matches!(
            PadicLocalField::from_eisenstein(
                coefficients.clone(),
                nonmonic,
                PadicLocalPrecision::new(5).unwrap(),
                None
            ),
            Err(PadicLocalFieldError::DefiningPolynomialNotMonic)
        ));

        let bad_middle = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![1], coefficients.element(1)),
                (vec![2], coefficients.one()),
            ])
            .unwrap();
        assert!(matches!(
            PadicLocalField::from_eisenstein(
                coefficients.clone(),
                bad_middle,
                PadicLocalPrecision::new(5).unwrap(),
                None
            ),
            Err(PadicLocalFieldError::NonEisensteinCoefficient { degree: 1, .. })
        ));

        let bad_constant = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(9)),
                (vec![2], coefficients.one()),
            ])
            .unwrap();
        assert!(matches!(
            PadicLocalField::from_eisenstein(
                coefficients,
                bad_constant,
                PadicLocalPrecision::new(5).unwrap(),
                None
            ),
            Err(PadicLocalFieldError::ConstantCoefficientNotValuationOne { .. })
        ));
        assert_eq!(
            PadicLocalPrecision::new(0),
            Err(PadicLocalPrecisionError::Zero)
        );
    }

    #[test]
    fn contexts_and_integral_elements_round_trip_and_revalidate() {
        let field = eisenstein_field(4, 7, "x", Some("varpi")).unwrap();
        let element = field
            .integral_element(&[
                field.coefficient_ring().element(5),
                field.coefficient_ring().element(7),
            ])
            .unwrap();

        assert_eq!(
            PadicLocalField::from_json(&field.to_json().unwrap()).unwrap(),
            field
        );
        assert_eq!(
            PadicLocalFieldElement::from_json(&element.to_json().unwrap()).unwrap(),
            element
        );
        assert_eq!(
            PadicLocalPrecision::from_json(&field.local_precision().to_json().unwrap()).unwrap(),
            field.local_precision()
        );
        assert!(
            PadicLocalField::from_json(&field.to_json().unwrap().replace(
                "\"absolute_uniformizer_precision\":7",
                "\"absolute_uniformizer_precision\":0"
            ))
            .is_err()
        );
    }

    #[test]
    fn coordinate_construction_checks_length_and_source_context() {
        let field = eisenstein_field(4, 7, "x", None).unwrap();
        assert!(matches!(
            field.integral_element(&[
                field.coefficient_ring().zero(),
                field.coefficient_ring().zero(),
                field.coefficient_ring().zero(),
            ]),
            Err(PadicLocalFieldError::TooManyCoordinates { .. })
        ));

        let other = QadicRing::new(field.residue_field().clone(), 3).unwrap();
        assert!(matches!(
            field.integral_element(&[other.zero()]),
            Err(PadicLocalFieldError::CoordinateContext { index: 0, .. })
        ));
    }

    #[test]
    fn quotient_arithmetic_respects_the_eisenstein_relation() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let pi = field.uniformizer();
        let three = field
            .integral_element(&[field.coefficient_ring().element(3)])
            .unwrap();
        let pi_squared = field.mul(&pi, &pi).unwrap();

        assert_eq!(pi.valuation(), PadicValuation::Exact(1));
        assert_eq!(pi.known_absolute_precision(), 7);
        assert_eq!(pi.relative_precision(), Some(6));
        assert_eq!(pi_squared.valuation(), PadicValuation::Exact(2));
        assert_eq!(pi_squared.known_absolute_precision(), 7);
        assert!(field.add(&pi_squared, &three).unwrap().is_zero());

        let one_plus_pi = field.add(&field.one(), &pi).unwrap();
        let one_minus_pi = field.sub(&field.one(), &pi).unwrap();
        let four = field
            .integral_element(&[field.coefficient_ring().element(4)])
            .unwrap();
        assert_eq!(field.mul(&one_plus_pi, &one_minus_pi).unwrap(), four);
        assert_eq!(
            field.neg(&field.neg(&one_plus_pi).unwrap()).unwrap(),
            one_plus_pi
        );
    }

    #[test]
    fn inverses_preserve_relative_and_adjust_absolute_precision() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let pi = field.uniformizer();
        let inverse_pi = pi.inverse().unwrap();

        assert_eq!(inverse_pi.valuation(), PadicValuation::Exact(-1));
        assert_eq!(inverse_pi.relative_precision(), Some(6));
        assert_eq!(inverse_pi.known_absolute_precision(), 5);
        assert_eq!(inverse_pi.inverse().unwrap(), pi);

        let product = field.mul(&pi, &inverse_pi).unwrap();
        assert_eq!(product.valuation(), PadicValuation::Exact(0));
        assert_eq!(product.known_absolute_precision(), 6);
        assert_eq!(
            product,
            field.restrict_value(&field.one(), product.known_absolute_precision())
        );

        let unit = field.add(&field.one(), &pi).unwrap();
        assert_eq!(
            field.mul(&unit, &unit.inverse().unwrap()).unwrap(),
            field.one()
        );
        assert_eq!(
            PadicLocalFieldElement::from_json(&inverse_pi.to_json().unwrap()).unwrap(),
            inverse_pi
        );
        assert!(field.zero().inverse().is_err());
    }

    #[test]
    fn checked_scalar_embeddings_scale_valuations_and_keep_extra_unit_digits() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let qadic_inverse = field
            .coefficient_field()
            .inverse(&field.coefficient_field().element(3))
            .unwrap();
        let embedded = field.element_from_qadic(&qadic_inverse).unwrap();

        assert_eq!(embedded.valuation(), PadicValuation::Exact(-2));
        assert_eq!(embedded.known_absolute_precision(), 4);
        assert_eq!(embedded.relative_precision(), Some(6));
        assert_eq!(
            embedded.normalized_coordinates()[0]
                .as_ref()
                .unwrap()
                .known_absolute_precision(),
            3
        );
        let three = field
            .integral_element(&[field.coefficient_ring().element(3)])
            .unwrap();
        assert_eq!(embedded.inverse().unwrap(), three);

        let padic_inverse = field
            .coefficient_field()
            .coefficient_field()
            .inverse(&field.coefficient_field().coefficient_field().element(3))
            .unwrap();
        assert_eq!(field.element_from_padic(&padic_inverse).unwrap(), embedded);

        let wrong = QadicField::new(field.residue_field().clone(), 3).unwrap();
        assert!(matches!(
            field.element_from_qadic(&wrong.one()),
            Err(PadicLocalFieldError::ScalarContextMismatch(_))
        ));
    }

    #[test]
    fn residue_and_teichmuller_use_the_explicit_residue_presentation() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let residue = field.residue_field().generator();
        let lift = field.teichmuller(&residue).unwrap();

        assert_eq!(field.residue(&lift).unwrap(), Some(residue));
        assert_eq!(
            field.residue(&field.uniformizer()).unwrap(),
            Some(field.residue_field().zero())
        );
        assert_eq!(
            field
                .residue(&field.uniformizer().inverse().unwrap())
                .unwrap(),
            None
        );
        assert_eq!(
            field.residue(&field.zero()).unwrap(),
            Some(field.residue_field().zero())
        );
    }

    #[test]
    fn precision_restriction_is_checked_distinct_and_induces_composite_maps() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let restriction = field
            .precision_restriction(PadicLocalPrecision::new(4).unwrap())
            .unwrap();

        assert_ne!(restriction.source(), restriction.target());
        assert_eq!(
            restriction.apply(&field.uniformizer()).unwrap(),
            restriction.target().uniformizer()
        );
        assert!(restriction
            .apply(&field.uniformizer().pow(4))
            .unwrap()
            .is_zero());
        assert_eq!(
            PadicLocalFieldRestriction::from_json(&restriction.to_json().unwrap()).unwrap(),
            restriction
        );
        assert!(matches!(
            restriction
                .target()
                .precision_restriction(PadicLocalPrecision::new(5).unwrap()),
            Err(PadicLocalFieldRestrictionError::InvalidPrecision { .. })
        ));

        let polynomials =
            PolynomialRing::new(field.clone(), ["z"], MonomialOrder::Lexicographic).unwrap();
        let polynomial = polynomials
            .from_u64_terms([(vec![0], field.uniformizer()), (vec![2], field.one())])
            .unwrap();
        let mapped = polynomial.map_coefficients(&restriction).unwrap();
        assert_eq!(mapped.ring().coefficient_ring(), restriction.target());
        assert_eq!(
            mapped.coefficient(&[Integer::zero()]).unwrap(),
            restriction.target().uniformizer()
        );
    }

    #[test]
    fn tower_invariants_match_the_hand_checked_ramified_unramified_example() {
        let field = eisenstein_field(4, 7, "x", Some("pi")).unwrap();
        let pi = field.uniformizer();

        let trace = pi.trace().unwrap();
        let norm = pi.norm().unwrap();
        let base = trace
            .field()
            .expect("local invariants land in a p-adic field");
        assert_eq!(base.absolute_precision(), 3);
        assert_eq!(trace, base.zero());
        assert_eq!(norm, base.element(9));
        assert_eq!(field.one().trace().unwrap(), base.element(4));
        assert_eq!(field.one().norm().unwrap(), base.one());

        let characteristic = pi.characteristic_polynomial().unwrap();
        assert_eq!(characteristic.ring().coefficient_ring(), base);
        assert_eq!(
            characteristic.coefficient(&[Integer::from(4)]).unwrap(),
            base.one()
        );
        assert_eq!(
            characteristic.coefficient(&[Integer::from(3)]).unwrap(),
            base.zero()
        );
        assert_eq!(
            characteristic.coefficient(&[Integer::from(2)]).unwrap(),
            base.element(6)
        );
        assert_eq!(
            characteristic.coefficient(&[Integer::one()]).unwrap(),
            base.zero()
        );
        assert_eq!(
            characteristic.coefficient(&[Integer::zero()]).unwrap(),
            base.element(9)
        );

        let inverse_pi = pi.inverse().unwrap();
        assert_eq!(
            inverse_pi.norm().unwrap().valuation(),
            PadicValuation::Exact(-2)
        );
        assert_eq!(
            inverse_pi.trace().unwrap().valuation(),
            PadicValuation::AtLeast(2)
        );

        let two = field
            .integral_element(&[field.coefficient_ring().element(2)])
            .unwrap();
        let left = field.add(&field.one(), &pi).unwrap();
        let right = field.add(&two, &pi).unwrap();
        let sum = field.add(&left, &right).unwrap();
        let trace_sum = sum.trace().unwrap();
        assert_eq!(
            trace_sum,
            base.add(&left.trace().unwrap(), &right.trace().unwrap())
                .unwrap()
        );
        let product = field.mul(&left, &right).unwrap();
        assert_eq!(
            product.norm().unwrap(),
            base.mul(&left.norm().unwrap(), &right.norm().unwrap())
                .unwrap()
        );
        let product_characteristic = product.characteristic_polynomial().unwrap();
        assert_eq!(
            product_characteristic
                .coefficient(&[Integer::zero()])
                .unwrap(),
            product.norm().unwrap()
        );

        let theta = field
            .element_from_qadic(&field.coefficient_field().generator())
            .unwrap();
        let primitive = field.add(&theta, &pi).unwrap();
        assert_eq!(
            primitive.minimal_polynomial().unwrap(),
            primitive.characteristic_polynomial().unwrap()
        );
        assert!(matches!(
            pi.minimal_polynomial(),
            Err(PadicLocalInvariantError::MinimalPolynomialNotCertified { degree: 4, .. })
        ));

        let coarse = eisenstein_field(4, 1, "x", Some("pi")).unwrap();
        assert!(matches!(
            coarse.one().trace(),
            Err(PadicLocalInvariantError::InsufficientPrecision { .. })
        ));
    }

    #[test]
    fn coefficient_frobenius_targets_the_conjugate_presentation() {
        let residue = FiniteField::extension(3_u64, 2).unwrap();
        let coefficients = QadicRing::new(residue, 4).unwrap();
        let generator = coefficients.generator();
        let middle = coefficients
            .mul(&coefficients.element(3), &generator)
            .unwrap();
        let polynomials =
            PolynomialRing::new(coefficients.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
        let eisenstein = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![1], middle),
                (vec![2], coefficients.one()),
            ])
            .unwrap();
        let field = PadicLocalField::from_eisenstein(
            coefficients.clone(),
            eisenstein,
            PadicLocalPrecision::new(7).unwrap(),
            Some("pi".to_owned()),
        )
        .unwrap();

        let conjugation = field.coefficient_frobenius_conjugate(1).unwrap();
        assert_ne!(conjugation.source(), conjugation.target());
        assert!(!conjugation.is_endomorphism());
        assert_eq!(
            conjugation.apply(&field.uniformizer()).unwrap(),
            conjugation.target().uniformizer()
        );
        let value = field
            .integral_element(&[generator.clone(), coefficients.one()])
            .unwrap();
        let expected = conjugation
            .target()
            .integral_element(&[generator.frobenius(1), coefficients.one()])
            .unwrap();
        assert_eq!(conjugation.apply(&value).unwrap(), expected);

        let product = field.mul(&value, &field.uniformizer()).unwrap();
        assert_eq!(
            conjugation.apply(&product).unwrap(),
            conjugation
                .target()
                .mul(
                    &conjugation.apply(&value).unwrap(),
                    &conjugation.apply(&field.uniformizer()).unwrap(),
                )
                .unwrap()
        );
        let outer = conjugation
            .target()
            .coefficient_frobenius_conjugate(1)
            .unwrap();
        let composed = outer.compose(&conjugation).unwrap();
        assert_eq!(composed.source(), &field);
        assert_eq!(composed.target(), &field);
        assert!(composed.is_endomorphism());
        assert_eq!(composed.apply(&value).unwrap(), value);
        assert_eq!(
            PadicLocalFieldCoefficientFrobenius::from_json(&conjugation.to_json().unwrap())
                .unwrap(),
            conjugation
        );
    }

    #[test]
    fn degree_one_uniformizer_uses_the_linear_relation() {
        let residue = FiniteField::prime(3_u8).unwrap();
        let coefficients = QadicRing::new(residue, 4).unwrap();
        let polynomials =
            PolynomialRing::new(coefficients.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
        let polynomial = polynomials
            .from_u64_terms([
                (vec![0], coefficients.element(3)),
                (vec![1], coefficients.one()),
            ])
            .unwrap();
        let field = PadicLocalField::from_eisenstein(
            coefficients,
            polynomial,
            PadicLocalPrecision::new(4).unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(field.ramification_index(), 1);
        let expected = field
            .integral_element(&[field.coefficient_ring().element(-3)])
            .unwrap();
        assert_eq!(field.uniformizer(), expected);
        assert_eq!(field.uniformizer().valuation(), PadicValuation::Exact(1));
        assert_eq!(field.uniformizer().relative_precision(), Some(3));
    }
}
