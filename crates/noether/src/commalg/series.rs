// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Context-parametric fixed-truncation power series with sticky native storage.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use crate::{
    CoefficientMap, CoefficientMappingError, CommutativeRing, ContextMismatchError, Integer,
    NoetherError, PrecisionContext, Ring, SemanticType, TypeDescription, UnitError, UnitInversion,
};

use super::polynomial::PolyCoreRingLane;
use super::{GradingError, WeightedGrading};

const PACKED_SERIES_TERM_PRODUCT_CROSSOVER: usize = 64;

static SERIES_TRUNCATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/SeriesTruncation@1",
    rust_path: "noether::commalg::SeriesTruncation",
    python_path: Some("noether.commalg.SeriesTruncation"),
};

static POWER_SERIES_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PowerSeriesRing@1",
    rust_path: "noether::commalg::PowerSeriesRing",
    python_path: Some("noether.commalg.PowerSeriesRing"),
};

static POWER_SERIES_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PowerSeries@1",
    rust_path: "noether::commalg::PowerSeries",
    python_path: Some("noether.commalg.PowerSeries"),
};

/// The finite monomial region retained by a power-series context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeriesTruncationMode {
    /// One variable, retaining exponents strictly below one precision bound.
    Univariate,
    /// Retain monomials whose total degree is strictly below one bound.
    TotalDegree,
    /// Retain monomials whose exponent in every variable is below its bound.
    Box,
    /// Retain monomials whose weighted degree is strictly below one bound.
    WeightedDegree,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TruncationShape {
    Univariate {
        precision: u64,
    },
    TotalDegree {
        bound: u64,
    },
    Box {
        bounds: Vec<u64>,
    },
    WeightedDegree {
        grading: WeightedGrading,
        bound: u64,
    },
}

/// A valid finite truncation policy fixed before series allocation.
///
/// Every bound is strict: univariate precision `N` retains `1, x, ..., x^(N-1)`;
/// total or weighted bound `N` retains monomials of degree `< N`; and a box
/// retains exponent vectors satisfying `e_i < bounds[i]` for every `i`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SeriesTruncation {
    shape: TruncationShape,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeriesTruncationWire {
    #[serde(rename = "type")]
    type_id: String,
    mode: String,
    #[serde(default)]
    bound: Option<u64>,
    #[serde(default)]
    bounds: Option<Vec<u64>>,
    #[serde(default)]
    grading: Option<WeightedGrading>,
}

/// Failure to construct or attach a finite series truncation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeriesTruncationError {
    /// A strict truncation bound was zero.
    ZeroBound,
    /// A box or weighted policy supplied no per-variable data.
    EmptyVariableData,
    /// A box bound was zero.
    ZeroBoxBound {
        /// Zero-based variable index.
        variable: usize,
    },
    /// A weighted-degree weight was zero.
    ZeroWeight {
        /// Zero-based variable index.
        variable: usize,
    },
    /// A first-class grading could not define the requested truncation.
    Grading(GradingError),
    /// The policy's per-variable data did not match the ring arity.
    ArityMismatch {
        /// Number of variables required by the policy.
        expected: usize,
        /// Number of variables in the power-series ring.
        actual: usize,
    },
}

impl fmt::Display for SeriesTruncationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBound => f.write_str("series truncation bound must be at least one"),
            Self::EmptyVariableData => {
                f.write_str("series truncation requires at least one variable bound or weight")
            }
            Self::ZeroBoxBound { variable } => {
                write!(
                    f,
                    "box truncation bound for variable {variable} must be positive"
                )
            }
            Self::ZeroWeight { variable } => {
                write!(f, "series weight for variable {variable} must be positive")
            }
            Self::Grading(error) => error.fmt(f),
            Self::ArityMismatch { expected, actual } => write!(
                f,
                "series truncation has arity {expected}; power-series ring has arity {actual}"
            ),
        }
    }
}

impl std::error::Error for SeriesTruncationError {}

impl SeriesTruncation {
    /// Retain one-variable powers `x^e` with `e < precision`.
    pub fn univariate(precision: u64) -> Result<Self, SeriesTruncationError> {
        if precision == 0 {
            return Err(SeriesTruncationError::ZeroBound);
        }
        Ok(Self {
            shape: TruncationShape::Univariate { precision },
        })
    }

    /// Retain exponent vectors whose total degree is less than `bound`.
    pub fn total_degree(bound: u64) -> Result<Self, SeriesTruncationError> {
        if bound == 0 {
            return Err(SeriesTruncationError::ZeroBound);
        }
        Ok(Self {
            shape: TruncationShape::TotalDegree { bound },
        })
    }

    /// Retain exponent vectors satisfying `exponent[i] < bounds[i]`.
    pub fn box_bounds(bounds: Vec<u64>) -> Result<Self, SeriesTruncationError> {
        if bounds.is_empty() {
            return Err(SeriesTruncationError::EmptyVariableData);
        }
        if let Some(variable) = bounds.iter().position(|bound| *bound == 0) {
            return Err(SeriesTruncationError::ZeroBoxBound { variable });
        }
        Ok(Self {
            shape: TruncationShape::Box { bounds },
        })
    }

    /// Retain exponent vectors of weighted degree less than `bound`.
    pub fn weighted_degree(weights: Vec<u64>, bound: u64) -> Result<Self, SeriesTruncationError> {
        let grading = WeightedGrading::new(weights).map_err(|error| match error {
            GradingError::NoVariables => SeriesTruncationError::EmptyVariableData,
            GradingError::ZeroWeight { variable } => SeriesTruncationError::ZeroWeight { variable },
            error @ (GradingError::ArityMismatch { .. }
            | GradingError::NegativeExponent { .. }
            | GradingError::DegreeOverflow
            | GradingError::RankMismatch { .. }
            | GradingError::NonpositiveFunctional { .. }) => SeriesTruncationError::Grading(error),
        })?;
        Self::graded(grading, bound)
    }

    /// Retain exponent vectors whose degree in `grading` is less than `bound`.
    pub fn graded(grading: WeightedGrading, bound: u64) -> Result<Self, SeriesTruncationError> {
        if bound == 0 {
            return Err(SeriesTruncationError::ZeroBound);
        }
        Ok(Self {
            shape: TruncationShape::WeightedDegree { grading, bound },
        })
    }

    /// Structural family of this truncation.
    pub fn mode(&self) -> SeriesTruncationMode {
        match self.shape {
            TruncationShape::Univariate { .. } => SeriesTruncationMode::Univariate,
            TruncationShape::TotalDegree { .. } => SeriesTruncationMode::TotalDegree,
            TruncationShape::Box { .. } => SeriesTruncationMode::Box,
            TruncationShape::WeightedDegree { .. } => SeriesTruncationMode::WeightedDegree,
        }
    }

    /// Single strict bound for univariate, total-degree, or weighted-degree
    /// truncation; `None` for a box.
    pub fn bound(&self) -> Option<u64> {
        match self.shape {
            TruncationShape::Univariate { precision } => Some(precision),
            TruncationShape::TotalDegree { bound }
            | TruncationShape::WeightedDegree { bound, .. } => Some(bound),
            TruncationShape::Box { .. } => None,
        }
    }

    /// Per-variable strict bounds for a box truncation.
    pub fn bounds(&self) -> Option<&[u64]> {
        match &self.shape {
            TruncationShape::Box { bounds } => Some(bounds),
            _ => None,
        }
    }

    /// Per-variable weights for weighted-degree truncation.
    pub fn weights(&self) -> Option<&[u64]> {
        match &self.shape {
            TruncationShape::WeightedDegree { grading, .. } => Some(grading.weights()),
            _ => None,
        }
    }

    /// Explicit grading for a weighted-degree truncation.
    pub fn explicit_grading(&self) -> Option<&WeightedGrading> {
        match &self.shape {
            TruncationShape::WeightedDegree { grading, .. } => Some(grading),
            _ => None,
        }
    }

    /// Serialize the exact finite monomial region as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a finite monomial region.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Whether this truncation retains the supplied exponent vector.
    pub fn contains_exponent(&self, exponent: &[u64]) -> bool {
        match &self.shape {
            TruncationShape::Univariate { precision } => {
                exponent.len() == 1 && exponent[0] < *precision
            }
            TruncationShape::TotalDegree { bound } => {
                checked_weighted_degree(exponent, core::iter::repeat_n(1_u64, exponent.len()))
                    .is_some_and(|degree| degree < *bound)
            }
            TruncationShape::Box { bounds } => {
                exponent.len() == bounds.len()
                    && exponent
                        .iter()
                        .zip(bounds)
                        .all(|(power, bound)| power < bound)
            }
            TruncationShape::WeightedDegree { grading, bound } => {
                grading.degree(exponent).is_ok_and(|degree| degree < *bound)
            }
        }
    }

    /// Whether `self` contains at least every coefficient position retained by
    /// `reference`. Policies of different structural families are deliberately
    /// incomparable.
    pub fn refines(&self, reference: &Self) -> bool {
        match (&self.shape, &reference.shape) {
            (
                TruncationShape::Univariate {
                    precision: candidate,
                },
                TruncationShape::Univariate {
                    precision: reference,
                },
            )
            | (
                TruncationShape::TotalDegree { bound: candidate },
                TruncationShape::TotalDegree { bound: reference },
            ) => candidate >= reference,
            (
                TruncationShape::Box { bounds: candidate },
                TruncationShape::Box { bounds: reference },
            ) => {
                candidate.len() == reference.len()
                    && candidate
                        .iter()
                        .zip(reference)
                        .all(|(candidate, reference)| candidate >= reference)
            }
            (
                TruncationShape::WeightedDegree {
                    grading: candidate_grading,
                    bound: candidate,
                },
                TruncationShape::WeightedDegree {
                    grading: reference_grading,
                    bound: reference,
                },
            ) => candidate_grading == reference_grading && candidate >= reference,
            _ => false,
        }
    }

    pub(crate) fn validate_arity(&self, arity: usize) -> Result<(), SeriesTruncationError> {
        let expected = match &self.shape {
            TruncationShape::Univariate { .. } => Some(1),
            TruncationShape::Box { bounds } => Some(bounds.len()),
            TruncationShape::WeightedDegree { grading, .. } => Some(grading.arity()),
            TruncationShape::TotalDegree { .. } => None,
        };
        if let Some(expected) = expected {
            if expected != arity {
                return Err(SeriesTruncationError::ArityMismatch {
                    expected,
                    actual: arity,
                });
            }
        }
        Ok(())
    }

    fn coefficient_capacity_bound(&self, arity: usize) -> Integer {
        match &self.shape {
            TruncationShape::Univariate { precision } => Integer::from(*precision),
            TruncationShape::TotalDegree { bound } => (0..arity)
                .fold(Integer::one(), |product, _| {
                    &product * &Integer::from(*bound)
                }),
            TruncationShape::Box { bounds } => {
                bounds.iter().fold(Integer::one(), |product, bound| {
                    &product * &Integer::from(*bound)
                })
            }
            TruncationShape::WeightedDegree { grading, bound } => {
                grading
                    .weights()
                    .iter()
                    .fold(Integer::one(), |product, weight| {
                        let positions = (bound - 1) / weight + 1;
                        &product * &Integer::from(positions)
                    })
            }
        }
    }

    fn nilpotence_bound(&self) -> u64 {
        match &self.shape {
            TruncationShape::Univariate { precision }
            | TruncationShape::TotalDegree { bound: precision } => *precision,
            TruncationShape::Box { bounds } => bounds.iter().fold(1_u64, |sum, bound| {
                sum.saturating_add(bound.saturating_sub(1))
            }),
            TruncationShape::WeightedDegree { grading, bound } => {
                let minimum = grading.weights().iter().copied().min().unwrap_or(1);
                (bound - 1) / minimum + 1
            }
        }
    }

    fn weighted_order(&self, exponent: &[u64]) -> Option<u64> {
        match &self.shape {
            TruncationShape::WeightedDegree { grading, .. } => grading.degree(exponent).ok(),
            TruncationShape::TotalDegree { .. } => {
                checked_weighted_degree(exponent, core::iter::repeat_n(1_u64, exponent.len()))
            }
            _ => None,
        }
    }

    fn derivative_truncation(&self, variable: usize) -> Option<Self> {
        let shape = match &self.shape {
            TruncationShape::Univariate { precision } => TruncationShape::Univariate {
                precision: precision.checked_sub(1).filter(|value| *value != 0)?,
            },
            TruncationShape::TotalDegree { bound } => TruncationShape::TotalDegree {
                bound: bound.checked_sub(1).filter(|value| *value != 0)?,
            },
            TruncationShape::Box { bounds } => {
                let mut target = bounds.clone();
                target[variable] = target[variable]
                    .checked_sub(1)
                    .filter(|value| *value != 0)?;
                TruncationShape::Box { bounds: target }
            }
            TruncationShape::WeightedDegree { grading, bound } => TruncationShape::WeightedDegree {
                grading: grading.clone(),
                bound: bound
                    .checked_sub(grading.weights()[variable])
                    .filter(|value| *value != 0)?,
            },
        };
        Some(Self { shape })
    }
}

impl Serialize for SeriesTruncation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.shape {
            TruncationShape::Univariate { precision } => {
                let mut state = serializer.serialize_struct("SeriesTruncation", 3)?;
                state.serialize_field("type", Self::TYPE_ID)?;
                state.serialize_field("mode", "univariate")?;
                state.serialize_field("bound", precision)?;
                state.end()
            }
            TruncationShape::TotalDegree { bound } => {
                let mut state = serializer.serialize_struct("SeriesTruncation", 3)?;
                state.serialize_field("type", Self::TYPE_ID)?;
                state.serialize_field("mode", "total_degree")?;
                state.serialize_field("bound", bound)?;
                state.end()
            }
            TruncationShape::Box { bounds } => {
                let mut state = serializer.serialize_struct("SeriesTruncation", 3)?;
                state.serialize_field("type", Self::TYPE_ID)?;
                state.serialize_field("mode", "box")?;
                state.serialize_field("bounds", bounds)?;
                state.end()
            }
            TruncationShape::WeightedDegree { grading, bound } => {
                let mut state = serializer.serialize_struct("SeriesTruncation", 4)?;
                state.serialize_field("type", Self::TYPE_ID)?;
                state.serialize_field("mode", "weighted_degree")?;
                state.serialize_field("grading", grading)?;
                state.serialize_field("bound", bound)?;
                state.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SeriesTruncation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SeriesTruncationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        match (wire.mode.as_str(), wire.bound, wire.bounds, wire.grading) {
            ("univariate", Some(bound), None, None) => Self::univariate(bound),
            ("total_degree", Some(bound), None, None) => Self::total_degree(bound),
            ("box", None, Some(bounds), None) => Self::box_bounds(bounds),
            ("weighted_degree", Some(bound), None, Some(grading)) => Self::graded(grading, bound),
            _ => {
                return Err(serde::de::Error::custom(
                    "serialized series truncation fields do not match its mode",
                ));
            }
        }
        .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for SeriesTruncation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.shape {
            TruncationShape::Univariate { precision } => write!(f, "order < {precision}"),
            TruncationShape::TotalDegree { bound } => write!(f, "total degree < {bound}"),
            TruncationShape::Box { bounds } => write!(f, "box {bounds:?}"),
            TruncationShape::WeightedDegree { grading, bound } => {
                write!(f, "weighted degree {:?} < {bound}", grading.weights())
            }
        }
    }
}

impl SemanticType for SeriesTruncation {
    const TYPE_ID: &'static str = "noether/commalg/SeriesTruncation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        match &self.shape {
            TruncationShape::Univariate { precision } => {
                output.push(0);
                output.extend_from_slice(&precision.to_be_bytes());
            }
            TruncationShape::TotalDegree { bound } => {
                output.push(1);
                output.extend_from_slice(&bound.to_be_bytes());
            }
            TruncationShape::Box { bounds } => {
                output.push(2);
                append_u64_slice(&mut output, bounds)?;
            }
            TruncationShape::WeightedDegree { grading, bound } => {
                output.push(3);
                append_u64_slice(&mut output, grading.weights())?;
                output.extend_from_slice(&bound.to_be_bytes());
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SERIES_TRUNCATION_DESCRIPTION
    }
}

/// Failure to construct a power-series coefficient context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeriesContextError {
    /// A power-series ring requires at least one indeterminate.
    NoVariables,
    /// One variable name was empty.
    EmptyVariableName {
        /// Zero-based variable index.
        variable: usize,
    },
    /// Two variables had the same name.
    DuplicateVariableName {
        /// Repeated name.
        name: String,
    },
    /// The finite truncation did not fit the variable list.
    Truncation(SeriesTruncationError),
}

impl fmt::Display for SeriesContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVariables => f.write_str("power-series ring requires at least one variable"),
            Self::EmptyVariableName { variable } => {
                write!(f, "power-series variable {variable} has an empty name")
            }
            Self::DuplicateVariableName { name } => {
                write!(f, "power-series variable name {name:?} is duplicated")
            }
            Self::Truncation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SeriesContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Truncation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SeriesTruncationError> for SeriesContextError {
    fn from(value: SeriesTruncationError) -> Self {
        Self::Truncation(value)
    }
}

/// Failure to construct, restrict, divide, differentiate, or compose power
/// series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerSeriesError {
    /// A value belonged to a different power-series or coefficient context.
    ContextMismatch(ContextMismatchError),
    /// An exponent vector had the wrong number of entries.
    ExponentArity {
        /// Required number of exponents.
        expected: usize,
        /// Supplied number of exponents.
        actual: usize,
    },
    /// A coefficient position lay outside the fixed truncation region.
    ExponentOutsideTruncation {
        /// Rejected exponent vector.
        exponent: Vec<u64>,
    },
    /// A coefficient did not belong to the series' coefficient ring.
    CoefficientOutsideContext,
    /// A variable index was outside the series-ring arity.
    VariableIndex {
        /// Rejected zero-based index.
        variable: usize,
        /// Number of variables in the context.
        arity: usize,
    },
    /// The requested truncation did not discard information from the source.
    InvalidRestriction {
        /// Source precision.
        source: SeriesTruncation,
        /// Requested target precision.
        target: SeriesTruncation,
    },
    /// Substitution did not preserve the source truncation ideal.
    InvalidComposition {
        /// Variable whose image violated the truncation relation.
        variable: usize,
        /// Stable mathematical reason for rejection.
        reason: &'static str,
    },
    /// Differentiation consumed all available precision in one variable.
    InsufficientDerivativePrecision {
        /// Variable with no positive target truncation left.
        variable: usize,
    },
    /// Exact division was requested by the zero series.
    DivisionByZero,
    /// The divisor was not a monomial times a unit series.
    ///
    /// This is the canonical class for which division is determined by the
    /// retained coefficients without choosing a term order or an arbitrary
    /// quotient lift.
    DivisorNotMonomialTimesUnit,
    /// Multiplication of the canonical quotient did not recover the dividend.
    NotExactlyDivisible,
}

impl fmt::Display for PowerSeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::ExponentArity { expected, actual } => write!(
                f,
                "power-series exponent has length {actual}; expected {expected}"
            ),
            Self::ExponentOutsideTruncation { exponent } => {
                write!(f, "exponent {exponent:?} lies outside the fixed series truncation")
            }
            Self::CoefficientOutsideContext => {
                f.write_str("power-series coefficient belongs to another ring context")
            }
            Self::VariableIndex { variable, arity } => {
                write!(f, "series variable index {variable} is outside arity {arity}")
            }
            Self::InvalidRestriction { source, target } => write!(
                f,
                "cannot restrict series truncation {source} to incomparable or finer {target}"
            ),
            Self::InvalidComposition { variable, reason } => write!(
                f,
                "image of series variable {variable} does not preserve the truncation ideal: {reason}"
            ),
            Self::InsufficientDerivativePrecision { variable } => write!(
                f,
                "differentiation in variable {variable} leaves no positive series precision"
            ),
            Self::DivisionByZero => f.write_str("cannot divide a power series by zero"),
            Self::DivisorNotMonomialTimesUnit => f.write_str(
                "power-series divisor is not a monomial times a series with unit constant coefficient",
            ),
            Self::NotExactlyDivisible => f.write_str(
                "power-series divisor does not multiply the canonical quotient to the dividend",
            ),
        }
    }
}

impl std::error::Error for PowerSeriesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for PowerSeriesError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A multivariate power-series ring with a finite immutable truncation.
#[derive(Clone)]
pub struct PowerSeriesRing<R: CommutativeRing> {
    coefficient_ring: R,
    variables: Vec<String>,
    truncation: SeriesTruncation,
    backend: Option<Arc<super::polynomial_flint::FlintSeriesContext>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct PowerSeriesRingWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    coefficient_ring: R,
    variables: Vec<String>,
    truncation: SeriesTruncation,
}

impl<R: CommutativeRing> PowerSeriesRing<R> {
    fn series_from_canonical_terms(&self, terms: BTreeMap<Vec<u64>, R::Element>) -> PowerSeries<R> {
        let storage = self
            .backend
            .as_ref()
            .and_then(|context| {
                super::polynomial_flint::series_from_terms::<R>(Arc::clone(context), &terms)
            })
            .map(|native| {
                PowerSeriesStorage::Flint(Arc::new(FlintBackedPowerSeries {
                    native,
                    terms: OnceLock::new(),
                }))
            })
            .unwrap_or_else(|| PowerSeriesStorage::Generic(Arc::new(terms)));
        PowerSeries {
            ring: self.clone(),
            storage,
        }
    }

    fn series_from_native(&self, native: super::polynomial_flint::FlintSeries) -> PowerSeries<R> {
        PowerSeries {
            ring: self.clone(),
            storage: PowerSeriesStorage::Flint(Arc::new(FlintBackedPowerSeries {
                native,
                terms: OnceLock::new(),
            })),
        }
    }

    /// Construct a fixed-truncation series ring over `coefficient_ring`.
    pub fn new<I, S>(
        coefficient_ring: R,
        variables: I,
        truncation: SeriesTruncation,
    ) -> Result<Self, SeriesContextError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let variables = variables.into_iter().map(Into::into).collect::<Vec<_>>();
        if variables.is_empty() {
            return Err(SeriesContextError::NoVariables);
        }
        if let Some(variable) = variables.iter().position(String::is_empty) {
            return Err(SeriesContextError::EmptyVariableName { variable });
        }
        let mut names = BTreeSet::new();
        for name in &variables {
            if !names.insert(name.clone()) {
                return Err(SeriesContextError::DuplicateVariableName { name: name.clone() });
            }
        }
        truncation.validate_arity(variables.len())?;
        let native_precision = match truncation.shape {
            TruncationShape::Univariate { precision } => Some(precision),
            TruncationShape::TotalDegree { bound } if variables.len() == 1 => Some(bound),
            _ => None,
        };
        let backend = native_precision.and_then(|precision| {
            super::polynomial_flint::series_context(&coefficient_ring, precision)
        });
        Ok(Self {
            coefficient_ring,
            variables,
            truncation,
            backend,
        })
    }

    /// Construct a series ring with the standard all-weights-one grading and
    /// strict absolute precision `precision`.
    pub fn with_precision<I, S>(
        coefficient_ring: R,
        variables: I,
        precision: u64,
    ) -> Result<Self, SeriesContextError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(
            coefficient_ring,
            variables,
            SeriesTruncation::total_degree(precision)?,
        )
    }

    /// Construct a series ring with a positive grading and strict absolute
    /// precision `precision` in that grading.
    pub fn with_grading<I, S>(
        coefficient_ring: R,
        variables: I,
        grading: WeightedGrading,
        precision: u64,
    ) -> Result<Self, SeriesContextError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(
            coefficient_ring,
            variables,
            SeriesTruncation::graded(grading, precision)?,
        )
    }

    /// Coefficient-ring context.
    pub fn coefficient_ring(&self) -> &R {
        &self.coefficient_ring
    }

    /// Ordered indeterminate names.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Immutable finite truncation policy.
    pub fn truncation(&self) -> &SeriesTruncation {
        &self.truncation
    }

    /// Serialize the complete coefficient, variable, and finite-precision context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a complete fixed-truncation series context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Positive grading used by a degree truncation. Box truncations have no
    /// single grading and return `None`.
    pub fn grading(&self) -> Option<WeightedGrading> {
        match &self.truncation.shape {
            TruncationShape::Univariate { .. } => WeightedGrading::standard(1).ok(),
            TruncationShape::TotalDegree { .. } => {
                WeightedGrading::standard(self.variables.len()).ok()
            }
            TruncationShape::Box { .. } => None,
            TruncationShape::WeightedDegree { grading, .. } => Some(grading.clone()),
        }
    }

    /// A deterministic upper bound on the number of retained coefficient
    /// positions, without allocating those positions.
    pub fn coefficient_capacity_bound(&self) -> Integer {
        self.truncation
            .coefficient_capacity_bound(self.variables.len())
    }

    /// Construct a sparse series from exponent/coefficient pairs.
    ///
    /// Duplicate exponents are summed. Coefficients outside their context and
    /// terms outside the fixed truncation are rejected rather than coerced or
    /// silently discarded.
    pub fn from_terms<I>(&self, terms: I) -> Result<PowerSeries<R>, PowerSeriesError>
    where
        I: IntoIterator<Item = (Vec<u64>, R::Element)>,
    {
        let mut canonical = BTreeMap::new();
        for (exponent, coefficient) in terms {
            self.validate_exponent(&exponent)?;
            if !self.coefficient_ring.contains(&coefficient) {
                return Err(PowerSeriesError::CoefficientOutsideContext);
            }
            if coefficient == self.coefficient_ring.zero() {
                continue;
            }
            let sum = if let Some(previous) = canonical.remove(&exponent) {
                self.coefficient_ring.add(&previous, &coefficient)?
            } else {
                coefficient
            };
            if sum != self.coefficient_ring.zero() {
                canonical.insert(exponent, sum);
            }
        }
        Ok(self.series_from_canonical_terms(canonical))
    }

    /// Embed a coefficient as a constant series.
    pub fn constant(&self, coefficient: R::Element) -> Result<PowerSeries<R>, PowerSeriesError> {
        self.from_terms([(vec![0; self.variables.len()], coefficient)])
    }

    /// Return the `variable`-th indeterminate.
    pub fn generator(&self, variable: usize) -> Result<PowerSeries<R>, PowerSeriesError> {
        if variable >= self.variables.len() {
            return Err(PowerSeriesError::VariableIndex {
                variable,
                arity: self.variables.len(),
            });
        }
        let mut exponent = vec![0; self.variables.len()];
        exponent[variable] = 1;
        if !self.truncation.contains_exponent(&exponent) {
            return self.from_terms([]);
        }
        self.from_terms([(exponent, self.coefficient_ring.one())])
    }

    /// All indeterminates in the context's declared order.
    pub fn generators(&self) -> Result<Vec<PowerSeries<R>>, PowerSeriesError> {
        (0..self.variables.len())
            .map(|variable| self.generator(variable))
            .collect()
    }

    /// Explicitly discard coefficient positions to enter a coarser context.
    pub fn restrict_truncation(
        &self,
        value: &PowerSeries<R>,
        target: SeriesTruncation,
    ) -> Result<PowerSeries<R>, PowerSeriesError> {
        self.require(value)?;
        target.validate_arity(self.variables.len()).map_err(|_| {
            PowerSeriesError::InvalidRestriction {
                source: self.truncation.clone(),
                target: target.clone(),
            }
        })?;
        if !self.truncation.refines(&target) {
            return Err(PowerSeriesError::InvalidRestriction {
                source: self.truncation.clone(),
                target,
            });
        }
        let target_ring = Self::new(
            self.coefficient_ring.clone(),
            self.variables.clone(),
            target,
        )
        .expect("validated coarser truncation remains a valid series context");
        target_ring.from_terms(
            value
                .terms_map()
                .iter()
                .filter(|(exponent, _)| target_ring.truncation.contains_exponent(exponent))
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone())),
        )
    }

    fn validate_exponent(&self, exponent: &[u64]) -> Result<(), PowerSeriesError> {
        if exponent.len() != self.variables.len() {
            return Err(PowerSeriesError::ExponentArity {
                expected: self.variables.len(),
                actual: exponent.len(),
            });
        }
        if !self.truncation.contains_exponent(exponent) {
            return Err(PowerSeriesError::ExponentOutsideTruncation {
                exponent: exponent.to_vec(),
            });
        }
        Ok(())
    }

    fn require(&self, value: &PowerSeries<R>) -> Result<(), ContextMismatchError> {
        if self == &value.ring {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: Self::TYPE_ID.to_owned(),
                actual: "unequal power-series context".to_owned(),
            })
        }
    }

    fn add_series(
        &self,
        left: &PowerSeries<R>,
        right: &PowerSeries<R>,
    ) -> Result<PowerSeries<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.add(right))
        {
            return Ok(self.series_from_native(native));
        }
        let mut terms = left.terms_map().clone();
        for (exponent, coefficient) in right.terms_map() {
            let sum = if let Some(previous) = terms.remove(exponent) {
                self.coefficient_ring.add(&previous, coefficient)?
            } else {
                coefficient.clone()
            };
            if sum != self.coefficient_ring.zero() {
                terms.insert(exponent.clone(), sum);
            }
        }
        Ok(self.series_from_canonical_terms(terms))
    }

    fn neg_series(&self, value: &PowerSeries<R>) -> Result<PowerSeries<R>, ContextMismatchError> {
        self.require(value)?;
        if let Some(native) = value.native().map(|value| value.neg()) {
            return Ok(self.series_from_native(native));
        }
        let terms = value
            .terms_map()
            .iter()
            .map(|(exponent, coefficient)| {
                self.coefficient_ring
                    .neg(coefficient)
                    .map(|coefficient| (exponent.clone(), coefficient))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(self.series_from_canonical_terms(terms))
    }

    fn sub_series(
        &self,
        left: &PowerSeries<R>,
        right: &PowerSeries<R>,
    ) -> Result<PowerSeries<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.sub(right))
        {
            return Ok(self.series_from_native(native));
        }
        self.add_series(left, &self.neg_series(right)?)
    }

    fn mul_series(
        &self,
        left: &PowerSeries<R>,
        right: &PowerSeries<R>,
    ) -> Result<PowerSeries<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.mul(right))
        {
            return Ok(self.series_from_native(native));
        }
        let work = left
            .terms_map()
            .len()
            .saturating_mul(right.terms_map().len());
        if work >= PACKED_SERIES_TERM_PRODUCT_CROSSOVER {
            if let Some(terms) = packed_series_product(self, left.terms_map(), right.terms_map()) {
                return Ok(self.series_from_canonical_terms(terms));
            }
        }
        // Multivariate paths discard a term pair before its coefficient is
        // computed or stored, which is important for weighted and box regions.
        let mut terms = BTreeMap::new();
        for (left_exponent, left_coefficient) in left.terms_map() {
            for (right_exponent, right_coefficient) in right.terms_map() {
                let Some(exponent) = add_exponents(left_exponent, right_exponent) else {
                    continue;
                };
                if !self.truncation.contains_exponent(&exponent) {
                    continue;
                }
                let product = self
                    .coefficient_ring
                    .mul(left_coefficient, right_coefficient)?;
                let coefficient = if let Some(previous) = terms.remove(&exponent) {
                    self.coefficient_ring.add(&previous, &product)?
                } else {
                    product
                };
                if coefficient != self.coefficient_ring.zero() {
                    terms.insert(exponent, coefficient);
                }
            }
        }
        Ok(self.series_from_canonical_terms(terms))
    }

    fn pow_series(
        &self,
        value: &PowerSeries<R>,
        exponent: u64,
    ) -> Result<PowerSeries<R>, ContextMismatchError> {
        self.require(value)?;
        let mut result = self.one();
        let mut base = value.clone();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.mul_series(&result, &base)?;
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self.mul_series(&base, &base)?;
            }
        }
        Ok(result)
    }

    fn natural_scale(
        &self,
        coefficient: &R::Element,
        factor: u64,
    ) -> Result<R::Element, ContextMismatchError> {
        let mut result = self.coefficient_ring.zero();
        let mut addend = coefficient.clone();
        let mut remaining = factor;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.coefficient_ring.add(&result, &addend)?;
            }
            remaining >>= 1;
            if remaining != 0 {
                addend = self.coefficient_ring.add(&addend, &addend)?;
            }
        }
        Ok(result)
    }
}

fn packed_series_product<R: CommutativeRing>(
    ring: &PowerSeriesRing<R>,
    left: &BTreeMap<Vec<u64>, R::Element>,
    right: &BTreeMap<Vec<u64>, R::Element>,
) -> Option<BTreeMap<Vec<u64>, R::Element>> {
    let arity = ring.variables.len();
    let truncation = match &ring.truncation.shape {
        TruncationShape::Univariate { precision } => poly_core::Truncation::TotalDegree {
            arity: 1,
            bound: u32::try_from(*precision).ok()?,
        },
        TruncationShape::TotalDegree { bound } => poly_core::Truncation::TotalDegree {
            arity,
            bound: u32::try_from(*bound).ok()?,
        },
        TruncationShape::Box { bounds } => poly_core::Truncation::Box {
            bounds: bounds
                .iter()
                .map(|bound| u32::try_from(*bound).ok())
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        },
        TruncationShape::WeightedDegree { grading, bound } => poly_core::Truncation::Weighted {
            weights: grading
                .weights()
                .iter()
                .map(|weight| u32::try_from(*weight).ok())
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
            bound: *bound,
        },
    };
    let capacity = left
        .len()
        .saturating_add(right.len())
        .saturating_mul(4)
        .max(64);
    let mut arena = poly_core::MonomialArena::with_capacity(
        arity,
        poly_core::MonomialOrder::GradedReverseLexicographic,
        capacity,
    );
    let lane = PolyCoreRingLane {
        ring: &ring.coefficient_ring,
    };
    let mut convert = |terms: &BTreeMap<Vec<u64>, R::Element>| {
        let terms = terms
            .iter()
            .map(|(exponent, coefficient)| {
                let exponent = exponent
                    .iter()
                    .map(|value| u32::try_from(*value).ok())
                    .collect::<Option<Vec<_>>>()?;
                arena
                    .intern_exponents(&exponent)
                    .ok()
                    .map(|monomial| (monomial, coefficient.clone()))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(poly_core::ArenaPolynomial::new(&arena, terms, &lane))
    };
    let left = convert(left)?;
    let right = convert(right)?;
    let product = left
        .multiply(&right, &mut arena, &lane, Some(&truncation))
        .ok()?;
    Some(
        product
            .iter()
            .map(|(monomial, coefficient)| {
                (
                    arena
                        .get(monomial)
                        .to_u32_vec()
                        .into_iter()
                        .map(u64::from)
                        .collect(),
                    coefficient.clone(),
                )
            })
            .collect(),
    )
}

impl<R> Serialize for PowerSeriesRing<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PowerSeriesRing", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("coefficient_ring", &self.coefficient_ring)?;
        state.serialize_field("variables", &self.variables)?;
        state.serialize_field("truncation", &self.truncation)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PowerSeriesRing<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PowerSeriesRingWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.coefficient_ring, wire.variables, wire.truncation)
            .map_err(serde::de::Error::custom)
    }
}

impl<R: CommutativeRing> PartialEq for PowerSeriesRing<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficient_ring == other.coefficient_ring
            && self.variables == other.variables
            && self.truncation == other.truncation
    }
}

impl<R: CommutativeRing> Eq for PowerSeriesRing<R> {}

impl<R: CommutativeRing> fmt::Debug for PowerSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PowerSeriesRing")
            .field("coefficient_type", &R::TYPE_ID)
            .field("variables", &self.variables)
            .field("truncation", &self.truncation)
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for PowerSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PowerSeriesRing<{}>[[{}]] ({})",
            R::TYPE_ID,
            self.variables.join(", "),
            self.truncation
        )
    }
}

impl<R: CommutativeRing> SemanticType for PowerSeriesRing<R> {
    const TYPE_ID: &'static str = "noether/commalg/PowerSeriesRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, R::TYPE_ID.as_bytes())?;
        append_component(&mut output, &self.coefficient_ring.canonical_bytes()?)?;
        append_length(&mut output, self.variables.len())?;
        for variable in &self.variables {
            append_component(&mut output, variable.as_bytes())?;
        }
        append_component(&mut output, &self.truncation.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POWER_SERIES_RING_DESCRIPTION
    }
}

impl<R: CommutativeRing> Ring for PowerSeriesRing<R> {
    type Element = PowerSeries<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring
    }

    fn zero(&self) -> Self::Element {
        self.series_from_canonical_terms(BTreeMap::new())
    }

    fn one(&self) -> Self::Element {
        let mut terms = BTreeMap::new();
        terms.insert(vec![0; self.variables.len()], self.coefficient_ring.one());
        self.series_from_canonical_terms(terms)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.add_series(a, b)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.sub_series(a, b)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.mul_series(a, b)
    }

    fn sum_products(
        &self,
        products: &[(&Self::Element, &Self::Element)],
    ) -> Result<Self::Element, ContextMismatchError> {
        for &(left, right) in products {
            self.require(left)?;
            self.require(right)?;
        }
        if products.is_empty() {
            return Ok(self.zero());
        }
        let native_products = products
            .iter()
            .map(|&(left, right)| Some((left.native()?, right.native()?)))
            .collect::<Option<Vec<_>>>();
        if let Some(native) = native_products
            .as_deref()
            .and_then(super::polynomial_flint::FlintSeries::sum_products)
        {
            return Ok(self.series_from_native(native));
        }

        let mut sum = self.zero();
        for &(left, right) in products {
            let product = self.mul_series(left, right)?;
            sum = self.add_series(&sum, &product)?;
        }
        Ok(sum)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.neg_series(value)
    }
}

impl<R: CommutativeRing> CommutativeRing for PowerSeriesRing<R> {}

impl<R: UnitInversion> UnitInversion for PowerSeriesRing<R> {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)?;
        value.inverse()
    }
}

impl<R: CommutativeRing> PrecisionContext for PowerSeriesRing<R> {
    type Precision = SeriesTruncation;

    fn precision(&self) -> &Self::Precision {
        &self.truncation
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate.refines(reference)
    }
}

struct FlintBackedPowerSeries<E> {
    native: super::polynomial_flint::FlintSeries,
    terms: OnceLock<BTreeMap<Vec<u64>, E>>,
}

enum PowerSeriesStorage<E> {
    Generic(Arc<BTreeMap<Vec<u64>, E>>),
    Flint(Arc<FlintBackedPowerSeries<E>>),
}

impl<E> Clone for PowerSeriesStorage<E> {
    fn clone(&self) -> Self {
        match self {
            Self::Generic(terms) => Self::Generic(Arc::clone(terms)),
            Self::Flint(series) => Self::Flint(Arc::clone(series)),
        }
    }
}

/// A fixed-truncation power series carrying its complete context.
///
/// Supported one-variable coefficient contexts keep an immutable FLINT
/// polynomial as their primary representation. Clones share that storage, and
/// the canonical sparse term view is materialized lazily only for inspection,
/// serialization, or a generic algorithm. Multivariate and unsupported
/// contexts retain the sparse truncation-aware representation.
#[derive(Clone)]
pub struct PowerSeries<R: CommutativeRing> {
    ring: PowerSeriesRing<R>,
    storage: PowerSeriesStorage<R::Element>,
}

#[derive(Serialize)]
struct PowerSeriesTermRef<'a, E> {
    exponent: &'a [u64],
    coefficient: &'a E,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "E: Deserialize<'de>"))]
struct PowerSeriesTermWire<E> {
    exponent: Vec<u64>,
    coefficient: E,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, E: Deserialize<'de>")
)]
struct PowerSeriesWire<R: CommutativeRing, E> {
    #[serde(rename = "type")]
    type_id: String,
    ring: PowerSeriesRing<R>,
    terms: Vec<PowerSeriesTermWire<E>>,
}

impl<R: CommutativeRing> PowerSeries<R> {
    fn terms_map(&self) -> &BTreeMap<Vec<u64>, R::Element> {
        match &self.storage {
            PowerSeriesStorage::Generic(terms) => terms,
            PowerSeriesStorage::Flint(series) => series
                .terms
                .get_or_init(|| series.native.to_terms(&self.ring.coefficient_ring)),
        }
    }

    fn native(&self) -> Option<&super::polynomial_flint::FlintSeries> {
        match &self.storage {
            PowerSeriesStorage::Generic(_) => None,
            PowerSeriesStorage::Flint(series) => Some(&series.native),
        }
    }

    #[cfg(test)]
    fn uses_flint_storage(&self) -> bool {
        matches!(self.storage, PowerSeriesStorage::Flint(_))
    }

    #[cfg(test)]
    fn sparse_view_is_materialized(&self) -> bool {
        match &self.storage {
            PowerSeriesStorage::Generic(_) => true,
            PowerSeriesStorage::Flint(series) => series.terms.get().is_some(),
        }
    }

    #[cfg(test)]
    fn shares_storage_with(&self, other: &Self) -> bool {
        match (&self.storage, &other.storage) {
            (PowerSeriesStorage::Generic(left), PowerSeriesStorage::Generic(right)) => {
                Arc::ptr_eq(left, right)
            }
            (PowerSeriesStorage::Flint(left), PowerSeriesStorage::Flint(right)) => {
                Arc::ptr_eq(left, right)
            }
            _ => false,
        }
    }

    /// Exact power-series context carried by this value.
    pub fn ring(&self) -> &PowerSeriesRing<R> {
        &self.ring
    }

    /// Nonzero terms in deterministic lexicographic exponent order.
    pub fn terms(&self) -> impl Iterator<Item = (&[u64], &R::Element)> {
        self.terms_map()
            .iter()
            .map(|(exponent, coefficient)| (exponent.as_slice(), coefficient))
    }

    /// Apply one checked scalar map to every retained coefficient while
    /// preserving variables, truncation identity, and exponents.
    pub fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<PowerSeries<M::Target>, CoefficientMappingError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        if self.ring.coefficient_ring() != map.source() {
            return Err(ContextMismatchError {
                expected: map.source().to_string(),
                actual: self.ring.coefficient_ring().to_string(),
            }
            .into());
        }
        let target_ring = PowerSeriesRing::new(
            map.target().clone(),
            self.ring.variables().iter().cloned(),
            self.ring.truncation().clone(),
        )
        .map_err(|error| CoefficientMappingError::TargetContext(error.to_string()))?;
        let mut terms = Vec::with_capacity(self.nonzero_term_count());
        for (exponent, coefficient) in self.terms() {
            terms.push((
                exponent.to_vec(),
                map.apply(coefficient)
                    .map_err(|error| CoefficientMappingError::Coefficient(error.to_string()))?,
            ));
        }
        target_ring
            .from_terms(terms)
            .map_err(|error| CoefficientMappingError::Value(error.to_string()))
    }

    /// Coefficient at a retained exponent position.
    ///
    /// Asking beyond the context's precision is an error, not a zero
    /// coefficient claim.
    pub fn coefficient(&self, exponent: &[u64]) -> Result<R::Element, PowerSeriesError> {
        self.ring.validate_exponent(exponent)?;
        if let Some(native) = self.native() {
            return Ok(native.coefficient(&self.ring.coefficient_ring, exponent[0]));
        }
        Ok(self
            .terms_map()
            .get(exponent)
            .cloned()
            .unwrap_or_else(|| self.ring.coefficient_ring.zero()))
    }

    /// Constant coefficient.
    pub fn constant_coefficient(&self) -> R::Element {
        if let Some(native) = self.native() {
            return native.coefficient(&self.ring.coefficient_ring, 0);
        }
        self.terms_map()
            .get(&vec![0; self.ring.variables.len()])
            .cloned()
            .unwrap_or_else(|| self.ring.coefficient_ring.zero())
    }

    /// Whether every retained coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.native()
            .map_or_else(|| self.terms_map().is_empty(), |native| native.is_zero())
    }

    /// Whether this is the multiplicative identity in its context.
    pub fn is_one(&self) -> bool {
        self == &self.ring.one()
    }

    /// Number of stored nonzero coefficients.
    pub fn nonzero_term_count(&self) -> usize {
        self.native()
            .map_or_else(|| self.terms_map().len(), |native| native.nonzero_count())
    }

    /// Serialize this series with its complete fixed-precision context and terms.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the context, exponent region, and coefficients.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Raise this series to a nonnegative power in its fixed context.
    pub fn pow(&self, exponent: u64) -> Result<Self, ContextMismatchError> {
        self.ring.pow_series(self, exponent)
    }

    /// Multiply every coefficient by one value from the coefficient ring.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, PowerSeriesError> {
        if !self.ring.coefficient_ring.contains(scalar) {
            return Err(PowerSeriesError::CoefficientOutsideContext);
        }
        if let Some(native) = self
            .native()
            .and_then(|native| native.scalar_mul::<R>(scalar))
        {
            return Ok(self.ring.series_from_native(native));
        }
        let mut terms = BTreeMap::new();
        for (exponent, coefficient) in self.terms_map() {
            let product = self.ring.coefficient_ring.mul(coefficient, scalar)?;
            if product != self.ring.coefficient_ring.zero() {
                terms.insert(exponent.clone(), product);
            }
        }
        Ok(self.ring.series_from_canonical_terms(terms))
    }

    /// Formal derivative with respect to one variable.
    pub fn derivative(&self, variable: usize) -> Result<Self, PowerSeriesError> {
        if variable >= self.ring.variables.len() {
            return Err(PowerSeriesError::VariableIndex {
                variable,
                arity: self.ring.variables.len(),
            });
        }
        let target_truncation = self
            .ring
            .truncation
            .derivative_truncation(variable)
            .ok_or(PowerSeriesError::InsufficientDerivativePrecision { variable })?;
        let target_ring = PowerSeriesRing::new(
            self.ring.coefficient_ring.clone(),
            self.ring.variables.clone(),
            target_truncation,
        )
        .expect("a derived truncation remains a valid series context");
        if let Some(derivative) = self.native().and_then(|native| {
            target_ring
                .backend
                .as_ref()
                .and_then(|target| native.derivative(Arc::clone(target)))
        }) {
            return Ok(target_ring.series_from_native(derivative));
        }
        let mut terms = BTreeMap::new();
        for (exponent, coefficient) in self.terms_map() {
            let factor = exponent[variable];
            if factor == 0 {
                continue;
            }
            let mut derivative_exponent = exponent.clone();
            derivative_exponent[variable] -= 1;
            if !target_ring
                .truncation
                .contains_exponent(&derivative_exponent)
            {
                continue;
            }
            let derivative_coefficient = self.ring.natural_scale(coefficient, factor)?;
            if derivative_coefficient != self.ring.coefficient_ring.zero() {
                terms.insert(derivative_exponent, derivative_coefficient);
            }
        }
        Ok(target_ring.series_from_canonical_terms(terms))
    }

    /// Substitute one image for each variable when the images preserve the
    /// truncation ideal.
    pub fn compose(&self, images: &[Self]) -> Result<Self, PowerSeriesError> {
        if images.len() != self.ring.variables.len() {
            return Err(PowerSeriesError::ExponentArity {
                expected: self.ring.variables.len(),
                actual: images.len(),
            });
        }
        for image in images {
            self.ring.require(image)?;
        }
        self.validate_composition(images)?;

        if let Some(composition) = self
            .native()
            .zip(images[0].native())
            .and_then(|(value, image)| value.compose(image))
        {
            return Ok(self.ring.series_from_native(composition));
        }
        if images.len() == 1 {
            let upper = self.native().map_or_else(
                || {
                    self.terms_map()
                        .keys()
                        .next_back()
                        .map_or(0, |exponent| exponent[0].saturating_add(1))
                },
                |native| u64::try_from(native.len()).expect("FLINT polynomial length fits in u64"),
            );
            let mut result = self.ring.zero();
            for exponent in (0..upper).rev() {
                result = self.ring.mul_series(&result, &images[0])?;
                let coefficient = self.coefficient(&[exponent])?;
                if coefficient != self.ring.coefficient_ring.zero() {
                    let constant = self.ring.constant(coefficient)?;
                    result = self.ring.add_series(&result, &constant)?;
                }
            }
            return Ok(result);
        }

        let mut result = self.ring.zero();
        let mut powers = (0..images.len())
            .map(|_| BTreeMap::<u64, Self>::new())
            .collect::<Vec<_>>();
        for (exponent, coefficient) in self.terms_map() {
            let mut term = self.ring.constant(coefficient.clone())?;
            for (variable, (power, image)) in exponent.iter().zip(images).enumerate() {
                if *power != 0 {
                    let image_power = if let Some(value) = powers[variable].get(power) {
                        value.clone()
                    } else {
                        let value = self.ring.pow_series(image, *power)?;
                        powers[variable].insert(*power, value.clone());
                        value
                    };
                    term = self.ring.mul_series(&term, &image_power)?;
                }
            }
            result = self.ring.add_series(&result, &term)?;
        }
        Ok(result)
    }

    /// Invert a series exactly when its constant coefficient is a unit.
    pub fn inverse(&self) -> Result<Self, UnitError>
    where
        R: UnitInversion,
    {
        let inverse_constant = match self
            .ring
            .coefficient_ring
            .inverse_unit(&self.constant_coefficient())
        {
            Ok(inverse) => inverse,
            Err(UnitError::ContextMismatch(error)) => {
                return Err(UnitError::ContextMismatch(error));
            }
            Err(UnitError::DivisionByZero(_) | UnitError::NotAUnit { .. }) => {
                return Err(UnitError::NotAUnit {
                    context: PowerSeriesRing::<R>::TYPE_ID,
                });
            }
        };
        if let Some(native) = self.native() {
            return Ok(self.ring.series_from_native(native.inverse()));
        }
        let constant = self
            .ring
            .constant(self.constant_coefficient())
            .map_err(|_| UnitError::NotAUnit {
                context: PowerSeriesRing::<R>::TYPE_ID,
            })?;
        let remainder = self.ring.sub_series(self, &constant)?;
        let inverse_constant_series =
            self.ring
                .constant(inverse_constant)
                .map_err(|_| UnitError::NotAUnit {
                    context: PowerSeriesRing::<R>::TYPE_ID,
                })?;
        let nilpotence_bound = self.ring.truncation.nilpotence_bound();
        if nilpotence_bound >= 8 {
            let two = self.ring.coefficient_ring.add(
                &self.ring.coefficient_ring.one(),
                &self.ring.coefficient_ring.one(),
            )?;
            let two = self.ring.constant(two).map_err(|_| UnitError::NotAUnit {
                context: PowerSeriesRing::<R>::TYPE_ID,
            })?;
            let mut inverse = inverse_constant_series;
            let mut vanishing_order = 1_u64;
            while vanishing_order < nilpotence_bound {
                let product = self.ring.mul_series(self, &inverse)?;
                let correction = self.ring.sub_series(&two, &product)?;
                inverse = self.ring.mul_series(&inverse, &correction)?;
                vanishing_order = vanishing_order.saturating_mul(2);
            }
            return Ok(inverse);
        }
        let normalized_remainder = self.ring.mul_series(&remainder, &inverse_constant_series)?;

        let mut sum = self.ring.one();
        let mut power = self.ring.one();
        for _ in 0..nilpotence_bound {
            power = self
                .ring
                .neg_series(&self.ring.mul_series(&power, &normalized_remainder)?)?;
            if power.is_zero() {
                break;
            }
            sum = self.ring.add_series(&sum, &power)?;
        }
        self.ring
            .mul_series(&sum, &inverse_constant_series)
            .map_err(UnitError::from)
    }

    /// Return the canonical exact quotient by a monomial times a unit series.
    ///
    /// Fixed truncation makes division by a nonunit nonunique: coefficients
    /// beyond the dividend's retained region cannot be recovered. For a
    /// divisor `x^a u`, with `u(0)` a unit, this method chooses the unique
    /// quotient obtained by shifting the retained dividend by `a`, filling
    /// the newly unknown boundary coefficients with zero, and dividing by
    /// `u`. It verifies the result by multiplication in this exact quotient
    /// ring. More general divisors are rejected rather than choosing a hidden
    /// term order or quotient lift.
    pub fn exact_div(&self, divisor: &Self) -> Result<Self, PowerSeriesError>
    where
        R: UnitInversion,
    {
        self.ring.require(divisor)?;
        let shift = divisor
            .monomial_valuation()
            .ok_or(PowerSeriesError::DivisionByZero)?;
        let shifted_divisor = divisor.shift_right(&shift)?;
        let shifted_dividend = self.shift_right(&shift)?;
        let inverse = shifted_divisor
            .inverse()
            .map_err(|_| PowerSeriesError::DivisorNotMonomialTimesUnit)?;
        let quotient = self.ring.mul_series(&shifted_dividend, &inverse)?;
        let reconstructed = self.ring.mul_series(divisor, &quotient)?;
        if &reconstructed == self {
            Ok(quotient)
        } else {
            Err(PowerSeriesError::NotExactlyDivisible)
        }
    }

    fn monomial_valuation(&self) -> Option<Vec<u64>> {
        if let Some(native) = self.native() {
            return native
                .valuation(&self.ring.coefficient_ring)
                .map(|valuation| vec![valuation]);
        }
        let mut terms = self.terms_map().keys();
        let mut valuation = terms.next()?.clone();
        for exponent in terms {
            for (minimum, value) in valuation.iter_mut().zip(exponent) {
                *minimum = (*minimum).min(*value);
            }
        }
        Some(valuation)
    }

    fn shift_right(&self, shift: &[u64]) -> Result<Self, PowerSeriesError> {
        debug_assert_eq!(shift.len(), self.ring.variables.len());
        if let Some(native) = self.native() {
            let shifted = native
                .shift_right(shift[0])
                .ok_or(PowerSeriesError::NotExactlyDivisible)?;
            return Ok(self.ring.series_from_native(shifted));
        }
        let mut terms = BTreeMap::new();
        for (exponent, coefficient) in self.terms_map() {
            let shifted = exponent
                .iter()
                .zip(shift)
                .map(|(value, amount)| value.checked_sub(*amount))
                .collect::<Option<Vec<_>>>()
                .ok_or(PowerSeriesError::NotExactlyDivisible)?;
            terms.insert(shifted, coefficient.clone());
        }
        Ok(self.ring.series_from_canonical_terms(terms))
    }

    /// Enter an explicitly coarser truncation context.
    pub fn restrict_truncation(&self, target: SeriesTruncation) -> Result<Self, PowerSeriesError> {
        self.ring.restrict_truncation(self, target)
    }

    fn validate_composition(&self, images: &[Self]) -> Result<(), PowerSeriesError> {
        match &self.ring.truncation.shape {
            TruncationShape::Univariate { precision } => {
                if !self.ring.pow_series(&images[0], *precision)?.is_zero() {
                    return Err(PowerSeriesError::InvalidComposition {
                        variable: 0,
                        reason: "the image does not satisfy x^precision = 0",
                    });
                }
            }
            TruncationShape::Box { bounds } => {
                for (variable, (image, bound)) in images.iter().zip(bounds).enumerate() {
                    if !self.ring.pow_series(image, *bound)?.is_zero() {
                        return Err(PowerSeriesError::InvalidComposition {
                            variable,
                            reason: "the image does not satisfy its box nilpotence relation",
                        });
                    }
                }
            }
            TruncationShape::TotalDegree { .. } => {
                for (variable, image) in images.iter().enumerate() {
                    if image
                        .minimum_weighted_order()
                        .is_some_and(|order| order < 1)
                    {
                        return Err(PowerSeriesError::InvalidComposition {
                            variable,
                            reason: "the image has a nonzero constant coefficient",
                        });
                    }
                }
            }
            TruncationShape::WeightedDegree { grading, .. } => {
                for (variable, (image, required)) in
                    images.iter().zip(grading.weights()).enumerate()
                {
                    if image
                        .minimum_weighted_order()
                        .is_some_and(|order| order < *required)
                    {
                        return Err(PowerSeriesError::InvalidComposition {
                            variable,
                            reason: "the image has smaller weighted order than its source variable",
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn minimum_weighted_order(&self) -> Option<u64> {
        self.terms_map()
            .keys()
            .filter_map(|exponent| self.ring.truncation.weighted_order(exponent))
            .min()
    }
}

impl<R> Serialize for PowerSeries<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let terms = self
            .terms()
            .map(|(exponent, coefficient)| PowerSeriesTermRef {
                exponent,
                coefficient,
            })
            .collect::<Vec<_>>();
        let mut state = serializer.serialize_struct("PowerSeries", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("terms", &terms)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PowerSeries<R>
where
    R: CommutativeRing + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PowerSeriesWire::<R, R::Element>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let mut previous: Option<&[u64]> = None;
        for term in &wire.terms {
            wire.ring
                .validate_exponent(&term.exponent)
                .map_err(serde::de::Error::custom)?;
            if !wire.ring.coefficient_ring.contains(&term.coefficient) {
                return Err(serde::de::Error::custom(
                    "power-series coefficient lies outside its declared context",
                ));
            }
            if term.coefficient == wire.ring.coefficient_ring.zero() {
                return Err(serde::de::Error::custom(
                    "serialized power-series terms must have nonzero coefficients",
                ));
            }
            if let Some(previous) = previous {
                if previous >= term.exponent.as_slice() {
                    return Err(serde::de::Error::custom(
                        "serialized power-series terms are not in strict lexicographic order",
                    ));
                }
            }
            previous = Some(&term.exponent);
        }
        wire.ring
            .from_terms(
                wire.terms
                    .into_iter()
                    .map(|term| (term.exponent, term.coefficient)),
            )
            .map_err(serde::de::Error::custom)
    }
}

impl<R: CommutativeRing> PartialEq for PowerSeries<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.ring != other.ring {
            return false;
        }
        match (&self.storage, &other.storage) {
            (PowerSeriesStorage::Generic(left), PowerSeriesStorage::Generic(right)) => {
                Arc::ptr_eq(left, right) || left == right
            }
            (PowerSeriesStorage::Flint(left), PowerSeriesStorage::Flint(right)) => {
                Arc::ptr_eq(left, right)
                    || left
                        .native
                        .equal(&right.native)
                        .unwrap_or_else(|| self.terms_map() == other.terms_map())
            }
            _ => self.terms_map() == other.terms_map(),
        }
    }
}

impl<R: CommutativeRing> Eq for PowerSeries<R> {}

impl<R: CommutativeRing> fmt::Debug for PowerSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PowerSeries")
            .field("ring", &self.ring)
            .field("nonzero_terms", &self.nonzero_term_count())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for PowerSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PowerSeries<{}>({} nonzero terms; {})",
            R::TYPE_ID,
            self.nonzero_term_count(),
            self.ring.truncation
        )
    }
}

impl<R: CommutativeRing> SemanticType for PowerSeries<R> {
    const TYPE_ID: &'static str = "noether/commalg/PowerSeries@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.ring.canonical_bytes()?)?;
        append_length(&mut output, self.terms_map().len())?;
        for (exponent, coefficient) in self.terms_map() {
            append_u64_slice(&mut output, exponent)?;
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POWER_SERIES_DESCRIPTION
    }
}

impl<R: CommutativeRing> Add<&PowerSeries<R>> for &PowerSeries<R> {
    type Output = Result<PowerSeries<R>, ContextMismatchError>;

    fn add(self, rhs: &PowerSeries<R>) -> Self::Output {
        self.ring.add(self, rhs)
    }
}

impl<R: CommutativeRing> Sub<&PowerSeries<R>> for &PowerSeries<R> {
    type Output = Result<PowerSeries<R>, ContextMismatchError>;

    fn sub(self, rhs: &PowerSeries<R>) -> Self::Output {
        self.ring.sub(self, rhs)
    }
}

impl<R: CommutativeRing> Mul<&PowerSeries<R>> for &PowerSeries<R> {
    type Output = Result<PowerSeries<R>, ContextMismatchError>;

    fn mul(self, rhs: &PowerSeries<R>) -> Self::Output {
        self.ring.mul(self, rhs)
    }
}

impl<R: CommutativeRing> Neg for &PowerSeries<R> {
    type Output = Result<PowerSeries<R>, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.ring.neg(self)
    }
}

fn checked_weighted_degree(
    exponent: &[u64],
    weights: impl IntoIterator<Item = u64>,
) -> Option<u64> {
    exponent
        .iter()
        .zip(weights)
        .try_fold(0_u64, |degree, (power, weight)| {
            degree.checked_add(power.checked_mul(weight)?)
        })
}

fn add_exponents(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.checked_add(*right))
        .collect()
}

fn append_length(output: &mut Vec<u8>, length: usize) -> Result<(), NoetherError> {
    let length = u64::try_from(length).map_err(|_| {
        NoetherError::Serialization("series component count exceeds u64".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_length(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

fn append_u64_slice(output: &mut Vec<u8>, values: &[u64]) -> Result<(), NoetherError> {
    append_length(output, values.len())?;
    for value in values {
        output.extend_from_slice(&value.to_be_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::commalg::ResidueRing;
    use crate::numthy::FiniteField;
    use crate::{IntegerRing, Rational, Q, Z};

    fn assert_sticky_native_arithmetic<R: CommutativeRing>(
        ring: PowerSeriesRing<R>,
        a_coefficient: R::Element,
        b_coefficient: R::Element,
        scalar: R::Element,
    ) {
        let expected_cubic_coefficient = ring
            .coefficient_ring()
            .mul(&a_coefficient, &b_coefficient)
            .unwrap();
        let left = ring
            .from_terms([
                (vec![1], a_coefficient),
                (vec![4], ring.coefficient_ring().one()),
            ])
            .unwrap();
        let right = ring
            .from_terms([
                (vec![2], b_coefficient),
                (vec![0], ring.coefficient_ring().one()),
            ])
            .unwrap();
        assert!(left.uses_flint_storage());
        assert!(!left.sparse_view_is_materialized());
        let cloned = left.clone();
        assert!(left.shares_storage_with(&cloned));

        let sum = ring.add(&left, &right).unwrap();
        let difference = ring.sub(&sum, &right).unwrap();
        let product = ring.mul(&left, &right).unwrap();
        let negative = ring.neg(&left).unwrap();
        let scaled = left.scalar_mul(&scalar).unwrap();
        for value in [&sum, &difference, &product, &negative, &scaled] {
            assert!(value.uses_flint_storage());
            assert!(!value.sparse_view_is_materialized());
        }

        assert!(product.coefficient(&[3]).unwrap() == expected_cubic_coefficient);
        assert!(!product.sparse_view_is_materialized());
        assert_eq!(difference, left);
        assert_eq!(ring.add(&left, &negative).unwrap(), ring.zero());
        for value in [
            &left,
            &right,
            &sum,
            &difference,
            &product,
            &negative,
            &scaled,
        ] {
            assert!(!value.sparse_view_is_materialized());
        }

        let generic_left = PowerSeries {
            ring: ring.clone(),
            storage: PowerSeriesStorage::Generic(Arc::new(left.terms_map().clone())),
        };
        let generic_right = PowerSeries {
            ring: ring.clone(),
            storage: PowerSeriesStorage::Generic(Arc::new(right.terms_map().clone())),
        };
        assert_eq!(sum, ring.add(&generic_left, &generic_right).unwrap());
        assert_eq!(
            difference,
            ring.sub(
                &ring.add(&generic_left, &generic_right).unwrap(),
                &generic_right,
            )
            .unwrap()
        );
        assert_eq!(product, ring.mul(&generic_left, &generic_right).unwrap());
        assert_eq!(negative, ring.neg(&generic_left).unwrap());
        assert_eq!(scaled, generic_left.scalar_mul(&scalar).unwrap());
    }

    fn assert_sticky_native_calculus<R: CommutativeRing + UnitInversion>(
        ring: PowerSeriesRing<R>,
        linear: R::Element,
        cubic: R::Element,
    ) {
        let terms = vec![
            (vec![0], ring.coefficient_ring().one()),
            (vec![1], linear),
            (vec![3], cubic),
            (vec![7], ring.coefficient_ring().one()),
        ];
        let value = ring.from_terms(terms.clone()).unwrap();
        let generator = ring.generator(0).unwrap();
        let image = ring.add(&generator, &generator.pow(2).unwrap()).unwrap();

        let mut generic_ring = ring.clone();
        generic_ring.backend = None;
        let generic_value = generic_ring.from_terms(terms).unwrap();
        let generic_generator = generic_ring.generator(0).unwrap();
        let generic_image = generic_ring
            .add(&generic_generator, &generic_generator.pow(2).unwrap())
            .unwrap();

        let derivative = value.derivative(0).unwrap();
        let inverse = value.inverse().unwrap();
        let composition = value.compose(std::slice::from_ref(&image)).unwrap();
        let fused = ring
            .sum_products(&[(&value, &image), (&image, &value), (&value, &value)])
            .unwrap();
        for result in [&derivative, &inverse, &composition, &fused] {
            assert!(result.uses_flint_storage());
            assert!(!result.sparse_view_is_materialized());
            assert!(result.nonzero_term_count() <= 8);
            let precision = match &result.ring.truncation.shape {
                TruncationShape::Univariate { precision } => *precision,
                TruncationShape::TotalDegree { bound } => *bound,
                _ => unreachable!("calculus fixture is univariate"),
            };
            assert!(result.native().unwrap().len() <= usize::try_from(precision).unwrap());
        }
        assert_eq!(derivative, generic_value.derivative(0).unwrap());
        assert_eq!(inverse, generic_value.inverse().unwrap());
        assert_eq!(
            composition,
            generic_value
                .compose(std::slice::from_ref(&generic_image))
                .unwrap()
        );
        assert_eq!(
            fused,
            generic_ring
                .sum_products(&[
                    (&generic_value, &generic_image),
                    (&generic_image, &generic_value),
                    (&generic_value, &generic_value),
                ])
                .unwrap()
        );
        assert_eq!(ring.mul(&value, &inverse).unwrap(), ring.one());
        assert!(matches!(
            derivative.coefficient(&[7]),
            Err(PowerSeriesError::ExponentOutsideTruncation { .. })
        ));
        assert!(!value.sparse_view_is_materialized());
    }

    fn assert_equal_contexts_stay_native<R: CommutativeRing>(
        left_ring: PowerSeriesRing<R>,
        right_ring: PowerSeriesRing<R>,
        left_coefficient: R::Element,
        right_coefficient: R::Element,
    ) {
        let left = left_ring.from_terms([(vec![1], left_coefficient)]).unwrap();
        let right = right_ring
            .from_terms([(vec![2], right_coefficient)])
            .unwrap();

        let sum = left_ring.add(&left, &right).unwrap();
        let product = left_ring.mul(&left, &right).unwrap();

        for value in [&left, &right, &sum, &product] {
            assert!(value.uses_flint_storage());
            assert!(!value.sparse_view_is_materialized());
        }
    }

    fn fresh_ring<R: CommutativeRing>(coefficients: R) -> PowerSeriesRing<R> {
        PowerSeriesRing::with_precision(coefficients, ["x"], 8).unwrap()
    }

    #[test]
    fn supported_exact_contexts_keep_univariate_series_in_flint() {
        assert_sticky_native_arithmetic(
            PowerSeriesRing::new(Z, ["x"], SeriesTruncation::univariate(8).unwrap()).unwrap(),
            Integer::from(7),
            Integer::from(-5),
            Integer::from(3),
        );
        assert_sticky_native_arithmetic(
            PowerSeriesRing::with_precision(Q, ["x"], 8).unwrap(),
            "7/3".parse().unwrap(),
            "-5/2".parse().unwrap(),
            "3/5".parse().unwrap(),
        );

        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_sticky_native_arithmetic(
                PowerSeriesRing::new(
                    residues.clone(),
                    ["x"],
                    SeriesTruncation::univariate(8).unwrap(),
                )
                .unwrap(),
                residues.element(7_u8),
                residues.element(11_u8),
                residues.element(3_u8),
            );
        }

        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_sticky_native_arithmetic(
                PowerSeriesRing::new(
                    field.clone(),
                    ["x"],
                    SeriesTruncation::univariate(8).unwrap(),
                )
                .unwrap(),
                field.element(2_u8),
                field.generator(),
                field.element(2_u8),
            );
        }
    }

    #[test]
    fn supported_exact_contexts_keep_univariate_calculus_in_flint() {
        assert_sticky_native_calculus(
            PowerSeriesRing::new(Z, ["x"], SeriesTruncation::univariate(8).unwrap()).unwrap(),
            Integer::from(2),
            Integer::from(-3),
        );
        assert_sticky_native_calculus(
            PowerSeriesRing::with_precision(Q, ["x"], 8).unwrap(),
            "2/3".parse().unwrap(),
            "-3/5".parse().unwrap(),
        );

        let word_field = FiniteField::prime(7_u8).unwrap();
        assert_sticky_native_calculus(
            PowerSeriesRing::with_precision(word_field.clone(), ["x"], 8).unwrap(),
            word_field.element(2_u8),
            word_field.element(3_u8),
        );

        let large_field = FiniteField::prime(
            "170141183460469231731687303715884105727"
                .parse::<Integer>()
                .unwrap(),
        )
        .unwrap();
        assert_sticky_native_calculus(
            PowerSeriesRing::with_precision(large_field.clone(), ["x"], 8).unwrap(),
            large_field.element(2_u8),
            large_field.element(3_u8),
        );

        let extension = FiniteField::extension(3, 2).unwrap();
        assert_sticky_native_calculus(
            PowerSeriesRing::with_precision(extension.clone(), ["x"], 8).unwrap(),
            extension.generator(),
            extension.element(2_u8),
        );
    }

    #[test]
    fn native_nonzero_count_is_hamming_weight_not_polynomial_length() {
        let ring = PowerSeriesRing::with_precision(Z, ["x"], 32).unwrap();
        let sparse = ring.from_terms([(vec![31], Integer::from(7))]).unwrap();

        assert!(sparse.uses_flint_storage());
        assert_eq!(sparse.nonzero_term_count(), 1);
        assert!(!sparse.sparse_view_is_materialized());
    }

    #[test]
    fn native_exact_division_keeps_flint_storage_sticky() {
        let ring = PowerSeriesRing::with_precision(Q, ["t"], 8).unwrap();
        let t = ring.generator(0).unwrap();
        let unit = ring.add(&ring.one(), &t).unwrap();
        let divisor = ring.mul(&t, &unit).unwrap();
        let quotient = ring
            .from_terms([
                (vec![0], Rational::from(1)),
                (vec![1], Rational::from(2)),
                (vec![3], Rational::from(-4)),
            ])
            .unwrap();
        let dividend = ring.mul(&divisor, &quotient).unwrap();

        let recovered = dividend.exact_div(&divisor).unwrap();
        assert!(recovered.uses_flint_storage());
        assert!(!recovered.sparse_view_is_materialized());
        assert_eq!(recovered, quotient);
        assert!(!recovered.sparse_view_is_materialized());
    }

    #[test]
    fn equal_separate_series_contexts_reuse_native_storage_without_materialization() {
        assert_equal_contexts_stay_native(fresh_ring(Z), fresh_ring(Z), 2.into(), 3.into());
        assert_equal_contexts_stay_native(
            fresh_ring(Q),
            fresh_ring(Q),
            "2/3".parse().unwrap(),
            "3/5".parse().unwrap(),
        );

        for modulus in [
            Integer::from(25),
            "18446744073709551629".parse::<Integer>().unwrap(),
        ] {
            let left_context = ResidueRing::new(modulus.clone()).unwrap();
            let right_context = ResidueRing::new(modulus).unwrap();
            assert_equal_contexts_stay_native(
                fresh_ring(left_context.clone()),
                fresh_ring(right_context.clone()),
                left_context.element(2),
                right_context.element(3),
            );
        }

        for characteristic in [
            Integer::from(7),
            "170141183460469231731687303715884105727"
                .parse::<Integer>()
                .unwrap(),
        ] {
            let left_context = FiniteField::prime(characteristic.clone()).unwrap();
            let right_context = FiniteField::prime(characteristic).unwrap();
            assert_equal_contexts_stay_native(
                fresh_ring(left_context.clone()),
                fresh_ring(right_context.clone()),
                left_context.element(2),
                right_context.element(3),
            );
        }

        let left_context = FiniteField::extension(3, 2).unwrap();
        let right_context = FiniteField::extension(3, 2).unwrap();
        assert_equal_contexts_stay_native(
            fresh_ring(left_context.clone()),
            fresh_ring(right_context.clone()),
            left_context.generator(),
            right_context.element(2),
        );
    }

    #[test]
    fn multivariate_weighted_series_keep_the_sparse_truncation_aware_floor() {
        let ring = PowerSeriesRing::with_grading(
            Z,
            ["x", "y", "z"],
            WeightedGrading::new(vec![1, 2, 3]).unwrap(),
            6,
        )
        .unwrap();
        let value = ring
            .from_terms([
                (vec![5, 0, 0], Integer::one()),
                (vec![1, 1, 0], Integer::from(2)),
                (vec![0, 0, 1], Integer::from(3)),
            ])
            .unwrap();

        assert!(!value.uses_flint_storage());
        assert!(value.sparse_view_is_materialized());
    }

    #[test]
    fn every_truncation_shape_matches_the_sparse_product_floor() {
        fn check(
            ring: &PowerSeriesRing<IntegerRing>,
            left_exponents: impl IntoIterator<Item = Vec<u64>>,
            right_exponents: impl IntoIterator<Item = Vec<u64>>,
        ) {
            let left = ring
                .from_terms(
                    left_exponents
                        .into_iter()
                        .enumerate()
                        .map(|(index, exponent)| (exponent, Integer::from(index + 1))),
                )
                .unwrap();
            let right = ring
                .from_terms(
                    right_exponents
                        .into_iter()
                        .enumerate()
                        .map(|(index, exponent)| (exponent, Integer::from(2 * index + 1))),
                )
                .unwrap();
            let mut expected = BTreeMap::new();
            for (left_exponent, left_coefficient) in left.terms_map() {
                for (right_exponent, right_coefficient) in right.terms_map() {
                    let exponent = left_exponent
                        .iter()
                        .zip(right_exponent)
                        .map(|(left, right)| left + right)
                        .collect::<Vec<_>>();
                    if !ring.truncation.contains_exponent(&exponent) {
                        continue;
                    }
                    let product = Z.mul(left_coefficient, right_coefficient).unwrap();
                    let coefficient = expected
                        .remove(&exponent)
                        .map_or(product.clone(), |previous| {
                            Z.add(&previous, &product).unwrap()
                        });
                    if !coefficient.is_zero() {
                        expected.insert(exponent, coefficient);
                    }
                }
            }
            assert_eq!(ring.mul(&left, &right).unwrap().terms_map(), &expected);
        }

        let univariate =
            PowerSeriesRing::new(Z, ["x"], SeriesTruncation::univariate(12).unwrap()).unwrap();
        check(
            &univariate,
            (0..8).map(|index| vec![index]).collect::<Vec<_>>(),
            (0..8).map(|index| vec![index]).collect::<Vec<_>>(),
        );

        for truncation in [
            SeriesTruncation::total_degree(11).unwrap(),
            SeriesTruncation::weighted_degree(vec![1, 2], 24).unwrap(),
            SeriesTruncation::box_bounds(vec![8, 8]).unwrap(),
        ] {
            let ring = PowerSeriesRing::new(Z, ["x", "y"], truncation).unwrap();
            check(
                &ring,
                (0..8)
                    .map(|index| vec![index, index % 3])
                    .collect::<Vec<_>>(),
                (0..8)
                    .map(|index| vec![index % 4, index])
                    .collect::<Vec<_>>(),
            );
        }
    }
}
