// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact Hilbert-series values in free-abelian grading group algebras.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Integer, NoetherError, SemanticType, TypeDescription};

use super::{Degree, FreeAbelianGrading, GradingError};

const MAX_HILBERT_SERIES_TERMS: usize = 100_000;
const MAX_STANDARD_HILBERT_DEGREE: usize = 4_096;

static HILBERT_SERIES_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/HilbertSeries@1",
    rust_path: "noether::commalg::HilbertSeries",
    python_path: Some("noether.commalg.HilbertSeries"),
};

/// Failure to construct or specialize an exact Hilbert series.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HilbertSeriesError {
    /// One numerator degree has the wrong grading rank.
    DegreeRank {
        /// Grading rank.
        expected: usize,
        /// Supplied degree rank.
        actual: usize,
    },
    /// A bounded exact representation exceeded its fixed term limit.
    TooManyTerms {
        /// Maximum admitted nonzero numerator terms.
        limit: usize,
    },
    /// An in-memory value does not use the canonical sparse representation.
    Noncanonical,
    /// The requested dense numerator requires the standard grading.
    NotStandardGraded,
    /// A dense standard numerator would exceed its fixed degree bound.
    StandardDegreeBound {
        /// Maximum admitted standard degree.
        limit: usize,
    },
}

impl fmt::Display for HilbertSeriesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegreeRank { expected, actual } => write!(
                formatter,
                "Hilbert numerator degree has rank {actual}; grading rank is {expected}"
            ),
            Self::TooManyTerms { limit } => {
                write!(formatter, "Hilbert numerator exceeds term limit {limit}")
            }
            Self::Noncanonical => {
                formatter.write_str("Hilbert numerator is not in canonical sparse form")
            }
            Self::NotStandardGraded => {
                formatter.write_str("a dense Hilbert numerator requires the standard grading")
            }
            Self::StandardDegreeBound { limit } => write!(
                formatter,
                "standard Hilbert numerator exceeds degree limit {limit}"
            ),
        }
    }
}

impl std::error::Error for HilbertSeriesError {}

/// Exact Hilbert series `K(T) / product_i (1 - T^degree(x_i))`.
///
/// The numerator is a canonical finite element of the group algebra of the
/// grading group. The denominator is determined by the ordered variable
/// degrees retained in [`Self::grading`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HilbertSeries {
    grading: FreeAbelianGrading,
    numerator: Vec<(Degree, Integer)>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HilbertSeriesWire {
    #[serde(rename = "type")]
    type_id: String,
    grading: FreeAbelianGrading,
    numerator: Vec<(Degree, Integer)>,
}

impl HilbertSeries {
    /// Construct and canonicalize an exact group-algebra numerator.
    pub fn new<I>(grading: FreeAbelianGrading, numerator: I) -> Result<Self, HilbertSeriesError>
    where
        I: IntoIterator<Item = (Degree, Integer)>,
    {
        let mut canonical = BTreeMap::<Degree, Integer>::new();
        for (degree, coefficient) in numerator {
            if degree.rank() != grading.rank() {
                return Err(HilbertSeriesError::DegreeRank {
                    expected: grading.rank(),
                    actual: degree.rank(),
                });
            }
            if coefficient.is_zero() {
                continue;
            }
            let coefficient = canonical
                .remove(&degree)
                .map_or(coefficient.clone(), |known| &known + &coefficient);
            if !coefficient.is_zero() {
                canonical.insert(degree, coefficient);
            }
            if canonical.len() > MAX_HILBERT_SERIES_TERMS {
                return Err(HilbertSeriesError::TooManyTerms {
                    limit: MAX_HILBERT_SERIES_TERMS,
                });
            }
        }
        Ok(Self {
            grading,
            numerator: canonical.into_iter().collect(),
        })
    }

    /// Exact free-abelian grading, including ordered denominator degrees.
    pub fn grading(&self) -> &FreeAbelianGrading {
        &self.grading
    }

    /// Canonical nonzero numerator terms in increasing degree order.
    pub fn numerator(&self) -> &[(Degree, Integer)] {
        &self.numerator
    }

    /// Independently recheck rank and canonicalization invariants.
    pub fn verify(&self) -> Result<(), HilbertSeriesError> {
        Self::new(self.grading.clone(), self.numerator.clone()).and_then(|rebuilt| {
            if &rebuilt == self {
                Ok(())
            } else {
                Err(HilbertSeriesError::Noncanonical)
            }
        })
    }

    /// Dense `K(t)` coefficients when this is the standard grading.
    pub fn standard_numerator(&self) -> Result<Vec<Integer>, HilbertSeriesError> {
        if self.grading != FreeAbelianGrading::standard(self.grading.arity()) {
            return Err(HilbertSeriesError::NotStandardGraded);
        }
        let mut degree_coefficients = Vec::new();
        for (degree, coefficient) in &self.numerator {
            let degree = degree
                .coordinates()
                .first()
                .and_then(Integer::to_u64_checked)
                .and_then(|degree| usize::try_from(degree).ok())
                .ok_or(HilbertSeriesError::StandardDegreeBound {
                    limit: MAX_STANDARD_HILBERT_DEGREE,
                })?;
            if degree > MAX_STANDARD_HILBERT_DEGREE {
                return Err(HilbertSeriesError::StandardDegreeBound {
                    limit: MAX_STANDARD_HILBERT_DEGREE,
                });
            }
            if degree_coefficients.len() <= degree {
                degree_coefficients.resize_with(degree + 1, Integer::zero);
            }
            degree_coefficients[degree] = coefficient.clone();
        }
        if degree_coefficients.is_empty() {
            degree_coefficients.push(Integer::zero());
        }
        Ok(degree_coefficients)
    }

    /// Serialize the complete exact series as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently revalidate an exact series.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for HilbertSeries {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        HilbertSeriesWire {
            type_id: Self::TYPE_ID.to_owned(),
            grading: self.grading.clone(),
            numerator: self.numerator.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for HilbertSeries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = HilbertSeriesWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.grading, wire.numerator).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for HilbertSeries {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Hilbert series with {} numerator terms over {}",
            self.numerator.len(),
            self.grading
        )
    }
}

impl SemanticType for HilbertSeries {
    const TYPE_ID: &'static str = "noether/commalg/HilbertSeries@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.grading.canonical_bytes()?)?;
        append_usize(&mut output, self.numerator.len())?;
        for (degree, coefficient) in &self.numerator {
            append_component(&mut output, &degree.canonical_bytes()?)?;
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &HILBERT_SERIES_DESCRIPTION
    }
}

impl From<GradingError> for HilbertSeriesError {
    fn from(error: GradingError) -> Self {
        match error {
            GradingError::RankMismatch { expected, actual } => {
                Self::DegreeRank { expected, actual }
            }
            _ => Self::NotStandardGraded,
        }
    }
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value)
        .map_err(|_| NoetherError::Serialization("length does not fit in u64".to_owned()))?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_canonicalize_and_typed_json_revalidates() {
        let grading = FreeAbelianGrading::standard(2);
        let series = HilbertSeries::new(
            grading,
            [
                (Degree::new(vec![Integer::zero()]), Integer::one()),
                (Degree::new(vec![Integer::from(2)]), Integer::from(-2)),
                (Degree::new(vec![Integer::from(2)]), Integer::one()),
            ],
        )
        .unwrap();
        assert_eq!(
            series.standard_numerator().unwrap(),
            vec![Integer::one(), Integer::zero(), Integer::from(-1)]
        );
        assert_eq!(
            HilbertSeries::from_json(&series.to_json().unwrap()).unwrap(),
            series
        );
    }
}
