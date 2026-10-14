// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact singly and multigraded Betti tables.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{Degree, FreeAbelianGrading, PolynomialRing};
use crate::{Field, NoetherError, SemanticType, TypeDescription};

use super::{FreeResolution, ResolutionCompleteness, ResolutionError};

const MAX_BETTI_ENTRIES: usize = 100_000;

static GRADED_BETTI_TABLE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/GradedBettiTable@1",
    rust_path: "noether::homalg::GradedBettiTable",
    python_path: Some("noether.homalg.GradedBettiTable"),
};

/// Failure to construct or derive an exact graded Betti table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GradedBettiTableError {
    /// Resolution construction or verification failed.
    Resolution(ResolutionError),
    /// Betti numbers require a verified minimal resolution.
    MinimalityRequired,
    /// One represented free module did not carry exact shifts.
    GradingRequired,
    /// One internal degree had the wrong grading rank.
    DegreeRank {
        /// Grading rank.
        expected: usize,
        /// Supplied degree rank.
        actual: usize,
    },
    /// An entry lay above the represented homological range.
    HomologicalDegree {
        /// Resolution length.
        length: usize,
        /// Rejected degree.
        degree: usize,
    },
    /// A multiplicity was zero.
    ZeroMultiplicity,
    /// A bounded table exceeded its fixed sparse-entry limit.
    TooManyEntries {
        /// Maximum admitted nonzero entries.
        limit: usize,
    },
    /// An in-memory table was not in canonical sparse order.
    Noncanonical,
}

impl fmt::Display for GradedBettiTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution(error) => error.fmt(formatter),
            Self::MinimalityRequired => {
                formatter.write_str("graded Betti numbers require a verified minimal resolution")
            }
            Self::GradingRequired => {
                formatter.write_str("graded Betti numbers require exact free-module shifts")
            }
            Self::DegreeRank { expected, actual } => write!(
                formatter,
                "Betti internal degree has rank {actual}; grading rank is {expected}"
            ),
            Self::HomologicalDegree { length, degree } => write!(
                formatter,
                "Betti homological degree {degree} exceeds resolution length {length}"
            ),
            Self::ZeroMultiplicity => {
                formatter.write_str("a sparse Betti entry has zero multiplicity")
            }
            Self::TooManyEntries { limit } => {
                write!(formatter, "graded Betti table exceeds entry limit {limit}")
            }
            Self::Noncanonical => formatter.write_str("graded Betti table is not canonical"),
        }
    }
}

impl std::error::Error for GradedBettiTableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ResolutionError> for GradedBettiTableError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

/// Canonical exact multigraded Betti numbers `beta_(i,a)`.
///
/// Internal degrees are the negatives of Noether's free-module shifts: a
/// summand `R(-a)` contributes one to `beta_(i,a)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedBettiTable {
    grading: FreeAbelianGrading,
    length: usize,
    completeness: ResolutionCompleteness,
    entries: Vec<(usize, Degree, usize)>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradedBettiTableWire {
    #[serde(rename = "type")]
    type_id: String,
    grading: FreeAbelianGrading,
    length: usize,
    completeness: ResolutionCompleteness,
    entries: Vec<(usize, Degree, usize)>,
}

impl GradedBettiTable {
    /// Construct and canonicalize exact sparse Betti entries.
    fn new<I>(
        grading: FreeAbelianGrading,
        length: usize,
        completeness: ResolutionCompleteness,
        entries: I,
    ) -> Result<Self, GradedBettiTableError>
    where
        I: IntoIterator<Item = (usize, Degree, usize)>,
    {
        let mut canonical = BTreeMap::<(usize, Degree), usize>::new();
        for (homological_degree, internal_degree, multiplicity) in entries {
            if homological_degree > length {
                return Err(GradedBettiTableError::HomologicalDegree {
                    length,
                    degree: homological_degree,
                });
            }
            if internal_degree.rank() != grading.rank() {
                return Err(GradedBettiTableError::DegreeRank {
                    expected: grading.rank(),
                    actual: internal_degree.rank(),
                });
            }
            if multiplicity == 0 {
                return Err(GradedBettiTableError::ZeroMultiplicity);
            }
            let value = canonical
                .entry((homological_degree, internal_degree))
                .or_default();
            *value =
                value
                    .checked_add(multiplicity)
                    .ok_or(GradedBettiTableError::TooManyEntries {
                        limit: MAX_BETTI_ENTRIES,
                    })?;
            if canonical.len() > MAX_BETTI_ENTRIES {
                return Err(GradedBettiTableError::TooManyEntries {
                    limit: MAX_BETTI_ENTRIES,
                });
            }
        }
        if let ResolutionCompleteness::ThroughHomologicalDegree { degree } = completeness {
            if degree != length {
                return Err(GradedBettiTableError::HomologicalDegree { length, degree });
            }
        }
        Ok(Self {
            grading,
            length,
            completeness,
            entries: canonical
                .into_iter()
                .map(|((homological_degree, internal_degree), multiplicity)| {
                    (homological_degree, internal_degree, multiplicity)
                })
                .collect(),
        })
    }

    pub(crate) fn from_exact_entries<I>(
        grading: FreeAbelianGrading,
        length: usize,
        completeness: ResolutionCompleteness,
        entries: I,
    ) -> Result<Self, GradedBettiTableError>
    where
        I: IntoIterator<Item = (usize, Degree, usize)>,
    {
        Self::new(grading, length, completeness, entries)
    }

    /// Exact polynomial-ring grading.
    pub fn grading(&self) -> &FreeAbelianGrading {
        &self.grading
    }

    /// Greatest represented homological degree.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Exact completeness or truncation claim inherited from the resolution.
    pub fn completeness(&self) -> &ResolutionCompleteness {
        &self.completeness
    }

    /// Canonical nonzero `(homological degree, internal degree, multiplicity)`
    /// entries.
    pub fn entries(&self) -> &[(usize, Degree, usize)] {
        &self.entries
    }

    /// One exact multigraded Betti number.
    pub fn get(&self, homological_degree: usize, internal_degree: &Degree) -> usize {
        self.entries
            .binary_search_by(|(known_homological, known_internal, _)| {
                (known_homological, known_internal).cmp(&(&homological_degree, internal_degree))
            })
            .ok()
            .map_or(0, |index| self.entries[index].2)
    }

    /// Total rank in one represented homological degree.
    pub fn total(&self, homological_degree: usize) -> usize {
        self.entries
            .iter()
            .filter(|(known, _, _)| *known == homological_degree)
            .map(|(_, _, multiplicity)| *multiplicity)
            .sum()
    }

    /// Projective dimension when the resolution is proved complete.
    pub fn projective_dimension(&self) -> Option<usize> {
        (self.completeness == ResolutionCompleteness::Complete).then_some(self.length)
    }

    /// Independently recheck all canonical sparse-table invariants.
    pub fn verify(&self) -> Result<(), GradedBettiTableError> {
        Self::new(
            self.grading.clone(),
            self.length,
            self.completeness.clone(),
            self.entries.clone(),
        )
        .and_then(|rebuilt| {
            if &rebuilt == self {
                Ok(())
            } else {
                Err(GradedBettiTableError::Noncanonical)
            }
        })
    }

    /// Serialize the complete exact table as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently revalidate an exact table.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    pub(crate) fn from_resolution<C: Field>(
        resolution: &FreeResolution<PolynomialRing<C>>,
    ) -> Result<Self, GradedBettiTableError> {
        if !resolution.is_minimal() {
            return Err(GradedBettiTableError::MinimalityRequired);
        }
        let mut entries = Vec::new();
        for homological_degree in 0..=resolution.length() {
            let shifts = resolution
                .free_module(homological_degree)
                .and_then(|module| module.degree_shifts())
                .ok_or(GradedBettiTableError::GradingRequired)?;
            entries.extend(
                shifts
                    .iter()
                    .map(|shift| (homological_degree, shift.neg(), 1)),
            );
        }
        Self::new(
            resolution
                .module()
                .generator_module()
                .ring()
                .grading()
                .clone(),
            resolution.length(),
            resolution.completeness().clone(),
            entries,
        )
    }
}

impl Serialize for GradedBettiTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        GradedBettiTableWire {
            type_id: Self::TYPE_ID.to_owned(),
            grading: self.grading.clone(),
            length: self.length,
            completeness: self.completeness.clone(),
            entries: self.entries.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradedBettiTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = GradedBettiTableWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.grading, wire.length, wire.completeness, wire.entries)
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for GradedBettiTable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GradedBettiTable(length={}, entries={}, {})",
            self.length,
            self.entries.len(),
            self.completeness
        )
    }
}

impl SemanticType for GradedBettiTable {
    const TYPE_ID: &'static str = "noether/homalg/GradedBettiTable@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_BETTI_TABLE_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::FreeAbelianGrading;
    use crate::Integer;

    #[test]
    fn sparse_entries_canonicalize_and_round_trip() {
        let table = GradedBettiTable::new(
            FreeAbelianGrading::standard(3),
            1,
            ResolutionCompleteness::Complete,
            [
                (0, Degree::new(vec![Integer::zero()]), 1),
                (1, Degree::new(vec![Integer::from(2)]), 2),
                (1, Degree::new(vec![Integer::from(2)]), 1),
            ],
        )
        .unwrap();
        assert_eq!(table.get(1, &Degree::new(vec![Integer::from(2)])), 3);
        assert_eq!(table.total(1), 3);
        assert_eq!(table.projective_dimension(), Some(1));
        table.verify().unwrap();
        assert_eq!(
            GradedBettiTable::from_json(&table.to_json().unwrap()).unwrap(),
            table
        );
    }
}
