// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Positive-definite integral lattices and exact enumeration results.

use core::fmt;

use lattice_enum::{cvp_enumerate, gso, svp_enumerate, EnumError};

use crate::{Integer, IntegerRing, NoetherError, Rational, SemanticType, TypeDescription};

use super::Matrix;

static LATTICE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/Lattice@1",
    rust_path: "noether::linalg::Lattice",
    python_path: Some("noether.linalg.Lattice"),
};

static LATTICE_ENUMERATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/LatticeEnumeration@1",
    rust_path: "noether::linalg::LatticeEnumeration",
    python_path: Some("noether.linalg.LatticeEnumeration"),
};

static CLOSEST_LATTICE_VECTORS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/ClosestLatticeVectors@1",
    rust_path: "noether::linalg::ClosestLatticeVectors",
    python_path: Some("noether.linalg.ClosestLatticeVectors"),
};

/// Failure to construct a positive-definite lattice or enumerate it exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum LatticeError {
    /// The supplied Gram matrix was not square.
    InvalidShape {
        /// Stable explanation of the shape mismatch.
        message: String,
    },
    /// The supplied Gram matrix was not symmetric.
    Asymmetric {
        /// First zero-based position at which symmetry failed.
        at: (usize, usize),
    },
    /// The supplied symmetric Gram matrix was not positive definite.
    NotPositiveDefinite,
    /// A norm bound was negative.
    InvalidBound {
        /// Rejected exact bound.
        bound: Integer,
    },
    /// A minimum was requested for a lattice with no nonzero vectors.
    NoNonzeroVectors,
    /// The estimated enumeration exceeded the bounded-work safety limit.
    WorkLimitExceeded {
        /// Estimated number of lattice points in the search ellipsoid.
        estimate: f64,
    },
    /// A coordinate exceeded the word-sized public representation or a
    /// floating sizing heuristic was unavailable.
    SearchRangeExceeded,
    /// A supposedly certified minimum-vector configuration was malformed or
    /// failed exact re-verification.
    InvalidMinimumConfiguration {
        /// Stable explanation of the failed invariant.
        message: String,
    },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { message } => write!(formatter, "invalid lattice shape: {message}"),
            Self::Asymmetric { at } => write!(
                formatter,
                "lattice Gram matrix is asymmetric at ({}, {})",
                at.0, at.1
            ),
            Self::NotPositiveDefinite => {
                formatter.write_str("lattice Gram matrix is not positive definite")
            }
            Self::InvalidBound { bound } => {
                write!(
                    formatter,
                    "lattice norm bound must be nonnegative, got {bound}"
                )
            }
            Self::NoNonzeroVectors => formatter
                .write_str("the rank-zero lattice has no nonzero vector and no attained minimum"),
            Self::WorkLimitExceeded { estimate } => write!(
                formatter,
                "lattice enumeration estimate {estimate:.3e} exceeds the bounded-work limit"
            ),
            Self::SearchRangeExceeded => formatter.write_str(
                "a lattice coordinate is outside i64 or a floating sizing heuristic is unavailable",
            ),
            Self::InvalidMinimumConfiguration { message } => {
                write!(formatter, "invalid minimum-vector configuration: {message}")
            }
        }
    }
}

impl std::error::Error for LatticeError {}

impl From<EnumError> for LatticeError {
    fn from(error: EnumError) -> Self {
        match error {
            EnumError::InvalidShape { detail } | EnumError::PruningShape { detail } => {
                Self::InvalidShape { message: detail }
            }
            EnumError::Asymmetric { at } => Self::Asymmetric { at },
            EnumError::NotPositiveDefinite => Self::NotPositiveDefinite,
            EnumError::InvalidBound { bound } => Self::InvalidBound { bound },
            EnumError::NoNonzeroVectors => Self::NoNonzeroVectors,
            EnumError::EnumerationOverflow { estimate } => Self::WorkLimitExceeded { estimate },
            EnumError::GramOverflow => Self::SearchRangeExceeded,
        }
    }
}

/// A positive-definite integral lattice presented by its Gram matrix.
///
/// The Gram presentation is the mathematical owner. Enumeration dispatches to
/// `lattice-enum`, whose search geometry, candidates, and norms use exact
/// FLINT-backed integer and rational arithmetic.
#[derive(Clone, PartialEq, Eq)]
pub struct Lattice {
    gram: Matrix<IntegerRing>,
}

impl Lattice {
    /// Construct and validate a lattice from a symmetric positive-definite Gram matrix.
    pub fn from_gram(gram: Matrix<IntegerRing>) -> Result<Self, LatticeError> {
        let rows = gram.rows().collect::<Vec<_>>();
        gso::validate_gram(&rows)?;
        gso::Cholesky::factor(&rows)?;
        Ok(Self { gram })
    }

    /// The exact integral Gram matrix.
    pub fn gram_matrix(&self) -> &Matrix<IntegerRing> {
        &self.gram
    }

    /// The lattice rank.
    pub fn rank(&self) -> usize {
        self.gram.shape().0
    }

    /// Enumerate one canonical representative from each `±` pair up to a norm bound.
    ///
    /// With no bound, this returns the complete shortest-vector shell. Supplying
    /// `max_pairs` permits a deterministic prefix and bypasses the kernel's
    /// whole-ellipsoid work estimate. The returned value states whether that
    /// prefix is complete. A rank-zero lattice accepts bounded empty-shell
    /// queries but rejects the unbounded form because no nonzero minimum is
    /// attained.
    pub fn shortest_vectors(
        &self,
        bound: Option<Integer>,
        max_pairs: Option<usize>,
    ) -> Result<LatticeEnumeration, LatticeError> {
        let gram = self.gram.rows().collect::<Vec<_>>();
        let result = svp_enumerate(&gram, bound, max_pairs)?;
        Ok(LatticeEnumeration {
            signed_count: result.count,
            max_norm: result.max_norm,
            bound: result.bound,
            complete: result.complete,
            vectors: result.vectors,
        })
    }

    /// Compute and certify the complete minimum-vector configuration.
    pub fn minimum_vector_configuration(
        &self,
    ) -> Result<super::MinimumVectorConfiguration, LatticeError> {
        if self.rank() == 0 {
            return Err(LatticeError::NoNonzeroVectors);
        }
        let enumeration = self.shortest_vectors(None, None)?;
        super::MinimumVectorConfiguration::from_lattice_enumeration(self, enumeration)
    }

    /// Find every closest lattice vector to an exact rational target.
    ///
    /// `target_numerators / target_denominator` gives coordinates in the Gram
    /// presentation. Returned coordinates are currently word-sized; search
    /// geometry, Gram arithmetic, distances and comparisons are exact and
    /// arbitrary precision.
    pub fn closest_vectors(
        &self,
        target_numerators: &[i64],
        target_denominator: i64,
        max_count: Option<usize>,
    ) -> Result<ClosestLatticeVectors, LatticeError> {
        let gram = self.gram.rows().collect::<Vec<_>>();
        let result = cvp_enumerate(&gram, target_numerators, target_denominator, max_count)?;
        let distance_squared = Rational::new(result.distance_sq_num, result.distance_sq_den)
            .expect("lattice-enum returns a positive squared denominator");
        Ok(ClosestLatticeVectors {
            distance_squared,
            vectors: result.vectors,
        })
    }
}

impl SemanticType for Lattice {
    const TYPE_ID: &'static str = "noether/linalg/Lattice@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.gram.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &LATTICE_DESCRIPTION
    }
}

impl fmt::Debug for Lattice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Lattice")
            .field("rank", &self.rank())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Lattice(rank={})", self.rank())
    }
}

/// Exact short-vector enumeration with one canonical representative per `±` pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeEnumeration {
    signed_count: Integer,
    max_norm: Integer,
    bound: Option<Integer>,
    complete: bool,
    vectors: Vec<Vec<i64>>,
}

impl LatticeEnumeration {
    /// Total number of signed vectors, including both members of every `±` pair.
    pub fn signed_count(&self) -> &Integer {
        &self.signed_count
    }

    /// Number of canonical `±`-pair representatives retained in this result.
    pub fn pair_count(&self) -> usize {
        self.vectors.len()
    }

    /// Largest exact norm attained by a retained vector.
    pub fn max_norm(&self) -> &Integer {
        &self.max_norm
    }

    /// Requested norm bound, or `None` for the minimum shell.
    pub fn bound(&self) -> Option<&Integer> {
        self.bound.as_ref()
    }

    /// Whether the search proved that no qualifying vector pair was omitted.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Canonically signed coordinate vectors in deterministic order.
    pub fn vectors(&self) -> &[Vec<i64>] {
        &self.vectors
    }

    /// Recheck counts, dimensions, signs and exact norms against a lattice.
    pub fn verify(&self, lattice: &Lattice) -> Result<(), LatticeError> {
        if self.signed_count != Integer::from(2_usize.saturating_mul(self.vectors.len())) {
            return Err(LatticeError::InvalidShape {
                message: "signed vector count does not equal twice the pair count".to_owned(),
            });
        }
        let gram = lattice.gram.rows().collect::<Vec<_>>();
        let mut attained = Integer::zero();
        for vector in &self.vectors {
            if vector.len() != lattice.rank() || !has_canonical_sign(vector) {
                return Err(LatticeError::InvalidShape {
                    message: "a short vector has the wrong dimension or canonical sign".to_owned(),
                });
            }
            let norm = gso::exact_quadratic(&gram, vector)?;
            if !norm.is_positive() || norm > self.max_norm {
                return Err(LatticeError::InvalidBound {
                    bound: self.max_norm.clone(),
                });
            }
            if norm > attained {
                attained = norm;
            }
        }
        if attained != self.max_norm {
            return Err(LatticeError::InvalidShape {
                message: "reported maximum norm is not attained".to_owned(),
            });
        }
        if self.complete {
            let verification_cap = self
                .pair_count()
                .checked_add(1)
                .ok_or(LatticeError::SearchRangeExceeded)?;
            let rerun = lattice.shortest_vectors(self.bound.clone(), Some(verification_cap))?;
            if !rerun.complete
                || rerun.signed_count != self.signed_count
                || rerun.max_norm != self.max_norm
                || rerun.vectors != self.vectors
            {
                return Err(LatticeError::InvalidMinimumConfiguration {
                    message: "complete short-vector claim did not reproduce".to_owned(),
                });
            }
        }
        Ok(())
    }
}

impl SemanticType for LatticeEnumeration {
    const TYPE_ID: &'static str = "noether/linalg/LatticeEnumeration@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.signed_count.to_string(),
            self.max_norm.to_string(),
            self.bound.as_ref().map(ToString::to_string),
            self.complete,
            &self.vectors,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &LATTICE_ENUMERATION_DESCRIPTION
    }
}

/// Exact closest-vector result for one rational target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosestLatticeVectors {
    distance_squared: Rational,
    vectors: Vec<Vec<i64>>,
}

impl ClosestLatticeVectors {
    /// The exact squared distance to every returned closest vector.
    pub fn distance_squared(&self) -> &Rational {
        &self.distance_squared
    }

    /// All closest integral coordinate vectors in deterministic order.
    pub fn vectors(&self) -> &[Vec<i64>] {
        &self.vectors
    }
}

impl SemanticType for ClosestLatticeVectors {
    const TYPE_ID: &'static str = "noether/linalg/ClosestLatticeVectors@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.distance_squared.numerator().to_string(),
            self.distance_squared.denominator().to_string(),
            &self.vectors,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &CLOSEST_LATTICE_VECTORS_DESCRIPTION
    }
}

fn has_canonical_sign(vector: &[i64]) -> bool {
    vector.iter().find(|coordinate| **coordinate != 0).copied() > Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Z;

    #[test]
    fn integral_gram_dispatches_to_verified_enumeration() {
        let gram = Matrix::from_rows(
            Z,
            [
                [Integer::from(2), Integer::from(-1)],
                [Integer::from(-1), Integer::from(2)],
            ],
        )
        .unwrap();
        let lattice = Lattice::from_gram(gram).unwrap();
        let shortest = lattice.shortest_vectors(None, None).unwrap();

        assert_eq!(shortest.signed_count(), &Integer::from(6));
        assert_eq!(shortest.max_norm(), &Integer::from(2));
        assert_eq!(shortest.pair_count(), 3);
        assert_eq!(shortest.bound(), None);
        assert!(shortest.is_complete());
        shortest.verify(&lattice).unwrap();
    }

    #[test]
    fn capped_enumeration_reports_proved_completeness() {
        let lattice = Lattice::from_gram(Matrix::identity(Z, 2)).unwrap();
        let truncated = lattice
            .shortest_vectors(Some(Integer::from(4)), Some(1))
            .unwrap();
        assert!(!truncated.is_complete());
        truncated.verify(&lattice).unwrap();

        let exact = lattice
            .shortest_vectors(Some(Integer::one()), Some(2))
            .unwrap();
        assert!(exact.is_complete());
        exact.verify(&lattice).unwrap();
    }

    #[test]
    fn closest_vector_distance_is_exact() {
        let lattice = Lattice::from_gram(Matrix::identity(Z, 2)).unwrap();
        let closest = lattice.closest_vectors(&[1, 1], 2, None).unwrap();

        assert_eq!(
            closest.distance_squared(),
            &Rational::new(Integer::from(1), Integer::from(2)).unwrap()
        );
        assert_eq!(closest.vectors().len(), 4);
    }
}
