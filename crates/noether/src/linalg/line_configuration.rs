// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite configurations of primitive integral lines and certified minima.

use core::fmt;

use crate::{Integer, IntegerRing, NoetherError, Rational, SemanticType, TypeDescription, Z};

use super::{Lattice, LatticeEnumeration, LatticeError, Matrix};

static INTEGRAL_LINE_CONFIGURATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegralLineConfiguration@1",
    rust_path: "noether::linalg::IntegralLineConfiguration",
    python_path: Some("noether.linalg.IntegralLineConfiguration"),
};

static MINIMUM_VECTOR_CONFIGURATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumVectorConfiguration@1",
    rust_path: "noether::linalg::MinimumVectorConfiguration",
    python_path: Some("noether.linalg.MinimumVectorConfiguration"),
};

/// Failure to construct or transform a configuration of primitive integral lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineConfigurationError {
    /// A vector did not have the declared ambient dimension.
    DimensionMismatch {
        /// Zero-based input vector index.
        line: usize,
        /// Declared ambient rank.
        expected: usize,
        /// Received vector length.
        actual: usize,
    },
    /// A supplied vector was zero and therefore did not define a line.
    ZeroVector {
        /// Zero-based input vector index.
        line: usize,
    },
    /// A supplied vector was not primitive.
    NonPrimitive {
        /// Zero-based input vector index.
        line: usize,
        /// Positive greatest common divisor of its coordinates.
        content: Integer,
    },
    /// A requested coordinate transformation had the wrong shape.
    TransformationShape {
        /// Required square shape.
        expected: (usize, usize),
        /// Received shape.
        actual: (usize, usize),
    },
    /// A requested coordinate transformation was not unimodular.
    NonUnimodularTransformation {
        /// Exact determinant of the rejected matrix.
        determinant: Integer,
    },
    /// A unimodular transformation did not preserve the line set.
    TransformationDoesNotPreserveConfiguration,
    /// A finite configuration stabilizer was requested for nonspanning lines.
    NonspanningStabilizer,
    /// Integral equivalence was requested for a nonspanning configuration.
    NonspanningEquivalence,
    /// Minimum-vector realization was requested for a nonspanning configuration.
    NonspanningRealization,
    /// Minimum-vector realization was requested in rank zero.
    RankZeroRealization,
    /// The ambient rank cannot be represented safely by matrix dimensions.
    AmbientRankTooLarge {
        /// Rejected ambient rank.
        ambient_rank: usize,
    },
    /// A line action could not fit the permutation point domain.
    PermutationDomainTooLarge {
        /// Number of unoriented lines.
        line_count: usize,
    },
    /// An external-line orbit action exceeded its fixed materialization cap.
    LineOrbitEnumerationLimitExceeded {
        /// Maximum number of tagged orbit points accepted.
        cap: usize,
    },
    /// The deterministic configuration-symmetry search exhausted its cap.
    SymmetrySearchBudgetExceeded {
        /// Maximum search nodes permitted.
        cap: usize,
    },
    /// An exact symmetry computation failed an internal algebraic invariant.
    SymmetryComputation {
        /// Stable explanation of the failed invariant.
        message: String,
    },
    /// An exact minimum-realization computation failed an internal invariant.
    RealizationComputation {
        /// Stable explanation of the failed invariant.
        message: String,
    },
    /// The determinant of an induced finite-order action was not `±1`.
    InvalidOrientationDeterminant {
        /// Exact rejected determinant.
        determinant: Rational,
    },
}

impl fmt::Display for LineConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                line,
                expected,
                actual,
            } => write!(
                formatter,
                "line {line} has dimension {actual}; expected {expected}"
            ),
            Self::ZeroVector { line } => {
                write!(formatter, "line {line} is represented by the zero vector")
            }
            Self::NonPrimitive { line, content } => write!(
                formatter,
                "line {line} has nonprimitive coordinate content {content}"
            ),
            Self::TransformationShape { expected, actual } => write!(
                formatter,
                "line transformation has shape {}x{}; expected {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::NonUnimodularTransformation { determinant } => write!(
                formatter,
                "line transformation has determinant {determinant}; expected ±1"
            ),
            Self::TransformationDoesNotPreserveConfiguration => formatter
                .write_str("unimodular transformation does not preserve the line configuration"),
            Self::NonspanningStabilizer => formatter
                .write_str("a finite integral-line stabilizer requires a spanning configuration"),
            Self::NonspanningEquivalence => formatter
                .write_str("integral-line equivalence currently requires spanning configurations"),
            Self::NonspanningRealization => formatter.write_str(
                "minimum-vector realization currently requires a spanning configuration",
            ),
            Self::RankZeroRealization => formatter.write_str(
                "minimum-vector realization requires positive ambient rank because the zero lattice has no nonzero vectors",
            ),
            Self::AmbientRankTooLarge { ambient_rank } => write!(
                formatter,
                "ambient rank {ambient_rank} has an unrepresentable symmetric-square dimension"
            ),
            Self::PermutationDomainTooLarge { line_count } => write!(
                formatter,
                "the {line_count}-line configuration exceeds the permutation point domain"
            ),
            Self::LineOrbitEnumerationLimitExceeded { cap } => write!(
                formatter,
                "external integral-line orbit enumeration exceeded its {cap}-point cap"
            ),
            Self::SymmetrySearchBudgetExceeded { cap } => write!(
                formatter,
                "configuration symmetry search exceeded its {cap}-node cap"
            ),
            Self::SymmetryComputation { message } => {
                write!(
                    formatter,
                    "configuration symmetry computation failed: {message}"
                )
            }
            Self::RealizationComputation { message } => {
                write!(
                    formatter,
                    "minimum-vector realization computation failed: {message}"
                )
            }
            Self::InvalidOrientationDeterminant { determinant } => write!(
                formatter,
                "induced rank-one action has determinant {determinant}; expected ±1"
            ),
        }
    }
}

impl std::error::Error for LineConfigurationError {}

/// A finite set of primitive rank-one direct summands of `Z^N`.
///
/// Each line is stored as one primitive vector modulo sign. The first nonzero
/// coordinate is positive, and the line rows are sorted and deduplicated. The
/// value retains its coordinate presentation; integral equivalence is an
/// operation rather than implicit object identity.
#[derive(Clone, PartialEq, Eq)]
pub struct IntegralLineConfiguration {
    ambient_rank: usize,
    lines: Vec<Vec<Integer>>,
}

impl IntegralLineConfiguration {
    /// Construct a checked, canonically ordered integral-line configuration.
    pub fn new(
        ambient_rank: usize,
        lines: impl IntoIterator<Item = Vec<Integer>>,
    ) -> Result<Self, LineConfigurationError> {
        checked_symmetric_dimension(ambient_rank)?;
        let mut canonical = Vec::new();
        for (index, mut line) in lines.into_iter().enumerate() {
            if line.len() != ambient_rank {
                return Err(LineConfigurationError::DimensionMismatch {
                    line: index,
                    expected: ambient_rank,
                    actual: line.len(),
                });
            }
            let content = line
                .iter()
                .fold(Integer::zero(), |gcd, entry| gcd.gcd(&entry.abs()));
            if content.is_zero() {
                return Err(LineConfigurationError::ZeroVector { line: index });
            }
            if !content.is_one() {
                return Err(LineConfigurationError::NonPrimitive {
                    line: index,
                    content,
                });
            }
            if line
                .iter()
                .find(|entry| !entry.is_zero())
                .is_some_and(Integer::is_negative)
            {
                for entry in &mut line {
                    *entry = -entry.clone();
                }
            }
            canonical.push(line);
        }
        canonical.sort();
        canonical.dedup();
        Ok(Self {
            ambient_rank,
            lines: canonical,
        })
    }

    /// Ambient lattice rank `N`.
    pub fn ambient_rank(&self) -> usize {
        self.ambient_rank
    }

    /// Number of unoriented integral lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Canonically signed and sorted primitive vector representatives.
    pub fn vectors(&self) -> &[Vec<Integer>] {
        &self.lines
    }

    /// Rank of the integral vectors over `Q`.
    pub fn span_rank(&self) -> usize {
        self.line_matrix().rank_backend()
    }

    /// Whether the lines span the ambient rational vector space.
    pub fn is_spanning(&self) -> bool {
        self.span_rank() == self.ambient_rank
    }

    /// Matrix whose rows are the upper-triangular coordinates of `v v^t`.
    pub fn rank_one_matrix(&self) -> Matrix<IntegerRing> {
        let symmetric_dimension = self.symmetric_dimension();
        if self.lines.is_empty() {
            return Matrix::zeros(Z, 0, symmetric_dimension);
        }
        Matrix::from_rows(
            Z,
            self.lines.iter().map(|line| {
                let mut row = Vec::with_capacity(symmetric_dimension);
                for left in 0..self.ambient_rank {
                    for right in left..self.ambient_rank {
                        row.push(&line[left] * &line[right]);
                    }
                }
                row
            }),
        )
        .expect("rank-one rows have a checked common dimension")
    }

    /// Dimension of the span of the forms `v v^t`.
    pub fn perfection_rank(&self) -> usize {
        self.rank_one_matrix().rank_backend()
    }

    /// Whether the rank-one forms are linearly independent.
    pub fn is_rank_one_independent(&self) -> bool {
        self.perfection_rank() == self.line_count()
    }

    /// Exact barycenter form `sum_v v v^t`.
    pub fn barycenter_form(&self) -> Matrix<IntegerRing> {
        let mut rows = vec![vec![Integer::zero(); self.ambient_rank]; self.ambient_rank];
        for line in &self.lines {
            for row in 0..self.ambient_rank {
                for column in 0..self.ambient_rank {
                    let term = &line[row] * &line[column];
                    rows[row][column] = &rows[row][column] + &term;
                }
            }
        }
        Matrix::from_rows(Z, rows).expect("barycenter rows have a checked square shape")
    }

    /// Apply a checked unimodular matrix to every column vector.
    pub fn transform(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<Self, LineConfigurationError> {
        let expected = (self.ambient_rank, self.ambient_rank);
        if transformation.shape() != expected {
            return Err(LineConfigurationError::TransformationShape {
                expected,
                actual: transformation.shape(),
            });
        }
        let determinant = transformation
            .det()
            .expect("an integer square matrix has a determinant");
        if !determinant.abs().is_one() {
            return Err(LineConfigurationError::NonUnimodularTransformation { determinant });
        }
        let matrix_rows = transformation.rows().collect::<Vec<_>>();
        let transformed = self.lines.iter().map(|line| {
            matrix_rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(line)
                        .fold(Integer::zero(), |sum, (left, right)| &sum + &(left * right))
                })
                .collect::<Vec<_>>()
        });
        Self::new(self.ambient_rank, transformed)
    }

    fn line_matrix(&self) -> Matrix<IntegerRing> {
        if self.lines.is_empty() {
            Matrix::zeros(Z, 0, self.ambient_rank)
        } else {
            Matrix::from_rows(Z, self.lines.clone())
                .expect("configuration lines have a checked common dimension")
        }
    }

    pub(super) fn symmetric_dimension(&self) -> usize {
        checked_symmetric_dimension(self.ambient_rank)
            .expect("the constructor checked the symmetric-square dimension")
    }
}

fn checked_symmetric_dimension(ambient_rank: usize) -> Result<usize, LineConfigurationError> {
    let successor = ambient_rank
        .checked_add(1)
        .ok_or(LineConfigurationError::AmbientRankTooLarge { ambient_rank })?;
    let (left, right) = if ambient_rank & 1 == 0 {
        (ambient_rank / 2, successor)
    } else {
        (ambient_rank, successor / 2)
    };
    let dimension = left
        .checked_mul(right)
        .ok_or(LineConfigurationError::AmbientRankTooLarge { ambient_rank })?;
    i64::try_from(dimension)
        .map(|_| dimension)
        .map_err(|_| LineConfigurationError::AmbientRankTooLarge { ambient_rank })
}

impl SemanticType for IntegralLineConfiguration {
    const TYPE_ID: &'static str = "noether/linalg/IntegralLineConfiguration@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let rows = self
            .lines
            .iter()
            .map(|line| line.iter().map(ToString::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        serde_json::to_vec(&(Self::TYPE_ID, self.ambient_rank, rows))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_LINE_CONFIGURATION_DESCRIPTION
    }
}

impl fmt::Debug for IntegralLineConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegralLineConfiguration")
            .field("ambient_rank", &self.ambient_rank)
            .field("line_count", &self.line_count())
            .field("perfection_rank", &self.perfection_rank())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntegralLineConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IntegralLineConfiguration(rank={}, lines={}, perfection_rank={})",
            self.ambient_rank,
            self.line_count(),
            self.perfection_rank()
        )
    }
}

/// A configuration certified as the complete minimum-vector set of a lattice.
///
/// The Gram matrix and minimum are evidence. Mathematical equality compares
/// the embedded integral-line configuration rather than the chosen witness.
#[derive(Clone)]
pub struct MinimumVectorConfiguration {
    configuration: IntegralLineConfiguration,
    gram: Matrix<IntegerRing>,
    minimum: Integer,
}

impl MinimumVectorConfiguration {
    pub(super) fn from_lattice_enumeration(
        lattice: &Lattice,
        enumeration: LatticeEnumeration,
    ) -> Result<Self, LatticeError> {
        if lattice.rank() == 0 {
            return Err(LatticeError::NoNonzeroVectors);
        }
        if enumeration.bound().is_some() || !enumeration.is_complete() {
            return Err(LatticeError::InvalidMinimumConfiguration {
                message: "minimum-vector evidence must be a complete minimum-shell enumeration"
                    .to_owned(),
            });
        }
        let lines = enumeration
            .vectors()
            .iter()
            .map(|line| line.iter().copied().map(Integer::from).collect::<Vec<_>>());
        let configuration =
            IntegralLineConfiguration::new(lattice.rank(), lines).map_err(|error| {
                LatticeError::InvalidMinimumConfiguration {
                    message: error.to_string(),
                }
            })?;
        let value = Self {
            configuration,
            gram: lattice.gram_matrix().clone(),
            minimum: enumeration.max_norm().clone(),
        };
        value.verify()?;
        Ok(value)
    }

    /// Certified primitive integral lines modulo sign.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        &self.configuration
    }

    /// Exact positive-definite Gram witness.
    pub fn gram_matrix(&self) -> &Matrix<IntegerRing> {
        &self.gram
    }

    /// Exact minimum attained by every stored line.
    pub fn minimum(&self) -> &Integer {
        &self.minimum
    }

    /// Re-enumerate the minimum shell and verify all evidence.
    pub fn verify(&self) -> Result<(), LatticeError> {
        if self.configuration.ambient_rank() == 0 {
            return Err(LatticeError::NoNonzeroVectors);
        }
        let lattice = Lattice::from_gram(self.gram.clone())?;
        let enumeration = lattice.shortest_vectors(None, None)?;
        if !enumeration.is_complete() || enumeration.max_norm() != &self.minimum {
            return Err(LatticeError::InvalidMinimumConfiguration {
                message: "Gram witness did not reproduce the certified minimum".to_owned(),
            });
        }
        let lines = enumeration
            .vectors()
            .iter()
            .map(|line| line.iter().copied().map(Integer::from).collect::<Vec<_>>());
        let reproduced =
            IntegralLineConfiguration::new(lattice.rank(), lines).map_err(|error| {
                LatticeError::InvalidMinimumConfiguration {
                    message: error.to_string(),
                }
            })?;
        if reproduced != self.configuration {
            return Err(LatticeError::InvalidMinimumConfiguration {
                message: "Gram witness has a different complete minimum-vector set".to_owned(),
            });
        }
        Ok(())
    }
}

impl PartialEq for MinimumVectorConfiguration {
    fn eq(&self, other: &Self) -> bool {
        self.configuration == other.configuration
    }
}

impl Eq for MinimumVectorConfiguration {}

impl SemanticType for MinimumVectorConfiguration {
    const TYPE_ID: &'static str = "noether/linalg/MinimumVectorConfiguration@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let rows = self
            .configuration
            .vectors()
            .iter()
            .map(|line| line.iter().map(ToString::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        serde_json::to_vec(&(Self::TYPE_ID, self.configuration.ambient_rank(), rows))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_VECTOR_CONFIGURATION_DESCRIPTION
    }
}

impl fmt::Debug for MinimumVectorConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumVectorConfiguration")
            .field("configuration", &self.configuration)
            .field("minimum", &self.minimum)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for MinimumVectorConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumVectorConfiguration(rank={}, lines={}, perfection_rank={}, minimum={})",
            self.configuration.ambient_rank(),
            self.configuration.line_count(),
            self.configuration.perfection_rank(),
            self.minimum
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integers(rows: &[&[i64]]) -> Vec<Vec<Integer>> {
        rows.iter()
            .map(|row| row.iter().copied().map(Integer::from).collect())
            .collect()
    }

    #[test]
    fn canonicalizes_sign_and_deduplicates_lines() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[-1, 0], &[1, 0], &[0, -1]])).unwrap();
        assert_eq!(configuration.vectors(), integers(&[&[0, 1], &[1, 0]]));
        assert_eq!(configuration.span_rank(), 2);
        assert_eq!(configuration.perfection_rank(), 2);
        assert!(configuration.is_spanning());
        assert!(configuration.is_rank_one_independent());
        assert_eq!(configuration.barycenter_form(), Matrix::identity(Z, 2));
    }

    #[test]
    fn computes_full_symmetric_square_rank() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1], &[1, -1]])).unwrap();
        assert_eq!(configuration.perfection_rank(), 3);
        assert_eq!(
            configuration.barycenter_form(),
            Matrix::from_rows(Z, integers(&[&[2, -1], &[-1, 2]]),).unwrap()
        );
    }

    #[test]
    fn rejects_zero_nonprimitive_and_wrong_dimension() {
        assert!(matches!(
            IntegralLineConfiguration::new(2, integers(&[&[0, 0]])),
            Err(LineConfigurationError::ZeroVector { .. })
        ));
        assert!(matches!(
            IntegralLineConfiguration::new(2, integers(&[&[2, 0]])),
            Err(LineConfigurationError::NonPrimitive { .. })
        ));
        assert!(matches!(
            IntegralLineConfiguration::new(2, integers(&[&[1]])),
            Err(LineConfigurationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_ambient_ranks_with_unrepresentable_symmetric_squares() {
        assert!(matches!(
            IntegralLineConfiguration::new(usize::MAX, Vec::<Vec<Integer>>::new()),
            Err(LineConfigurationError::AmbientRankTooLarge { .. })
        ));
        if usize::BITS >= 64 {
            let rank = usize::try_from(1_u64 << 32).unwrap();
            assert!(matches!(
                IntegralLineConfiguration::new(rank, Vec::<Vec<Integer>>::new()),
                Err(LineConfigurationError::AmbientRankTooLarge { .. })
            ));
        }
    }

    #[test]
    fn lattice_produces_certified_minimum_configuration() {
        let lattice =
            Lattice::from_gram(Matrix::from_rows(Z, integers(&[&[1, 0], &[0, 2]])).unwrap())
                .unwrap();
        let minimum = lattice.minimum_vector_configuration().unwrap();
        assert_eq!(minimum.minimum(), &Integer::one());
        assert_eq!(minimum.configuration().vectors(), integers(&[&[1, 0]]));
        assert_eq!(minimum.configuration().span_rank(), 1);
        minimum.verify().unwrap();
    }

    #[test]
    fn rank_zero_lattice_has_no_attained_minimum() {
        let lattice = Lattice::from_gram(Matrix::zeros(Z, 0, 0)).unwrap();
        assert_eq!(
            lattice.shortest_vectors(None, None),
            Err(LatticeError::NoNonzeroVectors)
        );
        let bounded = lattice
            .shortest_vectors(Some(Integer::zero()), None)
            .unwrap();
        assert!(bounded.vectors().is_empty());
        bounded.verify(&lattice).unwrap();
        assert_eq!(
            lattice.minimum_vector_configuration(),
            Err(LatticeError::NoNonzeroVectors)
        );
    }

    #[test]
    fn transforms_by_unimodular_matrices_only() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1]])).unwrap();
        let swap = Matrix::from_rows(Z, integers(&[&[0, 1], &[1, 0]])).unwrap();
        assert_eq!(configuration.transform(&swap).unwrap(), configuration);
        let scaling = Matrix::from_rows(Z, integers(&[&[2, 0], &[0, 1]])).unwrap();
        assert!(matches!(
            configuration.transform(&scaling),
            Err(LineConfigurationError::NonUnimodularTransformation { .. })
        ));
    }
}
