// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integer partitions and compositions.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{NoetherError, SemanticType, TypeDescription};

/// Largest Young diagram materialized by the 0.1 partition representation.
///
/// Both conjugation and hook lengths return one explicit integer per column or
/// cell. The bound makes those accepted, infallible operations safe for every
/// constructed partition instead of allowing an accidental unbounded
/// allocation.
pub const MAX_PARTITION_CELLS: usize = 1_000_000;

/// A failure while constructing an integer partition or composition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinError {
    /// A part was zero; parts in either object are positive.
    ZeroPart {
        /// Zero-based position of the rejected part.
        index: usize,
    },
    /// Partition parts were not weakly decreasing.
    PartitionOrder {
        /// Zero-based position of the first rejected adjacent pair.
        index: usize,
        /// Earlier part.
        left: usize,
        /// Later part.
        right: usize,
    },
    /// The sum of the parts overflowed the platform index type.
    SizeOverflow,
    /// A partition is too large for its explicit 0.1 algorithms.
    PartitionTooLarge {
        /// Number of cells in the Young diagram.
        size: usize,
        /// Largest accepted number of cells.
        max: usize,
    },
    /// A tableau row was not weakly increasing.
    TableauRowOrder {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column of the rejected adjacent pair's left entry.
        column: usize,
    },
    /// A tableau column was not strictly increasing.
    TableauColumnOrder {
        /// Zero-based row index of the rejected lower entry.
        row: usize,
        /// Zero-based column index.
        column: usize,
    },
    /// A tableau entry was not in the requested positive alphabet.
    TableauEntryOutsideAlphabet {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        column: usize,
        /// Rejected arbitrary-precision entry.
        entry: String,
        /// Largest admitted positive letter.
        alphabet_size: usize,
    },
    /// An exact tableau-cardinality formula failed its integrality check.
    InvalidTableauCardinality {
        /// Concise invariant failure.
        detail: String,
    },
    /// A symmetric function contained terms in distinct homogeneous degrees.
    MixedSymmetricDegrees {
        /// Degree established by the first nonzero term.
        expected: usize,
        /// Degree of the rejected term.
        actual: usize,
    },
    /// A symmetric-function calculation exceeded the explicit 0.1 degree bound.
    SymmetricDegreeTooLarge {
        /// Requested homogeneous degree.
        degree: usize,
        /// Largest accepted homogeneous degree.
        max: usize,
    },
    /// A symmetric-function basis conversion failed its exact linear-algebra check.
    SymmetricBasisConversion {
        /// Exact failure returned by the linear-algebra layer.
        detail: String,
    },
    /// A string did not name one of the five accepted symmetric-function bases.
    UnknownSymmetricBasis {
        /// Rejected basis label.
        basis: String,
    },
    /// A crystal rank lies outside the explicit finite-type-A 0.1 range.
    InvalidCrystalRank {
        /// Requested Cartan rank.
        rank: usize,
        /// Largest accepted Cartan rank.
        max: usize,
    },
    /// A type-A tableau shape has more rows than available letters.
    CrystalShapeTooTall {
        /// Number of rows in the shape.
        rows: usize,
        /// Size of the type-A letter alphabet.
        alphabet: usize,
    },
    /// A tableau shape has too many cells for the explicit crystal builder.
    CrystalShapeTooLarge {
        /// Number of cells in the shape.
        cells: usize,
        /// Largest accepted number of cells.
        max: usize,
    },
    /// An explicit crystal would exceed the accepted vertex bound.
    CrystalTooLarge {
        /// Requested or computed number of vertices.
        vertices: usize,
        /// Largest accepted number of vertices.
        max: usize,
    },
    /// A crystal allocation size overflowed the platform index type.
    CrystalSizeOverflow,
    /// A vertex identifier is not present in the crystal.
    CrystalVertexOutOfRange {
        /// Rejected vertex identifier.
        vertex: u32,
        /// Number of vertices in the crystal.
        vertices: usize,
    },
    /// A Kashiwara index is not present in the crystal's Cartan context.
    CrystalIndexOutOfRange {
        /// Rejected zero-based Kashiwara index.
        index: usize,
        /// Cartan rank of the crystal.
        rank: usize,
    },
    /// Two crystals have distinct Cartan contexts.
    CrystalContextMismatch,
    /// Stored crystal data violates a finite-crystal invariant.
    InvalidCrystal {
        /// Concise invariant failure.
        detail: String,
    },
}

impl fmt::Display for CombinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPart { index } => {
                write!(f, "part at index {index} is zero; parts must be positive")
            }
            Self::PartitionOrder { index, left, right } => write!(
                f,
                "partition parts must be weakly decreasing; parts[{index}]={left} < parts[{}]={right}",
                index + 1
            ),
            Self::SizeOverflow => f.write_str("sum of parts exceeds usize"),
            Self::PartitionTooLarge { size, max } => write!(
                f,
                "partition size {size} exceeds the explicit 0.1 limit {max}"
            ),
            Self::TableauRowOrder { row, column } => write!(
                f,
                "tableau row {row} is not weakly increasing at columns {column} and {}",
                column + 1
            ),
            Self::TableauColumnOrder { row, column } => write!(
                f,
                "tableau column {column} is not strictly increasing at rows {} and {row}",
                row.saturating_sub(1)
            ),
            Self::TableauEntryOutsideAlphabet {
                row,
                column,
                entry,
                alphabet_size,
            } => write!(
                f,
                "tableau entry {entry} at ({row}, {column}) is outside the alphabet 1..={alphabet_size}"
            ),
            Self::InvalidTableauCardinality { detail } => {
                write!(f, "semistandard-tableau cardinality is invalid: {detail}")
            }
            Self::MixedSymmetricDegrees { expected, actual } => write!(
                f,
                "symmetric function is not homogeneous: expected degree {expected}, got {actual}"
            ),
            Self::SymmetricDegreeTooLarge { degree, max } => write!(
                f,
                "symmetric-function degree {degree} exceeds the explicit 0.1 limit {max}"
            ),
            Self::SymmetricBasisConversion { detail } => {
                write!(f, "exact symmetric-function basis conversion failed: {detail}")
            }
            Self::UnknownSymmetricBasis { basis } => write!(
                f,
                "unknown symmetric-function basis {basis:?}; expected monomial, power_sum, elementary, complete_homogeneous, or schur"
            ),
            Self::InvalidCrystalRank { rank, max } => write!(
                f,
                "type-A crystal rank {rank} is outside the accepted range 1..={max}"
            ),
            Self::CrystalShapeTooTall { rows, alphabet } => write!(
                f,
                "tableau shape has {rows} rows but the type-A alphabet has {alphabet} letters"
            ),
            Self::CrystalShapeTooLarge { cells, max } => write!(
                f,
                "tableau shape has {cells} cells, exceeding the explicit crystal limit {max}"
            ),
            Self::CrystalTooLarge { vertices, max } => write!(
                f,
                "crystal has {vertices} vertices, exceeding the explicit 0.1 limit {max}"
            ),
            Self::CrystalSizeOverflow => {
                f.write_str("crystal allocation size exceeds the platform index type")
            }
            Self::CrystalVertexOutOfRange { vertex, vertices } => write!(
                f,
                "crystal vertex {vertex} is outside 0..{vertices}"
            ),
            Self::CrystalIndexOutOfRange { index, rank } => write!(
                f,
                "Kashiwara index {index} is outside 0..{rank}"
            ),
            Self::CrystalContextMismatch => {
                f.write_str("crystal tensor factors have different Cartan matrices")
            }
            Self::InvalidCrystal { detail } => write!(f, "invalid crystal: {detail}"),
        }
    }
}

impl std::error::Error for CombinError {}

fn checked_size(parts: &[usize]) -> Result<usize, CombinError> {
    parts.iter().try_fold(0_usize, |size, &part| {
        size.checked_add(part).ok_or(CombinError::SizeOverflow)
    })
}

/// An integer partition `lambda_1 >= ... >= lambda_k > 0`.
///
/// Parts are defining content and remain in canonical weakly-decreasing
/// order. The empty partition is the unique partition of zero.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Partition {
    parts: Vec<usize>,
    size: usize,
}

#[derive(Serialize, Deserialize)]
struct PartitionRepr {
    parts: Vec<usize>,
}

impl Partition {
    /// Construct an integer partition from canonical weakly-decreasing parts.
    pub fn new(parts: Vec<usize>) -> Result<Self, CombinError> {
        if let Some(index) = parts.iter().position(|&part| part == 0) {
            return Err(CombinError::ZeroPart { index });
        }
        if let Some((index, pair)) = parts
            .windows(2)
            .enumerate()
            .find(|(_, pair)| pair[0] < pair[1])
        {
            return Err(CombinError::PartitionOrder {
                index,
                left: pair[0],
                right: pair[1],
            });
        }
        let size = checked_size(&parts)?;
        if size > MAX_PARTITION_CELLS {
            return Err(CombinError::PartitionTooLarge {
                size,
                max: MAX_PARTITION_CELLS,
            });
        }
        Ok(Self { parts, size })
    }

    /// Canonical weakly-decreasing parts.
    #[must_use]
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Integer being partitioned.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Transpose the Young diagram.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        let Some(&width) = self.parts.first() else {
            return Self {
                parts: Vec::new(),
                size: 0,
            };
        };
        let mut counts = vec![0_usize; width + 1];
        for &part in &self.parts {
            counts[part] += 1;
        }
        let mut height = 0_usize;
        let mut parts = Vec::with_capacity(width);
        for column in (1..=width).rev() {
            height += counts[column];
            parts.push(height);
        }
        parts.reverse();
        Self {
            parts,
            size: self.size,
        }
    }

    /// Hook lengths in row-major Young-diagram order.
    #[must_use]
    pub fn hook_lengths(&self) -> Vec<usize> {
        let column_heights = self.conjugate();
        let mut hooks = Vec::with_capacity(self.size);
        for (row, &width) in self.parts.iter().enumerate() {
            for column in 0..width {
                hooks.push(width - column + column_heights.parts[column] - row - 1);
            }
        }
        hooks
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Partition({:?})", self.parts)
    }
}

impl Serialize for Partition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PartitionRepr {
            parts: self.parts.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Partition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PartitionRepr::deserialize(deserializer)?;
        Self::new(repr.parts).map_err(serde::de::Error::custom)
    }
}

static PARTITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Partition@1",
    rust_path: "noether::combin::Partition",
    python_path: Some("noether.combin.Partition"),
};

impl SemanticType for Partition {
    const TYPE_ID: &'static str = "noether/combin/Partition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &PARTITION_DESCRIPTION
    }
}

/// An ordered composition `alpha_1 + ... + alpha_k` into positive parts.
///
/// The empty composition is the unique composition of zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Composition {
    parts: Vec<usize>,
    size: usize,
}

#[derive(Serialize, Deserialize)]
struct CompositionRepr {
    parts: Vec<usize>,
}

impl Composition {
    /// Construct a composition from its ordered positive parts.
    pub fn new(parts: Vec<usize>) -> Result<Self, CombinError> {
        if let Some(index) = parts.iter().position(|&part| part == 0) {
            return Err(CombinError::ZeroPart { index });
        }
        let size = checked_size(&parts)?;
        Ok(Self { parts, size })
    }

    /// Ordered positive parts.
    #[must_use]
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Integer being composed.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl fmt::Display for Composition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Composition({:?})", self.parts)
    }
}

impl Serialize for Composition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        CompositionRepr {
            parts: self.parts.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Composition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = CompositionRepr::deserialize(deserializer)?;
        Self::new(repr.parts).map_err(serde::de::Error::custom)
    }
}

static COMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Composition@1",
    rust_path: "noether::combin::Composition",
    python_path: Some("noether.combin.Composition"),
};

impl SemanticType for Composition {
    const TYPE_ID: &'static str = "noether/combin/Composition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &COMPOSITION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_has_exact_conjugate_and_hooks() {
        let value = Partition::new(vec![3, 2]).unwrap();

        assert_eq!(value.size(), 5);
        assert_eq!(value.conjugate().parts(), &[2, 2, 1]);
        assert_eq!(value.hook_lengths(), vec![4, 3, 1, 2, 1]);
        assert_eq!(value.conjugate().conjugate(), value);
    }

    #[test]
    fn empty_partition_and_composition_are_canonical_zero_values() {
        let partition = Partition::new(Vec::new()).unwrap();
        let composition = Composition::new(Vec::new()).unwrap();

        assert_eq!(partition.size(), 0);
        assert_eq!(partition.conjugate(), partition);
        assert!(partition.hook_lengths().is_empty());
        assert_eq!(composition.size(), 0);
    }

    #[test]
    fn constructors_and_deserialization_fail_closed() {
        assert!(matches!(
            Partition::new(vec![2, 3]),
            Err(CombinError::PartitionOrder { .. })
        ));
        assert_eq!(
            Composition::new(vec![2, 0, 1]),
            Err(CombinError::ZeroPart { index: 1 })
        );
        assert!(serde_json::from_str::<Partition>(r#"{"parts":[2,3]}"#).is_err());
    }

    #[test]
    fn semantic_identity_and_json_are_stable() {
        let value = Composition::new(vec![2, 1, 2]).unwrap();
        let json = serde_json::to_string(&value).unwrap();

        assert_eq!(Composition::TYPE_ID, "noether/combin/Composition@1");
        assert_eq!(json, r#"{"parts":[2,1,2]}"#);
        assert_eq!(serde_json::from_str::<Composition>(&json).unwrap(), value);
    }
}
