// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Validated strong-Steiner directed complexes and their finite cells.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::combin::{GraphSymmetryError, Poset, PosetError};
use crate::groups::GroupError;
use crate::groups::Permutation;
use crate::homalg::{
    graded_homology_invariants_from_sparse_boundaries, ChainComplex, FinitelyGeneratedAbelianGroup,
    GradedHomology, HomologyError,
};
use crate::linalg::{LinalgError, Matrix, SparseMatrix};
use crate::{Integer, NoetherError, SemanticType, TypeDescription, Z};

use super::canonical::SteinerCanonicalLabel;
use super::homotopy::{FiniteOmegaCategory, OrientedBasepoint};
use super::nerve::{CategoricalHomologyPresentation, StreetNerveSkeleton};

static DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/category/SteinerCategory@1",
    rust_path: "noether::category::SteinerCategory",
    python_path: Some("noether.category.SteinerCategory"),
};

/// Largest total number of exact nonidentity cells materialized in one call.
pub const MAX_ENUMERATED_CELLS: usize = 1_000_000;

/// An atom in one graded basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AtomId {
    degree: usize,
    index: usize,
}

impl AtomId {
    /// Construct a degree-local atom identifier.
    #[must_use]
    pub const fn new(degree: usize, index: usize) -> Self {
        Self { degree, index }
    }

    /// Homological degree of the atom.
    #[must_use]
    pub const fn degree(self) -> usize {
        self.degree
    }

    /// Index in the basis of its degree.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Disjoint negative and positive supports of one atomic boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedBoundary {
    negative: Vec<usize>,
    positive: Vec<usize>,
}

impl SignedBoundary {
    /// Construct a normalized signed boundary from degree-local atom indices.
    pub fn new(negative: &[usize], positive: &[usize]) -> Result<Self, CategoryError> {
        let mut negative = negative.to_vec();
        let mut positive = positive.to_vec();
        negative.sort_unstable();
        positive.sort_unstable();
        if let Some(&atom) = negative
            .windows(2)
            .find_map(|pair| (pair[0] == pair[1]).then_some(&pair[0]))
        {
            return Err(CategoryError::DuplicateBoundaryAtom {
                side: "negative",
                atom,
            });
        }
        if let Some(&atom) = positive
            .windows(2)
            .find_map(|pair| (pair[0] == pair[1]).then_some(&pair[0]))
        {
            return Err(CategoryError::DuplicateBoundaryAtom {
                side: "positive",
                atom,
            });
        }
        if let Some(&atom) = negative
            .iter()
            .find(|atom| positive.binary_search(atom).is_ok())
        {
            return Err(CategoryError::OverlappingBoundarySupport { atom });
        }
        Ok(Self { negative, positive })
    }

    /// Negative support, in increasing degree-local order.
    #[must_use]
    pub fn negative(&self) -> &[usize] {
        &self.negative
    }

    /// Positive support, in increasing degree-local order.
    #[must_use]
    pub fn positive(&self) -> &[usize] {
        &self.positive
    }
}

/// A failure while constructing or deriving a finite strong Steiner category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoryError {
    /// The boundary table did not have one entry for every occupied degree.
    BoundaryDegreeCount {
        /// Number of atomic degrees.
        expected: usize,
        /// Number of received boundary degrees.
        actual: usize,
    },
    /// Degree zero was assigned boundary columns.
    DegreeZeroHasBoundary,
    /// A positive degree had the wrong number of boundary columns.
    BoundaryColumnCount {
        /// Degree of the malformed table.
        degree: usize,
        /// Number of atoms in that degree.
        expected: usize,
        /// Number of received columns.
        actual: usize,
    },
    /// A boundary support referred outside the preceding basis.
    BoundaryAtomOutOfRange {
        /// Degree of the atom whose boundary was malformed.
        degree: usize,
        /// Degree-local index of that atom.
        atom: usize,
        /// Out-of-range preceding atom.
        boundary_atom: usize,
        /// Size of the preceding basis.
        preceding_count: usize,
    },
    /// One support repeated an atom.
    DuplicateBoundaryAtom {
        /// Stable support name.
        side: &'static str,
        /// Repeated degree-local atom.
        atom: usize,
    },
    /// Positive and negative supports were not disjoint.
    OverlappingBoundarySupport {
        /// Atom occurring with both signs.
        atom: usize,
    },
    /// Consecutive boundaries did not compose to zero.
    BoundarySquaredNonzero {
        /// Degree of the original atom.
        degree: usize,
        /// Degree-local original atom.
        atom: usize,
        /// Degree-local atom detecting the nonzero composite.
        lower_atom: usize,
        /// Nonzero coefficient.
        coefficient: i64,
    },
    /// An iterated signed boundary did not augment to one.
    NotUnital {
        /// Degree of the original atom.
        degree: usize,
        /// Degree-local original atom.
        atom: usize,
        /// Iterated side.
        side: &'static str,
        /// Resulting augmentation.
        augmentation: i64,
    },
    /// The strong precedence relation contained a cycle.
    CyclicPrecedence,
    /// A requested atom did not exist.
    AtomOutOfRange {
        /// Requested degree.
        degree: usize,
        /// Requested degree-local index.
        index: usize,
    },
    /// A requested cell dimension was not present.
    CellDimensionOutOfRange {
        /// Requested dimension.
        dimension: usize,
        /// Largest nonidentity dimension.
        maximum: i64,
    },
    /// A proposed top chain did not define a cell.
    InvalidCellTopChain {
        /// Degree of the proposed cell.
        degree: usize,
        /// Stable explanation.
        reason: &'static str,
    },
    /// A zero-cell was asked for a source or target.
    ObjectHasNoBoundary,
    /// Two cells belonged to different category contexts.
    CellContextMismatch,
    /// A cell operation received incompatible dimensions.
    CellDimensionMismatch {
        /// Left dimension.
        left: usize,
        /// Right dimension.
        right: usize,
    },
    /// Two equal-dimensional cells did not have the required common face.
    CellsNotComposable {
        /// Dimension of the proposed composition.
        along: usize,
    },
    /// Exact cell materialization exceeded its count limit.
    CellEnumerationLimitExceeded {
        /// Accepted maximum.
        maximum: usize,
    },
    /// A named construction exceeded its deterministic representation limit.
    ConstructionTooLarge {
        /// Stable construction name.
        construction: &'static str,
        /// Requested parameter.
        parameter: usize,
    },
    /// An arithmetic size calculation overflowed.
    SizeOverflow,
    /// Exact linear algebra failed.
    LinearAlgebra(LinalgError),
    /// Exact homological algebra failed.
    Homology(HomologyError),
    /// Exact Poset construction failed.
    Poset(PosetError),
    /// Exact permutation-group computation failed.
    Group(GroupError),
    /// Exact coloured-incidence-graph canonicalization failed.
    GraphSymmetry(GraphSymmetryError),
    /// A canonicalization invariant failed internally.
    Canonicalization(&'static str),
    /// A basepoint or mapping-category precondition failed.
    InvalidBasepoint(&'static str),
    /// A bounded Street-nerve calculation failed.
    StreetNerve(&'static str),
}

impl fmt::Display for CategoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundaryDegreeCount { expected, actual } => write!(
                f,
                "Steiner boundary table has {actual} degrees; expected {expected}"
            ),
            Self::DegreeZeroHasBoundary => f.write_str("degree zero cannot have a boundary"),
            Self::BoundaryColumnCount {
                degree,
                expected,
                actual,
            } => write!(
                f,
                "Steiner degree {degree} has {actual} boundary columns; expected {expected}"
            ),
            Self::BoundaryAtomOutOfRange {
                degree,
                atom,
                boundary_atom,
                preceding_count,
            } => write!(
                f,
                "boundary of degree-{degree} atom {atom} uses atom {boundary_atom} outside [0, {preceding_count})"
            ),
            Self::DuplicateBoundaryAtom { side, atom } => {
                write!(f, "{side} boundary support repeats atom {atom}")
            }
            Self::OverlappingBoundarySupport { atom } => write!(
                f,
                "boundary atom {atom} occurs in both negative and positive supports"
            ),
            Self::BoundarySquaredNonzero {
                degree,
                atom,
                lower_atom,
                coefficient,
            } => write!(
                f,
                "boundary squared of degree-{degree} atom {atom} has coefficient {coefficient} at atom {lower_atom}"
            ),
            Self::NotUnital {
                degree,
                atom,
                side,
                augmentation,
            } => write!(
                f,
                "iterated {side} boundary of degree-{degree} atom {atom} has augmentation {augmentation}, not 1"
            ),
            Self::CyclicPrecedence => {
                f.write_str("the strong Steiner precedence relation contains a cycle")
            }
            Self::AtomOutOfRange { degree, index } => {
                write!(f, "atom ({degree}, {index}) is out of range")
            }
            Self::CellDimensionOutOfRange { dimension, maximum } => write!(
                f,
                "cell dimension {dimension} exceeds nonidentity dimension {maximum}"
            ),
            Self::InvalidCellTopChain { degree, reason } => {
                write!(f, "degree-{degree} top chain is not a Steiner cell: {reason}")
            }
            Self::ObjectHasNoBoundary => f.write_str("a zero-cell has no source or target"),
            Self::CellContextMismatch => {
                f.write_str("cells belong to different Steiner categories")
            }
            Self::CellDimensionMismatch { left, right } => {
                write!(f, "cell dimensions {left} and {right} do not agree")
            }
            Self::CellsNotComposable { along } => write!(
                f,
                "cells do not have matching target and source in dimension {along}"
            ),
            Self::CellEnumerationLimitExceeded { maximum } => {
                write!(f, "exact cell enumeration exceeded {maximum} cells")
            }
            Self::ConstructionTooLarge {
                construction,
                parameter,
            } => write!(
                f,
                "{construction} parameter {parameter} exceeds its deterministic limit"
            ),
            Self::SizeOverflow => f.write_str("Steiner category size calculation overflowed"),
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::Homology(error) => error.fmt(f),
            Self::Poset(error) => error.fmt(f),
            Self::Group(error) => error.fmt(f),
            Self::GraphSymmetry(error) => error.fmt(f),
            Self::Canonicalization(message)
            | Self::InvalidBasepoint(message)
            | Self::StreetNerve(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CategoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LinearAlgebra(error) => Some(error),
            Self::Homology(error) => Some(error),
            Self::Poset(error) => Some(error),
            Self::Group(error) => Some(error),
            Self::GraphSymmetry(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinalgError> for CategoryError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<HomologyError> for CategoryError {
    fn from(value: HomologyError) -> Self {
        Self::Homology(value)
    }
}

impl From<PosetError> for CategoryError {
    fn from(value: PosetError) -> Self {
        Self::Poset(value)
    }
}

impl From<GroupError> for CategoryError {
    fn from(value: GroupError) -> Self {
        Self::Group(value)
    }
}

impl From<GraphSymmetryError> for CategoryError {
    fn from(value: GraphSymmetryError) -> Self {
        Self::GraphSymmetry(value)
    }
}

/// Context-free double-sequence coordinates of one cell in `nu(K)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CellShape {
    negative: Vec<Vec<usize>>,
    positive: Vec<Vec<usize>>,
}

impl CellShape {
    fn dimension(&self) -> usize {
        self.negative.len() - 1
    }

    fn intrinsic_dimension(&self) -> usize {
        self.negative
            .iter()
            .rposition(|support| !support.is_empty())
            .unwrap_or(0)
    }

    fn identity(&self) -> Self {
        let mut result = self.clone();
        result.negative.push(Vec::new());
        result.positive.push(Vec::new());
        result
    }

    fn boundary_at(&self, dimension: usize, positive: bool) -> Option<Self> {
        if dimension >= self.dimension() {
            return None;
        }
        let mut negative = self.negative[..=dimension].to_vec();
        let mut positive_supports = self.positive[..=dimension].to_vec();
        let support = if positive {
            self.positive[dimension].clone()
        } else {
            self.negative[dimension].clone()
        };
        negative[dimension] = support.clone();
        positive_supports[dimension] = support;
        Some(Self {
            negative,
            positive: positive_supports,
        })
    }
}

#[derive(Debug)]
struct SteinerData {
    atom_counts: Vec<usize>,
    boundaries: Vec<Vec<SignedBoundary>>,
    cells: OnceLock<Vec<Vec<CellShape>>>,
}

impl PartialEq for SteinerData {
    fn eq(&self, other: &Self) -> bool {
        self.atom_counts == other.atom_counts && self.boundaries == other.boundaries
    }
}

impl Eq for SteinerData {}

/// A finite strong Steiner strict omega-category.
#[derive(Clone, Debug)]
pub struct SteinerCategory {
    data: Arc<SteinerData>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    #[serde(rename = "type")]
    type_id: String,
    atom_counts: Vec<usize>,
    boundaries: Vec<Vec<SignedBoundary>>,
}

impl SteinerCategory {
    /// Construct and fully validate a based augmented directed complex.
    ///
    /// `boundaries[q][i]` is the signed boundary of degree-`q` atom `i`.
    /// Degree zero must have an empty boundary list. The empty vectors encode
    /// the empty category.
    pub fn from_boundaries(
        mut atom_counts: Vec<usize>,
        mut boundaries: Vec<Vec<SignedBoundary>>,
    ) -> Result<Self, CategoryError> {
        if boundaries.len() != atom_counts.len() {
            return Err(CategoryError::BoundaryDegreeCount {
                expected: atom_counts.len(),
                actual: boundaries.len(),
            });
        }
        if boundaries
            .first()
            .is_some_and(|columns| !columns.is_empty())
        {
            return Err(CategoryError::DegreeZeroHasBoundary);
        }
        while atom_counts.last() == Some(&0) {
            if boundaries.last().is_some_and(|columns| !columns.is_empty()) {
                return Err(CategoryError::BoundaryColumnCount {
                    degree: atom_counts.len() - 1,
                    expected: 0,
                    actual: boundaries.last().map_or(0, Vec::len),
                });
            }
            atom_counts.pop();
            boundaries.pop();
        }
        for degree in 1..atom_counts.len() {
            if boundaries[degree].len() != atom_counts[degree] {
                return Err(CategoryError::BoundaryColumnCount {
                    degree,
                    expected: atom_counts[degree],
                    actual: boundaries[degree].len(),
                });
            }
            for (atom, boundary) in boundaries[degree].iter().enumerate() {
                if !boundary.negative.windows(2).all(|pair| pair[0] < pair[1]) {
                    return Err(CategoryError::DuplicateBoundaryAtom {
                        side: "negative",
                        atom: boundary
                            .negative
                            .windows(2)
                            .find(|pair| pair[0] >= pair[1])
                            .map_or(0, |pair| pair[1]),
                    });
                }
                if !boundary.positive.windows(2).all(|pair| pair[0] < pair[1]) {
                    return Err(CategoryError::DuplicateBoundaryAtom {
                        side: "positive",
                        atom: boundary
                            .positive
                            .windows(2)
                            .find(|pair| pair[0] >= pair[1])
                            .map_or(0, |pair| pair[1]),
                    });
                }
                if let Some(&overlap) = boundary
                    .negative
                    .iter()
                    .find(|value| boundary.positive.binary_search(value).is_ok())
                {
                    return Err(CategoryError::OverlappingBoundarySupport { atom: overlap });
                }
                for &boundary_atom in boundary.negative.iter().chain(&boundary.positive) {
                    if boundary_atom >= atom_counts[degree - 1] {
                        return Err(CategoryError::BoundaryAtomOutOfRange {
                            degree,
                            atom,
                            boundary_atom,
                            preceding_count: atom_counts[degree - 1],
                        });
                    }
                }
            }
        }

        let category = Self {
            data: Arc::new(SteinerData {
                atom_counts,
                boundaries,
                cells: OnceLock::new(),
            }),
        };
        category.verify_complex()?;
        category.verify_unital()?;
        category.verify_strong_loop_free()?;
        Ok(category)
    }

    /// The empty strict omega-category.
    pub fn empty() -> Self {
        Self::from_boundaries(Vec::new(), Vec::new())
            .expect("the empty directed complex is strong Steiner")
    }

    /// The terminal zero-category.
    pub fn point() -> Self {
        Self::from_boundaries(vec![1], vec![Vec::new()])
            .expect("one augmented zero-atom is strong Steiner")
    }

    /// The walking `dimension`-cell.
    pub fn globe(dimension: usize) -> Result<Self, CategoryError> {
        if dimension > 1_000 {
            return Err(CategoryError::ConstructionTooLarge {
                construction: "globe",
                parameter: dimension,
            });
        }
        if dimension == 0 {
            return Ok(Self::point());
        }
        let mut atom_counts = vec![2; dimension];
        atom_counts.push(1);
        let mut boundaries = vec![Vec::new(); dimension + 1];
        for degree in 1..=dimension {
            let column = SignedBoundary::new(&[0], &[1])?;
            boundaries[degree] = vec![column; atom_counts[degree]];
        }
        Self::from_boundaries(atom_counts, boundaries)
    }

    /// Street's oriented `dimension`-simplex.
    pub fn oriental(dimension: usize) -> Result<Self, CategoryError> {
        if dimension > 15 {
            return Err(CategoryError::ConstructionTooLarge {
                construction: "oriental",
                parameter: dimension,
            });
        }
        let vertices = dimension + 1;
        let mut faces = Vec::with_capacity(vertices);
        let mut face_indices = Vec::with_capacity(vertices);
        for degree in 0..=dimension {
            let degree_faces = combinations(vertices, degree + 1);
            let indices = degree_faces
                .iter()
                .enumerate()
                .map(|(index, face)| (face.clone(), index))
                .collect::<BTreeMap<_, _>>();
            faces.push(degree_faces);
            face_indices.push(indices);
        }
        let atom_counts = faces.iter().map(Vec::len).collect::<Vec<_>>();
        let mut boundaries = vec![Vec::new(); dimension + 1];
        for degree in 1..=dimension {
            for face in &faces[degree] {
                let mut negative = Vec::new();
                let mut positive = Vec::new();
                for removed in 0..face.len() {
                    let mut facet = face.clone();
                    facet.remove(removed);
                    let index = face_indices[degree - 1][&facet];
                    if removed % 2 == 0 {
                        positive.push(index);
                    } else {
                        negative.push(index);
                    }
                }
                boundaries[degree].push(SignedBoundary::new(&negative, &positive)?);
            }
        }
        Self::from_boundaries(atom_counts, boundaries)
    }

    /// The directed `dimension`-cube, formed by repeated Gray tensor product.
    pub fn cube(dimension: usize) -> Result<Self, CategoryError> {
        if dimension > 20 {
            return Err(CategoryError::ConstructionTooLarge {
                construction: "cube",
                parameter: dimension,
            });
        }
        let arrow = Self::globe(1)?;
        let mut result = Self::point();
        for _ in 0..dimension {
            result = result.gray_tensor(&arrow)?;
        }
        Ok(result)
    }

    /// Coproduct (disjoint union) of two finite strong Steiner categories.
    pub fn disjoint_union(&self, other: &Self) -> Result<Self, CategoryError> {
        if self.atom_count() == 0 {
            return Ok(other.clone());
        }
        if other.atom_count() == 0 {
            return Ok(self.clone());
        }
        let degree_count = self
            .data
            .atom_counts
            .len()
            .max(other.data.atom_counts.len());
        let mut atom_counts = Vec::with_capacity(degree_count);
        let mut boundaries = vec![Vec::new(); degree_count];
        for (degree, degree_boundaries) in boundaries.iter_mut().enumerate() {
            let left_count = self.atom_count_in_degree(degree);
            let right_count = other.atom_count_in_degree(degree);
            atom_counts.push(
                left_count
                    .checked_add(right_count)
                    .ok_or(CategoryError::SizeOverflow)?,
            );
            if degree == 0 {
                continue;
            }
            if degree < self.data.boundaries.len() {
                degree_boundaries.extend(self.data.boundaries[degree].iter().cloned());
            }
            if degree < other.data.boundaries.len() {
                for boundary in &other.data.boundaries[degree] {
                    let negative = boundary
                        .negative
                        .iter()
                        .map(|&atom| atom + self.atom_count_in_degree(degree - 1))
                        .collect::<Vec<_>>();
                    let positive = boundary
                        .positive
                        .iter()
                        .map(|&atom| atom + self.atom_count_in_degree(degree - 1))
                        .collect::<Vec<_>>();
                    degree_boundaries.push(SignedBoundary::new(&negative, &positive)?);
                }
            }
        }
        Self::from_boundaries(atom_counts, boundaries)
    }

    /// Largest degree containing an atom, or `-1` for the empty category.
    #[must_use]
    pub fn dimension(&self) -> i64 {
        self.data
            .atom_counts
            .iter()
            .rposition(|&count| count != 0)
            .map_or(-1, |degree| degree as i64)
    }

    /// Total number of atoms.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.data.atom_counts.iter().sum()
    }

    /// Numbers of atoms in each stored degree.
    #[must_use]
    pub fn atom_counts(&self) -> &[usize] {
        &self.data.atom_counts
    }

    /// Signed boundary of one atom.
    pub fn boundary(&self, atom: AtomId) -> Result<&SignedBoundary, CategoryError> {
        if atom.index >= self.atom_count_in_degree(atom.degree) {
            return Err(CategoryError::AtomOutOfRange {
                degree: atom.degree,
                index: atom.index,
            });
        }
        if atom.degree == 0 {
            return Err(CategoryError::ObjectHasNoBoundary);
        }
        Ok(&self.data.boundaries[atom.degree][atom.index])
    }

    /// Number of atoms in one degree; zero beyond the stored complex.
    #[must_use]
    pub fn atom_count_in_degree(&self, degree: usize) -> usize {
        self.data.atom_counts.get(degree).copied().unwrap_or(0)
    }

    /// Recheck `d^2 = 0`, unitality, and strong loop-freeness.
    pub fn verify(&self) -> Result<(), CategoryError> {
        self.verify_complex()?;
        self.verify_unital()?;
        self.verify_strong_loop_free()
    }

    /// The strong precedence Poset on the flattened atomic basis.
    pub fn precedence_poset(&self) -> Result<Poset, CategoryError> {
        let offsets = degree_offsets(&self.data.atom_counts)?;
        let mut relations = Vec::new();
        for degree in 1..self.data.boundaries.len() {
            for (atom, boundary) in self.data.boundaries[degree].iter().enumerate() {
                let upper = offsets[degree] + atom;
                relations.extend(
                    boundary
                        .negative
                        .iter()
                        .map(|&lower| (offsets[degree - 1] + lower, upper)),
                );
                relations.extend(
                    boundary
                        .positive
                        .iter()
                        .map(|&lower| (upper, offsets[degree - 1] + lower)),
                );
            }
        }
        let relations = relations
            .into_iter()
            .map(|(lower, upper)| {
                Ok((
                    u32::try_from(lower).map_err(|_| CategoryError::SizeOverflow)?,
                    u32::try_from(upper).map_err(|_| CategoryError::SizeOverflow)?,
                ))
            })
            .collect::<Result<Vec<_>, CategoryError>>()?;
        Ok(Poset::new(self.atom_count(), &relations)?)
    }

    /// Return one atomic cell.
    pub fn atom(&self, atom: AtomId) -> Result<SteinerCell, CategoryError> {
        if atom.index >= self.atom_count_in_degree(atom.degree) {
            return Err(CategoryError::AtomOutOfRange {
                degree: atom.degree,
                index: atom.index,
            });
        }
        self.cell(atom.degree, &[atom.index])
    }

    /// Construct the unwhiskered cell generated by a nonempty top atomic chain.
    pub fn cell(
        &self,
        dimension: usize,
        top_atoms: &[usize],
    ) -> Result<SteinerCell, CategoryError> {
        let mut top_atoms = top_atoms.to_vec();
        top_atoms.sort_unstable();
        if top_atoms.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(CategoryError::InvalidCellTopChain {
                degree: dimension,
                reason: "top support repeats an atom",
            });
        }
        if top_atoms
            .iter()
            .any(|&atom| atom >= self.atom_count_in_degree(dimension))
        {
            return Err(CategoryError::InvalidCellTopChain {
                degree: dimension,
                reason: "top support contains an out-of-range atom",
            });
        }
        let shape = self.minimal_cell_shape(dimension, &top_atoms)?;
        Ok(SteinerCell::from_shape(self.clone(), shape))
    }

    /// Reconstruct a cell from its complete Steiner double sequence.
    ///
    /// Unlike [`Self::cell`], this retains common lower-dimensional support
    /// and therefore represents whiskered cells faithfully. Each support must
    /// be a sorted degree-local subset. The two sequences are checked against
    /// all boundary, augmentation, and top-chain equations before the cell is
    /// returned.
    pub fn cell_from_supports(
        &self,
        negative_supports: Vec<Vec<usize>>,
        positive_supports: Vec<Vec<usize>>,
    ) -> Result<SteinerCell, CategoryError> {
        let shape = CellShape {
            negative: negative_supports,
            positive: positive_supports,
        };
        self.validate_cell_shape(&shape)?;
        Ok(SteinerCell::from_shape(self.clone(), shape))
    }

    /// Enumerate exact-dimensional nonidentity cells in one degree.
    pub fn exact_cells(&self, dimension: usize) -> Result<Vec<SteinerCell>, CategoryError> {
        if self.dimension() < dimension as i64 {
            return Ok(Vec::new());
        }
        Ok(self.all_cell_shapes()?[dimension]
            .iter()
            .filter(|shape| !shape.negative[dimension].is_empty())
            .cloned()
            .map(|shape| SteinerCell::from_shape(self.clone(), shape))
            .collect())
    }

    /// Enumerate all cells at one dimension, including iterated identities.
    pub fn cells(&self, dimension: usize) -> Result<Vec<SteinerCell>, CategoryError> {
        if self.dimension() < 0 {
            return Ok(Vec::new());
        }
        let maximum = self.dimension() as usize;
        let shapes = if dimension <= maximum {
            self.all_cell_shapes()?[dimension].clone()
        } else {
            let mut shapes = self.all_cell_shapes()?[maximum].clone();
            for _ in maximum..dimension {
                shapes = shapes.into_iter().map(|shape| shape.identity()).collect();
            }
            shapes
        };
        Ok(shapes
            .iter()
            .cloned()
            .map(|shape| SteinerCell::from_shape(self.clone(), shape))
            .collect())
    }

    /// Exact-dimensional nonidentity cell counts.
    pub fn exact_cell_counts(&self) -> Result<Vec<usize>, CategoryError> {
        Ok(self
            .all_cell_shapes()?
            .iter()
            .enumerate()
            .map(|(degree, cells)| {
                cells
                    .iter()
                    .filter(|cell| !cell.negative[degree].is_empty())
                    .count()
            })
            .collect())
    }

    /// Mapping omega-category between two objects.
    pub fn mapping_category(
        &self,
        source: &SteinerCell,
        target: &SteinerCell,
    ) -> Result<FiniteOmegaCategory, CategoryError> {
        FiniteOmegaCategory::new(self.clone(), vec![(source.clone(), target.clone())])
    }

    /// Gepner--Heine Poset of components.
    ///
    /// Its elements are the objects of this strong Steiner category, ordered
    /// by the existence of a 1-cell.
    pub fn component_poset(&self) -> Result<Poset, CategoryError> {
        super::homotopy::object_component_poset(self)
    }

    /// Gepner--Heine fundamental Poset at an ordered pair of objects.
    ///
    /// This uses the strong-Steiner description by directed atomic paths and
    /// atomic 2-cell relations, avoiding the generic iterated-boundary filter.
    pub fn fundamental_poset(
        &self,
        source: &SteinerCell,
        target: &SteinerCell,
    ) -> Result<Poset, CategoryError> {
        super::homotopy::fundamental_poset(self, source, target)
    }

    /// Enumerate monomorphic oriented basepoints of one dimension.
    pub fn monomorphic_oriented_basepoints(
        &self,
        dimension: usize,
    ) -> Result<Vec<OrientedBasepoint>, CategoryError> {
        OrientedBasepoint::enumerate_monomorphic(self, dimension)
    }

    /// Compute the Gepner--Heine homotopy Poset at a valid basepoint.
    pub fn homotopy_poset(&self, basepoint: &OrientedBasepoint) -> Result<Poset, CategoryError> {
        basepoint.homotopy_poset(self)
    }

    /// Strict truncation obtained by forgetting all atoms above `dimension`.
    pub fn truncate(&self, dimension: usize) -> Result<Self, CategoryError> {
        let keep = self.data.atom_counts.len().min(
            dimension
                .checked_add(1)
                .ok_or(CategoryError::SizeOverflow)?,
        );
        Self::from_boundaries(
            self.data.atom_counts[..keep].to_vec(),
            self.data.boundaries[..keep].to_vec(),
        )
    }

    /// Reverse the selected globular dimensions.
    ///
    /// A selected positive degree negates that differential, equivalently
    /// interchanging the negative and positive atomic boundary supports. The
    /// entries must be distinct positive dimensions present in the category.
    /// The result is returned only when this same atomic basis remains strong
    /// and loop-free; a strict categorical dual can exist outside that
    /// presentation class.
    pub fn dual(&self, dimensions: &[usize]) -> Result<Self, CategoryError> {
        let mut dimensions = dimensions.to_vec();
        dimensions.sort_unstable();
        if dimensions.first() == Some(&0)
            || dimensions.windows(2).any(|pair| pair[0] == pair[1])
            || dimensions
                .last()
                .is_some_and(|&degree| degree >= self.data.atom_counts.len())
        {
            return Err(CategoryError::InvalidBasepoint(
                "duality dimensions must be distinct positive dimensions present in the category",
            ));
        }
        let mut boundaries = self.data.boundaries.clone();
        for degree in dimensions {
            for boundary in &mut boundaries[degree] {
                core::mem::swap(&mut boundary.negative, &mut boundary.positive);
            }
        }
        Self::from_boundaries(self.data.atom_counts.clone(), boundaries)
    }

    /// Exact canonical labeling and complete atomic automorphism group.
    pub fn canonical_label(&self) -> Result<SteinerCanonicalLabel, CategoryError> {
        SteinerCanonicalLabel::compute(self)
    }

    /// Canonical representative under degree-preserving atom permutations.
    pub fn canonical_form(&self) -> Result<Self, CategoryError> {
        Ok(self.canonical_label()?.category().clone())
    }

    /// Whether two categories are strictly isomorphic.
    pub fn is_isomorphic(&self, other: &Self) -> Result<bool, CategoryError> {
        if self.atom_counts() != other.atom_counts() {
            return Ok(false);
        }
        Ok(self.canonical_form()? == other.canonical_form()?)
    }

    /// Complete automorphism group acting on flattened atomic ids.
    pub fn automorphism_group(&self) -> Result<crate::groups::PermutationGroup, CategoryError> {
        Ok(self.canonical_label()?.automorphism_group().clone())
    }

    /// Underlying integral atomic chain complex.
    pub fn underlying_chain_complex(
        &self,
    ) -> Result<ChainComplex<crate::IntegerRing>, CategoryError> {
        let mut boundaries = BTreeMap::new();
        if self.data.atom_counts.len() <= 1 {
            boundaries.insert(1, Matrix::zeros(Z, self.atom_count_in_degree(0), 0));
        } else {
            for degree in 1..self.data.atom_counts.len() {
                let rows = self.boundary_matrix_i64(degree);
                let rows = rows
                    .into_iter()
                    .map(|row| row.into_iter().map(Integer::from).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                boundaries.insert(
                    i32::try_from(degree).map_err(|_| CategoryError::SizeOverflow)?,
                    Matrix::from_rows(Z, rows)?,
                );
            }
        }
        Ok(ChainComplex::from_boundaries(Z, boundaries)?)
    }

    /// Integral polygraphic homology with source-retaining witnesses.
    pub fn polygraphic_homology(
        &self,
    ) -> Result<GradedHomology<crate::IntegerRing>, CategoryError> {
        Ok(self.underlying_chain_complex()?.graded_homology()?)
    }

    /// Lean integral polygraphic homology invariants by degree.
    pub fn polygraphic_homology_invariants(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, CategoryError> {
        let mut boundaries = BTreeMap::new();
        if self.data.atom_counts.len() <= 1 {
            boundaries.insert(1, SparseMatrix::zeros(Z, self.atom_count_in_degree(0), 0));
        } else {
            for degree in 1..self.data.atom_counts.len() {
                let rows = self.atom_count_in_degree(degree - 1);
                let columns = self.atom_count_in_degree(degree);
                let entries = self.data.boundaries[degree].iter().enumerate().flat_map(
                    |(column, boundary)| {
                        boundary
                            .negative
                            .iter()
                            .map(move |&row| (row, column, Integer::from(-1)))
                            .chain(
                                boundary
                                    .positive
                                    .iter()
                                    .map(move |&row| (row, column, Integer::one())),
                            )
                    },
                );
                boundaries.insert(
                    i32::try_from(degree).map_err(|_| CategoryError::SizeOverflow)?,
                    SparseMatrix::from_triplets(Z, rows, columns, entries)?,
                );
            }
        }
        Ok(graded_homology_invariants_from_sparse_boundaries(
            &boundaries,
        )?)
    }

    /// Non-symmetric Gray tensor product.
    pub fn gray_tensor(&self, other: &Self) -> Result<Self, CategoryError> {
        if self.atom_count() == 0 || other.atom_count() == 0 {
            return Ok(Self::empty());
        }
        let left_dimension = self.dimension() as usize;
        let right_dimension = other.dimension() as usize;
        let dimension = left_dimension
            .checked_add(right_dimension)
            .ok_or(CategoryError::SizeOverflow)?;
        let mut atom_counts = vec![0usize; dimension + 1];
        let mut pair_indices = vec![BTreeMap::new(); dimension + 1];
        for total in 0..=dimension {
            for left_degree in 0..=left_dimension.min(total) {
                let right_degree = total - left_degree;
                if right_degree > right_dimension {
                    continue;
                }
                for left_atom in 0..self.atom_count_in_degree(left_degree) {
                    for right_atom in 0..other.atom_count_in_degree(right_degree) {
                        let index = atom_counts[total];
                        atom_counts[total] = atom_counts[total]
                            .checked_add(1)
                            .ok_or(CategoryError::SizeOverflow)?;
                        pair_indices[total]
                            .insert((left_degree, left_atom, right_degree, right_atom), index);
                    }
                }
            }
        }

        let mut boundaries = vec![Vec::new(); dimension + 1];
        for total in 1..=dimension {
            for left_degree in 0..=left_dimension.min(total) {
                let right_degree = total - left_degree;
                if right_degree > right_dimension {
                    continue;
                }
                for left_atom in 0..self.atom_count_in_degree(left_degree) {
                    for right_atom in 0..other.atom_count_in_degree(right_degree) {
                        let mut negative = Vec::new();
                        let mut positive = Vec::new();
                        if left_degree > 0 {
                            let boundary = &self.data.boundaries[left_degree][left_atom];
                            for &atom in &boundary.negative {
                                negative.push(
                                    pair_indices[total - 1]
                                        [&(left_degree - 1, atom, right_degree, right_atom)],
                                );
                            }
                            for &atom in &boundary.positive {
                                positive.push(
                                    pair_indices[total - 1]
                                        [&(left_degree - 1, atom, right_degree, right_atom)],
                                );
                            }
                        }
                        if right_degree > 0 {
                            let boundary = &other.data.boundaries[right_degree][right_atom];
                            let (right_negative, right_positive) = if left_degree % 2 == 0 {
                                (&boundary.negative, &boundary.positive)
                            } else {
                                (&boundary.positive, &boundary.negative)
                            };
                            for &atom in right_negative {
                                negative.push(
                                    pair_indices[total - 1]
                                        [&(left_degree, left_atom, right_degree - 1, atom)],
                                );
                            }
                            for &atom in right_positive {
                                positive.push(
                                    pair_indices[total - 1]
                                        [&(left_degree, left_atom, right_degree - 1, atom)],
                                );
                            }
                        }
                        boundaries[total].push(SignedBoundary::new(&negative, &positive)?);
                    }
                }
            }
        }
        Self::from_boundaries(atom_counts, boundaries)
    }

    /// Exact Street-nerve skeleton through `max_simplicial_degree`.
    pub fn street_nerve_skeleton(
        &self,
        max_simplicial_degree: usize,
    ) -> Result<StreetNerveSkeleton, CategoryError> {
        StreetNerveSkeleton::compute(self, max_simplicial_degree)
    }

    /// Exact lazy presentation of categorical homology with coefficients in N.
    #[must_use]
    pub fn categorical_homology(&self) -> CategoricalHomologyPresentation {
        CategoricalHomologyPresentation::new(self.clone(), None)
    }

    /// Reduced categorical homology pointed at one object.
    pub fn reduced_categorical_homology(
        &self,
        object: &SteinerCell,
    ) -> Result<CategoricalHomologyPresentation, CategoryError> {
        if object.category != *self || object.dimension() != 0 {
            return Err(CategoryError::InvalidBasepoint(
                "categorical homology requires an object of the source category",
            ));
        }
        Ok(CategoricalHomologyPresentation::new(
            self.clone(),
            Some(object.clone()),
        ))
    }

    pub(crate) fn relabel(&self, local_images: &[Vec<usize>]) -> Result<Self, CategoryError> {
        if local_images.len() != self.data.atom_counts.len()
            || local_images
                .iter()
                .enumerate()
                .any(|(degree, images)| images.len() != self.atom_count_in_degree(degree))
        {
            return Err(CategoryError::Canonicalization(
                "degreewise relabeling has the wrong shape",
            ));
        }
        let mut boundaries = vec![Vec::new(); self.data.atom_counts.len()];
        for degree in 1..self.data.atom_counts.len() {
            boundaries[degree] =
                vec![SignedBoundary::new(&[], &[])?; self.atom_count_in_degree(degree)];
            for old_atom in 0..self.atom_count_in_degree(degree) {
                let new_atom = local_images[degree][old_atom];
                let old = &self.data.boundaries[degree][old_atom];
                let negative = old
                    .negative
                    .iter()
                    .map(|&atom| local_images[degree - 1][atom])
                    .collect::<Vec<_>>();
                let positive = old
                    .positive
                    .iter()
                    .map(|&atom| local_images[degree - 1][atom])
                    .collect::<Vec<_>>();
                boundaries[degree][new_atom] = SignedBoundary::new(&negative, &positive)?;
            }
        }
        Self::from_boundaries(self.data.atom_counts.clone(), boundaries)
    }

    pub(crate) fn unflatten_atom(&self, flattened: usize) -> Result<AtomId, CategoryError> {
        if flattened >= self.atom_count() {
            return Err(CategoryError::AtomOutOfRange {
                degree: self.data.atom_counts.len(),
                index: flattened,
            });
        }
        let offsets = degree_offsets(&self.data.atom_counts)?;
        for (degree, (&offset, &count)) in offsets.iter().zip(&self.data.atom_counts).enumerate() {
            if flattened < offset + count {
                return Ok(AtomId::new(degree, flattened - offset));
            }
        }
        Err(CategoryError::SizeOverflow)
    }

    fn verify_complex(&self) -> Result<(), CategoryError> {
        for degree in 2..self.data.atom_counts.len() {
            let lower = self.boundary_matrix_i64(degree - 1);
            let upper = self.boundary_matrix_i64(degree);
            let upper_columns = (0..self.atom_count_in_degree(degree))
                .map(|atom| upper.iter().map(|row| row[atom]).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for (atom, upper_column) in upper_columns.iter().enumerate() {
                for (lower_atom, lower_row) in lower.iter().enumerate() {
                    let coefficient = lower_row
                        .iter()
                        .zip(upper_column)
                        .map(|(left, right)| left * right)
                        .sum::<i64>();
                    if coefficient != 0 {
                        return Err(CategoryError::BoundarySquaredNonzero {
                            degree,
                            atom,
                            lower_atom,
                            coefficient,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn verify_unital(&self) -> Result<(), CategoryError> {
        for degree in 0..self.data.atom_counts.len() {
            for atom in 0..self.atom_count_in_degree(degree) {
                for (side, positive) in [("negative", false), ("positive", true)] {
                    let mut chain = vec![0i64; self.atom_count_in_degree(degree)];
                    chain[atom] = 1;
                    for current_degree in (1..=degree).rev() {
                        let boundary = self.apply_boundary(current_degree, &chain);
                        chain = boundary
                            .into_iter()
                            .map(|coefficient| {
                                if positive {
                                    coefficient.max(0)
                                } else {
                                    (-coefficient).max(0)
                                }
                            })
                            .collect();
                    }
                    let augmentation = chain.iter().sum::<i64>();
                    if augmentation != 1 {
                        return Err(CategoryError::NotUnital {
                            degree,
                            atom,
                            side,
                            augmentation,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn verify_strong_loop_free(&self) -> Result<(), CategoryError> {
        let offsets = degree_offsets(&self.data.atom_counts)?;
        let mut outgoing = vec![Vec::new(); self.atom_count()];
        let mut indegree = vec![0usize; self.atom_count()];
        for degree in 1..self.data.boundaries.len() {
            for (atom, boundary) in self.data.boundaries[degree].iter().enumerate() {
                let higher = offsets[degree] + atom;
                for &lower in &boundary.negative {
                    let lower = offsets[degree - 1] + lower;
                    outgoing[lower].push(higher);
                    indegree[higher] += 1;
                }
                for &lower in &boundary.positive {
                    let lower = offsets[degree - 1] + lower;
                    outgoing[higher].push(lower);
                    indegree[lower] += 1;
                }
            }
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(atom, &count)| (count == 0).then_some(atom))
            .collect::<VecDeque<_>>();
        let mut visited = 0;
        while let Some(atom) = ready.pop_front() {
            visited += 1;
            for &next in &outgoing[atom] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    ready.push_back(next);
                }
            }
        }
        if visited != self.atom_count() {
            return Err(CategoryError::CyclicPrecedence);
        }
        Ok(())
    }

    fn boundary_matrix_i64(&self, degree: usize) -> Vec<Vec<i64>> {
        let mut matrix = vec![
            vec![0i64; self.atom_count_in_degree(degree)];
            self.atom_count_in_degree(degree - 1)
        ];
        for (atom, boundary) in self.data.boundaries[degree].iter().enumerate() {
            for &lower in &boundary.negative {
                matrix[lower][atom] = -1;
            }
            for &lower in &boundary.positive {
                matrix[lower][atom] = 1;
            }
        }
        matrix
    }

    fn apply_boundary(&self, degree: usize, chain: &[i64]) -> Vec<i64> {
        let mut result = vec![0i64; self.atom_count_in_degree(degree - 1)];
        for (atom, &coefficient) in chain.iter().enumerate() {
            if coefficient == 0 {
                continue;
            }
            for &lower in &self.data.boundaries[degree][atom].negative {
                result[lower] -= coefficient;
            }
            for &lower in &self.data.boundaries[degree][atom].positive {
                result[lower] += coefficient;
            }
        }
        result
    }

    pub(crate) fn boundary_of_top(
        &self,
        degree: usize,
        top_atoms: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), CategoryError> {
        if degree == 0 {
            return Err(CategoryError::ObjectHasNoBoundary);
        }
        let mut chain = vec![0i64; self.atom_count_in_degree(degree)];
        for &atom in top_atoms {
            if atom >= chain.len() {
                return Err(CategoryError::InvalidCellTopChain {
                    degree,
                    reason: "top support contains an out-of-range atom",
                });
            }
            chain[atom] = 1;
        }
        let boundary = self.apply_boundary(degree, &chain);
        if boundary
            .iter()
            .any(|coefficient| !(-1..=1).contains(coefficient))
        {
            return Err(CategoryError::InvalidCellTopChain {
                degree,
                reason: "boundary coefficient is not -1, 0, or 1",
            });
        }
        let negative = boundary
            .iter()
            .enumerate()
            .filter_map(|(atom, &coefficient)| (coefficient == -1).then_some(atom))
            .collect();
        let positive = boundary
            .iter()
            .enumerate()
            .filter_map(|(atom, &coefficient)| (coefficient == 1).then_some(atom))
            .collect();
        Ok((negative, positive))
    }

    fn minimal_cell_shape(
        &self,
        degree: usize,
        top_atoms: &[usize],
    ) -> Result<CellShape, CategoryError> {
        if top_atoms.is_empty() {
            return Err(CategoryError::InvalidCellTopChain {
                degree,
                reason: "top support is empty",
            });
        }
        let mut negative = vec![Vec::new(); degree + 1];
        let mut positive = vec![Vec::new(); degree + 1];
        negative[degree] = top_atoms.to_vec();
        positive[degree] = top_atoms.to_vec();
        for current in (1..=degree).rev() {
            let from_negative = self.boundary_of_top(current, &negative[current])?;
            let from_positive = self.boundary_of_top(current, &positive[current])?;
            if from_negative != from_positive {
                return Err(CategoryError::InvalidCellTopChain {
                    degree,
                    reason: "positive and negative iterated boundaries disagree",
                });
            }
            negative[current - 1] = from_negative.0;
            positive[current - 1] = from_negative.1;
        }
        let shape = CellShape { negative, positive };
        self.validate_cell_shape(&shape)?;
        Ok(shape)
    }

    fn validate_cell_shape(&self, shape: &CellShape) -> Result<(), CategoryError> {
        if shape.negative.is_empty() || shape.negative.len() != shape.positive.len() {
            return Err(CategoryError::InvalidCellTopChain {
                degree: shape.negative.len().saturating_sub(1),
                reason: "cell double sequence has the wrong shape",
            });
        }
        for degree in 0..shape.negative.len() {
            for support in [&shape.negative[degree], &shape.positive[degree]] {
                if support.windows(2).any(|pair| pair[0] >= pair[1])
                    || support
                        .iter()
                        .any(|&atom| atom >= self.atom_count_in_degree(degree))
                {
                    return Err(CategoryError::InvalidCellTopChain {
                        degree,
                        reason: "cell support is not a sorted subset of the atomic basis",
                    });
                }
            }
        }
        let dimension = shape.dimension();
        if shape.negative[dimension] != shape.positive[dimension] {
            return Err(CategoryError::InvalidCellTopChain {
                degree: dimension,
                reason: "the two top chains of a cell do not agree",
            });
        }
        if shape.negative[0].len() != 1 || shape.positive[0].len() != 1 {
            return Err(CategoryError::InvalidCellTopChain {
                degree: dimension,
                reason: "the two zero-chains of a cell must each augment to one",
            });
        }
        for degree in 1..=dimension {
            let expected = (0..self.atom_count_in_degree(degree - 1))
                .map(|atom| {
                    i64::from(shape.positive[degree - 1].binary_search(&atom).is_ok())
                        - i64::from(shape.negative[degree - 1].binary_search(&atom).is_ok())
                })
                .collect::<Vec<_>>();
            for support in [&shape.negative[degree], &shape.positive[degree]] {
                if degree >= self.data.atom_counts.len() {
                    if !support.is_empty() || expected.iter().any(|&coefficient| coefficient != 0) {
                        return Err(CategoryError::InvalidCellTopChain {
                            degree,
                            reason: "a cell above the atomic dimension is not an identity",
                        });
                    }
                    continue;
                }
                let chain = (0..self.atom_count_in_degree(degree))
                    .map(|atom| i64::from(support.binary_search(&atom).is_ok()))
                    .collect::<Vec<_>>();
                if self.apply_boundary(degree, &chain) != expected {
                    return Err(CategoryError::InvalidCellTopChain {
                        degree,
                        reason: "cell double sequence does not satisfy its boundary equation",
                    });
                }
            }
        }
        Ok(())
    }

    fn compose_cell_shapes(
        &self,
        first: &CellShape,
        next: &CellShape,
        along: usize,
    ) -> Result<Option<CellShape>, CategoryError> {
        if first.dimension() != next.dimension() || along >= first.dimension() {
            return Ok(None);
        }
        if first.boundary_at(along, true) != next.boundary_at(along, false) {
            return Ok(None);
        }
        let dimension = first.dimension();
        let mut negative = vec![Vec::new(); dimension + 1];
        let mut positive = vec![Vec::new(); dimension + 1];
        for degree in 0..=dimension {
            if degree < along {
                negative[degree] = first.negative[degree].clone();
                positive[degree] = first.positive[degree].clone();
            } else if degree == along {
                negative[degree] = first.negative[degree].clone();
                positive[degree] = next.positive[degree].clone();
            } else {
                negative[degree] = disjoint_union_supports(
                    &first.negative[degree],
                    &next.negative[degree],
                    degree,
                )?;
                positive[degree] = disjoint_union_supports(
                    &first.positive[degree],
                    &next.positive[degree],
                    degree,
                )?;
            }
        }
        let result = CellShape { negative, positive };
        self.validate_cell_shape(&result)?;
        Ok(Some(result))
    }

    fn all_cell_shapes(&self) -> Result<&Vec<Vec<CellShape>>, CategoryError> {
        if let Some(cells) = self.data.cells.get() {
            return Ok(cells);
        }
        let mut all: Vec<Vec<CellShape>> = Vec::with_capacity(self.data.atom_counts.len());
        let mut total = 0usize;
        for degree in 0..self.data.atom_counts.len() {
            let atoms = self.atom_count_in_degree(degree);
            let mut seen = BTreeSet::new();
            let mut cells = Vec::new();
            for atom in 0..atoms {
                let shape = self.minimal_cell_shape(degree, &[atom])?;
                if seen.insert(shape.clone()) {
                    cells.push(shape);
                }
            }
            if degree > 0 {
                for cell in &all[degree - 1] {
                    let identity = cell.identity();
                    self.validate_cell_shape(&identity)?;
                    if seen.insert(identity.clone()) {
                        cells.push(identity);
                    }
                }
            }

            let mut cursor = 0usize;
            while cursor < cells.len() {
                let pivot = cells[cursor].clone();
                let existing = cells.clone();
                for other in existing {
                    for along in 0..degree {
                        for (first, next) in [(&pivot, &other), (&other, &pivot)] {
                            let Some(composite) = self.compose_cell_shapes(first, next, along)?
                            else {
                                continue;
                            };
                            if seen.insert(composite.clone()) {
                                cells.push(composite);
                                if total
                                    .checked_add(cells.len())
                                    .ok_or(CategoryError::SizeOverflow)?
                                    > MAX_ENUMERATED_CELLS
                                {
                                    return Err(CategoryError::CellEnumerationLimitExceeded {
                                        maximum: MAX_ENUMERATED_CELLS,
                                    });
                                }
                            }
                        }
                    }
                }
                cursor += 1;
            }
            cells.sort();
            total = total
                .checked_add(cells.len())
                .ok_or(CategoryError::SizeOverflow)?;
            if total > MAX_ENUMERATED_CELLS {
                return Err(CategoryError::CellEnumerationLimitExceeded {
                    maximum: MAX_ENUMERATED_CELLS,
                });
            }
            all.push(cells);
        }
        let _ = self.data.cells.set(all);
        Ok(self.data.cells.get().expect("cells were initialized"))
    }
}

fn disjoint_union_supports(
    left: &[usize],
    right: &[usize],
    degree: usize,
) -> Result<Vec<usize>, CategoryError> {
    let mut result = left.to_vec();
    result.extend_from_slice(right);
    result.sort_unstable();
    if result.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(CategoryError::InvalidCellTopChain {
            degree,
            reason: "composition gives an atomic coefficient greater than one",
        });
    }
    Ok(result)
}

impl PartialEq for SteinerCategory {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for SteinerCategory {}

impl SemanticType for SteinerCategory {
    const TYPE_ID: &'static str = "noether/category/SteinerCategory@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        output.extend_from_slice(&(self.data.atom_counts.len() as u64).to_be_bytes());
        for &count in &self.data.atom_counts {
            output.extend_from_slice(&(count as u64).to_be_bytes());
        }
        for degree in 1..self.data.boundaries.len() {
            for boundary in &self.data.boundaries[degree] {
                output.extend_from_slice(&(boundary.negative.len() as u64).to_be_bytes());
                for &atom in &boundary.negative {
                    output.extend_from_slice(&(atom as u64).to_be_bytes());
                }
                output.extend_from_slice(&(boundary.positive.len() as u64).to_be_bytes());
                for &atom in &boundary.positive {
                    output.extend_from_slice(&(atom as u64).to_be_bytes());
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &DESCRIPTION
    }
}

impl Serialize for SteinerCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Wire {
            type_id: Self::TYPE_ID.to_owned(),
            atom_counts: self.data.atom_counts.clone(),
            boundaries: self.data.boundaries.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SteinerCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Wire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_boundaries(wire.atom_counts, wire.boundaries).map_err(serde::de::Error::custom)
    }
}

/// One cell in a fixed finite strong Steiner category.
///
/// The two degreewise support sequences are Steiner's exact double-sequence
/// coordinates. Keeping both sequences is essential: common lower-dimensional
/// support records whiskering and cannot be reconstructed from the top chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteinerCell {
    pub(crate) category: SteinerCategory,
    shape: CellShape,
}

impl SteinerCell {
    fn from_shape(category: SteinerCategory, shape: CellShape) -> Self {
        Self { category, shape }
    }

    /// Ambient category context.
    #[must_use]
    pub fn category(&self) -> &SteinerCategory {
        &self.category
    }

    /// Current cell dimension, including identity lifts.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.shape.dimension()
    }

    /// Dimension of the underlying nonidentity cell.
    #[must_use]
    pub fn intrinsic_dimension(&self) -> usize {
        self.shape.intrinsic_dimension()
    }

    /// Whether this is an iterated identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.shape.negative[self.dimension()].is_empty()
    }

    /// Top atomic support of the underlying nonidentity cell.
    #[must_use]
    pub fn top_atoms(&self) -> &[usize] {
        &self.shape.negative[self.intrinsic_dimension()]
    }

    /// Negative atomic supports in degrees `0..=dimension`.
    #[must_use]
    pub fn negative_supports(&self) -> &[Vec<usize>] {
        &self.shape.negative
    }

    /// Positive atomic supports in degrees `0..=dimension`.
    #[must_use]
    pub fn positive_supports(&self) -> &[Vec<usize>] {
        &self.shape.positive
    }

    /// Identity one dimension above this cell.
    #[must_use]
    pub fn identity(&self) -> Self {
        Self::from_shape(self.category.clone(), self.shape.identity())
    }

    /// Immediate globular source.
    pub fn source(&self) -> Result<Self, CategoryError> {
        self.boundary_cell(false)
    }

    /// Immediate globular target.
    pub fn target(&self) -> Result<Self, CategoryError> {
        self.boundary_cell(true)
    }

    /// Iterated source in dimension `dimension`.
    pub fn source_at(&self, dimension: usize) -> Result<Self, CategoryError> {
        self.boundary_at(dimension, false)
    }

    /// Iterated target in dimension `dimension`.
    pub fn target_at(&self, dimension: usize) -> Result<Self, CategoryError> {
        self.boundary_at(dimension, true)
    }

    /// Compose this cell followed by `next` along a common `dimension`-face.
    ///
    /// Thus `first.compose_at(&second, k)` is defined precisely when the
    /// `k`-target of `first` equals the `k`-source of `second`. Both cells keep
    /// their common ambient dimension in the result.
    pub fn compose_at(&self, next: &Self, dimension: usize) -> Result<Self, CategoryError> {
        if self.category != next.category {
            return Err(CategoryError::CellContextMismatch);
        }
        if self.dimension() != next.dimension() {
            return Err(CategoryError::CellDimensionMismatch {
                left: self.dimension(),
                right: next.dimension(),
            });
        }
        if dimension >= self.dimension() {
            return Err(CategoryError::InvalidBasepoint(
                "composition dimension must be smaller than the cell dimension",
            ));
        }
        self.category
            .compose_cell_shapes(&self.shape, &next.shape, dimension)?
            .map(|shape| Self::from_shape(self.category.clone(), shape))
            .ok_or(CategoryError::CellsNotComposable { along: dimension })
    }

    /// Transport this cell under a verified atomic automorphism.
    pub fn apply_automorphism(&self, automorphism: &Permutation) -> Result<Self, CategoryError> {
        if !super::canonical::is_automorphism(&self.category, automorphism) {
            return Err(CategoryError::Canonicalization(
                "the supplied permutation is not an automorphism of the cell's category",
            ));
        }
        let offsets = degree_offsets(self.category.atom_counts())?;
        let mut local_images = self
            .category
            .atom_counts()
            .iter()
            .map(|&count| vec![0usize; count])
            .collect::<Vec<_>>();
        for (degree, degree_images) in local_images.iter_mut().enumerate() {
            let offset = offsets[degree];
            let degree_count = self.category.atom_count_in_degree(degree);
            for (atom, local_image) in degree_images.iter_mut().enumerate() {
                let flattened = offset + atom;
                let image = automorphism
                    .apply(u32::try_from(flattened).map_err(|_| CategoryError::SizeOverflow)?)
                    as usize;
                if image < offset || image >= offset + degree_count {
                    return Err(CategoryError::Canonicalization(
                        "an atomic automorphism did not preserve degree",
                    ));
                }
                *local_image = image - offset;
            }
        }
        self.relabel(&self.category, &local_images)
    }

    fn boundary_at(&self, dimension: usize, positive: bool) -> Result<Self, CategoryError> {
        if dimension >= self.dimension() {
            return Err(CategoryError::InvalidBasepoint(
                "requested globular boundary must have smaller dimension",
            ));
        }
        self.shape
            .boundary_at(dimension, positive)
            .map(|shape| Self::from_shape(self.category.clone(), shape))
            .ok_or(CategoryError::ObjectHasNoBoundary)
    }

    fn boundary_cell(&self, positive: bool) -> Result<Self, CategoryError> {
        if self.dimension() == 0 {
            return Err(CategoryError::ObjectHasNoBoundary);
        }
        self.boundary_at(self.dimension() - 1, positive)
    }

    pub(crate) fn relabel(
        &self,
        target: &SteinerCategory,
        local_images: &[Vec<usize>],
    ) -> Result<Self, CategoryError> {
        if self.category.atom_counts() != target.atom_counts() {
            return Err(CategoryError::CellContextMismatch);
        }
        let relabel_supports = |supports: &[Vec<usize>]| {
            supports
                .iter()
                .enumerate()
                .map(|(degree, support)| {
                    if degree >= local_images.len() {
                        return Vec::new();
                    }
                    let mut image = support
                        .iter()
                        .map(|&atom| local_images[degree][atom])
                        .collect::<Vec<_>>();
                    image.sort_unstable();
                    image
                })
                .collect::<Vec<_>>()
        };
        let shape = CellShape {
            negative: relabel_supports(&self.shape.negative),
            positive: relabel_supports(&self.shape.positive),
        };
        target.validate_cell_shape(&shape)?;
        Ok(Self::from_shape(target.clone(), shape))
    }
}

fn degree_offsets(atom_counts: &[usize]) -> Result<Vec<usize>, CategoryError> {
    let mut offsets = Vec::with_capacity(atom_counts.len());
    let mut offset = 0usize;
    for &count in atom_counts {
        offsets.push(offset);
        offset = offset
            .checked_add(count)
            .ok_or(CategoryError::SizeOverflow)?;
    }
    Ok(offsets)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            output.push(current.clone());
            return;
        }
        let remaining = k - current.len();
        for value in start..=n - remaining {
            current.push(value);
            visit(n, k, value + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    if k <= n {
        visit(n, k, 0, &mut Vec::new(), &mut output);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_noncomplex_and_cyclic_precedence_data() {
        let noncomplex = SteinerCategory::from_boundaries(
            vec![2, 2, 1],
            vec![
                vec![],
                vec![
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                ],
                vec![SignedBoundary::new(&[0], &[]).unwrap()],
            ],
        );
        assert!(matches!(
            noncomplex,
            Err(CategoryError::BoundarySquaredNonzero { .. })
        ));

        let cyclic = SteinerCategory::from_boundaries(
            vec![2, 2],
            vec![
                vec![],
                vec![
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                    SignedBoundary::new(&[1], &[0]).unwrap(),
                ],
            ],
        );
        assert!(matches!(cyclic, Err(CategoryError::CyclicPrecedence)));
    }

    #[test]
    fn globes_orientals_and_cubes_have_expected_atomic_vectors() {
        assert_eq!(SteinerCategory::empty().dimension(), -1);
        assert_eq!(SteinerCategory::globe(0).unwrap().atom_counts(), &[1]);
        assert_eq!(
            SteinerCategory::globe(3).unwrap().atom_counts(),
            &[2, 2, 2, 1]
        );
        assert_eq!(
            SteinerCategory::oriental(2).unwrap().atom_counts(),
            &[3, 3, 1]
        );
        assert_eq!(SteinerCategory::cube(2).unwrap().atom_counts(), &[4, 4, 1]);
    }

    #[test]
    fn two_oriental_cells_and_polygraphic_homology_are_exact() {
        let oriental = SteinerCategory::oriental(2).unwrap();
        assert_eq!(oriental.exact_cell_counts().unwrap(), vec![3, 4, 1]);
        let groups = oriental.polygraphic_homology_invariants().unwrap();
        assert_eq!(groups[&0].free_rank(), 1);
        assert_eq!(groups[&1].free_rank(), 0);
        assert_eq!(groups[&2].free_rank(), 0);
    }

    #[test]
    fn composition_uses_the_common_face_and_has_strict_units() {
        let category = SteinerCategory::oriental(2).unwrap();
        let first = category.atom(AtomId::new(1, 0)).unwrap();
        let second = category.atom(AtomId::new(1, 2)).unwrap();
        let composite = first.compose_at(&second, 0).unwrap();
        assert_eq!(composite.top_atoms(), &[0, 2]);
        assert_eq!(composite.source().unwrap(), first.source().unwrap());
        assert_eq!(composite.target().unwrap(), second.target().unwrap());

        let left_unit = first.source().unwrap().identity();
        let right_unit = first.target().unwrap().identity();
        assert_eq!(left_unit.compose_at(&first, 0).unwrap(), first);
        assert_eq!(first.compose_at(&right_unit, 0).unwrap(), first);
        assert!(second.compose_at(&first, 0).is_err());
    }

    #[test]
    fn double_sequences_retain_whiskering_below_the_top_chain() {
        let category = SteinerCategory::oriental(3).unwrap();
        let face_012 = category.atom(AtomId::new(2, 0)).unwrap();
        let edge_23 = category.atom(AtomId::new(1, 5)).unwrap();
        let whiskered = face_012.compose_at(&edge_23.identity(), 0).unwrap();

        assert_eq!(whiskered.top_atoms(), face_012.top_atoms());
        assert_ne!(whiskered, face_012);
        assert_eq!(whiskered.negative_supports()[1], vec![1, 5]);
        assert_eq!(whiskered.positive_supports()[1], vec![0, 3, 5]);
        assert_eq!(whiskered.source().unwrap().top_atoms(), &[1, 5]);
        assert_eq!(whiskered.target().unwrap().top_atoms(), &[0, 3, 5]);
        assert_eq!(
            category
                .cell_from_supports(
                    whiskered.negative_supports().to_vec(),
                    whiskered.positive_supports().to_vec(),
                )
                .unwrap(),
            whiskered
        );
        assert!(category.exact_cells(2).unwrap().contains(&whiskered));
    }

    #[test]
    fn truncation_disjoint_union_and_duality_preserve_the_model() {
        let oriental = SteinerCategory::oriental(2).unwrap();
        assert_eq!(oriental.truncate(1).unwrap().atom_counts(), &[3, 3]);

        let two_points = SteinerCategory::point()
            .disjoint_union(&SteinerCategory::point())
            .unwrap();
        assert_eq!(two_points.atom_counts(), &[2]);
        assert_eq!(two_points.exact_cell_counts().unwrap(), vec![2]);

        for dimensions in [&[1][..], &[2][..], &[1, 2][..]] {
            let dual = oriental.dual(dimensions).unwrap();
            assert_eq!(dual.atom_counts(), oriental.atom_counts());
            assert_eq!(dual.dual(dimensions).unwrap(), oriental);
        }
    }

    #[test]
    fn automorphisms_transport_cells_in_the_same_context() {
        let category = SteinerCategory::point()
            .disjoint_union(&SteinerCategory::point())
            .unwrap();
        let generator = category
            .canonical_label()
            .unwrap()
            .automorphism_group()
            .generators()[0]
            .clone();
        let first = category.atom(AtomId::new(0, 0)).unwrap();
        let second = first.apply_automorphism(&generator).unwrap();
        assert_eq!(second, category.atom(AtomId::new(0, 1)).unwrap());

        let wrong_degree = Permutation::identity(3);
        assert!(first.apply_automorphism(&wrong_degree).is_err());
    }

    #[test]
    fn gray_arrow_square_has_koszul_boundary_and_nine_atoms() {
        let arrow = SteinerCategory::globe(1).unwrap();
        let square = arrow.gray_tensor(&arrow).unwrap();
        assert_eq!(square.atom_counts(), &[4, 4, 1]);
        assert_eq!(square.atom_count(), 9);
        square.verify().unwrap();
        assert_eq!(SteinerCategory::point().gray_tensor(&arrow).unwrap(), arrow);
        assert_eq!(
            SteinerCategory::empty().gray_tensor(&arrow).unwrap(),
            SteinerCategory::empty()
        );
    }

    #[test]
    fn serde_round_trip_preserves_exact_identity() {
        let category = SteinerCategory::oriental(2).unwrap();
        let encoded = serde_json::to_string(&category).unwrap();
        let decoded: SteinerCategory = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, category);
        assert_eq!(
            decoded.canonical_bytes().unwrap(),
            category.canonical_bytes().unwrap()
        );
    }
}
