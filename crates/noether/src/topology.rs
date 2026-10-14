// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Simplicial objects, filtrations, persistence, and homology.
//!
//! A [`SimplicialComplex`] is an abstract finite complex: vertices are labels,
//! not coordinates.  Euclidean constructions live in [`crate::geometry`]
//! and produce this representation only after their geometric work is done.

use core::any::TypeId;
use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::combin::GraphSymmetryError;
use crate::homalg::{
    graded_homology_invariants_from_sparse_boundaries,
    integral_homology_invariants_from_sparse_boundaries, ChainComplex,
    FinitelyGeneratedAbelianGroup, GradedHomology, HomologyError, HomologyGroup,
};
use crate::linalg::{LinalgError, Matrix, SparseMatrix, Vector};
use crate::{
    ContextMismatchError, Field, Integer, IntegerRing, InversionError, NoetherError, RationalField,
    Ring, SemanticType, TypeDescription, Z,
};

mod cohomology;
mod cohomology_operation;
mod collapse;
mod cup_i;
mod eilenberg_maclane;
mod filtrations;
mod may_steenrod;
mod morse;
mod persistence;
mod simplicial_set;
mod simplicial_set_cohomology;
mod simplicial_set_collapse;
mod simplicial_set_symmetry;
mod strong;
mod symmetry;

pub use cohomology::{
    SimplicialCohomologyBasisElement, SimplicialCohomologyProduct, SimplicialCohomologyProductTerm,
    SimplicialCohomologyRing,
};
pub use cohomology_operation::AdmissibleCohomologyOperation;
pub use collapse::{
    ElementaryCollapse, SimplicialCollapseSequence, SimplicialSimpleHomotopyMove,
    SimplicialSimpleHomotopyZigzag,
};
pub use eilenberg_maclane::{
    EilenbergMacLaneCohomologyGenerator, EilenbergMacLaneCohomologyOperation,
    EilenbergMacLaneError, EilenbergMacLaneGeneratorOrigin, EilenbergMacLaneHomologyStrategy,
    EilenbergMacLaneIntegralCohomology, EilenbergMacLaneIntegralCohomologyGenerator,
    EilenbergMacLaneIntegralCohomologyPresentation,
    EilenbergMacLaneIntegralCohomologyPresentationStrategy,
    EilenbergMacLaneIntegralCohomologyRelation, EilenbergMacLaneIntegralCohomologyRelationTerm,
    EilenbergMacLaneIntegralGeneratorOrigin, EilenbergMacLaneIntegralHomology,
    EilenbergMacLaneModularCohomology, EilenbergMacLaneModularCohomologyStrategy,
    EilenbergMacLaneModularCohomologyTruncation,
    EilenbergMacLaneModularCohomologyTruncationStrategy, EilenbergMacLaneSpace,
};
pub use filtrations::DEFAULT_ALPHA_FACE_SLOTS;
pub use morse::SimplicialMorseReduction;
pub use persistence::{Barcode, FilteredComplex};
pub use simplicial_set::{SimplicialFace, SimplicialSet, SimplicialSetError};
pub use simplicial_set_cohomology::SimplicialSetCohomologyRing;
pub use simplicial_set_collapse::{
    SimplicialSetAnodyneMove, SimplicialSetAnodyneZigzag, SimplicialSetElementaryHornCollapse,
    SimplicialSetHornCollapseSequence,
};
pub use strong::{DominatedVertex, StrongCoreReduction};
#[allow(deprecated)]
pub use symmetry::{MAX_SIMPLICIAL_AUTOMORPHISM_DEGREE, MAX_SIMPLICIAL_AUTOMORPHISM_VERTICES};

static SIMPLICIAL_COMPLEX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialComplex@1",
    rust_path: "noether::topology::SimplicialComplex",
    python_path: Some("noether.topology.SimplicialComplex"),
};

// Below this size, constructing a Morse matching costs more than the sparse
// Smith work on the pinned small-complex corpus. Larger inputs still require a
// substantial boundary-size reduction before the Morse model is selected.
const SIMPLICIAL_MORSE_MIN_CELLS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegralHomologyStrategy {
    OrdinarySparseBoundaries,
    SimplicialMorseModel,
}

/// Failure to construct a chain complex or compute a topological invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// A simplicial dimension cannot be represented by the signed degree type.
    DimensionTooLarge {
        /// The unrepresentable simplicial dimension.
        dimension: usize,
    },
    /// Exact linear algebra failed.
    LinearAlgebra(LinalgError),
    /// Chain-complex construction or homology failed.
    Homology(HomologyError),
    /// A finite simplicial-set presentation is invalid.
    SimplicialSet(SimplicialSetError),
    /// A proposed real-valued filtration was not finite, face-closed, and monotone.
    InvalidFiltration {
        /// Precise failed invariant.
        reason: String,
    },
    /// A proposed barcode contained an invalid interval.
    InvalidBarcode {
        /// Precise failed invariant.
        reason: String,
    },
    /// An internal coefficient operation received an element from another context.
    ContextMismatch(ContextMismatchError),
    /// Persistent reduction could not invert a nonzero field coefficient.
    Inversion(InversionError),
    /// Exact permutation-group construction failed.
    Group(crate::groups::GroupError),
    /// Exact colored incidence-graph canonicalization failed.
    GraphSymmetry(GraphSymmetryError),
    /// Exact graded-commutative presentation construction failed.
    GradedCommutative(crate::commalg::GradedCommutativeError),
    /// A claimed strong collapse or core reduction failed exact replay.
    InvalidStrongCollapse {
        /// Precise failed invariant.
        reason: String,
    },
    /// A claimed elementary free-face collapse failed exact replay.
    InvalidElementaryCollapse {
        /// Precise failed invariant.
        reason: String,
    },
    /// A supplied simplicial cochain had an incompatible context or dimension.
    InvalidCochain {
        /// Precise failed invariant.
        reason: String,
    },
    /// A purported Steenrod-algebra word has an invalid prime, shape,
    /// admissibility inequality, or degree.
    InvalidCohomologyOperation {
        /// Precise failed invariant.
        reason: String,
    },
    /// Exhaustive automorphism search was requested on too many used vertices.
    SymmetrySearchTooLarge {
        /// Number of vertices in the complex.
        vertices: usize,
        /// Largest accepted vertex count.
        max: usize,
    },
    /// Sparse vertex labels would create an unreasonably large permutation domain.
    SymmetryDegreeTooLarge {
        /// Required permutation degree.
        degree: usize,
        /// Largest accepted permutation degree.
        max: usize,
    },
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionTooLarge { dimension } => {
                write!(f, "simplicial dimension {dimension} does not fit in i32")
            }
            Self::LinearAlgebra(error) => error.fmt(f),
            Self::Homology(error) => error.fmt(f),
            Self::SimplicialSet(error) => error.fmt(f),
            Self::InvalidFiltration { reason } => write!(f, "invalid filtered complex: {reason}"),
            Self::InvalidBarcode { reason } => write!(f, "invalid barcode: {reason}"),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Inversion(error) => error.fmt(f),
            Self::Group(error) => error.fmt(f),
            Self::GraphSymmetry(error) => error.fmt(f),
            Self::GradedCommutative(error) => error.fmt(f),
            Self::InvalidStrongCollapse { reason } => {
                write!(f, "invalid simplicial strong collapse: {reason}")
            }
            Self::InvalidElementaryCollapse { reason } => {
                write!(f, "invalid elementary simplicial collapse: {reason}")
            }
            Self::InvalidCochain { reason } => write!(f, "invalid simplicial cochain: {reason}"),
            Self::InvalidCohomologyOperation { reason } => {
                write!(f, "invalid cohomology operation: {reason}")
            }
            Self::SymmetrySearchTooLarge { vertices, max } => write!(
                f,
                "simplicial-complex automorphism search on {vertices} vertices exceeds the exhaustive 0.1 limit {max}"
            ),
            Self::SymmetryDegreeTooLarge { degree, max } => write!(
                f,
                "simplicial-complex vertex labels require permutation degree {degree}, exceeding the limit {max}"
            ),
        }
    }
}

impl std::error::Error for TopologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DimensionTooLarge { .. } => None,
            Self::LinearAlgebra(error) => Some(error),
            Self::Homology(error) => Some(error),
            Self::SimplicialSet(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::Inversion(error) => Some(error),
            Self::Group(error) => Some(error),
            Self::GraphSymmetry(error) => Some(error),
            Self::GradedCommutative(error) => Some(error),
            Self::InvalidFiltration { .. }
            | Self::InvalidBarcode { .. }
            | Self::InvalidStrongCollapse { .. }
            | Self::InvalidElementaryCollapse { .. }
            | Self::InvalidCochain { .. }
            | Self::InvalidCohomologyOperation { .. }
            | Self::SymmetrySearchTooLarge { .. }
            | Self::SymmetryDegreeTooLarge { .. } => None,
        }
    }
}

impl From<LinalgError> for TopologyError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<HomologyError> for TopologyError {
    fn from(value: HomologyError) -> Self {
        Self::Homology(value)
    }
}

impl From<SimplicialSetError> for TopologyError {
    fn from(value: SimplicialSetError) -> Self {
        Self::SimplicialSet(value)
    }
}

impl From<ContextMismatchError> for TopologyError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<InversionError> for TopologyError {
    fn from(value: InversionError) -> Self {
        Self::Inversion(value)
    }
}

impl From<crate::groups::GroupError> for TopologyError {
    fn from(value: crate::groups::GroupError) -> Self {
        Self::Group(value)
    }
}

impl From<GraphSymmetryError> for TopologyError {
    fn from(value: GraphSymmetryError) -> Self {
        Self::GraphSymmetry(value)
    }
}

impl From<crate::commalg::GradedCommutativeError> for TopologyError {
    fn from(value: crate::commalg::GradedCommutativeError) -> Self {
        Self::GradedCommutative(value)
    }
}

/// A simplex with strictly increasing vertex labels.
///
/// Construction sorts and de-duplicates labels.  The order fixes the usual
/// orientation: deleting the vertex in position `i` has sign `(-1)^i`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex(Vec<u32>);

impl Simplex {
    /// Construct a simplex from vertex labels in any order.
    pub fn new(mut vertices: Vec<u32>) -> Self {
        vertices.sort_unstable();
        vertices.dedup();
        Self(vertices)
    }

    /// Vertices in increasing order.
    pub fn vertices(&self) -> &[u32] {
        &self.0
    }

    /// Dimension, with the unrepresented empty simplex assigned zero.
    pub fn dimension(&self) -> usize {
        self.0.len().saturating_sub(1)
    }

    /// Short alias used by reduction kernels.
    #[doc(hidden)]
    pub fn dim(&self) -> usize {
        self.dimension()
    }

    /// Codimension-one faces in oriented boundary order.
    pub fn faces(&self) -> Vec<Self> {
        if self.0.len() <= 1 {
            return Vec::new();
        }
        (0..self.0.len())
            .map(|index| {
                let mut vertices = self.0.clone();
                vertices.remove(index);
                Self(vertices)
            })
            .collect()
    }

    fn is_face_of(&self, other: &Self) -> bool {
        self.0
            .iter()
            .all(|vertex| other.0.binary_search(vertex).is_ok())
    }
}

/// A finite abstract simplicial complex stored in deterministic index order.
///
/// [`from_facets`](Self::from_facets) takes the downward closure, sorts every
/// dimension lexicographically, and fixes that indexing for all subsequent
/// boundary matrices.  Empty facets are ignored; an empty facet list creates
/// the empty complex.
///
/// # Examples
///
/// ```
/// use noether::topology::SimplicialComplex;
/// use noether::Q;
///
/// let circle = SimplicialComplex::from_facets(&[
///     vec![0, 1], vec![1, 2], vec![0, 2],
/// ]);
/// assert_eq!(circle.euler_characteristic(), 0);
/// assert_eq!(circle.homology(Q, 1)?.dimension(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct SimplicialComplex {
    simplices: Vec<Vec<Simplex>>,
    index: Vec<BTreeMap<Simplex, usize>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplicialComplexWire {
    #[serde(rename = "type")]
    type_id: String,
    facets: Vec<Vec<u32>>,
}

impl SimplicialComplex {
    /// Construct the downward closure of a collection of facets.
    pub fn from_facets(facets: &[Vec<u32>]) -> Self {
        let mut by_dimension: Vec<Vec<Simplex>> = Vec::new();
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Simplex> = facets.iter().cloned().map(Simplex::new).collect();

        while let Some(simplex) = stack.pop() {
            if simplex.vertices().is_empty() || !seen.insert(simplex.clone()) {
                continue;
            }
            let dimension = simplex.dimension();
            if by_dimension.len() <= dimension {
                by_dimension.resize(dimension + 1, Vec::new());
            }
            for face in simplex.faces() {
                if !seen.contains(&face) {
                    stack.push(face);
                }
            }
            by_dimension[dimension].push(simplex);
        }

        for simplices in &mut by_dimension {
            simplices.sort();
        }
        let index = by_dimension
            .iter()
            .map(|simplices| {
                simplices
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(position, simplex)| (simplex, position))
                    .collect()
            })
            .collect();
        Self {
            simplices: by_dimension,
            index,
        }
    }

    /// Serialize the canonical facet presentation as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and normalize a typed facet presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Whether the complex has no simplices.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Top dimension, with the empty complex assigned dimension `-1`.
    pub fn dimension(&self) -> i64 {
        self.top_dimension().map_or(-1, |dimension| {
            i64::try_from(dimension).expect("simplicial-complex dimensions fit i64")
        })
    }

    fn top_dimension(&self) -> Option<usize> {
        self.simplices.len().checked_sub(1)
    }

    /// Number of simplices in dimension `dimension`.
    pub fn count(&self, dimension: usize) -> usize {
        self.simplices.get(dimension).map_or(0, Vec::len)
    }

    /// Total number of nonempty simplices.
    pub fn total_cells(&self) -> usize {
        self.simplices.iter().map(Vec::len).sum()
    }

    /// Simplices of one dimension, in canonical index order.
    pub fn simplices(&self, dimension: usize) -> &[Simplex] {
        self.simplices.get(dimension).map_or(&[], Vec::as_slice)
    }

    /// Maximal simplices in lexicographic order.
    pub fn facets(&self) -> Vec<&Simplex> {
        let mut facets: Vec<&Simplex> = self
            .simplices
            .iter()
            .flatten()
            .filter(|simplex| {
                self.simplices
                    .iter()
                    .skip(simplex.dimension() + 1)
                    .flatten()
                    .all(|candidate| !simplex.is_face_of(candidate))
            })
            .collect();
        facets.sort();
        facets
    }

    /// Index of a simplex within its dimension.
    pub fn position(&self, simplex: &Simplex) -> Option<usize> {
        self.index.get(simplex.dimension())?.get(simplex).copied()
    }

    /// Euler characteristic from cell counts.
    ///
    /// This is independent of homology and therefore provides a useful
    /// cross-check against the alternating sum of field-homology dimensions.
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .enumerate()
            .map(|(dimension, simplices)| {
                if dimension % 2 == 0 {
                    simplices.len() as i64
                } else {
                    -(simplices.len() as i64)
                }
            })
            .sum()
    }

    /// Boundary `d_k` over `F_2` as sparse columns of row indices.
    pub fn boundary_columns_f2(&self, degree: usize) -> Vec<Vec<u32>> {
        if degree == 0 || self.top_dimension().is_none_or(|top| degree > top) {
            return vec![Vec::new(); self.count(degree)];
        }
        self.simplices(degree)
            .iter()
            .map(|simplex| {
                let mut column: Vec<u32> = simplex
                    .faces()
                    .iter()
                    .filter_map(|face| self.position(face).map(|position| position as u32))
                    .collect();
                column.sort_unstable();
                column
            })
            .collect()
    }

    /// Signed boundary `d_k: C_k -> C_(k-1)` over an arbitrary ring.
    pub fn boundary_matrix<R: Ring>(
        &self,
        ring: R,
        degree: usize,
    ) -> Result<Matrix<R>, LinalgError> {
        let columns = self.count(degree);
        let rows = if degree == 0 {
            0
        } else {
            self.count(degree - 1)
        };
        if rows == 0 || columns == 0 {
            return Ok(Matrix::zeros(ring, rows, columns));
        }

        let mut entries = vec![vec![ring.zero(); columns]; rows];
        let positive = ring.one();
        let negative = ring.neg(&positive)?;
        for (column, simplex) in self.simplices(degree).iter().enumerate() {
            for (omitted, face) in simplex.faces().iter().enumerate() {
                let row = self
                    .position(face)
                    .expect("downward closure contains every face");
                entries[row][column] = if omitted % 2 == 0 {
                    positive.clone()
                } else {
                    negative.clone()
                };
            }
        }
        Matrix::from_rows(ring, entries)
    }

    /// Construct the canonical checked chain complex over `ring`.
    ///
    /// A zero-dimensional complex includes `d_0: C_0 -> 0`, because no
    /// adjacent nonzero differential exists from which `C_0` could otherwise
    /// be inferred.
    pub fn chain_complex<R: Ring>(&self, ring: R) -> Result<ChainComplex<R>, TopologyError> {
        let top_dimension = self.top_dimension().unwrap_or(0);
        let top = i32::try_from(top_dimension).map_err(|_| TopologyError::DimensionTooLarge {
            dimension: top_dimension,
        })?;
        let mut boundaries = BTreeMap::new();
        if top == 0 {
            boundaries.insert(0, self.boundary_matrix(ring.clone(), 0)?);
        } else {
            for degree in 1..=top {
                boundaries.insert(degree, self.boundary_matrix(ring.clone(), degree as usize)?);
            }
        }
        Ok(ChainComplex::from_boundaries(ring, boundaries)?)
    }

    /// Construct the canonical simplicial cochain complex over `ring`.
    ///
    /// Each coboundary is the transpose of the signed boundary in the fixed
    /// lexicographic simplex bases.
    pub fn cochain_complex<R: Ring>(&self, ring: R) -> Result<ChainComplex<R>, TopologyError> {
        let top_dimension = self.top_dimension().unwrap_or(0);
        let mut coboundaries = BTreeMap::new();
        for degree in 0..=top_dimension {
            let signed_degree = i32::try_from(degree)
                .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
            let upper = degree
                .checked_add(1)
                .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
            let coboundary = self.boundary_matrix(ring.clone(), upper)?.transpose();
            coboundaries.insert(signed_degree, coboundary);
        }
        Ok(ChainComplex::from_coboundaries(ring, coboundaries)?)
    }

    /// Alexander--Whitney front/back faces of one simplex, from split degree
    /// zero through its full dimension.
    pub fn alexander_whitney_diagonal(
        &self,
        dimension: usize,
        simplex: usize,
    ) -> Result<Vec<(Simplex, Simplex)>, TopologyError> {
        let value = self.simplices(dimension).get(simplex).ok_or_else(|| {
            TopologyError::InvalidCochain {
                reason: format!(
                    "there is no simplex {simplex} in dimension {dimension}; {} exist",
                    self.count(dimension)
                ),
            }
        })?;
        Ok((0..=dimension)
            .map(|split| {
                (
                    Simplex::new(value.vertices()[..=split].to_vec()),
                    Simplex::new(value.vertices()[split..].to_vec()),
                )
            })
            .collect())
    }

    /// Alexander--Whitney cup product of two simplicial cochains.
    ///
    /// Inputs and output use the same lexicographic simplex bases as the
    /// boundary and cochain complexes. This is a chain-level operation: inputs
    /// need not be cocycles.
    pub fn cup_product<R: Ring>(
        &self,
        left_degree: usize,
        left: &Vector<R>,
        right_degree: usize,
        right: &Vector<R>,
    ) -> Result<Vector<R>, TopologyError> {
        if left.ring() != right.ring() {
            return Err(TopologyError::InvalidCochain {
                reason: "cup-product inputs use different coefficient contexts".to_owned(),
            });
        }
        if left.dimension() != self.count(left_degree) {
            return Err(TopologyError::InvalidCochain {
                reason: format!(
                    "left degree-{left_degree} cochain has {} coefficients; expected {}",
                    left.dimension(),
                    self.count(left_degree)
                ),
            });
        }
        if right.dimension() != self.count(right_degree) {
            return Err(TopologyError::InvalidCochain {
                reason: format!(
                    "right degree-{right_degree} cochain has {} coefficients; expected {}",
                    right.dimension(),
                    self.count(right_degree)
                ),
            });
        }
        let degree =
            left_degree
                .checked_add(right_degree)
                .ok_or(TopologyError::DimensionTooLarge {
                    dimension: usize::MAX,
                })?;
        let ring = left.ring().clone();
        let mut values = Vec::with_capacity(self.count(degree));
        for simplex in self.simplices(degree) {
            let front = Simplex::new(simplex.vertices()[..=left_degree].to_vec());
            let back = Simplex::new(simplex.vertices()[left_degree..].to_vec());
            let left_index = self
                .position(&front)
                .expect("the downward closure contains every Alexander--Whitney front face");
            let right_index = self
                .position(&back)
                .expect("the downward closure contains every Alexander--Whitney back face");
            let left_value = left
                .get(left_index)
                .map_err(|_| TopologyError::InvalidCochain {
                    reason: "Alexander--Whitney front-face index escaped its checked cochain"
                        .to_owned(),
                })?;
            let right_value =
                right
                    .get(right_index)
                    .map_err(|_| TopologyError::InvalidCochain {
                        reason: "Alexander--Whitney back-face index escaped its checked cochain"
                            .to_owned(),
                    })?;
            values.push(ring.mul(&left_value, &right_value)?);
        }
        Ok(Vector::from_values(ring, values)?)
    }

    /// Construct a smaller chain model through verified discrete Morse theory.
    ///
    /// Matching uses only elementary free-face collapses, interspersed with
    /// deterministic critical-cell choices. The resulting signed Morse
    /// boundary is valid over every coefficient ring.
    pub fn morse_reduction<R: Ring>(
        &self,
        ring: R,
    ) -> Result<SimplicialMorseReduction<R>, TopologyError> {
        morse::reduce(self, ring)
    }

    /// Construct only the chain complex carried by the critical cells.
    pub fn morse_chain_complex<R: Ring>(&self, ring: R) -> Result<ChainComplex<R>, TopologyError> {
        Ok(self.morse_reduction(ring)?.into_reduced())
    }

    /// Return one Betti number over an exact field.
    ///
    /// This invariants-only path constructs and ranks only `d_n` and
    /// `d_(n+1)`.  It does not construct the full chain complex or explicit
    /// cycle and quotient bases; use [`Self::homology`] when those witnesses
    /// are needed.
    pub fn betti_number<R: Field>(&self, field: R, degree: usize) -> Result<usize, TopologyError> {
        let displayed_degree = i32::try_from(degree)
            .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
        let module_rank = self.count(degree);
        let rational_coefficients = TypeId::of::<R>() == TypeId::of::<RationalField>();
        let outgoing_rank = if degree == 0 || module_rank == 0 {
            0
        } else if rational_coefficients {
            self.sparse_integral_boundary(degree)?.integral_rank()
        } else {
            self.boundary_matrix(field.clone(), degree)?.rank()?
        };
        let incoming_rank = if self.top_dimension().is_some_and(|top| degree < top) {
            if rational_coefficients {
                self.sparse_integral_boundary(degree + 1)?.integral_rank()
            } else {
                self.boundary_matrix(field, degree + 1)?.rank()?
            }
        } else {
            0
        };
        field_homology_rank(displayed_degree, module_rank, outgoing_rank, incoming_rank)
    }

    fn sparse_integral_boundary(
        &self,
        degree: usize,
    ) -> Result<SparseMatrix<IntegerRing>, TopologyError> {
        let rows = degree.checked_sub(1).map_or(0, |lower| self.count(lower));
        let columns = self.count(degree);
        let mut entries = Vec::with_capacity(columns.saturating_mul(degree.saturating_add(1)));
        for (column, simplex) in self.simplices(degree).iter().enumerate() {
            for (omitted, face) in simplex.faces().iter().enumerate() {
                let row = self
                    .position(face)
                    .expect("downward closure contains every face");
                entries.push((
                    row,
                    column,
                    Integer::from(if omitted % 2 == 0 { 1 } else { -1 }),
                ));
            }
        }
        Ok(SparseMatrix::from_triplets(Z, rows, columns, entries)?)
    }

    /// Compute one field-homology group through the canonical chain complex.
    pub fn homology<R: Field>(
        &self,
        field: R,
        degree: usize,
    ) -> Result<HomologyGroup<R>, TopologyError> {
        let degree = i32::try_from(degree)
            .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
        Ok(self.chain_complex(field)?.homology(degree)?)
    }

    /// Compute field homology in every nonzero chain degree.
    pub fn graded_homology<R: Field>(&self, field: R) -> Result<GradedHomology<R>, TopologyError> {
        Ok(self.chain_complex(field)?.graded_homology()?)
    }

    /// Compute one integral homology group, including torsion.
    pub fn integral_homology(
        &self,
        degree: usize,
    ) -> Result<HomologyGroup<IntegerRing>, TopologyError> {
        let displayed_degree = i32::try_from(degree)
            .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
        let module_rank = self.count(degree);
        let invariants = self.homology_invariants(degree)?;
        let outgoing = self.boundary_matrix(Z, degree)?;
        let incoming = degree.checked_add(1).map_or_else(
            || Ok(Matrix::zeros(Z, module_rank, 0)),
            |upper| self.boundary_matrix(Z, upper),
        )?;
        Ok(HomologyGroup::from_integral_invariants(
            displayed_degree,
            false,
            outgoing,
            incoming,
            invariants.free_rank(),
            invariants.torsion_coefficients().to_vec(),
        ))
    }

    /// Compute integral homology as a lean finitely generated abelian group.
    ///
    /// Only sparse ranks and Smith invariant factors are retained. Use
    /// [`Self::integral_homology`] when the result must carry adjacent dense
    /// boundary matrices for an independent later verification.
    pub fn homology_invariants(
        &self,
        degree: usize,
    ) -> Result<FinitelyGeneratedAbelianGroup, TopologyError> {
        let displayed_degree = i32::try_from(degree)
            .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
        if self.total_cells() >= SIMPLICIAL_MORSE_MIN_CELLS {
            let mut groups = self.graded_integral_invariants_playbook()?;
            return Ok(groups
                .remove(&displayed_degree)
                .unwrap_or_else(|| FinitelyGeneratedAbelianGroup::from_invariants(0, Vec::new())));
        }
        let module_rank = self.count(degree);
        let outgoing = self.sparse_integral_boundary(degree)?;
        let incoming = degree.checked_add(1).map_or_else(
            || Ok(SparseMatrix::zeros(Z, module_rank, 0)),
            |upper| self.sparse_integral_boundary(upper),
        )?;
        Ok(integral_homology_invariants_from_sparse_boundaries(
            displayed_degree,
            module_rank,
            &outgoing,
            &incoming,
        )?)
    }

    /// Compute lean integral homology invariants in every chain degree.
    ///
    /// Each sparse boundary is constructed and Smith-reduced exactly once.
    /// The returned vector is indexed by homological degree and retains no
    /// boundary matrices or cycle representatives. The empty complex returns
    /// an empty vector.
    pub fn graded_homology_invariants(
        &self,
    ) -> Result<Vec<FinitelyGeneratedAbelianGroup>, TopologyError> {
        let Some(top) = self.top_dimension() else {
            return Ok(Vec::new());
        };
        let mut groups = self.graded_integral_invariants_playbook()?;
        (0..=top)
            .map(|degree| {
                let displayed_degree = i32::try_from(degree)
                    .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
                Ok(groups.remove(&displayed_degree).unwrap_or_else(|| {
                    FinitelyGeneratedAbelianGroup::from_invariants(0, Vec::new())
                }))
            })
            .collect()
    }

    fn graded_integral_invariants_playbook(
        &self,
    ) -> Result<BTreeMap<i32, FinitelyGeneratedAbelianGroup>, TopologyError> {
        let Some(top) = self.top_dimension() else {
            return Ok(BTreeMap::new());
        };
        if self.total_cells() >= SIMPLICIAL_MORSE_MIN_CELLS {
            if let Some(boundaries) = morse::integral_sparse_boundaries_when(self, |critical| {
                self.integral_homology_strategy(critical)
                    == IntegralHomologyStrategy::SimplicialMorseModel
            })? {
                return Ok(graded_homology_invariants_from_sparse_boundaries(
                    &boundaries,
                )?);
            }
        }

        let mut boundaries = BTreeMap::new();
        if top == 0 {
            boundaries.insert(0, self.sparse_integral_boundary(0)?);
        } else {
            for degree in 1..=top {
                let displayed_degree = i32::try_from(degree)
                    .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
                boundaries.insert(displayed_degree, self.sparse_integral_boundary(degree)?);
            }
        }
        Ok(graded_homology_invariants_from_sparse_boundaries(
            &boundaries,
        )?)
    }

    fn integral_homology_strategy(&self, critical: &[usize]) -> IntegralHomologyStrategy {
        if self.total_cells() < SIMPLICIAL_MORSE_MIN_CELLS {
            return IntegralHomologyStrategy::OrdinarySparseBoundaries;
        }
        let Some(top) = self.top_dimension() else {
            return IntegralHomologyStrategy::OrdinarySparseBoundaries;
        };
        let direct_work = (1..=top).fold(0usize, |work, degree| {
            work.saturating_add(self.count(degree - 1).saturating_mul(self.count(degree)))
        });
        let morse_work = (1..critical.len()).fold(0usize, |work, degree| {
            work.saturating_add(critical[degree - 1].saturating_mul(critical[degree]))
        });
        if morse_work.saturating_mul(4) < direct_work {
            IntegralHomologyStrategy::SimplicialMorseModel
        } else {
            IntegralHomologyStrategy::OrdinarySparseBoundaries
        }
    }

    /// Compute integral homology, including torsion, in every chain degree.
    pub fn graded_integral_homology(&self) -> Result<GradedHomology<IntegerRing>, TopologyError> {
        let groups = (0..self.simplices.len())
            .filter(|degree| self.count(*degree) != 0)
            .map(|degree| self.integral_homology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GradedHomology::from_groups(Z, groups))
    }
}

impl Serialize for SimplicialComplex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SimplicialComplexWire {
            type_id: Self::TYPE_ID.to_owned(),
            facets: self
                .facets()
                .into_iter()
                .map(|facet| facet.vertices().to_vec())
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialComplex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SimplicialComplexWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::from_facets(&wire.facets))
    }
}

pub(crate) fn field_homology_rank(
    degree: i32,
    module_rank: usize,
    outgoing_rank: usize,
    incoming_rank: usize,
) -> Result<usize, TopologyError> {
    module_rank
        .checked_sub(outgoing_rank)
        .and_then(|rank| rank.checked_sub(incoming_rank))
        .ok_or_else(|| {
            TopologyError::Homology(HomologyError::RankInconsistency {
                degree,
                module_rank,
                outgoing_rank,
                incoming_rank,
            })
        })
}

impl fmt::Debug for SimplicialComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplicialComplex")
            .field("dimension", &self.dimension())
            .field(
                "cell_counts",
                &self.simplices.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl fmt::Display for SimplicialComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimplicialComplex(dimension={}, cells={})",
            self.dimension(),
            self.total_cells()
        )
    }
}

impl SemanticType for SimplicialComplex {
    const TYPE_ID: &'static str = "noether/topology/SimplicialComplex@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        output.extend_from_slice(&(self.simplices.len() as u64).to_be_bytes());
        for simplices in &self.simplices {
            output.extend_from_slice(&(simplices.len() as u64).to_be_bytes());
            for simplex in simplices {
                output.extend_from_slice(&(simplex.vertices().len() as u64).to_be_bytes());
                for vertex in simplex.vertices() {
                    output.extend_from_slice(&vertex.to_be_bytes());
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SIMPLICIAL_COMPLEX_DESCRIPTION
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;
    use crate::{Q, Z};

    fn sphere(boundary_simplex_dimension: u32) -> SimplicialComplex {
        let all: Vec<u32> = (0..=boundary_simplex_dimension).collect();
        let facets = (0..=boundary_simplex_dimension as usize)
            .map(|index| {
                let mut facet = all.clone();
                facet.remove(index);
                facet
            })
            .collect::<Vec<_>>();
        SimplicialComplex::from_facets(&facets)
    }

    fn rp2() -> SimplicialComplex {
        SimplicialComplex::from_facets(&[
            vec![0, 1, 2],
            vec![0, 1, 4],
            vec![0, 2, 3],
            vec![0, 3, 5],
            vec![0, 4, 5],
            vec![1, 2, 5],
            vec![1, 3, 4],
            vec![1, 3, 5],
            vec![2, 3, 4],
            vec![2, 4, 5],
        ])
    }

    #[test]
    fn closure_indexing_and_facets_are_canonical() {
        let complex = SimplicialComplex::from_facets(&[vec![2, 0, 1], vec![1, 0, 2], vec![3]]);
        assert_eq!(
            (complex.count(0), complex.count(1), complex.count(2)),
            (4, 3, 1)
        );
        assert_eq!(complex.simplices(2)[0].vertices(), &[0, 1, 2]);
        assert_eq!(
            complex
                .facets()
                .iter()
                .map(|simplex| simplex.vertices())
                .collect::<Vec<_>>(),
            vec![&[0, 1, 2][..], &[3][..]]
        );
    }

    #[test]
    fn empty_and_zero_dimensional_complexes_have_distinct_dimensions() {
        let empty = SimplicialComplex::from_facets(&[]);
        let point = SimplicialComplex::from_facets(&[vec![0]]);

        assert!(empty.is_empty());
        assert_eq!(empty.dimension(), -1);
        assert!(!point.is_empty());
        assert_eq!(point.dimension(), 0);
    }

    #[test]
    fn signed_boundaries_form_a_complex() {
        for complex in [sphere(3), rp2()] {
            complex.chain_complex(Z).unwrap().verify().unwrap();
            complex.chain_complex(Q).unwrap().verify().unwrap();
        }
    }

    #[test]
    fn alexander_whitney_cup_products_use_the_fixed_simplex_bases() {
        let interval = SimplicialComplex::from_facets(&[vec![0, 1]]);
        let left = Vector::from_values(Z, [Integer::from(2), Integer::from(3)]).unwrap();
        let right = Vector::from_values(Z, [Integer::from(5)]).unwrap();
        let left_cup_right = interval.cup_product(0, &left, 1, &right).unwrap();
        let right_cup_left = interval.cup_product(1, &right, 0, &left).unwrap();
        assert_eq!(
            left_cup_right.values().collect::<Vec<_>>(),
            vec![Integer::from(10)]
        );
        assert_eq!(
            right_cup_left.values().collect::<Vec<_>>(),
            vec![Integer::from(15)]
        );

        let diagonal = interval.alexander_whitney_diagonal(1, 0).unwrap();
        assert_eq!(diagonal.len(), 2);
        assert_eq!(diagonal[0].0.vertices(), &[0]);
        assert_eq!(diagonal[0].1.vertices(), &[0, 1]);
        assert_eq!(diagonal[1].0.vertices(), &[0, 1]);
        assert_eq!(diagonal[1].1.vertices(), &[1]);
    }

    #[test]
    fn simplicial_cochains_are_transposed_signed_boundaries() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![0, 2], vec![1, 2]]);
        let cochains = circle.cochain_complex(Q).unwrap();
        cochains.verify().unwrap();
        assert_eq!(cochains.cohomology(0).unwrap().dimension(), 1);
        assert_eq!(cochains.cohomology(1).unwrap().dimension(), 1);
    }

    #[test]
    fn cup_products_reject_wrong_cochain_dimensions() {
        let edge = SimplicialComplex::from_facets(&[vec![0, 1]]);
        let wrong = Vector::from_values(Z, [Integer::from(1)]).unwrap();
        assert!(matches!(
            edge.cup_product(0, &wrong, 0, &wrong),
            Err(TopologyError::InvalidCochain { .. })
        ));
    }

    #[test]
    fn simplicial_morse_models_are_verified_and_preserve_homology() {
        for complex in [sphere(3), rp2()] {
            let integral = complex.morse_reduction(Z).unwrap();
            integral.verify().unwrap();
            assert!(integral.critical_cells().iter().sum::<usize>() < complex.total_cells());
            let reduced = integral.reduced().graded_homology_invariants().unwrap();
            for degree in 0..=complex.dimension() as i32 {
                assert_eq!(
                    reduced.get(&degree).cloned().unwrap_or_else(|| {
                        FinitelyGeneratedAbelianGroup::from_invariants(0, Vec::new())
                    }),
                    complex.homology_invariants(degree as usize).unwrap()
                );
            }

            let rational = complex.morse_reduction(Q).unwrap();
            rational.verify().unwrap();
            for degree in 0..=complex.dimension() as i32 {
                assert_eq!(
                    rational.reduced().betti_number(degree).unwrap(),
                    complex.betti_number(Q, degree as usize).unwrap()
                );
            }
        }
    }

    #[test]
    fn a_simplex_has_one_critical_cell() {
        for dimension in 1..=8_u32 {
            let simplex = SimplicialComplex::from_facets(&[(0..=dimension).collect()]);
            let reduction = simplex.morse_reduction(Z).unwrap();
            assert_eq!(reduction.critical_cells().iter().sum::<usize>(), 1);
            assert_eq!(reduction.matched_pairs() * 2 + 1, simplex.total_cells());
            reduction.verify().unwrap();
        }
    }

    #[test]
    fn simplicial_homology_playbook_uses_the_benchmarked_cutover() {
        let small = SimplicialComplex::from_facets(&[(0..=4).collect()]);
        let large = SimplicialComplex::from_facets(&[(0..=8).collect()]);
        let small_critical = morse::critical_cell_counts(&small);
        let large_critical = morse::critical_cell_counts(&large);
        assert_eq!(
            small.integral_homology_strategy(&small_critical),
            IntegralHomologyStrategy::OrdinarySparseBoundaries
        );
        assert_eq!(
            large.integral_homology_strategy(&large_critical),
            IntegralHomologyStrategy::SimplicialMorseModel
        );
    }

    #[test]
    fn isolated_vertices_survive_chain_construction() {
        let points = SimplicialComplex::from_facets(&[vec![0], vec![1], vec![2]]);
        let chains = points.chain_complex(Q).unwrap();
        assert_eq!(chains.degrees(), vec![0]);
        assert_eq!(chains.module_dimension(0), 3);
        assert_eq!(chains.homology(0).unwrap().dimension(), 3);
    }

    #[test]
    fn field_homology_distinguishes_torsion_witness() {
        let projective_plane = rp2();
        let f2 = FiniteField::prime(2).unwrap();
        assert_eq!(projective_plane.betti_number(Q, 1).unwrap(), 0);
        assert_eq!(projective_plane.betti_number(f2.clone(), 1).unwrap(), 1);
        assert_eq!(projective_plane.homology(Q, 1).unwrap().dimension(), 0);
        assert_eq!(projective_plane.homology(f2, 1).unwrap().dimension(), 1);
    }

    #[test]
    fn betti_numbers_match_full_homology_on_spheres() {
        for n in 1..=6u32 {
            let complex = sphere(n);
            for degree in 0..=complex.simplices.len() {
                assert_eq!(
                    complex.betti_number(Q, degree).unwrap(),
                    complex.homology(Q, degree).unwrap().dimension()
                );
            }
        }
    }

    #[test]
    fn integral_homology_names_projective_plane_torsion() {
        let projective_plane = rp2();
        let invariants = projective_plane.homology_invariants(1).unwrap();
        assert_eq!(invariants.free_rank(), 0);
        assert_eq!(invariants.torsion_coefficients(), &[Integer::from(2)]);
        invariants.verify().unwrap();

        let homology = projective_plane.graded_integral_homology().unwrap();
        assert_eq!(homology.get(0).unwrap().free_rank(), 1);
        assert_eq!(homology.get(1).unwrap().free_rank(), 0);
        assert_eq!(
            homology.get(1).unwrap().torsion_coefficients(),
            &[crate::Integer::from(2)]
        );
        assert!(homology.get(2).unwrap().torsion_coefficients().is_empty());
        homology.verify().unwrap();
    }

    #[test]
    fn graded_homology_invariants_reuse_boundaries_without_changing_results() {
        let empty = SimplicialComplex::from_facets(&[]);
        assert!(empty.graded_homology_invariants().unwrap().is_empty());

        for complex in [SimplicialComplex::from_facets(&[vec![7]]), sphere(2), rp2()] {
            let graded = complex.graded_homology_invariants().unwrap();
            assert_eq!(graded.len(), complex.dimension() as usize + 1);
            for (degree, group) in graded.iter().enumerate() {
                assert_eq!(group, &complex.homology_invariants(degree).unwrap());
                group.verify().unwrap();
            }
        }
    }

    #[test]
    fn euler_characteristic_of_spheres() {
        for n in 1..=6u32 {
            let complex = sphere(n);
            let dimension = n as i64 - 1;
            assert_eq!(
                complex.euler_characteristic(),
                1 + if dimension % 2 == 0 { 1 } else { -1 }
            );
        }
    }

    #[test]
    fn canonical_bytes_do_not_depend_on_facet_order() {
        let first = SimplicialComplex::from_facets(&[vec![0, 1, 2], vec![2, 3]]);
        let second = SimplicialComplex::from_facets(&[vec![3, 2], vec![2, 1, 0]]);
        assert_eq!(first, second);
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
    }
}
