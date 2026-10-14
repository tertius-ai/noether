// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite simplicial sets in canonical non-degenerate form.

use core::any::TypeId;
use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::homalg::{
    integral_homology_invariants_from_sparse_boundaries, ChainComplex, ChainReduction,
    FinitelyGeneratedAbelianGroup, GradedHomology, HomologyGroup,
};
use crate::linalg::{LinalgError, Matrix, SparseMatrix, Vector};
use crate::{
    Field, Integer, IntegerRing, NoetherError, RationalField, Ring, SemanticType, TypeDescription,
    Z,
};

use super::{field_homology_rank, TopologyError};

static SIMPLICIAL_SET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialSet@1",
    rust_path: "noether::topology::SimplicialSet",
    python_path: Some("noether.topology.SimplicialSet"),
};

/// One face of a non-degenerate simplex.
///
/// Every simplex of a simplicial set is uniquely a degeneracy of a
/// non-degenerate simplex.  A degenerate face therefore records both that
/// underlying simplex and the monotone surjection which produces the face.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum SimplicialFace {
    /// A non-degenerate simplex one dimension below, named by its index.
    Cell(usize),
    /// A degeneracy of a lower-dimensional non-degenerate simplex.
    Degenerate {
        /// Dimension of the underlying non-degenerate simplex.
        simplex_dimension: usize,
        /// Index of that simplex within its dimension.
        simplex: usize,
        /// Monotone surjection from the represented simplex to the underlying
        /// non-degenerate simplex, written as its values on ordered vertices.
        surjection: Vec<usize>,
    },
}

impl SimplicialFace {
    /// Construct a non-degenerate face one dimension below.
    pub const fn cell(index: usize) -> Self {
        Self::Cell(index)
    }

    /// Construct a degenerate face from its unique non-degenerate form.
    pub fn degenerate(
        simplex_dimension: usize,
        simplex: usize,
        surjection: impl Into<Vec<usize>>,
    ) -> Self {
        Self::Degenerate {
            simplex_dimension,
            simplex,
            surjection: surjection.into(),
        }
    }
}

/// Failure to construct or verify a finite simplicial set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimplicialSetError {
    /// A non-degenerate simplex has the wrong number of faces.
    FaceArity {
        /// Dimension of the simplex.
        dimension: usize,
        /// Index within that dimension.
        simplex: usize,
        /// Number of supplied faces.
        actual: usize,
        /// Required number of faces.
        expected: usize,
    },
    /// A face names a non-degenerate simplex which does not exist.
    MissingSimplex {
        /// Dimension of the simplex whose face is invalid.
        dimension: usize,
        /// Index of that simplex.
        simplex: usize,
        /// Face-map index.
        face: usize,
        /// Dimension named by the face.
        named_dimension: usize,
        /// Index named by the face.
        named_simplex: usize,
        /// Number of available simplices in the named dimension.
        available: usize,
    },
    /// A claimed degeneracy is not a canonical monotone surjection.
    InvalidDegeneracy {
        /// Dimension of the simplex whose face is invalid.
        dimension: usize,
        /// Index of that simplex.
        simplex: usize,
        /// Face-map index.
        face: usize,
        /// Precise failed condition.
        reason: String,
    },
    /// Two composites of face maps violate a simplicial identity.
    SimplicialIdentity {
        /// Dimension of the non-degenerate simplex.
        dimension: usize,
        /// Index of that simplex.
        simplex: usize,
        /// Smaller face-map index `i`.
        i: usize,
        /// Larger face-map index `j`.
        j: usize,
    },
}

impl fmt::Display for SimplicialSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FaceArity {
                dimension,
                simplex,
                actual,
                expected,
            } => write!(
                f,
                "{dimension}-simplex {simplex} has {actual} faces; expected {expected}"
            ),
            Self::MissingSimplex {
                dimension,
                simplex,
                face,
                named_dimension,
                named_simplex,
                available,
            } => write!(
                f,
                "d_{face} of {dimension}-simplex {simplex} names {named_dimension}-simplex {named_simplex}, but only {available} exist"
            ),
            Self::InvalidDegeneracy {
                dimension,
                simplex,
                face,
                reason,
            } => write!(
                f,
                "d_{face} of {dimension}-simplex {simplex} is not a valid degeneracy: {reason}"
            ),
            Self::SimplicialIdentity {
                dimension,
                simplex,
                i,
                j,
            } => write!(
                f,
                "face maps of {dimension}-simplex {simplex} violate d_{i} d_{j} = d_{} d_{i}",
                j - 1
            ),
        }
    }
}

impl std::error::Error for SimplicialSetError {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SimplexReference {
    simplex_dimension: usize,
    simplex: usize,
    surjection: Vec<usize>,
}

/// A finite simplicial set, stored by its non-degenerate simplices and faces.
///
/// `faces[n][x]` contains `d_0(x), ..., d_n(x)` for non-degenerate
/// `n`-simplex `x`.  Degenerate simplices are represented canonically rather
/// than stored, so the value stays finite.  Construction checks arities,
/// references, degeneracy maps, and every generating face identity.
///
/// # Examples
///
/// ```
/// use noether::topology::{SimplicialFace, SimplicialSet};
/// use noether::Q;
///
/// // The interval: two vertices and one edge, with d_0(e) = v_1 and
/// // d_1(e) = v_0.
/// let interval = SimplicialSet::new(vec![
///     vec![vec![], vec![]],
///     vec![vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)]],
/// ])?;
/// assert_eq!(interval.homology(Q, 0)?.dimension(), 1);
/// assert_eq!(interval.homology(Q, 1)?.dimension(), 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct SimplicialSet {
    pub(super) faces: Vec<Vec<Vec<SimplicialFace>>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplicialSetWire {
    #[serde(rename = "type")]
    type_id: String,
    faces: Vec<Vec<Vec<SimplicialFace>>>,
}

impl SimplicialSet {
    /// Construct from the faces of every non-degenerate simplex.
    pub fn new(mut faces: Vec<Vec<Vec<SimplicialFace>>>) -> Result<Self, TopologyError> {
        while faces.last().is_some_and(Vec::is_empty) {
            faces.pop();
        }
        let result = Self { faces };
        result.verify()?;
        Ok(result)
    }

    /// Serialize the complete non-degenerate presentation as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reverify a complete non-degenerate presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// The minimal simplicial circle: one vertex and one loop.
    pub fn circle() -> Self {
        Self {
            faces: vec![
                vec![vec![]],
                vec![vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)]],
            ],
        }
    }

    /// The minimal simplicial `n`-sphere.
    ///
    /// For positive `n`, this is the quotient of the standard `n`-simplex by
    /// its boundary: one vertex and one non-degenerate `n`-simplex.
    pub fn sphere(dimension: usize) -> Self {
        if dimension == 0 {
            return Self {
                faces: vec![vec![vec![], vec![]]],
            };
        }
        if dimension == 1 {
            return Self::circle();
        }
        let mut faces = vec![Vec::new(); dimension + 1];
        faces[0] = vec![vec![]];
        faces[dimension] = vec![vec![
            SimplicialFace::Degenerate {
                simplex_dimension: 0,
                simplex: 0,
                surjection: vec![0; dimension],
            };
            dimension + 1
        ]];
        Self { faces }
    }

    /// Whether there are no non-degenerate simplices.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Top dimension containing a non-degenerate simplex.
    ///
    /// The empty simplicial set has dimension `-1`.
    pub fn dimension(&self) -> i64 {
        self.top_dimension().map_or(-1, |dimension| {
            i64::try_from(dimension).expect("simplicial-set dimensions fit i64")
        })
    }

    fn top_dimension(&self) -> Option<usize> {
        self.faces.len().checked_sub(1)
    }

    /// Number of non-degenerate simplices in one dimension.
    pub fn count(&self, dimension: usize) -> usize {
        self.faces.get(dimension).map_or(0, Vec::len)
    }

    /// Total number of non-degenerate simplices.
    pub fn total_cells(&self) -> usize {
        self.faces.iter().map(Vec::len).sum()
    }

    /// Faces of one non-degenerate simplex, or `None` when it does not exist.
    pub fn faces_of(&self, dimension: usize, simplex: usize) -> Option<&[SimplicialFace]> {
        self.faces
            .get(dimension)
            .and_then(|level| level.get(simplex))
            .map(Vec::as_slice)
    }

    /// Euler characteristic from non-degenerate cell counts.
    pub fn euler_characteristic(&self) -> i64 {
        self.faces
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

    /// Normalized boundary over `F_2` as sparse columns of row indices.
    pub fn boundary_columns_f2(&self, degree: usize) -> Vec<Vec<u32>> {
        if degree == 0 || self.top_dimension().is_none_or(|top| degree > top) {
            return vec![Vec::new(); self.count(degree)];
        }
        self.faces[degree]
            .iter()
            .map(|faces| {
                let mut parity = vec![false; self.count(degree - 1)];
                for face in faces {
                    if let SimplicialFace::Cell(row) = face {
                        parity[*row] = !parity[*row];
                    }
                }
                parity
                    .iter()
                    .enumerate()
                    .filter_map(|(row, present)| present.then_some(row as u32))
                    .collect()
            })
            .collect()
    }

    /// Verify all references, degeneracy maps, and generating simplicial
    /// identities.
    pub fn verify(&self) -> Result<(), SimplicialSetError> {
        for (dimension, level) in self.faces.iter().enumerate() {
            for (simplex, faces) in level.iter().enumerate() {
                let expected = if dimension == 0 { 0 } else { dimension + 1 };
                if faces.len() != expected {
                    return Err(SimplicialSetError::FaceArity {
                        dimension,
                        simplex,
                        actual: faces.len(),
                        expected,
                    });
                }
                if dimension == 0 {
                    continue;
                }
                for (face, value) in faces.iter().enumerate() {
                    self.validate_face(dimension, simplex, face, value)?;
                }
            }
        }

        for dimension in 2..self.faces.len() {
            for (simplex, faces) in self.faces[dimension].iter().enumerate() {
                for i in 0..dimension {
                    for j in (i + 1)..=dimension {
                        let left = self
                            .face_of_reference(&self.reference(&faces[j], dimension - 1), i)
                            .ok_or(SimplicialSetError::SimplicialIdentity {
                                dimension,
                                simplex,
                                i,
                                j,
                            })?;
                        let right = self
                            .face_of_reference(&self.reference(&faces[i], dimension - 1), j - 1)
                            .ok_or(SimplicialSetError::SimplicialIdentity {
                                dimension,
                                simplex,
                                i,
                                j,
                            })?;
                        if left != right {
                            return Err(SimplicialSetError::SimplicialIdentity {
                                dimension,
                                simplex,
                                i,
                                j,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Signed normalized boundary `d_k` over an arbitrary ring.
    pub fn boundary_matrix<R: Ring>(
        &self,
        ring: R,
        degree: usize,
    ) -> Result<Matrix<R>, TopologyError> {
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
        let negative = ring.neg(&positive).map_err(LinalgError::from)?;
        for (column, faces) in self.faces.get(degree).into_iter().flatten().enumerate() {
            for (face, value) in faces.iter().enumerate() {
                let SimplicialFace::Cell(row) = value else {
                    continue;
                };
                let coefficient = if face % 2 == 0 { &positive } else { &negative };
                entries[*row][column] = ring
                    .add(&entries[*row][column], coefficient)
                    .map_err(LinalgError::from)?;
            }
        }
        Ok(Matrix::from_rows(ring, entries)?)
    }

    /// Construct the canonical normalized chain complex over `ring`.
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

    /// Construct normalized simplicial cochains over `ring`.
    pub fn cochain_complex<R: Ring>(&self, ring: R) -> Result<ChainComplex<R>, TopologyError> {
        let top_dimension = self.top_dimension().unwrap_or(0);
        let mut coboundaries = BTreeMap::new();
        for degree in 0..=top_dimension {
            let signed_degree = i32::try_from(degree)
                .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
            let upper = degree
                .checked_add(1)
                .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
            coboundaries.insert(
                signed_degree,
                self.boundary_matrix(ring.clone(), upper)?.transpose(),
            );
        }
        Ok(ChainComplex::from_coboundaries(ring, coboundaries)?)
    }

    /// Alexander--Whitney cup product on normalized simplicial cochains.
    ///
    /// Inputs and output use the nondegenerate simplex bases. A degenerate
    /// front or back face contributes zero. Inputs need not be cocycles.
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
        let left_removed = if right_degree == 0 {
            Vec::new()
        } else {
            (left_degree + 1..=degree).collect()
        };
        let right_removed = (0..left_degree).collect::<Vec<_>>();
        let ring = left.ring().clone();
        let mut output = Vec::with_capacity(self.count(degree));
        for simplex in 0..self.count(degree) {
            let Some(front) = self.normalized_iterated_face(degree, simplex, &left_removed) else {
                output.push(ring.zero());
                continue;
            };
            let Some(back) = self.normalized_iterated_face(degree, simplex, &right_removed) else {
                output.push(ring.zero());
                continue;
            };
            let front = left.get(front).map_err(|_| TopologyError::InvalidCochain {
                reason: "Alexander--Whitney front face escaped its checked cochain".to_owned(),
            })?;
            let back = right.get(back).map_err(|_| TopologyError::InvalidCochain {
                reason: "Alexander--Whitney back face escaped its checked cochain".to_owned(),
            })?;
            output.push(ring.mul(&front, &back)?);
        }
        Ok(Vector::from_values(ring, output)?)
    }

    /// Reduce normalized chains by deterministic universal-unit cancellation.
    ///
    /// The returned homological-algebra witness records each `1` or `-1`
    /// pivot and can be replayed over the supplied coefficient ring. This is
    /// deliberately a reduction of normalized chains rather than a second,
    /// simplicial-set-specific Morse implementation.
    pub fn normalized_chain_reduction<R: Ring>(
        &self,
        ring: R,
    ) -> Result<ChainReduction<R>, TopologyError> {
        Ok(self.chain_complex(ring)?.algebraic_morse_reduction()?)
    }

    /// Return one Betti number over an exact field.
    ///
    /// This ranks only the two adjacent normalized boundary matrices.  Use
    /// [`Self::homology`] when explicit cycle and quotient bases are needed.
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
        for (column, faces) in self.faces.get(degree).into_iter().flatten().enumerate() {
            for (face, value) in faces.iter().enumerate() {
                if let SimplicialFace::Cell(row) = value {
                    entries.push((
                        *row,
                        column,
                        Integer::from(if face % 2 == 0 { 1 } else { -1 }),
                    ));
                }
            }
        }
        Ok(SparseMatrix::from_triplets(Z, rows, columns, entries)?)
    }

    /// Compute one field-homology group through normalized chains.
    pub fn homology<R: Field>(
        &self,
        field: R,
        degree: usize,
    ) -> Result<HomologyGroup<R>, TopologyError> {
        let degree = i32::try_from(degree)
            .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
        Ok(self.chain_complex(field)?.homology(degree)?)
    }

    /// Compute field homology in every nonzero normalized chain degree.
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

    /// Compute integral normalized homology as a lean finitely generated
    /// abelian group.
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

    /// Compute integral homology in every normalized chain degree.
    pub fn graded_integral_homology(&self) -> Result<GradedHomology<IntegerRing>, TopologyError> {
        let groups = (0..self.faces.len())
            .filter(|degree| self.count(*degree) != 0)
            .map(|degree| self.integral_homology(degree))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GradedHomology::from_groups(Z, groups))
    }

    fn validate_face(
        &self,
        dimension: usize,
        simplex: usize,
        face: usize,
        value: &SimplicialFace,
    ) -> Result<(), SimplicialSetError> {
        let face_dimension = dimension - 1;
        let (named_dimension, named_simplex) = match value {
            SimplicialFace::Cell(index) => (face_dimension, *index),
            SimplicialFace::Degenerate {
                simplex_dimension,
                simplex: named_simplex,
                surjection,
            } => {
                if *simplex_dimension >= face_dimension {
                    return Err(SimplicialSetError::InvalidDegeneracy {
                        dimension,
                        simplex,
                        face,
                        reason: format!(
                            "underlying dimension {simplex_dimension} is not smaller than represented face dimension {face_dimension}"
                        ),
                    });
                }
                if surjection.len() != dimension {
                    return Err(SimplicialSetError::InvalidDegeneracy {
                        dimension,
                        simplex,
                        face,
                        reason: format!(
                            "surjection has {} values; expected {dimension}",
                            surjection.len()
                        ),
                    });
                }
                if !is_monotone_surjection(surjection, *simplex_dimension) {
                    return Err(SimplicialSetError::InvalidDegeneracy {
                        dimension,
                        simplex,
                        face,
                        reason: format!(
                            "{:?} is not a monotone surjection onto 0..={simplex_dimension}",
                            surjection
                        ),
                    });
                }
                (*simplex_dimension, *named_simplex)
            }
        };
        let available = self.count(named_dimension);
        if named_simplex >= available {
            return Err(SimplicialSetError::MissingSimplex {
                dimension,
                simplex,
                face,
                named_dimension,
                named_simplex,
                available,
            });
        }
        Ok(())
    }

    fn reference(&self, face: &SimplicialFace, face_dimension: usize) -> SimplexReference {
        match face {
            SimplicialFace::Cell(simplex) => SimplexReference {
                simplex_dimension: face_dimension,
                simplex: *simplex,
                surjection: (0..=face_dimension).collect(),
            },
            SimplicialFace::Degenerate {
                simplex_dimension,
                simplex,
                surjection,
            } => SimplexReference {
                simplex_dimension: *simplex_dimension,
                simplex: *simplex,
                surjection: surjection.clone(),
            },
        }
    }

    fn face_of_reference(
        &self,
        reference: &SimplexReference,
        removed: usize,
    ) -> Option<SimplexReference> {
        if removed >= reference.surjection.len() {
            return None;
        }
        let mut restricted = reference.surjection.clone();
        restricted.remove(removed);
        if is_monotone_surjection(&restricted, reference.simplex_dimension) {
            return Some(SimplexReference {
                simplex_dimension: reference.simplex_dimension,
                simplex: reference.simplex,
                surjection: restricted,
            });
        }

        if reference.simplex_dimension == 0 {
            return None;
        }
        let missing =
            (0..=reference.simplex_dimension).find(|value| !restricted.contains(value))?;
        let base_face = self
            .faces
            .get(reference.simplex_dimension)?
            .get(reference.simplex)?
            .get(missing)?;
        let base_reference = self.reference(base_face, reference.simplex_dimension - 1);
        let relabelled = restricted
            .iter()
            .map(|value| if *value < missing { *value } else { *value - 1 });
        let surjection = relabelled
            .map(|value| base_reference.surjection.get(value).copied())
            .collect::<Option<Vec<_>>>()?;
        Some(SimplexReference {
            simplex_dimension: base_reference.simplex_dimension,
            simplex: base_reference.simplex,
            surjection,
        })
    }

    pub(super) fn normalized_iterated_face(
        &self,
        dimension: usize,
        simplex: usize,
        removed: &[usize],
    ) -> Option<usize> {
        let mut reference = SimplexReference {
            simplex_dimension: dimension,
            simplex,
            surjection: (0..=dimension).collect(),
        };
        for &face in removed.iter().rev() {
            reference = self.face_of_reference(&reference, face)?;
        }
        let target_dimension = dimension.checked_sub(removed.len())?;
        (reference.simplex_dimension == target_dimension
            && reference
                .surjection
                .iter()
                .copied()
                .eq(0..=target_dimension))
        .then_some(reference.simplex)
    }
}

impl Serialize for SimplicialSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SimplicialSetWire {
            type_id: Self::TYPE_ID.to_owned(),
            faces: self.faces.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SimplicialSetWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.faces).map_err(serde::de::Error::custom)
    }
}

fn is_monotone_surjection(values: &[usize], target_dimension: usize) -> bool {
    values.first() == Some(&0)
        && values.last() == Some(&target_dimension)
        && values
            .windows(2)
            .all(|window| window[0] <= window[1] && window[1] <= window[0] + 1)
}

impl fmt::Debug for SimplicialSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplicialSet")
            .field("dimension", &self.dimension())
            .field(
                "non_degenerate_counts",
                &self.faces.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl fmt::Display for SimplicialSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimplicialSet(dimension={}, non_degenerate={})",
            self.dimension(),
            self.total_cells()
        )
    }
}

impl SemanticType for SimplicialSet {
    const TYPE_ID: &'static str = "noether/topology/SimplicialSet@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        output.extend_from_slice(&(self.faces.len() as u64).to_be_bytes());
        for level in &self.faces {
            output.extend_from_slice(&(level.len() as u64).to_be_bytes());
            for faces in level {
                output.extend_from_slice(&(faces.len() as u64).to_be_bytes());
                for face in faces {
                    match face {
                        SimplicialFace::Cell(simplex) => {
                            output.push(0);
                            output.extend_from_slice(&(*simplex as u64).to_be_bytes());
                        }
                        SimplicialFace::Degenerate {
                            simplex_dimension,
                            simplex,
                            surjection,
                        } => {
                            output.push(1);
                            output.extend_from_slice(&(*simplex_dimension as u64).to_be_bytes());
                            output.extend_from_slice(&(*simplex as u64).to_be_bytes());
                            output.extend_from_slice(&(surjection.len() as u64).to_be_bytes());
                            for value in surjection {
                                output.extend_from_slice(&(*value as u64).to_be_bytes());
                            }
                        }
                    }
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SIMPLICIAL_SET_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Q, Z};

    fn triangle() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(1)],
            ],
            vec![vec![
                SimplicialFace::Cell(2),
                SimplicialFace::Cell(1),
                SimplicialFace::Cell(0),
            ]],
        ])
        .unwrap()
    }

    #[test]
    fn malformed_presentations_fail_closed() {
        let wrong_arity =
            SimplicialSet::new(vec![vec![vec![]], vec![vec![SimplicialFace::Cell(0)]]]);
        assert!(matches!(
            wrong_arity,
            Err(TopologyError::SimplicialSet(
                SimplicialSetError::FaceArity { .. }
            ))
        ));

        let missing = SimplicialSet::new(vec![
            vec![vec![]],
            vec![vec![SimplicialFace::Cell(0), SimplicialFace::Cell(7)]],
        ]);
        assert!(matches!(
            missing,
            Err(TopologyError::SimplicialSet(
                SimplicialSetError::MissingSimplex { .. }
            ))
        ));

        let bad_surjection = SimplicialSet::new(vec![
            vec![vec![]],
            vec![],
            vec![vec![
                SimplicialFace::degenerate(0, 0, [0, 1]),
                SimplicialFace::degenerate(0, 0, [0, 0]),
                SimplicialFace::degenerate(0, 0, [0, 0]),
            ]],
        ]);
        assert!(matches!(
            bad_surjection,
            Err(TopologyError::SimplicialSet(
                SimplicialSetError::InvalidDegeneracy { .. }
            ))
        ));
    }

    #[test]
    fn face_identities_are_checked() {
        let invalid = SimplicialSet::new(vec![
            vec![vec![], vec![]],
            vec![
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(1)],
            ],
            vec![vec![
                SimplicialFace::Cell(0),
                SimplicialFace::Cell(1),
                SimplicialFace::Cell(2),
            ]],
        ]);
        assert!(matches!(
            invalid,
            Err(TopologyError::SimplicialSet(
                SimplicialSetError::SimplicialIdentity { .. }
            ))
        ));
    }

    #[test]
    fn faces_of_a_degenerate_edge_follow_simplicial_identities() {
        let interval = SimplicialSet::new(vec![
            vec![vec![], vec![]],
            vec![vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)]],
        ])
        .unwrap();
        let s0_edge = SimplexReference {
            simplex_dimension: 1,
            simplex: 0,
            surjection: vec![0, 0, 1],
        };

        let edge = SimplexReference {
            simplex_dimension: 1,
            simplex: 0,
            surjection: vec![0, 1],
        };
        assert_eq!(interval.face_of_reference(&s0_edge, 0), Some(edge.clone()));
        assert_eq!(interval.face_of_reference(&s0_edge, 1), Some(edge));
        assert_eq!(
            interval.face_of_reference(&s0_edge, 2),
            Some(SimplexReference {
                simplex_dimension: 0,
                simplex: 0,
                surjection: vec![0, 0],
            })
        );
    }

    #[test]
    fn empty_and_zero_dimensional_sets_have_distinct_dimensions() {
        let empty = SimplicialSet::new(Vec::new()).unwrap();
        let point = SimplicialSet::new(vec![vec![vec![]]]).unwrap();

        assert!(empty.is_empty());
        assert_eq!(empty.dimension(), -1);
        assert!(!point.is_empty());
        assert_eq!(point.dimension(), 0);
    }

    #[test]
    fn standard_triangle_is_contractible() {
        let triangle = triangle();
        triangle.verify().unwrap();
        triangle.chain_complex(Z).unwrap().verify().unwrap();
        for (degree, expected) in [(0, 1), (1, 0), (2, 0), (3, 0)] {
            assert_eq!(triangle.betti_number(Q, degree).unwrap(), expected);
            assert_eq!(triangle.homology(Q, degree).unwrap().dimension(), expected);
        }
    }

    #[test]
    fn normalized_chain_reduction_is_replayable() {
        let triangle = triangle();
        let reduction = triangle.normalized_chain_reduction(Z).unwrap();
        assert_eq!(reduction.source(), &triangle.chain_complex(Z).unwrap());
        assert_eq!(reduction.removed_pairs(), 3);
        assert_eq!(reduction.reduced().module_dimension(0), 1);
        assert_eq!(reduction.reduced().module_dimension(1), 0);
        assert_eq!(reduction.reduced().module_dimension(2), 0);
        reduction.verify().unwrap();

        let replayed = reduction
            .source()
            .reduction_from_cancellations(reduction.cancellations())
            .unwrap();
        assert_eq!(replayed, reduction);
    }

    #[test]
    fn minimal_circle_has_signed_zero_boundary() {
        let circle = SimplicialSet::circle();
        let boundary = circle.boundary_matrix(Z, 1).unwrap();
        assert_eq!(
            boundary.rows().collect::<Vec<_>>(),
            vec![vec![crate::Integer::from(0)]]
        );
        let invariants = circle.homology_invariants(1).unwrap();
        assert_eq!(invariants.free_rank(), 1);
        assert!(invariants.torsion_coefficients().is_empty());
        let homology = circle.graded_integral_homology().unwrap();
        assert_eq!(homology.get(0).unwrap().free_rank(), 1);
        assert_eq!(homology.get(1).unwrap().free_rank(), 1);
        homology.verify().unwrap();
    }

    #[test]
    fn minimal_spheres_have_the_expected_homology() {
        for dimension in 0..=6 {
            let sphere = SimplicialSet::sphere(dimension);
            sphere.verify().unwrap();
            sphere.chain_complex(Z).unwrap().verify().unwrap();
            let homology = sphere.graded_integral_homology().unwrap();
            if dimension == 0 {
                assert_eq!(homology.get(0).unwrap().free_rank(), 2);
            } else {
                assert_eq!(homology.get(0).unwrap().free_rank(), 1);
                assert_eq!(homology.get(dimension as i32).unwrap().free_rank(), 1);
            }
            assert_eq!(
                sphere.euler_characteristic(),
                1 + if dimension % 2 == 0 { 1 } else { -1 }
            );
        }
    }

    #[test]
    fn canonical_bytes_include_degeneracy_data() {
        let sphere = SimplicialSet::sphere(3);
        assert_eq!(sphere, SimplicialSet::sphere(3));
        assert_eq!(
            sphere.canonical_bytes().unwrap(),
            SimplicialSet::sphere(3).canonical_bytes().unwrap()
        );
        assert_ne!(
            sphere.canonical_bytes().unwrap(),
            SimplicialSet::sphere(2).canonical_bytes().unwrap()
        );
    }

    #[test]
    fn typed_json_round_trip_revalidates_faces_and_degeneracies() {
        for value in [triangle(), SimplicialSet::sphere(2)] {
            let json = value.to_json().unwrap();
            assert_eq!(SimplicialSet::from_json(&json).unwrap(), value);
        }
    }
}
