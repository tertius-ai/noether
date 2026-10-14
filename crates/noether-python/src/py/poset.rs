// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public finite-poset facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use noether::combin::{
    BeatPoint, BeatPointKind, CoreReduction, Poset, PosetCanonicalLabel, PosetError, VertexId,
};

use crate::py::facade_scalar::integer_to_python;
use crate::py::graph::PyGraph;
use crate::py::groups::{PyPermGroup, PyPermutation};
use crate::py::homalg::PyFinitelyGeneratedAbelianGroup;
use crate::py::topology_facade::PySimplicialComplex;

fn poset_err(error: PosetError) -> PyErr {
    match error {
        PosetError::ElementOutOfRange { .. } => PyIndexError::new_err(error.to_string()),
        PosetError::OrderTooLarge { .. }
        | PosetError::RelationOutOfRange { .. }
        | PosetError::ReflexiveRelation { .. }
        | PosetError::CyclicRelations
        | PosetError::Graph(_) => PyValueError::new_err(error.to_string()),
        PosetError::Group(_)
        | PosetError::Topology(_)
        | PosetError::CanonicalSearchBudgetExceeded { .. }
        | PosetError::OrderComplexFacetLimitExceeded { .. }
        | PosetError::OrderComplexSimplexLimitExceeded { .. }
        | PosetError::AutomorphismVerificationFailed => PyRuntimeError::new_err(error.to_string()),
    }
}

/// One checked up- or down-beat-point certificate.
#[pyclass(
    module = "noether.combin",
    name = "BeatPoint",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyBeatPoint {
    inner: BeatPoint,
}

#[pymethods]
impl PyBeatPoint {
    /// Element which can be deleted.
    #[getter]
    fn element(&self) -> VertexId {
        self.inner.element()
    }

    /// `"down"` for a greatest strict predecessor, otherwise `"up"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            BeatPointKind::Down => "down",
            BeatPointKind::Up => "up",
        }
    }

    /// Neighbor certifying the beat-point condition.
    #[getter]
    fn witness(&self) -> VertexId {
        self.inner.witness()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "BeatPoint(element={}, kind='{}', witness={})",
            self.element(),
            self.kind(),
            self.witness()
        ))
    }
}

/// Exact Poset-owned canonical labeling and symmetry evidence.
#[pyclass(
    module = "noether.combin",
    name = "PosetCanonicalLabel",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPosetCanonicalLabel {
    inner: PosetCanonicalLabel,
}

#[pymethods]
impl PyPosetCanonicalLabel {
    /// Canonically relabeled poset.
    #[getter]
    fn poset(&self) -> PyPoset {
        PyPoset::from_core(self.inner.poset().clone())
    }

    /// Permutation from original element ids to canonical ids.
    #[getter]
    fn labeling(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.labeling().clone(),
        }
    }

    /// Principal-downset integers in canonical element order.
    #[getter]
    fn incidence_sequence(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .incidence_sequence()
            .iter()
            .map(|value| integer_to_python(py, value))
            .collect()
    }

    /// Complete automorphism group acting on the original element ids.
    #[getter]
    fn automorphism_group(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.automorphism_group().clone(),
        }
    }

    /// Verify the labeling and stored evidence against an original poset.
    fn verify(&self, original: &PyPoset) -> bool {
        self.inner.verify(&original.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PosetCanonicalLabel(order={}, automorphism_order={})",
            self.inner.poset().order(),
            self.inner.automorphism_group().order().to_decimal()
        ))
    }
}

/// Replayable beat-point reduction to a canonical core.
#[pyclass(
    module = "noether.combin",
    name = "CoreReduction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCoreReduction {
    inner: CoreReduction,
}

#[pymethods]
impl PyCoreReduction {
    /// Terminal core before its final canonical relabeling.
    #[getter]
    fn reduced(&self) -> PyPoset {
        PyPoset::from_core(self.inner.reduced().clone())
    }

    /// Canonical representative of the terminal core.
    #[getter]
    fn core(&self) -> PyPoset {
        PyPoset::from_core(self.inner.core().clone())
    }

    /// Beat-point deletions in replay order and original ids.
    #[getter]
    fn deletions(&self) -> Vec<PyBeatPoint> {
        self.inner
            .deletions()
            .iter()
            .copied()
            .map(|inner| PyBeatPoint { inner })
            .collect()
    }

    /// Map from reduced ids to retained original ids.
    #[getter]
    fn retained_vertices(&self) -> Vec<VertexId> {
        self.inner.retained_vertices().to_vec()
    }

    /// Final permutation from reduced ids to canonical-core ids.
    #[getter]
    fn canonical_labeling(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.canonical_labeling().clone(),
        }
    }

    /// Compute lean integral homology from the canonical core.
    fn graded_homology_invariants(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<PyFinitelyGeneratedAbelianGroup>> {
        py.detach(|| self.inner.graded_homology_invariants())
            .map(|groups| {
                groups
                    .into_iter()
                    .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
                    .collect()
            })
            .map_err(poset_err)
    }

    /// Replay and verify the reduction against its source poset.
    fn verify(&self, original: &PyPoset) -> bool {
        self.inner.verify(&original.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CoreReduction(core_order={}, deletions={})",
            self.inner.core().order(),
            self.inner.deletions().len()
        ))
    }
}

/// The thin Python handle around Noether's validated Rust [`Poset`].
///
/// Construction accepts strict generating relations on `0..n_elements`.
/// Redundant transitive relations and duplicates are removed, so `covers` is
/// always the directed Hasse graph. Python stores no parallel order relation or
/// caller-supplied claims about lattice or graded structure.
///
/// ```python
/// from noether.combin import Poset
///
/// diamond = Poset(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
/// assert diamond.leq(0, 3)
/// assert diamond.covers.edge_count == 4
/// ```
#[pyclass(module = "noether.combin", name = "Poset", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPoset {
    pub(crate) inner: Poset,
}

impl Serialize for PyPoset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyPoset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Poset::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyPoset {
    pub const TYPE_ID: &'static str = <Poset as noether::SemanticType>::TYPE_ID;

    pub(crate) fn from_core(inner: Poset) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPoset {
    /// Construct a finite poset from strict generating relation pairs.
    #[new]
    fn new(n_elements: usize, relations: Vec<Vec<i64>>) -> PyResult<Self> {
        let mut pairs = Vec::with_capacity(relations.len());
        for relation in relations {
            let [lower, upper] = relation.as_slice() else {
                return Err(PyValueError::new_err(format!(
                    "each poset relation must be a pair, got {relation:?}"
                )));
            };
            let lower = VertexId::try_from(*lower).map_err(|_| {
                PyValueError::new_err(format!(
                    "poset relation endpoint {lower} is not a nonnegative 32-bit element id"
                ))
            })?;
            let upper = VertexId::try_from(*upper).map_err(|_| {
                PyValueError::new_err(format!(
                    "poset relation endpoint {upper} is not a nonnegative 32-bit element id"
                ))
            })?;
            pairs.push((lower, upper));
        }
        Poset::new(n_elements, &pairs)
            .map(Self::from_core)
            .map_err(poset_err)
    }

    /// The stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize only the validated defining Hasse diagram.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
            .map_err(|error| PyValueError::new_err(format!("Poset serialization failed: {error}")))
    }

    /// Deserialize and revalidate a finite poset.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Poset deserialization failed: {error}"))
        })
    }

    /// Number of elements.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// The directed Hasse diagram as the public graph type.
    #[getter]
    fn covers(&self) -> PyGraph {
        PyGraph::from_core(self.inner.covers().clone())
    }

    /// Return whether `lower <= upper`.
    fn leq(&self, lower: VertexId, upper: VertexId) -> PyResult<bool> {
        self.inner.leq(lower, upper).map_err(poset_err)
    }

    /// Compute the exact labeling minimizing the principal-downset tuple.
    fn canonical_label(&self, py: Python<'_>) -> PyResult<PyPosetCanonicalLabel> {
        py.detach(|| self.inner.canonical_label())
            .map(|inner| PyPosetCanonicalLabel { inner })
            .map_err(poset_err)
    }

    /// Return the representative minimizing the principal-downset tuple.
    fn canonical_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| self.inner.canonical_form())
            .map(Self::from_core)
            .map_err(poset_err)
    }

    /// Compute the exact automorphism group on the element ids.
    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| self.inner.automorphism_group())
            .map(|inner| PyPermGroup { inner })
            .map_err(poset_err)
    }

    /// Strict predecessor counts indexed by element id.
    fn strict_lower_counts(&self) -> Vec<usize> {
        self.inner.strict_lower_counts()
    }

    /// Strict successor counts indexed by element id.
    fn strict_upper_counts(&self) -> Vec<usize> {
        self.inner.strict_upper_counts()
    }

    /// Minimal elements in ascending id order.
    fn minimal_elements(&self) -> Vec<VertexId> {
        self.inner.minimal_elements()
    }

    /// Maximal elements in ascending id order.
    fn maximal_elements(&self) -> Vec<VertexId> {
        self.inner.maximal_elements()
    }

    /// Number of strict comparable pairs.
    fn comparable_pair_count(&self) -> usize {
        self.inner.comparable_pair_count()
    }

    /// Weakly connected components of the Hasse diagram.
    fn connected_components(&self) -> Vec<Vec<VertexId>> {
        self.inner.connected_components()
    }

    /// Number of elements in a longest chain.
    fn height(&self) -> usize {
        self.inner.height()
    }

    /// Maximum antichain size.
    fn width(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.width())
    }

    /// Reverse every comparison while retaining element ids.
    fn opposite(&self) -> PyResult<Self> {
        self.inner
            .opposite()
            .map(Self::from_core)
            .map_err(poset_err)
    }

    /// Whether every pair has a meet and a join.
    fn is_lattice(&self, py: Python<'_>) -> bool {
        py.detach(|| self.inner.is_lattice())
    }

    /// Whether this poset is isomorphic to its opposite.
    fn is_self_dual(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.is_self_dual()).map_err(poset_err)
    }

    /// Current up- and down-beat-point certificates.
    fn beat_points(&self) -> Vec<PyBeatPoint> {
        self.inner
            .beat_points()
            .into_iter()
            .map(|inner| PyBeatPoint { inner })
            .collect()
    }

    /// Whether this poset has no beat points.
    fn is_core(&self) -> bool {
        self.inner.is_core()
    }

    /// Delete beat points and return replayable reduction evidence.
    fn core(&self, py: Python<'_>) -> PyResult<PyCoreReduction> {
        py.detach(|| self.inner.core())
            .map(|inner| PyCoreReduction { inner })
            .map_err(poset_err)
    }

    /// Compute lean integral homology after beat-point reduction.
    fn graded_homology_invariants(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<PyFinitelyGeneratedAbelianGroup>> {
        py.detach(|| self.inner.graded_homology_invariants())
            .map(|groups| {
                groups
                    .into_iter()
                    .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
                    .collect()
            })
            .map_err(poset_err)
    }

    /// Construct the order complex, preserving poset element ids.
    fn order_complex(&self, py: Python<'_>) -> PyResult<PySimplicialComplex> {
        py.detach(|| self.inner.order_complex())
            .map(PySimplicialComplex::from_core)
            .map_err(poset_err)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("order", self.inner.order())?;
        summary.set_item("cover_count", self.inner.covers().edge_count())?;
        Ok(summary.unbind())
    }

    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        &self,
        py: Python<'_>,
        include: Option<Bound<'_, PyAny>>,
        exclude: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        crate::py::repr::mimebundle(py, self.__repr__(), vec![], include, exclude)
    }
}
