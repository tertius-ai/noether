// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of finite strong Steiner categories.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::collections::BTreeMap;

use noether::category::{
    AtomId, CategoricalHomologyComponentSlice, CategoricalHomologyPresentation, CategoryError,
    FiniteOmegaCategory, OrientedBasepoint, SignedBoundary, SteinerCanonicalLabel, SteinerCategory,
    SteinerCell, StreetNerveSkeleton, StreetSimplex,
};
use noether::SemanticType;
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::groups::{PyPermGroup, PyPermutation};
use super::homalg::{
    ChainComplexInner, GradedHomologyInner, PyChainComplex, PyFinitelyGeneratedAbelianGroup,
    PyGradedHomology,
};
use super::poset::PyPoset;

fn category_error(error: CategoryError) -> PyErr {
    match error {
        CategoryError::AtomOutOfRange { .. } | CategoryError::CellDimensionOutOfRange { .. } => {
            PyIndexError::new_err(error.to_string())
        }
        CategoryError::CellEnumerationLimitExceeded { .. }
        | CategoryError::ConstructionTooLarge { .. }
        | CategoryError::SizeOverflow => PyOverflowError::new_err(error.to_string()),
        CategoryError::LinearAlgebra(_)
        | CategoryError::Homology(_)
        | CategoryError::Poset(_)
        | CategoryError::Group(_)
        | CategoryError::Canonicalization(_)
        | CategoryError::StreetNerve(_) => PyRuntimeError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Degree-local atom identifier.
#[pyclass(module = "noether.category", name = "AtomId", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyAtomId {
    inner: AtomId,
}

#[pymethods]
impl PyAtomId {
    #[new]
    fn new(degree: usize, index: usize) -> Self {
        Self {
            inner: AtomId::new(degree, index),
        }
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "AtomId(degree={}, index={})",
            self.degree(),
            self.index()
        ))
    }
}

/// Disjoint signed support of one atomic boundary.
#[pyclass(
    module = "noether.category",
    name = "SignedBoundary",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySignedBoundary {
    inner: SignedBoundary,
}

#[pymethods]
impl PySignedBoundary {
    #[new]
    #[pyo3(signature = (negative=Vec::new(), positive=Vec::new()))]
    fn new(negative: Vec<usize>, positive: Vec<usize>) -> PyResult<Self> {
        SignedBoundary::new(&negative, &positive)
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    #[getter]
    fn negative(&self) -> Vec<usize> {
        self.inner.negative().to_vec()
    }

    #[getter]
    fn positive(&self) -> Vec<usize> {
        self.inner.positive().to_vec()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SignedBoundary(negative={:?}, positive={:?})",
            self.inner.negative(),
            self.inner.positive()
        ))
    }
}

/// Validated finite strong Steiner category.
#[pyclass(
    module = "noether.category",
    name = "SteinerCategory",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySteinerCategory {
    pub(crate) inner: SteinerCategory,
}

impl Serialize for PySteinerCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PySteinerCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        SteinerCategory::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PySteinerCategory {
    pub const TYPE_ID: &'static str = SteinerCategory::TYPE_ID;

    fn from_core(inner: SteinerCategory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySteinerCategory {
    #[new]
    fn new(atom_counts: Vec<usize>, boundaries: Vec<Vec<PySignedBoundary>>) -> PyResult<Self> {
        SteinerCategory::from_boundaries(
            atom_counts,
            boundaries
                .into_iter()
                .map(|degree| degree.into_iter().map(|boundary| boundary.inner).collect())
                .collect(),
        )
        .map(Self::from_core)
        .map_err(category_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    #[staticmethod]
    fn empty() -> Self {
        Self::from_core(SteinerCategory::empty())
    }

    #[staticmethod]
    fn point() -> Self {
        Self::from_core(SteinerCategory::point())
    }

    #[staticmethod]
    fn globe(dimension: usize) -> PyResult<Self> {
        SteinerCategory::globe(dimension)
            .map(Self::from_core)
            .map_err(category_error)
    }

    #[staticmethod]
    /// Construct the Street oriental of the requested dimension.
    fn oriental(dimension: usize) -> PyResult<Self> {
        SteinerCategory::oriental(dimension)
            .map(Self::from_core)
            .map_err(category_error)
    }

    #[staticmethod]
    fn cube(dimension: usize) -> PyResult<Self> {
        SteinerCategory::cube(dimension)
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn disjoint_union(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .disjoint_union(&other.inner)
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("SteinerCategory serialization failed: {error}"))
        })
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("SteinerCategory deserialization failed: {error}"))
        })
    }

    #[getter]
    fn dimension(&self) -> i64 {
        self.inner.dimension()
    }

    #[getter]
    fn atom_count(&self) -> usize {
        self.inner.atom_count()
    }

    #[getter]
    fn atom_counts(&self) -> Vec<usize> {
        self.inner.atom_counts().to_vec()
    }

    fn atom_count_in_degree(&self, degree: usize) -> usize {
        self.inner.atom_count_in_degree(degree)
    }

    fn boundary(&self, atom: &PyAtomId) -> PyResult<PySignedBoundary> {
        self.inner
            .boundary(atom.inner)
            .cloned()
            .map(|inner| PySignedBoundary { inner })
            .map_err(category_error)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(category_error)
    }

    fn precedence_poset(&self) -> PyResult<PyPoset> {
        self.inner
            .precedence_poset()
            .map(PyPoset::from_core)
            .map_err(category_error)
    }

    fn atom(&self, atom: &PyAtomId) -> PyResult<PySteinerCell> {
        self.inner
            .atom(atom.inner)
            .map(|inner| PySteinerCell { inner })
            .map_err(category_error)
    }

    fn cell(&self, dimension: usize, top_atoms: Vec<usize>) -> PyResult<PySteinerCell> {
        self.inner
            .cell(dimension, &top_atoms)
            .map(|inner| PySteinerCell { inner })
            .map_err(category_error)
    }

    fn cell_from_supports(
        &self,
        negative_supports: Vec<Vec<usize>>,
        positive_supports: Vec<Vec<usize>>,
    ) -> PyResult<PySteinerCell> {
        self.inner
            .cell_from_supports(negative_supports, positive_supports)
            .map(|inner| PySteinerCell { inner })
            .map_err(category_error)
    }

    fn exact_cells(&self, py: Python<'_>, dimension: usize) -> PyResult<Vec<PySteinerCell>> {
        py.detach(|| self.inner.exact_cells(dimension))
            .map(|cells| {
                cells
                    .into_iter()
                    .map(|inner| PySteinerCell { inner })
                    .collect()
            })
            .map_err(category_error)
    }

    fn cells(&self, py: Python<'_>, dimension: usize) -> PyResult<Vec<PySteinerCell>> {
        py.detach(|| self.inner.cells(dimension))
            .map(|cells| {
                cells
                    .into_iter()
                    .map(|inner| PySteinerCell { inner })
                    .collect()
            })
            .map_err(category_error)
    }

    fn exact_cell_counts(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        py.detach(|| self.inner.exact_cell_counts())
            .map_err(category_error)
    }

    fn mapping_category(
        &self,
        py: Python<'_>,
        source: &PySteinerCell,
        target: &PySteinerCell,
    ) -> PyResult<PyFiniteOmegaCategory> {
        py.detach(|| self.inner.mapping_category(&source.inner, &target.inner))
            .map(|inner| PyFiniteOmegaCategory { inner })
            .map_err(category_error)
    }

    fn component_poset(&self, py: Python<'_>) -> PyResult<PyPoset> {
        py.detach(|| self.inner.component_poset())
            .map(PyPoset::from_core)
            .map_err(category_error)
    }

    fn fundamental_poset(
        &self,
        py: Python<'_>,
        source: &PySteinerCell,
        target: &PySteinerCell,
    ) -> PyResult<PyPoset> {
        py.detach(|| self.inner.fundamental_poset(&source.inner, &target.inner))
            .map(PyPoset::from_core)
            .map_err(category_error)
    }

    fn monomorphic_oriented_basepoints(
        &self,
        py: Python<'_>,
        dimension: usize,
    ) -> PyResult<Vec<PyOrientedBasepoint>> {
        py.detach(|| self.inner.monomorphic_oriented_basepoints(dimension))
            .map(|values| {
                values
                    .into_iter()
                    .map(|inner| PyOrientedBasepoint { inner })
                    .collect()
            })
            .map_err(category_error)
    }

    fn homotopy_poset(&self, py: Python<'_>, basepoint: &PyOrientedBasepoint) -> PyResult<PyPoset> {
        py.detach(|| self.inner.homotopy_poset(&basepoint.inner))
            .map(PyPoset::from_core)
            .map_err(category_error)
    }

    fn truncate(&self, py: Python<'_>, dimension: usize) -> PyResult<Self> {
        py.detach(|| self.inner.truncate(dimension))
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn dual(&self, py: Python<'_>, dimensions: Vec<usize>) -> PyResult<Self> {
        py.detach(|| self.inner.dual(&dimensions))
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn canonical_label(&self, py: Python<'_>) -> PyResult<PySteinerCanonicalLabel> {
        py.detach(|| self.inner.canonical_label())
            .map(|inner| PySteinerCanonicalLabel { inner })
            .map_err(category_error)
    }

    fn canonical_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| self.inner.canonical_form())
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn is_isomorphic(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        py.detach(|| self.inner.is_isomorphic(&other.inner))
            .map_err(category_error)
    }

    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| self.inner.automorphism_group())
            .map(|inner| PyPermGroup { inner })
            .map_err(category_error)
    }

    fn underlying_chain_complex(&self, py: Python<'_>) -> PyResult<PyChainComplex> {
        py.detach(|| self.inner.underlying_chain_complex())
            .map(|complex| PyChainComplex {
                inner: ChainComplexInner::Integer(complex),
            })
            .map_err(category_error)
    }

    fn polygraphic_homology(&self, py: Python<'_>) -> PyResult<PyGradedHomology> {
        py.detach(|| self.inner.polygraphic_homology())
            .map(|homology| PyGradedHomology {
                inner: GradedHomologyInner::Integer(homology),
            })
            .map_err(category_error)
    }

    fn polygraphic_homology_invariants(
        &self,
        py: Python<'_>,
    ) -> PyResult<BTreeMap<i32, PyFinitelyGeneratedAbelianGroup>> {
        py.detach(|| self.inner.polygraphic_homology_invariants())
            .map(|groups| {
                groups
                    .into_iter()
                    .map(|(degree, inner)| (degree, PyFinitelyGeneratedAbelianGroup { inner }))
                    .collect()
            })
            .map_err(category_error)
    }

    fn gray_tensor(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| self.inner.gray_tensor(&other.inner))
            .map(Self::from_core)
            .map_err(category_error)
    }

    fn street_nerve_skeleton(
        &self,
        py: Python<'_>,
        max_degree: usize,
    ) -> PyResult<PyStreetNerveSkeleton> {
        py.detach(|| self.inner.street_nerve_skeleton(max_degree))
            .map(|inner| PyStreetNerveSkeleton { inner })
            .map_err(category_error)
    }

    fn categorical_homology(&self, py: Python<'_>) -> PyCategoricalHomologyPresentation {
        py.detach(|| PyCategoricalHomologyPresentation {
            inner: self.inner.categorical_homology(),
        })
    }

    fn reduced_categorical_homology(
        &self,
        py: Python<'_>,
        object: &PySteinerCell,
    ) -> PyResult<PyCategoricalHomologyPresentation> {
        py.detach(|| self.inner.reduced_categorical_homology(&object.inner))
            .map(|inner| PyCategoricalHomologyPresentation { inner })
            .map_err(category_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SteinerCategory(atom_counts={:?}, dimension={})",
            self.inner.atom_counts(),
            self.inner.dimension()
        ))
    }
}

/// One exact cell in a Steiner-category context.
#[pyclass(
    module = "noether.category",
    name = "SteinerCell",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySteinerCell {
    inner: SteinerCell,
}

#[pymethods]
impl PySteinerCell {
    #[getter]
    fn category(&self) -> PySteinerCategory {
        PySteinerCategory::from_core(self.inner.category().clone())
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn intrinsic_dimension(&self) -> usize {
        self.inner.intrinsic_dimension()
    }

    #[getter]
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    #[getter]
    fn top_atoms(&self) -> Vec<usize> {
        self.inner.top_atoms().to_vec()
    }

    #[getter]
    fn negative_supports(&self) -> Vec<Vec<usize>> {
        self.inner.negative_supports().to_vec()
    }

    #[getter]
    fn positive_supports(&self) -> Vec<Vec<usize>> {
        self.inner.positive_supports().to_vec()
    }

    fn identity(&self) -> Self {
        Self {
            inner: self.inner.identity(),
        }
    }

    #[getter]
    fn source(&self) -> PyResult<Self> {
        self.inner
            .source()
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    #[getter]
    fn target(&self) -> PyResult<Self> {
        self.inner
            .target()
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn source_at(&self, dimension: usize) -> PyResult<Self> {
        self.inner
            .source_at(dimension)
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn target_at(&self, dimension: usize) -> PyResult<Self> {
        self.inner
            .target_at(dimension)
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn compose_at(&self, next: &Self, dimension: usize) -> PyResult<Self> {
        self.inner
            .compose_at(&next.inner, dimension)
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn apply_automorphism(&self, automorphism: &PyPermutation) -> PyResult<Self> {
        self.inner
            .apply_automorphism(&automorphism.inner)
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SteinerCell(dimension={}, intrinsic_dimension={}, top_atoms={:?})",
            self.inner.dimension(),
            self.inner.intrinsic_dimension(),
            self.inner.top_atoms()
        ))
    }
}

/// Exact category canonicalization and symmetry evidence.
#[pyclass(
    module = "noether.category",
    name = "SteinerCanonicalLabel",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySteinerCanonicalLabel {
    inner: SteinerCanonicalLabel,
}

#[pymethods]
impl PySteinerCanonicalLabel {
    #[getter]
    fn category(&self) -> PySteinerCategory {
        PySteinerCategory::from_core(self.inner.category().clone())
    }

    #[getter]
    fn labeling(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.labeling().clone(),
        }
    }

    #[getter]
    fn degreewise_labeling(&self) -> Vec<Vec<usize>> {
        self.inner.degreewise_labeling().to_vec()
    }

    #[getter]
    fn encoding(&self) -> Vec<u64> {
        self.inner.encoding().to_vec()
    }

    #[getter]
    fn automorphism_group(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.automorphism_group().clone(),
        }
    }

    fn transport_cell(&self, cell: &PySteinerCell) -> PyResult<PySteinerCell> {
        self.inner
            .transport_cell(&cell.inner)
            .map(|inner| PySteinerCell { inner })
            .map_err(category_error)
    }

    fn verify(&self, original: &PySteinerCategory) -> bool {
        self.inner.verify(&original.inner)
    }
}

/// Explicit finite iterated mapping category.
#[pyclass(
    module = "noether.category",
    name = "FiniteOmegaCategory",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteOmegaCategory {
    inner: FiniteOmegaCategory,
}

#[pymethods]
impl PyFiniteOmegaCategory {
    #[getter]
    fn ambient(&self) -> PySteinerCategory {
        PySteinerCategory::from_core(self.inner.ambient().clone())
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    #[getter]
    fn dimension(&self) -> PyResult<i64> {
        self.inner.dimension().map_err(category_error)
    }

    fn objects(&self) -> PyResult<Vec<PySteinerCell>> {
        self.inner
            .objects()
            .map(|cells| {
                cells
                    .into_iter()
                    .map(|inner| PySteinerCell { inner })
                    .collect()
            })
            .map_err(category_error)
    }

    fn exact_cells(&self, py: Python<'_>, dimension: usize) -> PyResult<Vec<PySteinerCell>> {
        py.detach(|| self.inner.exact_cells(dimension))
            .map(|cells| {
                cells
                    .into_iter()
                    .map(|inner| PySteinerCell { inner })
                    .collect()
            })
            .map_err(category_error)
    }

    fn exact_cell_counts(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        py.detach(|| self.inner.exact_cell_counts())
            .map_err(category_error)
    }

    fn strong_steiner_model(&self, py: Python<'_>) -> PyResult<Option<PySteinerCategory>> {
        py.detach(|| self.inner.strong_steiner_model())
            .map(|model| model.map(PySteinerCategory::from_core))
            .map_err(category_error)
    }

    fn mapping_category(
        &self,
        py: Python<'_>,
        source: &PySteinerCell,
        target: &PySteinerCell,
    ) -> PyResult<Self> {
        py.detach(|| self.inner.mapping_category(&source.inner, &target.inner))
            .map(|inner| Self { inner })
            .map_err(category_error)
    }

    fn component_poset(&self, py: Python<'_>) -> PyResult<PyPoset> {
        py.detach(|| self.inner.component_poset())
            .map(PyPoset::from_core)
            .map_err(category_error)
    }
}

/// One oriented globe-boundary basepoint.
#[pyclass(
    module = "noether.category",
    name = "OrientedBasepoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyOrientedBasepoint {
    inner: OrientedBasepoint,
}

#[pymethods]
impl PyOrientedBasepoint {
    #[staticmethod]
    fn from_pairs(
        category: &PySteinerCategory,
        pairs: Vec<(PySteinerCell, PySteinerCell)>,
    ) -> PyResult<Self> {
        OrientedBasepoint::new(
            &category.inner,
            pairs
                .into_iter()
                .map(|(source, target)| (source.inner, target.inner))
                .collect(),
        )
        .map(|inner| Self { inner })
        .map_err(category_error)
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn homotopy_degree(&self) -> usize {
        self.inner.homotopy_degree()
    }

    #[getter]
    fn pairs(&self) -> Vec<(PySteinerCell, PySteinerCell)> {
        self.inner
            .pairs()
            .iter()
            .cloned()
            .map(|(source, target)| {
                (
                    PySteinerCell { inner: source },
                    PySteinerCell { inner: target },
                )
            })
            .collect()
    }

    #[getter]
    fn is_monomorphic(&self) -> bool {
        self.inner.is_monomorphic()
    }

    fn iterated_mapping_category(
        &self,
        py: Python<'_>,
        category: &PySteinerCategory,
    ) -> PyResult<PyFiniteOmegaCategory> {
        py.detach(|| self.inner.iterated_mapping_category(&category.inner))
            .map(|inner| PyFiniteOmegaCategory { inner })
            .map_err(category_error)
    }

    fn homotopy_poset(&self, py: Python<'_>, category: &PySteinerCategory) -> PyResult<PyPoset> {
        py.detach(|| self.inner.homotopy_poset(&category.inner))
            .map(PyPoset::from_core)
            .map_err(category_error)
    }
}

/// One Street-nerve simplex.
#[pyclass(
    module = "noether.category",
    name = "StreetSimplex",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyStreetSimplex {
    inner: StreetSimplex,
}

#[pymethods]
impl PyStreetSimplex {
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn images(&self) -> Vec<Vec<Vec<usize>>> {
        self.inner.images().to_vec()
    }

    #[getter]
    fn is_thin(&self) -> bool {
        self.inner.is_thin()
    }
}

/// Certified finite Street-nerve skeleton.
#[pyclass(
    module = "noether.category",
    name = "StreetNerveSkeleton",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyStreetNerveSkeleton {
    inner: StreetNerveSkeleton,
}

#[pymethods]
impl PyStreetNerveSkeleton {
    #[getter]
    fn source(&self) -> PySteinerCategory {
        PySteinerCategory::from_core(self.inner.source().clone())
    }

    #[getter]
    fn max_simplicial_degree(&self) -> usize {
        self.inner.max_simplicial_degree()
    }

    fn simplices(&self, degree: usize) -> Option<Vec<PyStreetSimplex>> {
        self.inner.simplices(degree).map(|simplices| {
            simplices
                .iter()
                .cloned()
                .map(|inner| PyStreetSimplex { inner })
                .collect()
        })
    }

    #[getter]
    fn simplex_counts(&self) -> Vec<usize> {
        self.inner.simplex_counts()
    }

    #[getter]
    fn thin_simplex_counts(&self) -> Vec<usize> {
        self.inner.thin_simplex_counts()
    }

    fn faces(&self, degree: usize, simplex: usize) -> Option<Vec<usize>> {
        self.inner.faces(degree, simplex).map(<[usize]>::to_vec)
    }

    fn degeneracies(&self, degree: usize, simplex: usize) -> Option<Vec<usize>> {
        self.inner
            .degeneracies(degree, simplex)
            .map(<[usize]>::to_vec)
    }

    fn verify(&self) -> bool {
        self.inner.verify()
    }
}

/// Exact categorical-homology presentation over the coefficient monoid N.
#[pyclass(
    module = "noether.category",
    name = "CategoricalHomologyPresentation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCategoricalHomologyPresentation {
    inner: CategoricalHomologyPresentation,
}

#[pymethods]
impl PyCategoricalHomologyPresentation {
    #[getter]
    fn coefficient_monoid(&self) -> &'static str {
        self.inner.coefficient_monoid()
    }

    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model()
    }

    #[getter]
    fn spectrum_model(&self) -> &'static str {
        self.inner.spectrum_model()
    }

    #[getter]
    fn infinite_loop_model(&self) -> &'static str {
        self.inner.infinite_loop_model()
    }

    #[getter]
    fn source(&self) -> PySteinerCategory {
        PySteinerCategory::from_core(self.inner.source().clone())
    }

    #[getter]
    fn basepoint(&self) -> Option<PySteinerCell> {
        self.inner
            .basepoint()
            .cloned()
            .map(|inner| PySteinerCell { inner })
    }

    #[getter]
    fn is_reduced(&self) -> bool {
        self.inner.is_reduced()
    }

    fn skeleton(&self, py: Python<'_>, max_degree: usize) -> PyResult<PyStreetNerveSkeleton> {
        py.detach(|| self.inner.skeleton(max_degree))
            .map(|inner| PyStreetNerveSkeleton { inner })
            .map_err(category_error)
    }

    fn component_poset_slice(
        &self,
        py: Python<'_>,
        particle_number: usize,
    ) -> PyResult<PyCategoricalHomologyComponentSlice> {
        py.detach(|| self.inner.component_poset_slice(particle_number))
            .map(|inner| PyCategoricalHomologyComponentSlice { inner })
            .map_err(category_error)
    }

    fn fundamental_poset(
        &self,
        py: Python<'_>,
        source_word: Vec<PySteinerCell>,
        target_word: Vec<PySteinerCell>,
    ) -> PyResult<PyPoset> {
        let source_word = source_word
            .into_iter()
            .map(|cell| cell.inner)
            .collect::<Vec<_>>();
        let target_word = target_word
            .into_iter()
            .map(|cell| cell.inner)
            .collect::<Vec<_>>();
        py.detach(|| self.inner.fundamental_poset(&source_word, &target_word))
            .map(PyPoset::from_core)
            .map_err(category_error)
    }

    fn particle_homotopy_poset(
        &self,
        py: Python<'_>,
        factors: Vec<PyOrientedBasepoint>,
    ) -> PyResult<PyPoset> {
        let factors = factors
            .into_iter()
            .map(|factor| factor.inner)
            .collect::<Vec<_>>();
        py.detach(|| self.inner.particle_homotopy_poset(&factors))
            .map(PyPoset::from_core)
            .map_err(category_error)
    }
}

/// One finite particle-number slice of categorical-homology components.
#[pyclass(
    module = "noether.category",
    name = "CategoricalHomologyComponentSlice",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCategoricalHomologyComponentSlice {
    inner: CategoricalHomologyComponentSlice,
}

#[pymethods]
impl PyCategoricalHomologyComponentSlice {
    #[getter]
    fn particle_number(&self) -> usize {
        self.inner.particle_number()
    }

    #[getter]
    fn object_multisets(&self) -> Vec<Vec<usize>> {
        self.inner.object_multisets().to_vec()
    }

    #[getter]
    fn poset(&self) -> PyPoset {
        PyPoset::from_core(self.inner.poset().clone())
    }
}
