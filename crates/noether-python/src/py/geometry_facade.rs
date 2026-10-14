// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical Python projection of the public Euclidean-geometry facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::geometry::{
    delaunay, regular_triangulation, GeometryError, Metric, PointCloud, Triangulation,
};
use noether::SemanticType;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn geometry_error(error: GeometryError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn point_cloud_error(error: String) -> PyErr {
    PyValueError::new_err(format!("invalid point cloud: {error}"))
}

/// A checked finite point cloud in Euclidean space.
#[pyclass(
    module = "noether.geometry",
    name = "PointCloud",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPointCloud {
    pub(crate) inner: PointCloud,
}

#[pymethods]
impl PyPointCloud {
    /// Construct from finite coordinate rows of one common dimension.
    #[new]
    fn new(points: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: PointCloud::new(&points).map_err(point_cloud_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PointCloud::TYPE_ID
    }

    /// Serialize the finite coordinate rows.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| {
            PyValueError::new_err(format!("PointCloud serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate dimensions, finiteness, and signed zeros.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value)
            .map(|inner| Self { inner })
            .map_err(|error| {
                PyValueError::new_err(format!("PointCloud deserialization failed: {error}"))
            })
    }

    /// Number of sites.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cloud has no sites.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Coordinate count of each site.
    #[getter]
    fn ambient_dimension(&self) -> usize {
        self.inner.ambient_dim()
    }

    /// Sites in input order.
    #[getter]
    fn points(&self) -> Vec<Vec<f64>> {
        self.inner.to_rows()
    }

    /// Coordinates of one site.
    fn __getitem__(&self, index: isize) -> PyResult<Vec<f64>> {
        let len = self.inner.len() as isize;
        let index = if index < 0 { index + len } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("PointCloud index out of range"));
        }
        Ok(self.inner.point(index as usize).to_vec())
    }

    /// Euclidean distance between two indexed sites.
    fn distance(&self, i: usize, j: usize) -> PyResult<f64> {
        let len = self.inner.len();
        if i >= len || j >= len {
            return Err(PyIndexError::new_err(format!(
                "point indices ({i}, {j}) are outside 0..{len}"
            )));
        }
        Ok(self.inner.distance(i, j))
    }

    /// Smallest radius about an input site containing the whole cloud.
    fn enclosing_radius(&self) -> f64 {
        Metric::enclosing_radius(&self.inner)
    }

    /// Dimension of the affine hull occupied by the sites.
    fn affine_rank(&self) -> usize {
        self.inner.affine_rank()
    }

    /// Re-express the cloud isometrically in its affine-hull coordinates.
    fn project_to_affine_hull(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self
                    .inner
                    .project_to_affine_hull()
                    .map_err(point_cloud_error)?,
            })
        })
    }

    /// Exact-predicate Delaunay triangulation.
    fn delaunay(&self, py: Python<'_>) -> PyResult<PyTriangulation> {
        py.detach(|| PyTriangulation::from_delaunay(&self.inner))
    }

    /// Exact-predicate regular triangulation with one finite weight per site.
    fn regular_triangulation(
        &self,
        py: Python<'_>,
        weights: Vec<f64>,
    ) -> PyResult<PyTriangulation> {
        py.detach(|| PyTriangulation::from_regular(&self.inner, &weights))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PointCloud(points={}, ambient_dimension={}, affine_rank={})",
                self.inner.len(),
                self.inner.ambient_dim(),
                self.inner.affine_rank()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", PointCloud::TYPE_ID)?;
        summary.set_item("points", self.inner.len())?;
        summary.set_item("ambient_dimension", self.inner.ambient_dim())?;
        summary.set_item("affine_rank", self.inner.affine_rank())?;
        Ok(summary.unbind())
    }
}

/// A Delaunay or regular triangulation backed directly by `geom-core`.
#[pyclass(
    module = "noether.geometry",
    name = "Triangulation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyTriangulation {
    pub(crate) inner: Triangulation,
}

impl PyTriangulation {
    fn from_delaunay(points: &PointCloud) -> PyResult<Self> {
        Ok(Self {
            inner: delaunay(points).map_err(geometry_error)?,
        })
    }

    fn from_regular(points: &PointCloud, weights: &[f64]) -> PyResult<Self> {
        Ok(Self {
            inner: regular_triangulation(points, weights).map_err(geometry_error)?,
        })
    }

    fn canonical_cells(&self) -> Vec<Vec<u32>> {
        let mut cells = self
            .inner
            .cells()
            .map(|cell| {
                let mut cell = cell.to_vec();
                cell.sort_unstable();
                cell
            })
            .collect::<Vec<_>>();
        cells.sort_unstable();
        cells
    }
}

#[pymethods]
impl PyTriangulation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Triangulation::TYPE_ID
    }

    /// Serialize the checked sites and optional regular weights.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| {
            PyValueError::new_err(format!("Triangulation serialization failed: {error}"))
        })
    }

    /// Deserialize by reconstructing the deterministic triangulation.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value)
            .map(|inner| Self { inner })
            .map_err(|error| {
                PyValueError::new_err(format!("Triangulation deserialization failed: {error}"))
            })
    }

    /// Affine dimension actually triangulated.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dim()
    }

    /// Short dimension alias retained for standard triangulation notation.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Sites in input order.
    #[getter]
    fn points(&self) -> Vec<Vec<f64>> {
        self.inner.points().to_vec()
    }

    /// Maximal simplices as vertex-index lists, in canonical order.
    #[getter]
    fn simplices(&self) -> Vec<Vec<u32>> {
        self.canonical_cells()
    }

    /// Conventional geometry alias for `simplices`.
    #[getter]
    fn cells(&self) -> Vec<Vec<u32>> {
        self.canonical_cells()
    }

    /// Number of maximal simplices.
    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.n_cells()
    }

    /// Whether this is a regular rather than ordinary Delaunay triangulation.
    #[getter]
    fn weighted(&self) -> bool {
        self.inner.weights().is_some()
    }

    /// Regular weights, or `None` for an ordinary Delaunay triangulation.
    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights().map(<[f64]>::to_vec)
    }

    /// Circumcentre and radius of every maximal simplex.
    fn circumcentres(&self) -> Vec<(Vec<f64>, f64)> {
        self.canonical_cells()
            .iter()
            .filter_map(|cell| {
                let rows = cell
                    .iter()
                    .map(|vertex| &self.inner.points()[*vertex as usize][..])
                    .collect::<Vec<_>>();
                let weights = self.inner.weights().map(|all_weights| {
                    cell.iter()
                        .map(|vertex| all_weights[*vertex as usize])
                        .collect::<Vec<_>>()
                });
                noether::geometry::circumsphere(&rows, weights.as_deref()).map(|ball| {
                    let radius = ball.radius();
                    (ball.center, radius)
                })
            })
            .collect()
    }

    /// Recompute the deterministic checked construction and compare its value.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify())
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Triangulation(dimension={}, points={}, simplices={}{})",
                self.inner.dim(),
                self.inner.points().len(),
                self.inner.n_cells(),
                if self.inner.weights().is_some() {
                    ", weighted"
                } else {
                    ""
                }
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", Triangulation::TYPE_ID)?;
        summary.set_item("dimension", self.inner.dim())?;
        summary.set_item("points", self.inner.points().len())?;
        summary.set_item("simplices", self.inner.n_cells())?;
        summary.set_item("weighted", self.inner.weights().is_some())?;
        Ok(summary.unbind())
    }
}

/// Canonical Delaunay entry point for the Python boundary.
#[pyfunction(name = "_facade_delaunay")]
pub fn facade_delaunay(points: &PyPointCloud) -> PyResult<PyTriangulation> {
    PyTriangulation::from_delaunay(&points.inner)
}

/// Canonical regular-triangulation entry point for the Python boundary.
#[pyfunction(name = "_facade_regular_triangulation")]
pub fn facade_regular_triangulation(
    points: &PyPointCloud,
    weights: Vec<f64>,
) -> PyResult<PyTriangulation> {
    PyTriangulation::from_regular(&points.inner, &weights)
}
