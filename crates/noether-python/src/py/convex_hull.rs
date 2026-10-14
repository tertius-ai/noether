// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of Noether's exact bounded-polytope facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::geometry::{convex_hull, GeometryError, Polytope};
use noether::{Rational, SemanticType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::facade_scalar::{extract_rational, rational_to_python};

fn geometry_error(error: GeometryError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_rows(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Rational>>> {
    let rows: Vec<Vec<Bound<'_, PyAny>>> = value.extract()?;
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| extract_rational(py, &entry))
                .collect()
        })
        .collect()
}

fn extract_values(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<Rational>> {
    let values: Vec<Bound<'_, PyAny>> = value.extract()?;
    values
        .into_iter()
        .map(|entry| extract_rational(py, &entry))
        .collect()
}

fn row_to_python(py: Python<'_>, row: &[Rational]) -> PyResult<Vec<Py<PyAny>>> {
    row.iter()
        .map(|coordinate| rational_to_python(py, coordinate))
        .collect()
}

fn rows_to_python(py: Python<'_>, rows: &[Vec<Rational>]) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    rows.iter().map(|row| row_to_python(py, row)).collect()
}

/// A nonempty bounded convex polytope with exact rational coordinates.
#[pyclass(module = "noether.geometry", name = "Polytope", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyPolytope {
    inner: Polytope,
}

impl PyPolytope {
    fn new(inner: Polytope) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPolytope {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Polytope::TYPE_ID
    }

    /// Construct the exact convex hull of rational vertices.
    #[staticmethod]
    fn from_vertices(py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<Self> {
        let points = extract_rows(py, points)?;
        py.detach(move || {
            Ok(Self::new(
                Polytope::from_vertices(&points).map_err(geometry_error)?,
            ))
        })
    }

    /// Construct a bounded polytope from `normal_i · x <= rhs_i`.
    #[staticmethod]
    fn from_halfspaces(
        py: Python<'_>,
        normals: &Bound<'_, PyAny>,
        rhs: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let normals = extract_rows(py, normals)?;
        let rhs = extract_values(py, rhs)?;
        py.detach(move || {
            Ok(Self::new(
                Polytope::from_halfspaces(&normals, &rhs).map_err(geometry_error)?,
            ))
        })
    }

    /// Deserialize and recompute every derived invariant.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self::new(
            Polytope::from_json(value).map_err(geometry_error)?,
        ))
    }

    /// Serialize exact defining vertices and semantic identity.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(geometry_error)
    }

    /// Ambient coordinate dimension.
    #[getter]
    fn ambient_dimension(&self) -> usize {
        self.inner.ambient_dimension()
    }

    /// Affine dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Whether the affine hull is the whole ambient space.
    #[getter]
    fn is_full_dimensional(&self) -> bool {
        self.inner.is_full_dimensional()
    }

    /// Exact vertices as `fractions.Fraction` values.
    #[getter]
    fn vertices(&self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        rows_to_python(py, self.inner.vertices())
    }

    /// Exact ambient inequalities `(normal, rhs)`.
    #[getter]
    fn facets(&self, py: Python<'_>) -> PyResult<Vec<(Vec<Py<PyAny>>, Py<PyAny>)>> {
        self.inner
            .facets()
            .map_err(geometry_error)?
            .iter()
            .map(|facet| {
                Ok((
                    row_to_python(py, &facet.normal)?,
                    rational_to_python(py, &facet.rhs)?,
                ))
            })
            .collect()
    }

    /// Vertex indices lying on each facet.
    #[getter]
    fn vertex_facet_incidence(&self) -> Vec<Vec<usize>> {
        self.inner.vertex_facet_incidence().to_vec()
    }

    /// Face counts `(f_-1, f_0, ..., f_dimension)`.
    #[getter]
    fn f_vector(&self) -> Vec<usize> {
        self.inner.f_vector()
    }

    /// Faces as `(dimension, vertex_indices)`, including the empty face.
    fn face_lattice(&self, py: Python<'_>) -> Vec<(isize, Vec<usize>)> {
        py.detach(|| {
            self.inner
                .face_lattice()
                .faces
                .into_iter()
                .map(|face| (face.dim, face.vertices))
                .collect()
        })
    }

    /// Hasse covering pairs indexing `face_lattice()`.
    fn face_lattice_covers(&self, py: Python<'_>) -> Vec<(usize, usize)> {
        py.detach(|| self.inner.face_lattice().covers)
    }

    /// Whether an exact ambient point is a vertex.
    fn is_vertex(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner
            .is_vertex(&extract_values(py, point)?)
            .map_err(geometry_error)
    }

    /// Whether an exact ambient inequality defines a facet.
    fn is_facet(
        &self,
        py: Python<'_>,
        normal: &Bound<'_, PyAny>,
        rhs: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.inner
            .is_facet(&extract_values(py, normal)?, &extract_rational(py, rhs)?)
            .map_err(geometry_error)
    }

    /// Recompute the exact hull and all derived combinatorics.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(geometry_error))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", Polytope::TYPE_ID)?;
        summary.set_item("dimension", self.inner.dimension())?;
        summary.set_item("ambient_dimension", self.inner.ambient_dimension())?;
        summary.set_item("vertices", self.inner.vertices().len())?;
        summary.set_item("facets", self.inner.vertex_facet_incidence().len())?;
        Ok(summary.unbind())
    }
}

/// Canonical exact convex-hull entry point for the Python boundary.
#[pyfunction(name = "_facade_convex_hull")]
pub fn facade_convex_hull(py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<PyPolytope> {
    let points = extract_rows(py, points)?;
    py.detach(|| convex_hull(&points))
        .map(PyPolytope::new)
        .map_err(geometry_error)
}
