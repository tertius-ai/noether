// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public exact-matroid facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::combin::{Matroid, MatroidError, VertexId};
use noether::SemanticType;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFrozenSet, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::polynomial_facade::PyPolynomial;
fn matroid_error(error: MatroidError) -> PyErr {
    match error {
        MatroidError::PolynomialContext(_) | MatroidError::Polynomial(_) => {
            PyRuntimeError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// A thin Python handle around Noether's validated Rust [`Matroid`].
#[pyclass(module = "noether.combin", name = "Matroid", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMatroid {
    inner: Matroid,
}

impl Serialize for PyMatroid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyMatroid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Matroid::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyMatroid {
    pub const TYPE_ID: &'static str = <Matroid as SemanticType>::TYPE_ID;

    fn set_family(py: Python<'_>, values: Vec<Vec<VertexId>>) -> PyResult<Py<PyAny>> {
        let sets = values
            .into_iter()
            .map(|elements| PyFrozenSet::new(py, elements).map(Bound::into_any))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, sets)?.unbind().into_any())
    }
}

#[pymethods]
impl PyMatroid {
    /// Construct from a labeled ground set and the complete basis family.
    #[staticmethod]
    fn from_bases(ground_set: Vec<VertexId>, bases: Vec<Vec<VertexId>>) -> PyResult<Self> {
        Matroid::from_bases(ground_set, bases)
            .map(|inner| Self { inner })
            .map_err(matroid_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize only the canonical ground set and complete basis family.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("Matroid serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate every matroid axiom.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Matroid deserialization failed: {error}"))
        })
    }

    /// Canonical ordered element identifiers.
    #[getter]
    fn ground_set(&self) -> Vec<VertexId> {
        self.inner.ground_set().to_vec()
    }

    /// Complete canonical basis family.
    #[getter]
    fn bases(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::set_family(py, self.inner.bases())
    }

    /// Rank of a subset of the ground set.
    fn rank(&self, subset: Vec<VertexId>) -> PyResult<usize> {
        self.inner.rank(&subset).map_err(matroid_error)
    }

    /// Minimal dependent subsets.
    #[getter]
    fn circuits(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::set_family(py, self.inner.circuits())
    }

    /// Delete one element while preserving the remaining labels.
    fn delete(&self, element: VertexId) -> PyResult<Self> {
        self.inner
            .delete(element)
            .map(|inner| Self { inner })
            .map_err(matroid_error)
    }

    /// Contract one element while preserving the remaining labels.
    fn contract(&self, element: VertexId) -> PyResult<Self> {
        self.inner
            .contract(element)
            .map(|inner| Self { inner })
            .map_err(matroid_error)
    }

    /// Exact Tutte polynomial in `Z[x, y]`.
    fn tutte_polynomial(&self, py: Python<'_>) -> PyResult<PyPolynomial> {
        py.detach(|| self.inner.tutte_polynomial())
            .map(PyPolynomial::from_integer)
            .map_err(matroid_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("order", self.inner.ground_set().len())?;
        summary.set_item(
            "rank",
            self.inner
                .rank(self.inner.ground_set())
                .map_err(matroid_error)?,
        )?;
        summary.set_item("basis_count", self.inner.bases().len())?;
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
