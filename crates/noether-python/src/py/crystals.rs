// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of finite type-A Kashiwara crystals.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::combin::{CombinError, KashiwaraCrystal, Partition, VertexId};
use noether::SemanticType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::partition::PyPartition;
fn combin_error(error: CombinError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_partition(value: &Bound<'_, PyAny>) -> PyResult<Partition> {
    if let Ok(partition) = value.extract::<PyRef<'_, PyPartition>>() {
        return Ok(partition.core().clone());
    }
    Partition::new(value.extract::<Vec<usize>>()?).map_err(combin_error)
}

/// A thin Python handle around Noether's validated finite type-A crystal.
#[pyclass(
    module = "noether.combin",
    name = "KashiwaraCrystal",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyKashiwaraCrystal {
    inner: KashiwaraCrystal,
}

impl Serialize for PyKashiwaraCrystal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyKashiwaraCrystal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        KashiwaraCrystal::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyKashiwaraCrystal {
    pub const TYPE_ID: &'static str = <KashiwaraCrystal as SemanticType>::TYPE_ID;

    fn from_core(inner: KashiwaraCrystal) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKashiwaraCrystal {
    /// Construct the standard letter crystal `B(Lambda_1)` of type `A_rank`.
    #[staticmethod]
    fn type_a_letters(rank: usize) -> PyResult<Self> {
        KashiwaraCrystal::type_a_letters(rank)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    /// Construct the semistandard-tableau crystal `B(shape)` of type `A_rank`.
    #[staticmethod]
    fn type_a_tableaux(rank: usize, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = extract_partition(shape)?;
        KashiwaraCrystal::type_a_tableaux(rank, &shape)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize the Cartan matrix, weights, and lowering table.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("KashiwaraCrystal serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate every colored-string and weight invariant.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("KashiwaraCrystal deserialization failed: {error}"))
        })
    }

    /// Cartan rank, equal to the number of Kashiwara colors.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Number of vertices in the explicit colored graph.
    #[getter]
    fn vertex_count(&self) -> usize {
        self.inner.vertex_count()
    }

    /// Defining type-A Cartan matrix.
    #[getter]
    fn cartan_matrix(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let rows = self
            .inner
            .cartan_matrix()
            .iter()
            .map(|row| PyTuple::new(py, row.iter().copied()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    /// Weight in fundamental-weight coordinates.
    fn weight(&self, vertex: VertexId, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let weight = self.inner.weight(vertex).map_err(combin_error)?;
        Ok(PyTuple::new(py, weight.iter().copied())?.unbind())
    }

    /// Apply the Kashiwara raising operator `e_index`.
    fn e(&self, index: usize, vertex: VertexId) -> PyResult<Option<VertexId>> {
        self.inner.e(index, vertex).map_err(combin_error)
    }

    /// Apply the Kashiwara lowering operator `f_index`.
    fn f(&self, index: usize, vertex: VertexId) -> PyResult<Option<VertexId>> {
        self.inner.f(index, vertex).map_err(combin_error)
    }

    /// Return the upward and downward distances along one colored string.
    fn epsilon_phi(&self, index: usize, vertex: VertexId) -> PyResult<(usize, usize)> {
        self.inner.epsilon_phi(index, vertex).map_err(combin_error)
    }

    /// Vertices killed by every raising operator.
    fn highest_weight_vertices(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.highest_weight_vertices())?.unbind())
    }

    /// Weight-multiplicity character in deterministic weight order.
    fn character(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let entries = self
            .inner
            .character()
            .into_iter()
            .map(|(weight, multiplicity)| {
                let weight = PyTuple::new(py, weight)?.into_any();
                PyTuple::new(py, [weight, multiplicity.into_pyobject(py)?.into_any()])
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.unbind())
    }

    /// Form the right-to-left Kashiwara tensor product.
    fn tensor(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .tensor(&rhs.inner)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("cartan_type", format!("A{}", self.inner.rank()))?;
        summary.set_item("rank", self.inner.rank())?;
        summary.set_item("vertex_count", self.inner.vertex_count())?;
        summary.set_item(
            "highest_weight_count",
            self.inner.highest_weight_vertices().len(),
        )?;
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
