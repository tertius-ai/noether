// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of integer partitions and compositions.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::combin::{CombinError, Composition, Partition};
use noether::SemanticType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

fn combin_error(error: CombinError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// A thin Python handle around Noether's canonical integer partition.
#[pyclass(module = "noether.combin", name = "Partition", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPartition {
    inner: Partition,
}

impl Serialize for PyPartition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyPartition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Partition::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyPartition {
    pub const TYPE_ID: &'static str = <Partition as SemanticType>::TYPE_ID;

    pub fn from_core(inner: Partition) -> Self {
        Self { inner }
    }

    pub(crate) fn core(&self) -> &Partition {
        &self.inner
    }
}

#[pymethods]
impl PyPartition {
    /// Construct from weakly-decreasing positive parts.
    #[new]
    fn new(parts: Vec<usize>) -> PyResult<Self> {
        Partition::new(parts)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize only the canonical parts.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("Partition serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate the partition.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Partition deserialization failed: {error}"))
        })
    }

    /// Canonical weakly-decreasing parts.
    #[getter]
    fn parts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.parts().iter().copied())?.unbind())
    }

    /// Integer being partitioned.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Transpose the Young diagram.
    fn conjugate(&self) -> Self {
        Self::from_core(self.inner.conjugate())
    }

    /// Hook lengths in row-major Young-diagram order.
    fn hook_lengths(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.hook_lengths())?.unbind())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("size", self.inner.size())?;
        summary.set_item("length", self.inner.parts().len())?;
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

/// A thin Python handle around Noether's canonical integer composition.
#[pyclass(
    module = "noether.combin",
    name = "Composition",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyComposition {
    inner: Composition,
}

impl Serialize for PyComposition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyComposition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Composition::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyComposition {
    pub const TYPE_ID: &'static str = <Composition as SemanticType>::TYPE_ID;
}

#[pymethods]
impl PyComposition {
    /// Construct from ordered positive parts.
    #[new]
    fn new(parts: Vec<usize>) -> PyResult<Self> {
        Composition::new(parts)
            .map(|inner| Self { inner })
            .map_err(combin_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize only the canonical parts.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("Composition serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate the composition.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Composition deserialization failed: {error}"))
        })
    }

    /// Ordered positive parts.
    #[getter]
    fn parts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.parts().iter().copied())?.unbind())
    }

    /// Integer being composed.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("size", self.inner.size())?;
        summary.set_item("length", self.inner.parts().len())?;
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
