// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of semistandard tableaux and RSK.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::combin::{CombinError, SemistandardTableaux, Tableau};
use noether::{Integer, SemanticType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::facade_scalar::{extract_integer, integer_to_python};
use super::partition::PyPartition;
fn combin_error(error: CombinError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_rows(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Integer>>> {
    let rows: Vec<Vec<Bound<'_, PyAny>>> = value.extract()?;
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| extract_integer(&entry))
                .collect()
        })
        .collect()
}

fn extract_word(value: &Bound<'_, PyAny>) -> PyResult<Vec<Integer>> {
    let entries: Vec<Bound<'_, PyAny>> = value.extract()?;
    entries
        .into_iter()
        .map(|entry| extract_integer(&entry))
        .collect()
}

fn rows_to_python(py: Python<'_>, rows: &[Vec<Integer>]) -> PyResult<Py<PyTuple>> {
    let rows = rows
        .iter()
        .map(|row| {
            let entries = row
                .iter()
                .map(|entry| integer_to_python(py, entry))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, entries)?.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, rows)?.unbind())
}

/// A thin Python handle around Noether's semistandard integer tableau.
#[pyclass(module = "noether.combin", name = "Tableau", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyTableau {
    inner: Tableau,
}

impl Serialize for PyTableau {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyTableau {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Tableau::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PyTableau {
    pub const TYPE_ID: &'static str = <Tableau as SemanticType>::TYPE_ID;

    pub fn from_core(inner: Tableau) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTableau {
    /// Construct a semistandard tableau from row-major Python integers.
    #[new]
    fn new(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        Tableau::new(extract_rows(rows)?)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize canonical row data; arbitrary-precision entries use decimal strings.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("Tableau serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate shape, rows, and columns.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("Tableau deserialization failed: {error}"))
        })
    }

    /// Immutable row-major entries as Python integers.
    #[getter]
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        rows_to_python(py, self.inner.rows())
    }

    /// Canonical partition shape.
    #[getter]
    fn shape(&self) -> PyPartition {
        PyPartition::from_core(self.inner.shape().clone())
    }

    /// Content vector in the positive alphabet `1..=alphabet_size`.
    fn content(&self, alphabet_size: usize, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let content = self.inner.content(alphabet_size).map_err(combin_error)?;
        let content = content
            .iter()
            .map(|entry| integer_to_python(py, entry))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, content)?.unbind())
    }

    /// Lazily enumerate semistandard tableaux of one shape and alphabet.
    #[staticmethod]
    fn semistandard(shape: &PyPartition, maximum_entry: usize) -> PyResult<PySemistandardTableaux> {
        SemistandardTableaux::new(shape.core().clone(), maximum_entry)
            .map(PySemistandardTableaux::from_core)
            .map_err(combin_error)
    }

    /// Robinson--Schensted--Knuth row insertion for an integer word.
    #[staticmethod]
    fn rsk(word: &Bound<'_, PyAny>) -> PyResult<(Self, Self)> {
        Tableau::rsk(&extract_word(word)?)
            .map(|(insertion, recording)| (Self::from_core(insertion), Self::from_core(recording)))
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
        let parts = self.inner.shape().parts();
        summary.set_item(
            "shape",
            parts
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        if parts.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("shape_truncated", true)?;
        }
        summary.set_item("size", self.inner.shape().size())?;
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

/// A lazy finite iterator over semistandard tableaux of fixed shape.
#[pyclass(
    module = "noether.combin",
    name = "SemistandardTableaux",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySemistandardTableaux {
    inner: SemistandardTableaux,
}

impl PySemistandardTableaux {
    fn from_core(inner: SemistandardTableaux) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySemistandardTableaux {
    /// Construct a lazy enumeration in row-major lexicographic order.
    #[new]
    fn new(shape: &PyPartition, maximum_entry: usize) -> PyResult<Self> {
        SemistandardTableaux::new(shape.core().clone(), maximum_entry)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    /// Shared shape of every yielded tableau.
    #[getter]
    fn shape(&self) -> PyPartition {
        PyPartition::from_core(self.inner.shape().clone())
    }

    /// Largest positive letter in the alphabet.
    #[getter]
    fn maximum_entry(&self) -> usize {
        self.inner.maximum_entry()
    }

    /// Exact hook-content cardinality as an arbitrary-size Python integer.
    #[getter]
    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.cardinality())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyTableau> {
        slf.inner.next().map(PyTableau::from_core)
    }

    fn __length_hint__(&self) -> usize {
        self.inner.size_hint().1.unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SemistandardTableaux(shape={:?}, maximum_entry={}, cardinality={})",
            self.inner.shape().parts(),
            self.inner.maximum_entry(),
            self.inner.cardinality()
        ))
    }
}
