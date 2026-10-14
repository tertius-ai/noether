// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of exact rational symmetric functions.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use core::str::FromStr;

use noether::combin::{CombinError, Partition, SymmetricBasis, SymmetricFunction};
use noether::SemanticType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::facade_scalar::{extract_rational, rational_to_python};
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

fn extract_terms(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Partition, noether::Rational)>> {
    let terms = value.extract::<Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>>()?;
    terms
        .into_iter()
        .map(|(partition, coefficient)| {
            Ok((
                extract_partition(&partition)?,
                extract_rational(py, &coefficient)?,
            ))
        })
        .collect()
}

/// A thin Python handle around Noether's exact rational symmetric function.
#[pyclass(
    module = "noether.combin",
    name = "SymmetricFunction",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PySymmetricFunction {
    inner: SymmetricFunction,
}

impl Serialize for PySymmetricFunction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PySymmetricFunction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        SymmetricFunction::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl PySymmetricFunction {
    pub const TYPE_ID: &'static str = <SymmetricFunction as SemanticType>::TYPE_ID;

    fn from_core(inner: SymmetricFunction) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySymmetricFunction {
    /// Construct from a canonical basis label and `(partition, coefficient)` terms.
    #[new]
    fn new(py: Python<'_>, basis: &str, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let basis = SymmetricBasis::from_str(basis).map_err(combin_error)?;
        SymmetricFunction::new(basis, extract_terms(py, terms)?)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        Self::TYPE_ID
    }

    /// Serialize the basis and typed exact terms.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|error| {
            PyValueError::new_err(format!("SymmetricFunction serialization failed: {error}"))
        })
    }

    /// Deserialize and revalidate partitions, coefficients, and homogeneity.
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        serde_json::from_str(value).map_err(|error| {
            PyValueError::new_err(format!("SymmetricFunction deserialization failed: {error}"))
        })
    }

    /// Canonical basis label.
    #[getter]
    fn basis(&self) -> &'static str {
        self.inner.basis().label()
    }

    /// Homogeneous degree, or `None` for zero.
    #[getter]
    fn degree(&self) -> Option<usize> {
        self.inner.degree()
    }

    /// Immutable `(Partition, Fraction)` terms in deterministic order.
    #[getter]
    fn terms(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let terms = self
            .inner
            .terms()
            .map(|(partition, coefficient)| {
                let partition = Py::new(py, PyPartition::from_core(partition.clone()))?.into_any();
                let coefficient = rational_to_python(py, coefficient)?;
                Ok(PyTuple::new(py, [partition, coefficient])?.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, terms)?.unbind())
    }

    /// Re-express this value exactly in another classical basis.
    fn convert_basis(&self, basis: &str) -> PyResult<Self> {
        let basis = SymmetricBasis::from_str(basis).map_err(combin_error)?;
        self.inner
            .convert_basis(basis)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .add(&rhs.inner)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .sub(&rhs.inner)
            .map(Self::from_core)
            .map_err(combin_error)
    }

    fn __mul__(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .mul(&rhs.inner)
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
        summary.set_item("basis", self.inner.basis().label())?;
        summary.set_item("degree", self.inner.degree())?;
        summary.set_item("n_terms", self.inner.terms().len())?;
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
