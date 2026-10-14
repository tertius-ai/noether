// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projections of canonical cross-domain coefficient maps.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::{
    CoefficientMap, IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction,
    PrimeSubfieldEmbedding, PrimeSubfieldEmbeddingError, SemanticType,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use serde::{Deserialize, Serialize};

use super::commalg::{PyResidueClass, PyResidueRing};
use super::facade_scalar::{extract_integer, rational_to_python};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::root::{PyIntegerRing, PyRationalField};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn prime_embedding_error(error: PrimeSubfieldEmbeddingError) -> PyErr {
    match error {
        PrimeSubfieldEmbeddingError::ContextMismatch(error) => {
            super::errors::context_mismatch(error)
        }
        error => value_error(error),
    }
}

fn canonical_hash(value: &impl SemanticType) -> isize {
    let mut hasher = DefaultHasher::new();
    match value.canonical_bytes() {
        Ok(bytes) => bytes.hash(&mut hasher),
        Err(error) => error.to_string().hash(&mut hasher),
    }
    let hash = hasher.finish() as isize;
    if hash == -1 {
        -2
    } else {
        hash
    }
}

/// The canonical embedding of the integers into the rationals.
#[pyclass(
    module = "noether",
    name = "IntegerRationalEmbedding",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyIntegerRationalEmbedding {
    pub(crate) inner: IntegerRationalEmbedding,
}

#[pymethods]
impl PyIntegerRationalEmbedding {
    #[classattr]
    fn type_id() -> &'static str {
        IntegerRationalEmbedding::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyIntegerRing {
        PyIntegerRing
    }

    #[getter]
    fn target(&self) -> PyRationalField {
        PyRationalField
    }

    fn apply(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let image = self
            .inner
            .apply(&extract_integer(value)?)
            .map_err(super::errors::context_mismatch)?;
        rational_to_python(py, &image)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: IntegerRationalEmbedding::from_json(value).map_err(value_error)?,
        })
    }

    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.apply(py, value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("IntegerRationalEmbedding({})", self.inner))
    }
}

/// Canonical reduction from the integers into one residue ring.
#[pyclass(module = "noether", name = "IntegerReduction", frozen, from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyIntegerReduction {
    pub(crate) inner: IntegerReduction,
}

#[pymethods]
impl PyIntegerReduction {
    #[classattr]
    fn type_id() -> &'static str {
        IntegerReduction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyIntegerRing {
        PyIntegerRing
    }

    #[getter]
    fn target(&self) -> PyResidueRing {
        PyResidueRing {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &Bound<'_, PyAny>) -> PyResult<PyResidueClass> {
        Ok(PyResidueClass {
            inner: self
                .inner
                .apply(&extract_integer(value)?)
                .map_err(super::errors::context_mismatch)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: IntegerReduction::from_json(value).map_err(value_error)?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyResidueClass> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("IntegerReduction({})", self.inner))
    }
}

/// The canonical unital map from the integers into one finite field.
#[pyclass(
    module = "noether",
    name = "IntegerFiniteFieldMap",
    frozen,
    from_py_object
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyIntegerFiniteFieldMap {
    pub(crate) inner: IntegerFiniteFieldMap,
}

#[pymethods]
impl PyIntegerFiniteFieldMap {
    #[classattr]
    fn type_id() -> &'static str {
        IntegerFiniteFieldMap::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyIntegerRing {
        PyIntegerRing
    }

    #[getter]
    fn target(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &Bound<'_, PyAny>) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .apply(&extract_integer(value)?)
                .map_err(super::errors::context_mismatch)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: IntegerFiniteFieldMap::from_json(value).map_err(value_error)?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyFiniteFieldElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("IntegerFiniteFieldMap({})", self.inner))
    }
}

/// The checked canonical embedding of a prime field into an extension field.
#[pyclass(
    module = "noether",
    name = "PrimeSubfieldEmbedding",
    frozen,
    from_py_object
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyPrimeSubfieldEmbedding {
    pub(crate) inner: PrimeSubfieldEmbedding,
}

#[pymethods]
impl PyPrimeSubfieldEmbedding {
    #[classattr]
    fn type_id() -> &'static str {
        PrimeSubfieldEmbedding::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(prime_embedding_error)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PrimeSubfieldEmbedding::from_json(value).map_err(value_error)?,
        })
    }

    fn __call__(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PrimeSubfieldEmbedding({})", self.inner))
    }
}
