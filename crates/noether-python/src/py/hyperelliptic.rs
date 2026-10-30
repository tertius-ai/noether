// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Transport for Rust-owned hyperelliptic models and integral Frobenius evidence.

use super::linalg::{PyFrobeniusSemilinearMap, PyMatrix};
use super::numthy::PyFiniteField;
use super::polynomial_facade::PyPolynomial;
use super::surface::{
    PyCyclicCoverSmoothnessEvidence, PyCyclicHypersurfaceCover, PyFrobeniusPolynomial,
    PyZetaFunction,
};
use noether::commalg::{HyperellipticCurve, HyperellipticError, HyperellipticFrobeniusEvidence};
use noether::{SemanticType, UnsupportedOperationError};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

fn error(value: HyperellipticError) -> PyErr {
    match value {
        HyperellipticError::ContextMismatch(e) => super::errors::context_mismatch(e),
        HyperellipticError::InvalidEvidence(message) => {
            super::errors::VerificationError::new_err(message)
        }
        HyperellipticError::Unsupported(reason) => super::errors::unsupported_operation(
            UnsupportedOperationError::new(HyperellipticCurve::TYPE_ID, reason),
        ),
        value => PyValueError::new_err(value.to_string()),
    }
}

/// A checked odd-characteristic smooth projective hyperelliptic model.
#[pyclass(
    module = "noether.commalg",
    name = "HyperellipticCurve",
    frozen,
    from_py_object
)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct PyHyperellipticCurve {
    pub(crate) inner: HyperellipticCurve,
}

#[pymethods]
impl PyHyperellipticCurve {
    #[new]
    fn new(py: Python<'_>, f: &PyPolynomial) -> PyResult<Self> {
        let f = f
            .as_finite_field()
            .ok_or_else(|| {
                PyTypeError::new_err("HyperellipticCurve requires a finite-field polynomial")
            })?
            .clone();
        Ok(Self {
            inner: py.detach(|| HyperellipticCurve::new(f)).map_err(error)?,
        })
    }
    #[classmethod]
    fn from_equation(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        f: &PyPolynomial,
        h: &PyPolynomial,
    ) -> PyResult<Self> {
        let f = f
            .as_finite_field()
            .ok_or_else(|| PyTypeError::new_err("f must be a finite-field polynomial"))?
            .clone();
        let h = h
            .as_finite_field()
            .ok_or_else(|| PyTypeError::new_err("h must be a finite-field polynomial"))?
            .clone();
        Ok(Self {
            inner: py
                .detach(|| HyperellipticCurve::from_equation(f, h))
                .map_err(error)?,
        })
    }
    #[classattr]
    fn type_id() -> &'static str {
        HyperellipticCurve::TYPE_ID
    }
    #[getter]
    fn base(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.base().clone(),
        }
    }
    #[getter]
    fn f(&self) -> PyPolynomial {
        PyPolynomial::from_finite_field(self.inner.f().clone())
    }
    #[getter]
    fn h(&self) -> PyPolynomial {
        PyPolynomial::from_finite_field(self.inner.h().clone())
    }
    #[getter]
    fn branch_equation(&self) -> PyPolynomial {
        PyPolynomial::from_finite_field(self.inner.branch_equation().clone())
    }
    #[getter]
    fn genus(&self) -> usize {
        self.inner.genus()
    }
    #[getter]
    fn cover(&self) -> PyCyclicHypersurfaceCover {
        PyCyclicHypersurfaceCover::from_finite(self.inner.cover().clone())
    }
    #[getter]
    fn infinity_residue_degrees(&self) -> Vec<usize> {
        self.inner.infinity_residue_degrees()
    }
    fn completion_shift(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_finite_field(
            self.inner.completion_shift().map_err(error)?,
        ))
    }
    fn smoothness_evidence(&self, py: Python<'_>) -> PyResult<PyCyclicCoverSmoothnessEvidence> {
        Ok(PyCyclicCoverSmoothnessEvidence::from_finite(
            py.detach(|| self.inner.smoothness_evidence())
                .map_err(error)?,
        ))
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify()).map_err(error)
    }
    fn frobenius_polynomial(&self, py: Python<'_>) -> PyResult<PyFrobeniusPolynomial> {
        Ok(PyFrobeniusPolynomial {
            inner: py
                .detach(|| self.inner.frobenius_polynomial())
                .map_err(error)?,
        })
    }
    #[pyo3(signature=(precision=None))]
    fn frobenius_evidence(
        &self,
        py: Python<'_>,
        precision: Option<u64>,
    ) -> PyResult<PyHyperellipticFrobeniusEvidence> {
        Ok(PyHyperellipticFrobeniusEvidence {
            inner: py
                .detach(|| self.inner.frobenius_evidence(precision))
                .map_err(error)?,
        })
    }
    fn zeta_function(&self, py: Python<'_>) -> PyResult<PyZetaFunction> {
        Ok(PyZetaFunction {
            inner: py.detach(|| self.inner.zeta_function()).map_err(error)?,
        })
    }

    #[pyo3(signature=(extension_degree=1))]
    fn point_count(&self, py: Python<'_>, extension_degree: u64) -> PyResult<Py<PyAny>> {
        let count = py
            .detach(|| self.inner.point_count(extension_degree))
            .map_err(error)?;
        super::facade_scalar::integer_to_python(py, &count)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json())
            .map_err(super::errors::noether_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: String) -> PyResult<Self> {
        Ok(Self {
            inner: py
                .detach(|| HyperellipticCurve::from_json(&value))
                .map_err(super::errors::noether_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "HyperellipticCurve(genus={}, field_degree={})",
            self.inner.genus(),
            self.inner.base().degree()
        ))
    }
}

/// Replayable precision bounds and integral p- and q-Frobenius matrices.
#[pyclass(
    module = "noether.commalg",
    name = "HyperellipticFrobeniusEvidence",
    frozen,
    from_py_object
)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct PyHyperellipticFrobeniusEvidence {
    pub(crate) inner: HyperellipticFrobeniusEvidence,
}

#[pymethods]
impl PyHyperellipticFrobeniusEvidence {
    fn frobenius_polynomial(&self, py: Python<'_>) -> PyResult<PyFrobeniusPolynomial> {
        Ok(PyFrobeniusPolynomial {
            inner: py
                .detach(|| self.inner.frobenius_polynomial())
                .map_err(error)?,
        })
    }
    #[classattr]
    fn type_id() -> &'static str {
        HyperellipticFrobeniusEvidence::TYPE_ID
    }
    #[getter]
    fn curve(&self) -> PyHyperellipticCurve {
        PyHyperellipticCurve {
            inner: self.inner.curve().clone(),
        }
    }

    #[getter]
    fn normalized_branch(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_finite_field(
            self.inner.normalized_branch().map_err(error)?,
        ))
    }

    #[getter]
    fn quadratic_twist_sign(&self) -> i8 {
        self.inner.quadratic_twist_sign()
    }
    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision()
    }
    #[getter]
    fn working_precision(&self) -> u64 {
        self.inner.working_precision()
    }
    #[getter]
    fn binomial_terms(&self) -> usize {
        self.inner.binomial_terms()
    }
    #[getter]
    fn denominator_loss(&self) -> u64 {
        self.inner.denominator_loss()
    }
    #[getter]
    fn frobenius(&self) -> PyFrobeniusSemilinearMap {
        PyFrobeniusSemilinearMap::from_qadic(self.inner.frobenius().clone())
    }
    #[getter]
    fn linear_frobenius(&self) -> PyMatrix {
        PyMatrix::from_qadic(self.inner.linear_frobenius().clone())
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify()).map_err(error)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json())
            .map_err(super::errors::noether_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: String) -> PyResult<Self> {
        Ok(Self {
            inner: py
                .detach(|| HyperellipticFrobeniusEvidence::from_json(&value))
                .map_err(super::errors::noether_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "HyperellipticFrobeniusEvidence(genus={}, precision={})",
            self.inner.curve().genus(),
            self.inner.precision()
        ))
    }
}
