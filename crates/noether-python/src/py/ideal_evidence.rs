// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python transport for ideal-operation evidence.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    ArithmeticFiberSupport, ArithmeticFiberSupportClaim, ArithmeticNullstellensatzCertificate,
    IdealElimination, IdealReduction, RationalFunctionField,
};
use noether::numthy::FiniteField;
use noether::{IntegerRing, RationalField, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::{extract_integer, integer_to_python};
use super::number_field::{IdealInner, PyIdeal};
use super::numthy::PyPartialIntegerFactorization;
use super::polynomial_facade::{PyPolynomial, PyPolynomialRing};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
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

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum IdealReductionInner {
    Integer(IdealReduction<IntegerRing>),
    Rational(IdealReduction<RationalField>),
    FiniteField(IdealReduction<FiniteField>),
    RationalFunction(IdealReduction<RationalFunctionField<RationalField>>),
    FiniteRationalFunction(IdealReduction<RationalFunctionField<FiniteField>>),
}

/// Exact normal-form and ideal-membership multiplier evidence.
#[pyclass(
    module = "noether.commalg",
    name = "IdealReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIdealReduction {
    pub(crate) inner: IdealReductionInner,
}

impl PyIdealReduction {
    pub(crate) fn from_integer(inner: IdealReduction<IntegerRing>) -> Self {
        Self {
            inner: IdealReductionInner::Integer(inner),
        }
    }

    pub(crate) fn from_rational(inner: IdealReduction<RationalField>) -> Self {
        Self {
            inner: IdealReductionInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: IdealReduction<FiniteField>) -> Self {
        Self {
            inner: IdealReductionInner::FiniteField(inner),
        }
    }

    pub(crate) fn from_rational_function(
        inner: IdealReduction<RationalFunctionField<RationalField>>,
    ) -> Self {
        Self {
            inner: IdealReductionInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_rational_function(
        inner: IdealReduction<RationalFunctionField<FiniteField>>,
    ) -> Self {
        Self {
            inner: IdealReductionInner::FiniteRationalFunction(inner),
        }
    }
}

#[pymethods]
impl PyIdealReduction {
    #[classattr]
    fn type_id() -> &'static str {
        IdealReduction::<IntegerRing>::TYPE_ID
    }

    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            IdealReductionInner::Integer(value) => {
                PyIdeal::from_integer_polynomial(value.ideal().clone())
            }
            IdealReductionInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            IdealReductionInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
            IdealReductionInner::RationalFunction(value) => {
                PyIdeal::from_rational_function_polynomial(value.ideal().clone())
            }
            IdealReductionInner::FiniteRationalFunction(value) => {
                PyIdeal::from_finite_rational_function_polynomial(value.ideal().clone())
            }
        }
    }

    #[getter]
    fn value(&self) -> PyPolynomial {
        match &self.inner {
            IdealReductionInner::Integer(value) => {
                PyPolynomial::from_integer(value.value().clone())
            }
            IdealReductionInner::Rational(value) => {
                PyPolynomial::from_rational(value.value().clone())
            }
            IdealReductionInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.value().clone())
            }
            IdealReductionInner::RationalFunction(value) => {
                PyPolynomial::from_rational_function(value.value().clone())
            }
            IdealReductionInner::FiniteRationalFunction(value) => {
                PyPolynomial::from_finite_rational_function(value.value().clone())
            }
        }
    }

    #[getter]
    fn multipliers(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            IdealReductionInner::Integer(value) => value
                .multipliers()
                .iter()
                .cloned()
                .map(PyPolynomial::from_integer)
                .collect(),
            IdealReductionInner::Rational(value) => value
                .multipliers()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            IdealReductionInner::FiniteField(value) => value
                .multipliers()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
            IdealReductionInner::RationalFunction(value) => value
                .multipliers()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational_function)
                .collect(),
            IdealReductionInner::FiniteRationalFunction(value) => value
                .multipliers()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_rational_function)
                .collect(),
        }
    }

    #[getter]
    fn remainder(&self) -> PyPolynomial {
        match &self.inner {
            IdealReductionInner::Integer(value) => {
                PyPolynomial::from_integer(value.remainder().clone())
            }
            IdealReductionInner::Rational(value) => {
                PyPolynomial::from_rational(value.remainder().clone())
            }
            IdealReductionInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.remainder().clone())
            }
            IdealReductionInner::RationalFunction(value) => {
                PyPolynomial::from_rational_function(value.remainder().clone())
            }
            IdealReductionInner::FiniteRationalFunction(value) => {
                PyPolynomial::from_finite_rational_function(value.remainder().clone())
            }
        }
    }

    #[getter]
    fn is_member(&self) -> bool {
        match &self.inner {
            IdealReductionInner::Integer(value) => value.is_member(),
            IdealReductionInner::Rational(value) => value.is_member(),
            IdealReductionInner::FiniteField(value) => value.is_member(),
            IdealReductionInner::RationalFunction(value) => value.is_member(),
            IdealReductionInner::FiniteRationalFunction(value) => value.is_member(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            IdealReductionInner::Integer(value) => value.verify(),
            IdealReductionInner::Rational(value) => value.verify(),
            IdealReductionInner::FiniteField(value) => value.verify(),
            IdealReductionInner::RationalFunction(value) => value.verify(),
            IdealReductionInner::FiniteRationalFunction(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            IdealReductionInner::Integer(value) => value.to_json(),
            IdealReductionInner::Rational(value) => value.to_json(),
            IdealReductionInner::FiniteField(value) => value.to_json(),
            IdealReductionInner::RationalFunction(value) => value.to_json(),
            IdealReductionInner::FiniteRationalFunction(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        match serialized_ideal_coefficient_type(&payload)? {
            type_id if type_id == IntegerRing::TYPE_ID => Ok(Self::from_integer(
                IdealReduction::<IntegerRing>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == RationalField::TYPE_ID => Ok(Self::from_rational(
                IdealReduction::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                IdealReduction::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == RationalFunctionField::<FiniteField>::TYPE_ID => {
                match serialized_rational_function_base_type(&payload)? {
                    base if base == RationalField::TYPE_ID => Ok(Self::from_rational_function(
                        IdealReduction::<RationalFunctionField<RationalField>>::from_json(value)
                            .map_err(value_error)?,
                    )),
                    base if base == FiniteField::TYPE_ID => {
                        Ok(Self::from_finite_rational_function(
                            IdealReduction::<RationalFunctionField<FiniteField>>::from_json(value)
                                .map_err(value_error)?,
                        ))
                    }
                    base => Err(PyTypeError::new_err(format!(
                        "unsupported rational-function base field {base:?}"
                    ))),
                }
            }
            type_id => Err(PyTypeError::new_err(format!(
                "unsupported ideal-reduction coefficient context {type_id:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            IdealReductionInner::Integer(value) => canonical_hash(value),
            IdealReductionInner::Rational(value) => canonical_hash(value),
            IdealReductionInner::FiniteField(value) => canonical_hash(value),
            IdealReductionInner::RationalFunction(value) => canonical_hash(value),
            IdealReductionInner::FiniteRationalFunction(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "IdealReduction(member={}, multipliers={})",
            self.is_member(),
            self.multipliers().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum IdealEliminationInner {
    Rational(IdealElimination<RationalField>),
    FiniteField(IdealElimination<FiniteField>),
    RationalFunction(IdealElimination<RationalFunctionField<RationalField>>),
    FiniteRationalFunction(IdealElimination<RationalFunctionField<FiniteField>>),
}

/// Replayable exact elimination result over a field.
#[pyclass(
    module = "noether.commalg",
    name = "IdealElimination",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIdealElimination {
    pub(crate) inner: IdealEliminationInner,
}

impl PyIdealElimination {
    pub(crate) fn from_rational(inner: IdealElimination<RationalField>) -> Self {
        Self {
            inner: IdealEliminationInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: IdealElimination<FiniteField>) -> Self {
        Self {
            inner: IdealEliminationInner::FiniteField(inner),
        }
    }

    pub(crate) fn from_rational_function(
        inner: IdealElimination<RationalFunctionField<RationalField>>,
    ) -> Self {
        Self {
            inner: IdealEliminationInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_rational_function(
        inner: IdealElimination<RationalFunctionField<FiniteField>>,
    ) -> Self {
        Self {
            inner: IdealEliminationInner::FiniteRationalFunction(inner),
        }
    }
}

#[pymethods]
impl PyIdealElimination {
    #[classattr]
    fn type_id() -> &'static str {
        IdealElimination::<RationalField>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            IdealEliminationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            IdealEliminationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
            IdealEliminationInner::RationalFunction(value) => {
                PyIdeal::from_rational_function_polynomial(value.source().clone())
            }
            IdealEliminationInner::FiniteRationalFunction(value) => {
                PyIdeal::from_finite_rational_function_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn variables(&self) -> Vec<usize> {
        match &self.inner {
            IdealEliminationInner::Rational(value) => value.variables().to_vec(),
            IdealEliminationInner::FiniteField(value) => value.variables().to_vec(),
            IdealEliminationInner::RationalFunction(value) => value.variables().to_vec(),
            IdealEliminationInner::FiniteRationalFunction(value) => value.variables().to_vec(),
        }
    }

    #[getter]
    fn result(&self) -> PyIdeal {
        match &self.inner {
            IdealEliminationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.result().clone())
            }
            IdealEliminationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.result().clone())
            }
            IdealEliminationInner::RationalFunction(value) => {
                PyIdeal::from_rational_function_polynomial(value.result().clone())
            }
            IdealEliminationInner::FiniteRationalFunction(value) => {
                PyIdeal::from_finite_rational_function_polynomial(value.result().clone())
            }
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            IdealEliminationInner::Rational(value) => value.verify(),
            IdealEliminationInner::FiniteField(value) => value.verify(),
            IdealEliminationInner::RationalFunction(value) => value.verify(),
            IdealEliminationInner::FiniteRationalFunction(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            IdealEliminationInner::Rational(value) => value.to_json(),
            IdealEliminationInner::FiniteField(value) => value.to_json(),
            IdealEliminationInner::RationalFunction(value) => value.to_json(),
            IdealEliminationInner::FiniteRationalFunction(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        match serialized_source_ideal_coefficient_type(&payload)? {
            type_id if type_id == RationalField::TYPE_ID => Ok(Self::from_rational(
                IdealElimination::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                IdealElimination::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            type_id if type_id == RationalFunctionField::<FiniteField>::TYPE_ID => {
                match serialized_source_rational_function_base_type(&payload)? {
                    base if base == RationalField::TYPE_ID => Ok(Self::from_rational_function(
                        IdealElimination::<RationalFunctionField<RationalField>>::from_json(value)
                            .map_err(value_error)?,
                    )),
                    base if base == FiniteField::TYPE_ID => {
                        Ok(Self::from_finite_rational_function(
                            IdealElimination::<RationalFunctionField<FiniteField>>::from_json(
                                value,
                            )
                            .map_err(value_error)?,
                        ))
                    }
                    base => Err(PyTypeError::new_err(format!(
                        "unsupported rational-function base field {base:?}"
                    ))),
                }
            }
            type_id => Err(PyTypeError::new_err(format!(
                "unsupported ideal-elimination coefficient context {type_id:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            IdealEliminationInner::Rational(value) => canonical_hash(value),
            IdealEliminationInner::FiniteField(value) => canonical_hash(value),
            IdealEliminationInner::RationalFunction(value) => canonical_hash(value),
            IdealEliminationInner::FiniteRationalFunction(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "IdealElimination(variables={:?}, generators={})",
            self.variables(),
            self.result().inner_generator_count()
        ))
    }
}

/// Exact arithmetic Nullstellensatz multiplier identity over `Z`.
#[pyclass(
    module = "noether.commalg",
    name = "ArithmeticNullstellensatzCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyArithmeticNullstellensatzCertificate {
    pub(crate) inner: ArithmeticNullstellensatzCertificate,
}

#[pymethods]
impl PyArithmeticNullstellensatzCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        ArithmeticNullstellensatzCertificate::TYPE_ID
    }

    /// Build and verify an identity against the source ideal's canonical basis.
    #[classmethod]
    fn from_identity(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        source: &PyIdeal,
        integer: &Bound<'_, PyAny>,
        multipliers: Vec<Py<PyPolynomial>>,
    ) -> PyResult<Self> {
        let IdealInner::IntegerPolynomial(source) = &source.inner else {
            return Err(PyTypeError::new_err(
                "an arithmetic certificate requires a polynomial ideal over Z",
            ));
        };
        let multipliers =
            multipliers
                .into_iter()
                .map(|multiplier| {
                    multiplier.borrow(py).as_integer().cloned().ok_or_else(|| {
                        PyTypeError::new_err("multipliers must be polynomials over Z")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: ArithmeticNullstellensatzCertificate::from_identity(
                source.clone(),
                extract_integer(integer)?,
                multipliers,
            )
            .map_err(value_error)?,
        })
    }

    #[getter]
    fn ring(&self) -> PyPolynomialRing {
        PyPolynomialRing::from_integer(self.inner.ring().clone())
    }

    #[getter]
    fn generators(&self) -> Vec<PyPolynomial> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(PyPolynomial::from_integer)
            .collect()
    }

    #[getter]
    fn integer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.integer())
    }

    #[getter]
    fn principal_ideal(&self) -> PyIdeal {
        PyIdeal::from_integer(self.inner.principal_ideal().clone())
    }

    #[getter]
    fn multipliers(&self) -> Vec<PyPolynomial> {
        self.inner
            .multipliers()
            .iter()
            .cloned()
            .map(PyPolynomial::from_integer)
            .collect()
    }

    fn is_exact_contraction(&self) -> PyResult<bool> {
        self.inner.is_exact_contraction().map_err(value_error)
    }

    fn arithmetic_fiber_support(
        &self,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> PyResult<PyArithmeticFiberSupport> {
        Ok(PyArithmeticFiberSupport {
            inner: self
                .inner
                .arithmetic_fiber_support(trial_bound, primality_proof_bits)
                .map_err(value_error)?,
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
            inner: ArithmeticNullstellensatzCertificate::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact or containing arithmetic support with bounded factorization evidence.
#[pyclass(
    module = "noether.commalg",
    name = "ArithmeticFiberSupport",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyArithmeticFiberSupport {
    pub(crate) inner: ArithmeticFiberSupport,
}

#[pymethods]
impl PyArithmeticFiberSupport {
    #[classattr]
    fn type_id() -> &'static str {
        ArithmeticFiberSupport::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyPolynomialRing {
        PyPolynomialRing::from_integer(self.inner.ring().clone())
    }

    #[getter]
    fn generators(&self) -> Vec<PyPolynomial> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(PyPolynomial::from_integer)
            .collect()
    }

    #[getter]
    fn support_ideal(&self) -> PyIdeal {
        PyIdeal::from_integer(self.inner.support_ideal().clone())
    }

    #[getter]
    fn claim(&self) -> &'static str {
        match self.inner.claim() {
            ArithmeticFiberSupportClaim::ExactContraction => "exact_contraction",
            ArithmeticFiberSupportClaim::CertifiedContainment => "certified_containment",
        }
    }

    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    #[getter]
    fn certificate(&self) -> Option<PyArithmeticNullstellensatzCertificate> {
        self.inner
            .certificate()
            .cloned()
            .map(|inner| PyArithmeticNullstellensatzCertificate { inner })
    }

    #[getter]
    fn factorization(&self) -> PyPartialIntegerFactorization {
        PyPartialIntegerFactorization {
            inner: self.inner.factorization().clone(),
        }
    }

    #[getter]
    fn proved_primes(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u64)>> {
        self.inner
            .proved_primes()
            .iter()
            .map(|(prime, exponent)| Ok((integer_to_python(py, prime)?, *exponent)))
            .collect()
    }

    #[getter]
    fn prime_support_is_complete(&self) -> bool {
        self.inner.prime_support_is_complete()
    }

    #[getter]
    fn unfactored_cofactor(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .unfactored_cofactor()
            .map(|cofactor| integer_to_python(py, cofactor))
            .transpose()
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
            inner: ArithmeticFiberSupport::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

fn serialized_ideal_coefficient_type(payload: &serde_json::Value) -> PyResult<&str> {
    payload
        .get("ideal")
        .and_then(|ideal| ideal.get("ring"))
        .and_then(|ring| ring.get("coefficient_ring"))
        .and_then(|ring| ring.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| value_error("serialized ideal reduction has no coefficient context"))
}

fn serialized_rational_function_base_type(payload: &serde_json::Value) -> PyResult<&str> {
    payload
        .get("ideal")
        .and_then(|ideal| ideal.get("ring"))
        .and_then(|ring| ring.get("coefficient_ring"))
        .and_then(|field| field.get("coefficient_ring"))
        .and_then(|ring| ring.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| value_error("serialized ideal reduction has no function-field base"))
}

fn serialized_source_ideal_coefficient_type(payload: &serde_json::Value) -> PyResult<&str> {
    payload
        .get("source")
        .and_then(|ideal| ideal.get("ring"))
        .and_then(|ring| ring.get("coefficient_ring"))
        .and_then(|ring| ring.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| value_error("serialized ideal elimination has no coefficient context"))
}

fn serialized_source_rational_function_base_type(payload: &serde_json::Value) -> PyResult<&str> {
    payload
        .get("source")
        .and_then(|ideal| ideal.get("ring"))
        .and_then(|ring| ring.get("coefficient_ring"))
        .and_then(|field| field.get("coefficient_ring"))
        .and_then(|ring| ring.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| value_error("serialized ideal elimination has no function-field base"))
}

trait PyIdealPrivateCount {
    fn inner_generator_count(&self) -> usize;
}

impl PyIdealPrivateCount for PyIdeal {
    fn inner_generator_count(&self) -> usize {
        match &self.inner {
            IdealInner::Integer(value) => value.generators().len(),
            IdealInner::NumberField(value) => value.generators().len(),
            IdealInner::IntegerPolynomial(value) => value.generators().len(),
            IdealInner::RationalPolynomial(value) => value.generators().len(),
            IdealInner::FiniteFieldPolynomial(value) => value.generators().len(),
            IdealInner::RationalFunctionPolynomial(value) => value.generators().len(),
            IdealInner::FiniteRationalFunctionPolynomial(value) => value.generators().len(),
        }
    }
}
