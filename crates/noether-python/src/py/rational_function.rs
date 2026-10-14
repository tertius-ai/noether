// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public rational-function facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    RationalFunction, RationalFunctionError, RationalFunctionField, RationalFunctionFieldMap,
    RationalFunctionFieldMapError,
};
use noether::numthy::{FiniteField, FiniteFieldElement};
use noether::{
    CoefficientMap, Field, PrimeSubfieldEmbedding, RationalField, Ring, SemanticType, Q,
};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyType};
use serde::{Deserialize, Serialize};

use super::coefficient_maps::PyPrimeSubfieldEmbedding;
use super::facade_scalar::{extract_integer, extract_rational, rational_to_python};
use super::linalg::{PyMatrix, PyVector};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::polynomial_facade::{PyPolynomial, PyPolynomialRing};
use super::root::PyRationalField;

fn rational_function_error(error: RationalFunctionError) -> PyErr {
    match error {
        RationalFunctionError::ZeroDenominator
        | RationalFunctionError::DivisionByZero
        | RationalFunctionError::Pole => PyZeroDivisionError::new_err(error.to_string()),
        RationalFunctionError::ContextMismatch(error) => super::errors::context_mismatch(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn rational_function_field_map_error(error: RationalFunctionFieldMapError) -> PyErr {
    match error {
        RationalFunctionFieldMapError::ContextMismatch(error)
        | RationalFunctionFieldMapError::RationalFunction(
            RationalFunctionError::ContextMismatch(error),
        ) => super::errors::context_mismatch(error),
        error => PyValueError::new_err(error.to_string()),
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

fn serialized_coefficient_type(value: &str, nested_field: bool) -> PyResult<String> {
    let payload: serde_json::Value = serde_json::from_str(value)
        .map_err(|error| PyValueError::new_err(format!("invalid JSON: {error}")))?;
    let field = if nested_field {
        payload.get("field")
    } else {
        Some(&payload)
    }
    .ok_or_else(|| PyValueError::new_err("rational-function JSON is missing its field"))?;
    field
        .get("coefficient_ring")
        .and_then(|context| context.get("type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PyValueError::new_err("rational-function JSON is missing its coefficient-context type")
        })
}

fn finite_field_coefficient(
    field: &FiniteField,
    value: &Bound<'_, PyAny>,
) -> PyResult<FiniteFieldElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(field.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyFiniteFieldElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(PyValueError::new_err(
            "finite-field coefficient belongs to another context",
        ));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or FiniteFieldElement",
    ))
}

#[derive(Clone)]
enum RationalFunctionFieldInner {
    Rational(RationalFunctionField<RationalField>),
    FiniteField(RationalFunctionField<FiniteField>),
}

impl PartialEq for RationalFunctionFieldInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for RationalFunctionFieldInner {}

/// A rational function field over `Q` or one concrete finite field.
#[pyclass(
    module = "noether.commalg",
    name = "RationalFunctionField",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRationalFunctionField {
    inner: RationalFunctionFieldInner,
}

impl PyRationalFunctionField {
    pub(crate) fn as_rational(&self) -> Option<&RationalFunctionField<RationalField>> {
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => Some(field),
            RationalFunctionFieldInner::FiniteField(_) => None,
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&RationalFunctionField<FiniteField>> {
        match &self.inner {
            RationalFunctionFieldInner::FiniteField(field) => Some(field),
            RationalFunctionFieldInner::Rational(_) => None,
        }
    }

    pub(crate) fn from_finite_field(field: RationalFunctionField<FiniteField>) -> Self {
        Self {
            inner: RationalFunctionFieldInner::FiniteField(field),
        }
    }

    pub(crate) fn from_rational(field: RationalFunctionField<RationalField>) -> Self {
        Self {
            inner: RationalFunctionFieldInner::Rational(field),
        }
    }

    fn polynomial_ring_value(&self) -> PyPolynomialRing {
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => {
                PyPolynomialRing::from_rational(field.polynomial_ring().clone())
            }
            RationalFunctionFieldInner::FiniteField(field) => {
                PyPolynomialRing::from_finite_field(field.polynomial_ring().clone())
            }
        }
    }

    fn polynomial_from_python(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyPolynomial> {
        if let Ok(polynomial) = value.extract::<PyRef<'_, PyPolynomial>>() {
            return match &self.inner {
                RationalFunctionFieldInner::Rational(field) => polynomial
                    .as_rational()
                    .filter(|value| field.polynomial_ring().contains(value))
                    .cloned()
                    .map(PyPolynomial::from_rational)
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "polynomial belongs to another coefficient or variable context",
                        )
                    }),
                RationalFunctionFieldInner::FiniteField(field) => polynomial
                    .as_finite_field()
                    .filter(|value| field.polynomial_ring().contains(value))
                    .cloned()
                    .map(PyPolynomial::from_finite_field)
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "polynomial belongs to another coefficient or variable context",
                        )
                    }),
            };
        }
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => Ok(PyPolynomial::from_rational(
                field
                    .polynomial_ring()
                    .constant(extract_rational(py, value)?)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            )),
            RationalFunctionFieldInner::FiniteField(field) => Ok(PyPolynomial::from_finite_field(
                field
                    .polynomial_ring()
                    .constant(finite_field_coefficient(field.coefficient_ring(), value)?)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            )),
        }
    }

    fn constant_from_python(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRationalFunction> {
        let inner = match &self.inner {
            RationalFunctionFieldInner::Rational(field) => RationalFunctionInner::Rational(
                field
                    .constant(extract_rational(py, value)?)
                    .map_err(rational_function_error)?,
            ),
            RationalFunctionFieldInner::FiniteField(field) => RationalFunctionInner::FiniteField(
                field
                    .constant(finite_field_coefficient(field.coefficient_ring(), value)?)
                    .map_err(rational_function_error)?,
            ),
        };
        Ok(PyRationalFunction { inner })
    }

    fn from_polynomial_value(&self, value: &PyPolynomial) -> PyResult<PyRationalFunction> {
        let inner = match (&self.inner, value.as_rational(), value.as_finite_field()) {
            (RationalFunctionFieldInner::Rational(field), Some(value), _) => {
                RationalFunctionInner::Rational(
                    field
                        .from_polynomial(value.clone())
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                )
            }
            (RationalFunctionFieldInner::FiniteField(field), _, Some(value)) => {
                RationalFunctionInner::FiniteField(
                    field
                        .from_polynomial(value.clone())
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "polynomial belongs to another coefficient context",
                ));
            }
        };
        Ok(PyRationalFunction { inner })
    }

    pub(crate) fn coerce(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRationalFunction> {
        if let Ok(function) = value.extract::<PyRef<'_, PyRationalFunction>>() {
            if function.field_value().inner == self.inner {
                return Ok(function.clone());
            }
            return Err(PyValueError::new_err(
                "rational function belongs to another field context",
            ));
        }
        if let Ok(polynomial) = value.extract::<PyRef<'_, PyPolynomial>>() {
            return self.from_polynomial_value(&polynomial);
        }
        self.constant_from_python(py, value)
    }
}

#[pymethods]
impl PyRationalFunctionField {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RationalFunctionField::<RationalField>::TYPE_ID
    }

    #[new]
    fn new(coefficient_ring: &Bound<'_, PyAny>, variable: String) -> PyResult<Self> {
        let inner = if coefficient_ring.is_instance_of::<PyRationalField>() {
            RationalFunctionFieldInner::Rational(
                RationalFunctionField::new(Q, variable).map_err(rational_function_error)?,
            )
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyFiniteField>>() {
            RationalFunctionFieldInner::FiniteField(
                RationalFunctionField::new(field.inner.clone(), variable)
                    .map_err(rational_function_error)?,
            )
        } else {
            return Err(PyTypeError::new_err(
                "RationalFunctionField currently supports Q and FiniteField coefficient contexts",
            ));
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn coefficient_ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            RationalFunctionFieldInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            RationalFunctionFieldInner::FiniteField(field) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: field.coefficient_ring().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn variable(&self) -> String {
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => field.variable().to_owned(),
            RationalFunctionFieldInner::FiniteField(field) => field.variable().to_owned(),
        }
    }

    #[getter]
    fn polynomial_ring(&self) -> PyPolynomialRing {
        self.polynomial_ring_value()
    }

    fn zero(&self) -> PyRationalFunction {
        let inner = match &self.inner {
            RationalFunctionFieldInner::Rational(field) => {
                RationalFunctionInner::Rational(field.zero())
            }
            RationalFunctionFieldInner::FiniteField(field) => {
                RationalFunctionInner::FiniteField(field.zero())
            }
        };
        PyRationalFunction { inner }
    }

    fn one(&self) -> PyRationalFunction {
        let inner = match &self.inner {
            RationalFunctionFieldInner::Rational(field) => {
                RationalFunctionInner::Rational(field.one())
            }
            RationalFunctionFieldInner::FiniteField(field) => {
                RationalFunctionInner::FiniteField(field.one())
            }
        };
        PyRationalFunction { inner }
    }

    fn generator(&self) -> PyRationalFunction {
        let inner = match &self.inner {
            RationalFunctionFieldInner::Rational(field) => {
                RationalFunctionInner::Rational(field.generator())
            }
            RationalFunctionFieldInner::FiniteField(field) => {
                RationalFunctionInner::FiniteField(field.generator())
            }
        };
        PyRationalFunction { inner }
    }

    fn constant(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyRationalFunction> {
        self.constant_from_python(py, value)
    }

    fn from_polynomial(&self, value: &PyPolynomial) -> PyResult<PyRationalFunction> {
        self.from_polynomial_value(value)
    }

    fn fraction(
        &self,
        py: Python<'_>,
        numerator: &Bound<'_, PyAny>,
        denominator: &Bound<'_, PyAny>,
    ) -> PyResult<PyRationalFunction> {
        let numerator = self.polynomial_from_python(py, numerator)?;
        let denominator = self.polynomial_from_python(py, denominator)?;
        let inner = match (
            &self.inner,
            numerator.as_rational(),
            denominator.as_rational(),
        ) {
            (RationalFunctionFieldInner::Rational(field), Some(numerator), Some(denominator)) => {
                RationalFunctionInner::Rational(
                    field
                        .fraction(numerator.clone(), denominator.clone())
                        .map_err(rational_function_error)?,
                )
            }
            (RationalFunctionFieldInner::FiniteField(field), _, _) => {
                let numerator = numerator.as_finite_field().ok_or_else(|| {
                    PyValueError::new_err("numerator has the wrong coefficient context")
                })?;
                let denominator = denominator.as_finite_field().ok_or_else(|| {
                    PyValueError::new_err("denominator has the wrong coefficient context")
                })?;
                RationalFunctionInner::FiniteField(
                    field
                        .fraction(numerator.clone(), denominator.clone())
                        .map_err(rational_function_error)?,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "fraction polynomials have the wrong coefficient context",
                ));
            }
        };
        Ok(PyRationalFunction { inner })
    }

    #[pyo3(signature = (numerator, denominator=None))]
    fn element(
        &self,
        py: Python<'_>,
        numerator: &Bound<'_, PyAny>,
        denominator: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRationalFunction> {
        match denominator {
            Some(denominator) => self.fraction(py, numerator, denominator),
            None => self.coerce(py, numerator),
        }
    }

    fn inverse(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyRationalFunction> {
        self.coerce(py, value)?.inverse_value()
    }

    /// Extend one checked prime-subfield embedding coefficientwise and map the
    /// source generator to the target generator.
    fn induced_map(
        &self,
        target: &Self,
        coefficient_map: &PyPrimeSubfieldEmbedding,
    ) -> PyResult<PyRationalFunctionFieldMap> {
        let source = self.as_finite_field().ok_or_else(|| {
            PyTypeError::new_err(
                "this coefficient map induces only maps between finite rational-function fields",
            )
        })?;
        let target = target.as_finite_field().ok_or_else(|| {
            PyTypeError::new_err(
                "this coefficient map induces only maps between finite rational-function fields",
            )
        })?;
        Ok(PyRationalFunctionFieldMap {
            inner: RationalFunctionFieldMap::new(
                source.clone(),
                target.clone(),
                coefficient_map.inner.clone(),
            )
            .map_err(rational_function_field_map_error)?,
        })
    }

    /// Serialize the complete coefficient-field and variable context.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => field.to_json(),
            RationalFunctionFieldInner::FiniteField(field) => field.to_json(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a supported rational-function field.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient_type(value, false)?.as_str() {
            RationalField::TYPE_ID => Ok(Self {
                inner: RationalFunctionFieldInner::Rational(
                    RationalFunctionField::<RationalField>::from_json(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                ),
            }),
            FiniteField::TYPE_ID => Ok(Self {
                inner: RationalFunctionFieldInner::FiniteField(
                    RationalFunctionField::<FiniteField>::from_json(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                ),
            }),
            type_id => Err(PyValueError::new_err(format!(
                "unsupported rational-function coefficient-context type {type_id}"
            ))),
        }
    }

    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyRationalFunction> {
        self.coerce(py, value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            RationalFunctionFieldInner::Rational(field) => canonical_hash(field),
            RationalFunctionFieldInner::FiniteField(field) => canonical_hash(field),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "RationalFunctionField(coefficient_ring={}, variable={:?})",
                match &self.inner {
                    RationalFunctionFieldInner::Rational(_) => "Q".to_owned(),
                    RationalFunctionFieldInner::FiniteField(field) => {
                        field.coefficient_ring().to_string()
                    }
                },
                self.variable()
            )
        })
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[derive(Clone)]
enum RationalFunctionInner {
    Rational(RationalFunction<RationalField>),
    FiniteField(RationalFunction<FiniteField>),
}

impl PartialEq for RationalFunctionInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for RationalFunctionInner {}

/// One reduced rational function carrying its complete field context.
#[pyclass(
    module = "noether.commalg",
    name = "RationalFunction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRationalFunction {
    inner: RationalFunctionInner,
}

impl PyRationalFunction {
    pub(crate) fn as_rational(&self) -> Option<&RationalFunction<RationalField>> {
        match &self.inner {
            RationalFunctionInner::Rational(value) => Some(value),
            RationalFunctionInner::FiniteField(_) => None,
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&RationalFunction<FiniteField>> {
        match &self.inner {
            RationalFunctionInner::FiniteField(value) => Some(value),
            RationalFunctionInner::Rational(_) => None,
        }
    }

    pub(crate) fn from_finite_field(value: RationalFunction<FiniteField>) -> Self {
        Self {
            inner: RationalFunctionInner::FiniteField(value),
        }
    }

    pub(crate) fn from_rational(value: RationalFunction<RationalField>) -> Self {
        Self {
            inner: RationalFunctionInner::Rational(value),
        }
    }

    fn field_value(&self) -> PyRationalFunctionField {
        let inner = match &self.inner {
            RationalFunctionInner::Rational(value) => {
                RationalFunctionFieldInner::Rational(value.field().clone())
            }
            RationalFunctionInner::FiniteField(value) => {
                RationalFunctionFieldInner::FiniteField(value.field().clone())
            }
        };
        PyRationalFunctionField { inner }
    }

    fn coerce(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyRationalFunction> {
        self.field_value().coerce(py, value)
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (RationalFunctionInner::Rational(left), RationalFunctionInner::Rational(right)) => {
                RationalFunctionInner::Rational(binary_function(left, right, operation)?)
            }
            (
                RationalFunctionInner::FiniteField(left),
                RationalFunctionInner::FiniteField(right),
            ) => RationalFunctionInner::FiniteField(binary_function(left, right, operation)?),
            _ => {
                return Err(PyValueError::new_err(
                    "rational functions belong to different field contexts",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn inverse_value(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            RationalFunctionInner::Rational(value) => RationalFunctionInner::Rational(
                value
                    .field()
                    .inverse(value)
                    .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))?,
            ),
            RationalFunctionInner::FiniteField(value) => RationalFunctionInner::FiniteField(
                value
                    .field()
                    .inverse(value)
                    .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))?,
            ),
        };
        Ok(Self { inner })
    }

    fn pow_signed(&self, exponent: i64) -> PyResult<Self> {
        if exponent < 0 {
            return self
                .inverse_value()?
                .pow_signed_unsigned(exponent.unsigned_abs());
        }
        self.pow_signed_unsigned(exponent as u64)
    }

    fn pow_signed_unsigned(&self, exponent: u64) -> PyResult<Self> {
        let inner = match &self.inner {
            RationalFunctionInner::Rational(value) => RationalFunctionInner::Rational(
                value.pow(exponent).map_err(rational_function_error)?,
            ),
            RationalFunctionInner::FiniteField(value) => RationalFunctionInner::FiniteField(
                value.pow(exponent).map_err(rational_function_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn defer_to_outer_polynomial(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> Option<Py<PyAny>> {
        if value
            .extract::<PyRef<'_, PyPolynomial>>()
            .is_ok_and(|value| {
                value.as_rational_function().is_some()
                    || value.as_finite_rational_function().is_some()
            })
            || value.is_instance_of::<PyMatrix>()
            || value.is_instance_of::<PyVector>()
        {
            Some(py.NotImplemented())
        } else {
            None
        }
    }
}

/// A checked coefficientwise map between finite rational-function fields.
#[pyclass(
    module = "noether.commalg",
    name = "RationalFunctionFieldMap",
    frozen,
    from_py_object
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyRationalFunctionFieldMap {
    pub(crate) inner: RationalFunctionFieldMap<FiniteField, FiniteField, PrimeSubfieldEmbedding>,
}

#[pymethods]
impl PyRationalFunctionFieldMap {
    #[classattr]
    fn type_id() -> &'static str {
        RationalFunctionFieldMap::<FiniteField, FiniteField, PrimeSubfieldEmbedding>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyRationalFunctionField {
        PyRationalFunctionField::from_finite_field(self.inner.source().clone())
    }

    #[getter]
    fn target(&self) -> PyRationalFunctionField {
        PyRationalFunctionField::from_finite_field(self.inner.target().clone())
    }

    #[getter]
    fn coefficient_map(&self) -> PyPrimeSubfieldEmbedding {
        PyPrimeSubfieldEmbedding {
            inner: self.inner.coefficient_map().clone(),
        }
    }

    fn apply(&self, value: &PyRationalFunction) -> PyResult<PyRationalFunction> {
        let value = value.as_finite_field().ok_or_else(|| {
            PyTypeError::new_err("the map expects a rational function over its finite source field")
        })?;
        Ok(PyRationalFunction::from_finite_field(
            self.inner
                .apply(value)
                .map_err(rational_function_field_map_error)?,
        ))
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RationalFunctionFieldMap::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyRationalFunction) -> PyResult<PyRationalFunction> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("RationalFunctionFieldMap({})", self.inner))
    }
}

fn binary_function<R: Field>(
    left: &RationalFunction<R>,
    right: &RationalFunction<R>,
    operation: &str,
) -> PyResult<RationalFunction<R>> {
    match operation {
        "add" => (left + right).map_err(rational_function_error),
        "sub" => (left - right).map_err(rational_function_error),
        "mul" => (left * right).map_err(rational_function_error),
        "div" => (left / right).map_err(rational_function_error),
        _ => Err(PyValueError::new_err("unknown rational-function operation")),
    }
}

#[pymethods]
impl PyRationalFunction {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RationalFunction::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn field(&self) -> PyRationalFunctionField {
        self.field_value()
    }

    #[getter]
    fn numerator(&self) -> PyPolynomial {
        match &self.inner {
            RationalFunctionInner::Rational(value) => {
                PyPolynomial::from_rational(value.numerator().clone())
            }
            RationalFunctionInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.numerator().clone())
            }
        }
    }

    #[getter]
    fn denominator(&self) -> PyPolynomial {
        match &self.inner {
            RationalFunctionInner::Rational(value) => {
                PyPolynomial::from_rational(value.denominator().clone())
            }
            RationalFunctionInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.denominator().clone())
            }
        }
    }

    fn is_zero(&self) -> bool {
        match &self.inner {
            RationalFunctionInner::Rational(value) => value.is_zero(),
            RationalFunctionInner::FiniteField(value) => value.is_zero(),
        }
    }

    fn is_one(&self) -> bool {
        match &self.inner {
            RationalFunctionInner::Rational(value) => value.is_one(),
            RationalFunctionInner::FiniteField(value) => value.is_one(),
        }
    }

    /// Whether this rational function is an exact positive integral power.
    fn is_nth_power(&self, exponent: u64) -> PyResult<bool> {
        match &self.inner {
            RationalFunctionInner::Rational(value) => value.is_nth_power(exponent),
            RationalFunctionInner::FiniteField(value) => value.is_nth_power(exponent),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Extract a deterministic exact root, or return `None` when none exists.
    fn nth_root(&self, exponent: u64) -> PyResult<Option<Self>> {
        let inner = match &self.inner {
            RationalFunctionInner::Rational(value) => value
                .nth_root(exponent)
                .map(|root| root.map(RationalFunctionInner::Rational)),
            RationalFunctionInner::FiniteField(value) => value
                .nth_root(exponent)
                .map(|root| root.map(RationalFunctionInner::FiniteField)),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(inner.map(|inner| Self { inner }))
    }

    fn specialize(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            RationalFunctionInner::Rational(function) => rational_to_python(
                py,
                &function
                    .specialize(&extract_rational(py, value)?)
                    .map_err(rational_function_error)?,
            ),
            RationalFunctionInner::FiniteField(function) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: function
                        .specialize(&finite_field_coefficient(
                            function.field().coefficient_ring(),
                            value,
                        )?)
                        .map_err(rational_function_error)?,
                },
            )?
            .into_any()),
        }
    }

    fn pow(&self, exponent: i64) -> PyResult<Self> {
        self.pow_signed(exponent)
    }

    fn inverse(&self) -> PyResult<Self> {
        self.inverse_value()
    }

    /// Serialize the field context and canonical reduced fraction.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            RationalFunctionInner::Rational(value) => value.to_json(),
            RationalFunctionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a supported canonical rational function.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient_type(value, true)?.as_str() {
            RationalField::TYPE_ID => Ok(Self {
                inner: RationalFunctionInner::Rational(
                    RationalFunction::<RationalField>::from_json(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                ),
            }),
            FiniteField::TYPE_ID => Ok(Self {
                inner: RationalFunctionInner::FiniteField(
                    RationalFunction::<FiniteField>::from_json(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                ),
            }),
            type_id => Err(PyValueError::new_err(format!(
                "unsupported rational-function coefficient-context type {type_id}"
            ))),
        }
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Some(deferred) = self.defer_to_outer_polynomial(py, other) {
            return Ok(deferred);
        }
        Ok(Py::new(py, self.binary(&self.coerce(py, other)?, "add")?)?.into_any())
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, self.binary(&self.coerce(py, other)?, "add")?)?.into_any())
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Some(deferred) = self.defer_to_outer_polynomial(py, other) {
            return Ok(deferred);
        }
        Ok(Py::new(py, self.binary(&self.coerce(py, other)?, "sub")?)?.into_any())
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, self.coerce(py, other)?.binary(self, "sub")?)?.into_any())
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Some(deferred) = self.defer_to_outer_polynomial(py, other) {
            return Ok(deferred);
        }
        Ok(Py::new(py, self.binary(&self.coerce(py, other)?, "mul")?)?.into_any())
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, self.binary(&self.coerce(py, other)?, "mul")?)?.into_any())
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce(py, other)?, "div")
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.coerce(py, other)?.binary(self, "div")
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            RationalFunctionInner::Rational(value) => {
                RationalFunctionInner::Rational((-value).map_err(rational_function_error)?)
            }
            RationalFunctionInner::FiniteField(value) => {
                RationalFunctionInner::FiniteField((-value).map_err(rational_function_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn __pow__(&self, exponent: i64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "RationalFunction power does not accept Python's third modulus argument",
            ));
        }
        self.pow_signed(exponent)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            RationalFunctionInner::Rational(value) => canonical_hash(value),
            RationalFunctionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "RationalFunction(variable={:?}, numerator_terms={}, denominator_terms={})",
                self.field_value().variable(),
                self.numerator().term_count(),
                self.denominator().term_count()
            )
        })
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", "noether/commalg/RationalFunction@1")?;
        summary.set_item("variable", self.field_value().variable())?;
        summary.set_item(
            "numerator_nonzero_term_count",
            self.numerator().term_count(),
        )?;
        summary.set_item(
            "denominator_nonzero_term_count",
            self.denominator().term_count(),
        )?;
        Ok(summary.unbind())
    }
}
