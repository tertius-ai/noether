// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of elliptic curves, Miller functions, and power classes.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]
#![allow(clippy::large_enum_variant)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{RationalFunction, RationalFunctionField};
use noether::numthy::{
    EllipticCurve, EllipticCurveError, EllipticCurvePoint, FiniteField, MillerFunction,
    NthPowerClass, NthPowerClassGroup, NumberField, PowerClassError,
};
use noether::{Integer, RationalField, SemanticType, UnsupportedOperationError, Q};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};

use super::curve::{PyCurve, PyCurveFunction, PyCurvePoint, PyWeilDivisor};
use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::number_field::{extract_number_field_element, PyNumberField, PyNumberFieldElement};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::rational_function::{PyRationalFunction, PyRationalFunctionField};
use super::root::PyRationalField;

type RationalFunctions = RationalFunctionField<RationalField>;
type FiniteFunctions = RationalFunctionField<FiniteField>;

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

fn elliptic_error(error: EllipticCurveError) -> PyErr {
    match error {
        EllipticCurveError::ContextMismatch(error) => super::errors::context_mismatch(error),
        EllipticCurveError::Curve(noether::commalg::CurveError::ContextMismatch(error)) => {
            super::errors::context_mismatch(error)
        }
        EllipticCurveError::Inversion(_) => PyZeroDivisionError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn power_error(error: PowerClassError) -> PyErr {
    match error {
        PowerClassError::ContextMismatch(error) => super::errors::context_mismatch(error),
        PowerClassError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[derive(Clone, Copy)]
enum EllipticContext {
    Rational,
    Finite,
    NumberField,
    RationalFunction,
    FiniteFunction,
}

fn context_from_field_json(field: &serde_json::Value) -> PyResult<EllipticContext> {
    let type_id = field
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("serialized field context has no type"))?;
    match type_id {
        RationalField::TYPE_ID => Ok(EllipticContext::Rational),
        FiniteField::TYPE_ID => Ok(EllipticContext::Finite),
        NumberField::TYPE_ID => Ok(EllipticContext::NumberField),
        RationalFunctionField::<RationalField>::TYPE_ID => {
            let coefficient_type = field
                .get("coefficient_ring")
                .and_then(|value| value.get("type"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "serialized rational-function field has no coefficient context",
                    )
                })?;
            match coefficient_type {
                RationalField::TYPE_ID => Ok(EllipticContext::RationalFunction),
                FiniteField::TYPE_ID => Ok(EllipticContext::FiniteFunction),
                _ => Err(PyTypeError::new_err(
                    "unsupported rational-function coefficient context",
                )),
            }
        }
        _ => Err(PyTypeError::new_err(
            "unsupported elliptic-curve coefficient context",
        )),
    }
}

fn find_elliptic_base(value: &serde_json::Value) -> Option<&serde_json::Value> {
    match value {
        serde_json::Value::Object(object) => {
            if object.get("type").and_then(serde_json::Value::as_str)
                == Some(EllipticCurve::<RationalField>::TYPE_ID)
            {
                return object.get("base");
            }
            object.values().find_map(find_elliptic_base)
        }
        serde_json::Value::Array(values) => values.iter().find_map(find_elliptic_base),
        _ => None,
    }
}

fn serialized_elliptic_context(value: &str) -> PyResult<EllipticContext> {
    let value: serde_json::Value =
        serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    context_from_field_json(find_elliptic_base(&value).ok_or_else(|| {
        PyValueError::new_err("serialized value contains no elliptic-curve base field")
    })?)
}

fn serialized_power_context(value: &str) -> PyResult<EllipticContext> {
    let value: serde_json::Value =
        serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    context_from_field_json(
        value.get("field").ok_or_else(|| {
            PyValueError::new_err("serialized power-class value contains no field")
        })?,
    )
}

fn coerce_rational_function(
    py: Python<'_>,
    field: &RationalFunctions,
    value: &Bound<'_, PyAny>,
) -> PyResult<RationalFunction<RationalField>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        if let Some(value) = value.as_rational().filter(|value| value.field() == field) {
            return Ok(value.clone());
        }
    }
    field
        .constant(extract_rational(py, value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn coerce_finite_function(
    field: &FiniteFunctions,
    value: &Bound<'_, PyAny>,
) -> PyResult<RationalFunction<FiniteField>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        if let Some(value) = value
            .as_finite_field()
            .filter(|value| value.field() == field)
        {
            return Ok(value.clone());
        }
    }
    field
        .constant(super::polynomial_facade::extract_finite_field(
            field.coefficient_ring(),
            value,
        )?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[derive(Clone, PartialEq, Eq)]
enum EllipticCurveInner {
    Rational(EllipticCurve<RationalField>),
    Finite(EllipticCurve<FiniteField>),
    NumberField(EllipticCurve<NumberField>),
    RationalFunction(EllipticCurve<RationalFunctions>),
    FiniteFunction(EllipticCurve<FiniteFunctions>),
}

/// A nonsingular generalized Weierstrass elliptic curve.
#[pyclass(
    module = "noether.numthy",
    name = "EllipticCurve",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEllipticCurve {
    inner: EllipticCurveInner,
}

impl PyEllipticCurve {
    fn context_name(&self) -> &'static str {
        match &self.inner {
            EllipticCurveInner::Rational(_) => RationalField::TYPE_ID,
            EllipticCurveInner::Finite(_) => FiniteField::TYPE_ID,
            EllipticCurveInner::NumberField(_) => NumberField::TYPE_ID,
            EllipticCurveInner::RationalFunction(_) | EllipticCurveInner::FiniteFunction(_) => {
                RationalFunctionField::<RationalField>::TYPE_ID
            }
        }
    }
}

#[pymethods]
impl PyEllipticCurve {
    #[new]
    fn new(
        py: Python<'_>,
        base: &Bound<'_, PyAny>,
        coefficients: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let values = coefficients
            .try_iter()
            .map_err(|_| PyTypeError::new_err("elliptic coefficients must be iterable"))?
            .collect::<PyResult<Vec<_>>>()?;
        if values.len() != 5 {
            return Err(PyValueError::new_err(
                "a generalized Weierstrass equation needs five coefficients",
            ));
        }
        let inner = if base.is_instance_of::<PyRationalField>() {
            let coefficients = values
                .iter()
                .map(|value| extract_rational(py, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            EllipticCurveInner::Rational(
                EllipticCurve::new(Q, coefficients).map_err(elliptic_error)?,
            )
        } else if let Ok(field) = base.extract::<PyRef<'_, PyFiniteField>>() {
            let coefficients = values
                .iter()
                .map(|value| super::polynomial_facade::extract_finite_field(&field.inner, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            EllipticCurveInner::Finite(
                EllipticCurve::new(field.inner.clone(), coefficients).map_err(elliptic_error)?,
            )
        } else if let Ok(field) = base.extract::<PyRef<'_, PyNumberField>>() {
            let coefficients = values
                .iter()
                .map(|value| extract_number_field_element(py, &field.inner, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            EllipticCurveInner::NumberField(
                EllipticCurve::new(field.inner.clone(), coefficients).map_err(elliptic_error)?,
            )
        } else if let Ok(field) = base.extract::<PyRef<'_, PyRationalFunctionField>>() {
            if let Some(context) = field.as_rational() {
                let coefficients = values
                    .iter()
                    .map(|value| coerce_rational_function(py, context, value))
                    .collect::<PyResult<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
                EllipticCurveInner::RationalFunction(
                    EllipticCurve::new(context.clone(), coefficients).map_err(elliptic_error)?,
                )
            } else if let Some(context) = field.as_finite_field() {
                let coefficients = values
                    .iter()
                    .map(|value| coerce_finite_function(context, value))
                    .collect::<PyResult<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
                EllipticCurveInner::FiniteFunction(
                    EllipticCurve::new(context.clone(), coefficients).map_err(elliptic_error)?,
                )
            } else {
                return Err(PyTypeError::new_err(
                    "unsupported rational-function coefficient field",
                ));
            }
        } else {
            return Err(PyTypeError::new_err(
                "EllipticCurve supports Q, finite fields, number fields, Q(t), and finite-field rational-function fields",
            ));
        };
        Ok(Self { inner })
    }

    #[classattr]
    fn type_id() -> &'static str {
        EllipticCurve::<RationalField>::TYPE_ID
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            EllipticCurveInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            EllipticCurveInner::Finite(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.base().clone(),
                },
            )?
            .into_any()),
            EllipticCurveInner::NumberField(value) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: value.base().clone(),
                },
            )?
            .into_any()),
            EllipticCurveInner::RationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(value.base().clone()),
            )?
            .into_any()),
            EllipticCurveInner::FiniteFunction(value) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_finite_field(value.base().clone()),
            )?
            .into_any()),
        }
    }

    #[getter]
    fn underlying_curve(&self) -> PyCurve {
        match &self.inner {
            EllipticCurveInner::Rational(value) => {
                PyCurve::from_rational(value.underlying_curve().clone())
            }
            EllipticCurveInner::Finite(value) => {
                PyCurve::from_finite(value.underlying_curve().clone())
            }
            EllipticCurveInner::NumberField(value) => {
                PyCurve::from_number_field(value.underlying_curve().clone())
            }
            EllipticCurveInner::RationalFunction(value) => {
                PyCurve::from_rational_function(value.underlying_curve().clone())
            }
            EllipticCurveInner::FiniteFunction(value) => {
                PyCurve::from_finite_function(value.underlying_curve().clone())
            }
        }
    }

    #[getter]
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            EllipticCurveInner::Rational(value) => value
                .coefficients()
                .iter()
                .map(|value| rational_to_python(py, value))
                .collect(),
            EllipticCurveInner::Finite(value) => value
                .coefficients()
                .iter()
                .cloned()
                .map(|inner| Py::new(py, PyFiniteFieldElement { inner }).map(Py::into_any))
                .collect(),
            EllipticCurveInner::NumberField(value) => value
                .coefficients()
                .iter()
                .cloned()
                .map(|inner| Py::new(py, PyNumberFieldElement { inner }).map(Py::into_any))
                .collect(),
            EllipticCurveInner::RationalFunction(value) => value
                .coefficients()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyRationalFunction::from_rational(inner)).map(Py::into_any)
                })
                .collect(),
            EllipticCurveInner::FiniteFunction(value) => value
                .coefficients()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyRationalFunction::from_finite_field(inner)).map(Py::into_any)
                })
                .collect(),
        }
    }

    #[getter]
    fn discriminant(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            EllipticCurveInner::Rational(value) => rational_to_python(py, value.discriminant()),
            EllipticCurveInner::Finite(value) => Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.discriminant().clone(),
                },
            )
            .map(Py::into_any),
            EllipticCurveInner::NumberField(value) => Py::new(
                py,
                PyNumberFieldElement {
                    inner: value.discriminant().clone(),
                },
            )
            .map(Py::into_any),
            EllipticCurveInner::RationalFunction(value) => Py::new(
                py,
                PyRationalFunction::from_rational(value.discriminant().clone()),
            )
            .map(Py::into_any),
            EllipticCurveInner::FiniteFunction(value) => Py::new(
                py,
                PyRationalFunction::from_finite_field(value.discriminant().clone()),
            )
            .map(Py::into_any),
        }
    }

    fn origin(&self) -> PyResult<PyEllipticCurvePoint> {
        let inner = match &self.inner {
            EllipticCurveInner::Rational(value) => {
                EllipticPointInner::Rational(value.origin().map_err(elliptic_error)?)
            }
            EllipticCurveInner::Finite(value) => {
                EllipticPointInner::Finite(value.origin().map_err(elliptic_error)?)
            }
            EllipticCurveInner::NumberField(value) => {
                EllipticPointInner::NumberField(value.origin().map_err(elliptic_error)?)
            }
            EllipticCurveInner::RationalFunction(value) => {
                EllipticPointInner::RationalFunction(value.origin().map_err(elliptic_error)?)
            }
            EllipticCurveInner::FiniteFunction(value) => {
                EllipticPointInner::FiniteFunction(value.origin().map_err(elliptic_error)?)
            }
        };
        Ok(PyEllipticCurvePoint { inner })
    }

    fn point(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<PyEllipticCurvePoint> {
        let inner = match &self.inner {
            EllipticCurveInner::Rational(value) => EllipticPointInner::Rational(
                value
                    .point(extract_rational(py, x)?, extract_rational(py, y)?)
                    .map_err(elliptic_error)?,
            ),
            EllipticCurveInner::Finite(value) => EllipticPointInner::Finite(
                value
                    .point(
                        super::polynomial_facade::extract_finite_field(value.base(), x)?,
                        super::polynomial_facade::extract_finite_field(value.base(), y)?,
                    )
                    .map_err(elliptic_error)?,
            ),
            EllipticCurveInner::NumberField(value) => EllipticPointInner::NumberField(
                value
                    .point(
                        extract_number_field_element(py, value.base(), x)?,
                        extract_number_field_element(py, value.base(), y)?,
                    )
                    .map_err(elliptic_error)?,
            ),
            EllipticCurveInner::RationalFunction(value) => EllipticPointInner::RationalFunction(
                value
                    .point(
                        coerce_rational_function(py, value.base(), x)?,
                        coerce_rational_function(py, value.base(), y)?,
                    )
                    .map_err(elliptic_error)?,
            ),
            EllipticCurveInner::FiniteFunction(value) => EllipticPointInner::FiniteFunction(
                value
                    .point(
                        coerce_finite_function(value.base(), x)?,
                        coerce_finite_function(value.base(), y)?,
                    )
                    .map_err(elliptic_error)?,
            ),
        };
        Ok(PyEllipticCurvePoint { inner })
    }

    fn contains(&self, point: &PyEllipticCurvePoint) -> bool {
        matches!(
            (&self.inner, &point.inner),
            (EllipticCurveInner::Rational(c), EllipticPointInner::Rational(p)) if c.contains(p)
        ) || matches!(
            (&self.inner, &point.inner),
            (EllipticCurveInner::Finite(c), EllipticPointInner::Finite(p)) if c.contains(p)
        ) || matches!(
            (&self.inner, &point.inner),
            (EllipticCurveInner::NumberField(c), EllipticPointInner::NumberField(p)) if c.contains(p)
        ) || matches!(
            (&self.inner, &point.inner),
            (EllipticCurveInner::RationalFunction(c), EllipticPointInner::RationalFunction(p)) if c.contains(p)
        ) || matches!(
            (&self.inner, &point.inner),
            (EllipticCurveInner::FiniteFunction(c), EllipticPointInner::FiniteFunction(p)) if c.contains(p)
        )
    }

    fn x_function(&self) -> PyResult<PyCurveFunction> {
        match &self.inner {
            EllipticCurveInner::Rational(v) => v
                .x_function()
                .map(PyCurveFunction::from_rational)
                .map_err(elliptic_error),
            EllipticCurveInner::Finite(v) => v
                .x_function()
                .map(PyCurveFunction::from_finite)
                .map_err(elliptic_error),
            EllipticCurveInner::NumberField(v) => v
                .x_function()
                .map(PyCurveFunction::from_number_field)
                .map_err(elliptic_error),
            EllipticCurveInner::RationalFunction(v) => v
                .x_function()
                .map(PyCurveFunction::from_rational_function)
                .map_err(elliptic_error),
            EllipticCurveInner::FiniteFunction(v) => v
                .x_function()
                .map(PyCurveFunction::from_finite_function)
                .map_err(elliptic_error),
        }
    }

    fn y_function(&self) -> PyResult<PyCurveFunction> {
        match &self.inner {
            EllipticCurveInner::Rational(v) => v
                .y_function()
                .map(PyCurveFunction::from_rational)
                .map_err(elliptic_error),
            EllipticCurveInner::Finite(v) => v
                .y_function()
                .map(PyCurveFunction::from_finite)
                .map_err(elliptic_error),
            EllipticCurveInner::NumberField(v) => v
                .y_function()
                .map(PyCurveFunction::from_number_field)
                .map_err(elliptic_error),
            EllipticCurveInner::RationalFunction(v) => v
                .y_function()
                .map(PyCurveFunction::from_rational_function)
                .map_err(elliptic_error),
            EllipticCurveInner::FiniteFunction(v) => v
                .y_function()
                .map(PyCurveFunction::from_finite_function)
                .map_err(elliptic_error),
        }
    }

    fn vertical_line_function(&self, point: &PyEllipticCurvePoint) -> PyResult<PyCurveFunction> {
        match (&self.inner, &point.inner) {
            (EllipticCurveInner::Rational(c), EllipticPointInner::Rational(p)) => c
                .vertical_line_function(p)
                .map(PyCurveFunction::from_rational)
                .map_err(elliptic_error),
            (EllipticCurveInner::Finite(c), EllipticPointInner::Finite(p)) => c
                .vertical_line_function(p)
                .map(PyCurveFunction::from_finite)
                .map_err(elliptic_error),
            (EllipticCurveInner::NumberField(c), EllipticPointInner::NumberField(p)) => c
                .vertical_line_function(p)
                .map(PyCurveFunction::from_number_field)
                .map_err(elliptic_error),
            (EllipticCurveInner::RationalFunction(c), EllipticPointInner::RationalFunction(p)) => c
                .vertical_line_function(p)
                .map(PyCurveFunction::from_rational_function)
                .map_err(elliptic_error),
            (EllipticCurveInner::FiniteFunction(c), EllipticPointInner::FiniteFunction(p)) => c
                .vertical_line_function(p)
                .map(PyCurveFunction::from_finite_function)
                .map_err(elliptic_error),
            _ => Err(PyTypeError::new_err(
                "point belongs to another elliptic-curve context",
            )),
        }
    }

    fn line_function(
        &self,
        left: &PyEllipticCurvePoint,
        right: &PyEllipticCurvePoint,
    ) -> PyResult<PyCurveFunction> {
        match (&self.inner, &left.inner, &right.inner) {
            (
                EllipticCurveInner::Rational(c),
                EllipticPointInner::Rational(p),
                EllipticPointInner::Rational(q),
            ) => c
                .line_function(p, q)
                .map(PyCurveFunction::from_rational)
                .map_err(elliptic_error),
            (
                EllipticCurveInner::Finite(c),
                EllipticPointInner::Finite(p),
                EllipticPointInner::Finite(q),
            ) => c
                .line_function(p, q)
                .map(PyCurveFunction::from_finite)
                .map_err(elliptic_error),
            (
                EllipticCurveInner::NumberField(c),
                EllipticPointInner::NumberField(p),
                EllipticPointInner::NumberField(q),
            ) => c
                .line_function(p, q)
                .map(PyCurveFunction::from_number_field)
                .map_err(elliptic_error),
            (
                EllipticCurveInner::RationalFunction(c),
                EllipticPointInner::RationalFunction(p),
                EllipticPointInner::RationalFunction(q),
            ) => c
                .line_function(p, q)
                .map(PyCurveFunction::from_rational_function)
                .map_err(elliptic_error),
            (
                EllipticCurveInner::FiniteFunction(c),
                EllipticPointInner::FiniteFunction(p),
                EllipticPointInner::FiniteFunction(q),
            ) => c
                .line_function(p, q)
                .map(PyCurveFunction::from_finite_function)
                .map_err(elliptic_error),
            _ => Err(PyTypeError::new_err(
                "points belong to another elliptic-curve context",
            )),
        }
    }

    /// Compute Miller's function with divisor `n(P) - (nP) - (n-1)(O)`.
    fn miller_function(
        &self,
        point: &PyEllipticCurvePoint,
        multiplier: &Bound<'_, PyAny>,
    ) -> PyResult<PyMillerFunction> {
        let multiplier = extract_integer(multiplier)?;
        let inner = match (&self.inner, &point.inner) {
            (EllipticCurveInner::Rational(c), EllipticPointInner::Rational(p)) => {
                MillerInner::Rational(c.miller_function(p, &multiplier).map_err(elliptic_error)?)
            }
            (EllipticCurveInner::Finite(c), EllipticPointInner::Finite(p)) => {
                MillerInner::Finite(c.miller_function(p, &multiplier).map_err(elliptic_error)?)
            }
            (EllipticCurveInner::NumberField(c), EllipticPointInner::NumberField(p)) => {
                MillerInner::NumberField(c.miller_function(p, &multiplier).map_err(elliptic_error)?)
            }
            (EllipticCurveInner::RationalFunction(c), EllipticPointInner::RationalFunction(p)) => {
                MillerInner::RationalFunction(
                    c.miller_function(p, &multiplier).map_err(elliptic_error)?,
                )
            }
            (EllipticCurveInner::FiniteFunction(c), EllipticPointInner::FiniteFunction(p)) => {
                MillerInner::FiniteFunction(
                    c.miller_function(p, &multiplier).map_err(elliptic_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "point belongs to another elliptic-curve context",
                ))
            }
        };
        Ok(PyMillerFunction { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            EllipticCurveInner::Rational(v) => v.verify(),
            EllipticCurveInner::Finite(v) => v.verify(),
            EllipticCurveInner::NumberField(v) => v.verify(),
            EllipticCurveInner::RationalFunction(v) => v.verify(),
            EllipticCurveInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            EllipticCurveInner::Rational(v) => v.to_json(),
            EllipticCurveInner::Finite(v) => v.to_json(),
            EllipticCurveInner::NumberField(v) => v.to_json(),
            EllipticCurveInner::RationalFunction(v) => v.to_json(),
            EllipticCurveInner::FiniteFunction(v) => v.to_json(),
        }
        .map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_elliptic_context(value)? {
            EllipticContext::Rational => EllipticCurveInner::Rational(
                EllipticCurve::<RationalField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::Finite => EllipticCurveInner::Finite(
                EllipticCurve::<FiniteField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::NumberField => EllipticCurveInner::NumberField(
                EllipticCurve::<NumberField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::RationalFunction => EllipticCurveInner::RationalFunction(
                EllipticCurve::<RationalFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::FiniteFunction => EllipticCurveInner::FiniteFunction(
                EllipticCurve::<FiniteFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            EllipticCurveInner::Rational(v) => canonical_hash(v),
            EllipticCurveInner::Finite(v) => canonical_hash(v),
            EllipticCurveInner::NumberField(v) => canonical_hash(v),
            EllipticCurveInner::RationalFunction(v) => canonical_hash(v),
            EllipticCurveInner::FiniteFunction(v) => canonical_hash(v),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("EllipticCurve(base={})", self.context_name()))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EllipticCurve::<RationalField>::TYPE_ID)?;
        summary.set_item("base_type", self.context_name())?;
        summary.set_item("model", "generalized_weierstrass")?;
        summary.set_item("nonsingular", true)?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum EllipticPointInner {
    Rational(EllipticCurvePoint<RationalField>),
    Finite(EllipticCurvePoint<FiniteField>),
    NumberField(EllipticCurvePoint<NumberField>),
    RationalFunction(EllipticCurvePoint<RationalFunctions>),
    FiniteFunction(EllipticCurvePoint<FiniteFunctions>),
}

/// An affine point or the chosen origin on an elliptic curve.
#[pyclass(
    module = "noether.numthy",
    name = "EllipticCurvePoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEllipticCurvePoint {
    inner: EllipticPointInner,
}

impl PyEllipticCurvePoint {
    fn binary(&self, other: &Self, subtract: bool) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {{
                let value = if subtract {
                    ($left - $right).map_err(elliptic_error)?
                } else {
                    ($left + $right).map_err(elliptic_error)?
                };
                EllipticPointInner::$variant(value)
            }};
        }
        let inner = match (&self.inner, &other.inner) {
            (EllipticPointInner::Rational(a), EllipticPointInner::Rational(b)) => {
                apply!(a, b, Rational)
            }
            (EllipticPointInner::Finite(a), EllipticPointInner::Finite(b)) => {
                apply!(a, b, Finite)
            }
            (EllipticPointInner::NumberField(a), EllipticPointInner::NumberField(b)) => {
                apply!(a, b, NumberField)
            }
            (EllipticPointInner::RationalFunction(a), EllipticPointInner::RationalFunction(b)) => {
                apply!(a, b, RationalFunction)
            }
            (EllipticPointInner::FiniteFunction(a), EllipticPointInner::FiniteFunction(b)) => {
                apply!(a, b, FiniteFunction)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "elliptic points belong to different contexts",
                ))
            }
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyEllipticCurvePoint {
    #[classattr]
    fn type_id() -> &'static str {
        EllipticCurvePoint::<RationalField>::TYPE_ID
    }

    #[getter]
    fn curve(&self) -> PyEllipticCurve {
        let inner = match &self.inner {
            EllipticPointInner::Rational(v) => EllipticCurveInner::Rational(v.curve().clone()),
            EllipticPointInner::Finite(v) => EllipticCurveInner::Finite(v.curve().clone()),
            EllipticPointInner::NumberField(v) => {
                EllipticCurveInner::NumberField(v.curve().clone())
            }
            EllipticPointInner::RationalFunction(v) => {
                EllipticCurveInner::RationalFunction(v.curve().clone())
            }
            EllipticPointInner::FiniteFunction(v) => {
                EllipticCurveInner::FiniteFunction(v.curve().clone())
            }
        };
        PyEllipticCurve { inner }
    }

    #[getter]
    fn is_infinity(&self) -> bool {
        match &self.inner {
            EllipticPointInner::Rational(v) => v.is_infinity(),
            EllipticPointInner::Finite(v) => v.is_infinity(),
            EllipticPointInner::NumberField(v) => v.is_infinity(),
            EllipticPointInner::RationalFunction(v) => v.is_infinity(),
            EllipticPointInner::FiniteFunction(v) => v.is_infinity(),
        }
    }

    #[getter]
    fn affine_coordinates(&self, py: Python<'_>) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        match &self.inner {
            EllipticPointInner::Rational(v) => v
                .affine_coordinates()
                .map(|(x, y)| Ok((rational_to_python(py, x)?, rational_to_python(py, y)?)))
                .transpose(),
            EllipticPointInner::Finite(v) => v
                .affine_coordinates()
                .map(|(x, y)| {
                    Ok((
                        Py::new(py, PyFiniteFieldElement { inner: x.clone() })?.into_any(),
                        Py::new(py, PyFiniteFieldElement { inner: y.clone() })?.into_any(),
                    ))
                })
                .transpose(),
            EllipticPointInner::NumberField(v) => v
                .affine_coordinates()
                .map(|(x, y)| {
                    Ok((
                        Py::new(py, PyNumberFieldElement { inner: x.clone() })?.into_any(),
                        Py::new(py, PyNumberFieldElement { inner: y.clone() })?.into_any(),
                    ))
                })
                .transpose(),
            EllipticPointInner::RationalFunction(v) => v
                .affine_coordinates()
                .map(|(x, y)| {
                    Ok((
                        Py::new(py, PyRationalFunction::from_rational(x.clone()))?.into_any(),
                        Py::new(py, PyRationalFunction::from_rational(y.clone()))?.into_any(),
                    ))
                })
                .transpose(),
            EllipticPointInner::FiniteFunction(v) => v
                .affine_coordinates()
                .map(|(x, y)| {
                    Ok((
                        Py::new(py, PyRationalFunction::from_finite_field(x.clone()))?.into_any(),
                        Py::new(py, PyRationalFunction::from_finite_field(y.clone()))?.into_any(),
                    ))
                })
                .transpose(),
        }
    }

    #[getter]
    fn underlying_curve_point(&self) -> PyCurvePoint {
        match &self.inner {
            EllipticPointInner::Rational(v) => {
                PyCurvePoint::from_rational(v.underlying_curve_point().clone())
            }
            EllipticPointInner::Finite(v) => {
                PyCurvePoint::from_finite(v.underlying_curve_point().clone())
            }
            EllipticPointInner::NumberField(v) => {
                PyCurvePoint::from_number_field(v.underlying_curve_point().clone())
            }
            EllipticPointInner::RationalFunction(v) => {
                PyCurvePoint::from_rational_function(v.underlying_curve_point().clone())
            }
            EllipticPointInner::FiniteFunction(v) => {
                PyCurvePoint::from_finite_function(v.underlying_curve_point().clone())
            }
        }
    }

    fn scalar_mul(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_integer(scalar)?;
        let inner = match &self.inner {
            EllipticPointInner::Rational(v) => {
                EllipticPointInner::Rational(v.scalar_mul(&scalar).map_err(elliptic_error)?)
            }
            EllipticPointInner::Finite(v) => {
                EllipticPointInner::Finite(v.scalar_mul(&scalar).map_err(elliptic_error)?)
            }
            EllipticPointInner::NumberField(v) => {
                EllipticPointInner::NumberField(v.scalar_mul(&scalar).map_err(elliptic_error)?)
            }
            EllipticPointInner::RationalFunction(v) => {
                EllipticPointInner::RationalFunction(v.scalar_mul(&scalar).map_err(elliptic_error)?)
            }
            EllipticPointInner::FiniteFunction(v) => {
                EllipticPointInner::FiniteFunction(v.scalar_mul(&scalar).map_err(elliptic_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn has_exact_order(&self, candidate: &Bound<'_, PyAny>) -> PyResult<bool> {
        let candidate = extract_integer(candidate)?;
        match &self.inner {
            EllipticPointInner::Rational(v) => v.has_exact_order(&candidate),
            EllipticPointInner::Finite(v) => v.has_exact_order(&candidate),
            EllipticPointInner::NumberField(v) => v.has_exact_order(&candidate),
            EllipticPointInner::RationalFunction(v) => v.has_exact_order(&candidate),
            EllipticPointInner::FiniteFunction(v) => v.has_exact_order(&candidate),
        }
        .map_err(elliptic_error)
    }

    fn verify_exact_order(&self, candidate: &Bound<'_, PyAny>) -> PyResult<()> {
        let candidate = extract_integer(candidate)?;
        match &self.inner {
            EllipticPointInner::Rational(v) => v.verify_exact_order(&candidate),
            EllipticPointInner::Finite(v) => v.verify_exact_order(&candidate),
            EllipticPointInner::NumberField(v) => v.verify_exact_order(&candidate),
            EllipticPointInner::RationalFunction(v) => v.verify_exact_order(&candidate),
            EllipticPointInner::FiniteFunction(v) => v.verify_exact_order(&candidate),
        }
        .map_err(elliptic_error)
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            EllipticPointInner::Rational(v) => v.verify(),
            EllipticPointInner::Finite(v) => v.verify(),
            EllipticPointInner::NumberField(v) => v.verify(),
            EllipticPointInner::RationalFunction(v) => v.verify(),
            EllipticPointInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            EllipticPointInner::Rational(v) => v.to_json(),
            EllipticPointInner::Finite(v) => v.to_json(),
            EllipticPointInner::NumberField(v) => v.to_json(),
            EllipticPointInner::RationalFunction(v) => v.to_json(),
            EllipticPointInner::FiniteFunction(v) => v.to_json(),
        }
        .map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_elliptic_context(value)? {
            EllipticContext::Rational => EllipticPointInner::Rational(
                EllipticCurvePoint::<RationalField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::Finite => EllipticPointInner::Finite(
                EllipticCurvePoint::<FiniteField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::NumberField => EllipticPointInner::NumberField(
                EllipticCurvePoint::<NumberField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::RationalFunction => EllipticPointInner::RationalFunction(
                EllipticCurvePoint::<RationalFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::FiniteFunction => EllipticPointInner::FiniteFunction(
                EllipticCurvePoint::<FiniteFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other, false)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other, true)
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            EllipticPointInner::Rational(v) => {
                EllipticPointInner::Rational(v.negated().map_err(elliptic_error)?)
            }
            EllipticPointInner::Finite(v) => {
                EllipticPointInner::Finite(v.negated().map_err(elliptic_error)?)
            }
            EllipticPointInner::NumberField(v) => {
                EllipticPointInner::NumberField(v.negated().map_err(elliptic_error)?)
            }
            EllipticPointInner::RationalFunction(v) => {
                EllipticPointInner::RationalFunction(v.negated().map_err(elliptic_error)?)
            }
            EllipticPointInner::FiniteFunction(v) => {
                EllipticPointInner::FiniteFunction(v.negated().map_err(elliptic_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scalar_mul(scalar)
    }

    fn __rmul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scalar_mul(scalar)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            EllipticPointInner::Rational(v) => canonical_hash(v),
            EllipticPointInner::Finite(v) => canonical_hash(v),
            EllipticPointInner::NumberField(v) => canonical_hash(v),
            EllipticPointInner::RationalFunction(v) => canonical_hash(v),
            EllipticPointInner::FiniteFunction(v) => canonical_hash(v),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EllipticCurvePoint(infinity={})",
            self.is_infinity()
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EllipticCurvePoint::<RationalField>::TYPE_ID)?;
        summary.set_item("infinity", self.is_infinity())?;
        summary.set_item("coordinate_count", if self.is_infinity() { 0 } else { 2 })?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum MillerInner {
    Rational(MillerFunction<RationalField>),
    Finite(MillerFunction<FiniteField>),
    NumberField(MillerFunction<NumberField>),
    RationalFunction(MillerFunction<RationalFunctions>),
    FiniteFunction(MillerFunction<FiniteFunctions>),
}

/// A Miller function with its exact divisor formula.
#[pyclass(
    module = "noether.numthy",
    name = "MillerFunction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMillerFunction {
    inner: MillerInner,
}

impl PyMillerFunction {
    fn multiplier_value(&self) -> &Integer {
        match &self.inner {
            MillerInner::Rational(v) => v.multiplier(),
            MillerInner::Finite(v) => v.multiplier(),
            MillerInner::NumberField(v) => v.multiplier(),
            MillerInner::RationalFunction(v) => v.multiplier(),
            MillerInner::FiniteFunction(v) => v.multiplier(),
        }
    }
}

#[pymethods]
impl PyMillerFunction {
    #[classattr]
    fn type_id() -> &'static str {
        MillerFunction::<RationalField>::TYPE_ID
    }

    #[getter]
    fn curve(&self) -> PyEllipticCurve {
        let inner = match &self.inner {
            MillerInner::Rational(v) => EllipticCurveInner::Rational(v.curve().clone()),
            MillerInner::Finite(v) => EllipticCurveInner::Finite(v.curve().clone()),
            MillerInner::NumberField(v) => EllipticCurveInner::NumberField(v.curve().clone()),
            MillerInner::RationalFunction(v) => {
                EllipticCurveInner::RationalFunction(v.curve().clone())
            }
            MillerInner::FiniteFunction(v) => EllipticCurveInner::FiniteFunction(v.curve().clone()),
        };
        PyEllipticCurve { inner }
    }

    #[getter]
    fn point(&self) -> PyEllipticCurvePoint {
        let inner = match &self.inner {
            MillerInner::Rational(v) => EllipticPointInner::Rational(v.point().clone()),
            MillerInner::Finite(v) => EllipticPointInner::Finite(v.point().clone()),
            MillerInner::NumberField(v) => EllipticPointInner::NumberField(v.point().clone()),
            MillerInner::RationalFunction(v) => {
                EllipticPointInner::RationalFunction(v.point().clone())
            }
            MillerInner::FiniteFunction(v) => EllipticPointInner::FiniteFunction(v.point().clone()),
        };
        PyEllipticCurvePoint { inner }
    }

    #[getter]
    fn multiplier(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.multiplier_value())
    }

    #[getter]
    fn terminal_point(&self) -> PyEllipticCurvePoint {
        let inner = match &self.inner {
            MillerInner::Rational(v) => EllipticPointInner::Rational(v.terminal_point().clone()),
            MillerInner::Finite(v) => EllipticPointInner::Finite(v.terminal_point().clone()),
            MillerInner::NumberField(v) => {
                EllipticPointInner::NumberField(v.terminal_point().clone())
            }
            MillerInner::RationalFunction(v) => {
                EllipticPointInner::RationalFunction(v.terminal_point().clone())
            }
            MillerInner::FiniteFunction(v) => {
                EllipticPointInner::FiniteFunction(v.terminal_point().clone())
            }
        };
        PyEllipticCurvePoint { inner }
    }

    #[getter]
    fn function(&self) -> PyCurveFunction {
        match &self.inner {
            MillerInner::Rational(v) => {
                PyCurveFunction::from_rational(v.as_curve_function().clone())
            }
            MillerInner::Finite(v) => PyCurveFunction::from_finite(v.as_curve_function().clone()),
            MillerInner::NumberField(v) => {
                PyCurveFunction::from_number_field(v.as_curve_function().clone())
            }
            MillerInner::RationalFunction(v) => {
                PyCurveFunction::from_rational_function(v.as_curve_function().clone())
            }
            MillerInner::FiniteFunction(v) => {
                PyCurveFunction::from_finite_function(v.as_curve_function().clone())
            }
        }
    }

    #[getter]
    fn divisor(&self) -> PyResult<PyWeilDivisor> {
        match &self.inner {
            MillerInner::Rational(v) => v
                .divisor_formula()
                .map(PyWeilDivisor::from_rational)
                .map_err(elliptic_error),
            MillerInner::Finite(v) => v
                .divisor_formula()
                .map(PyWeilDivisor::from_finite)
                .map_err(elliptic_error),
            MillerInner::NumberField(v) => v
                .divisor_formula()
                .map(PyWeilDivisor::from_number_field)
                .map_err(elliptic_error),
            MillerInner::RationalFunction(v) => v
                .divisor_formula()
                .map(PyWeilDivisor::from_rational_function)
                .map_err(elliptic_error),
            MillerInner::FiniteFunction(v) => v
                .divisor_formula()
                .map(PyWeilDivisor::from_finite_function)
                .map_err(elliptic_error),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            MillerInner::Rational(v) => v.verify(),
            MillerInner::Finite(v) => v.verify(),
            MillerInner::NumberField(v) => v.verify(),
            MillerInner::RationalFunction(v) => v.verify(),
            MillerInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn verify_principal_divisor(&self) -> PyResult<()> {
        match &self.inner {
            MillerInner::Rational(v) => v
                .verify_principal_divisor()
                .map_err(super::errors::verification),
            MillerInner::Finite(v) => v
                .verify_principal_divisor()
                .map_err(super::errors::verification),
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/numthy/MillerFunction/verify_principal_divisor@1",
                    self.curve().context_name(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            MillerInner::Rational(v) => v.to_json(),
            MillerInner::Finite(v) => v.to_json(),
            MillerInner::NumberField(v) => v.to_json(),
            MillerInner::RationalFunction(v) => v.to_json(),
            MillerInner::FiniteFunction(v) => v.to_json(),
        }
        .map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_elliptic_context(value)? {
            EllipticContext::Rational => MillerInner::Rational(
                MillerFunction::<RationalField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::Finite => MillerInner::Finite(
                MillerFunction::<FiniteField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::NumberField => MillerInner::NumberField(
                MillerFunction::<NumberField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::RationalFunction => MillerInner::RationalFunction(
                MillerFunction::<RationalFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::FiniteFunction => MillerInner::FiniteFunction(
                MillerFunction::<FiniteFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            MillerInner::Rational(v) => canonical_hash(v),
            MillerInner::Finite(v) => canonical_hash(v),
            MillerInner::NumberField(v) => canonical_hash(v),
            MillerInner::RationalFunction(v) => canonical_hash(v),
            MillerInner::FiniteFunction(v) => canonical_hash(v),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "MillerFunction(multiplier={}, terminal_infinity={})",
            self.multiplier_value(),
            self.terminal_point().is_infinity()
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", MillerFunction::<RationalField>::TYPE_ID)?;
        summary.set_item(
            "multiplier",
            integer_to_python(py, self.multiplier_value())?,
        )?;
        summary.set_item("terminal_infinity", self.terminal_point().is_infinity())?;
        summary.set_item("divisor_formula", "n(P)-(nP)-(n-1)(O)")?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PowerGroupInner {
    Rational(NthPowerClassGroup<RationalField>),
    RationalFunction(NthPowerClassGroup<RationalFunctions>),
}

/// An effective multiplicative quotient context `K*/K*^n`.
#[pyclass(
    module = "noether.numthy",
    name = "NthPowerClassGroup",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNthPowerClassGroup {
    inner: PowerGroupInner,
}

impl PyNthPowerClassGroup {
    fn exponent_value(&self) -> &Integer {
        match &self.inner {
            PowerGroupInner::Rational(v) => v.exponent(),
            PowerGroupInner::RationalFunction(v) => v.exponent(),
        }
    }
}

#[pymethods]
impl PyNthPowerClassGroup {
    #[new]
    fn new(base: &Bound<'_, PyAny>, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exponent = extract_integer(exponent)?;
        let inner = if base.is_instance_of::<PyRationalField>() {
            PowerGroupInner::Rational(NthPowerClassGroup::rational(exponent).map_err(power_error)?)
        } else if let Ok(field) = base.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let context = field.as_rational().ok_or_else(|| {
                super::errors::unsupported_operation(
                    UnsupportedOperationError::new(
                        "noether/numthy/NthPowerClassGroup/new@1",
                        RationalFunctionField::<FiniteField>::TYPE_ID,
                    )
                    .with_supported_contexts([
                        RationalField::TYPE_ID,
                        RationalFunctionField::<RationalField>::TYPE_ID,
                    ])
                    .with_detail("effective power classes initially require Q or Q(t)"),
                )
            })?;
            PowerGroupInner::RationalFunction(
                NthPowerClassGroup::rational_functions(context.clone(), exponent)
                    .map_err(power_error)?,
            )
        } else {
            return Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/numthy/NthPowerClassGroup/new@1",
                    "unsupported Python coefficient context",
                )
                .with_supported_contexts([
                    RationalField::TYPE_ID,
                    RationalFunctionField::<RationalField>::TYPE_ID,
                ])
                .with_detail("effective power classes initially require Q or Q(t)"),
            ));
        };
        Ok(Self { inner })
    }

    #[classattr]
    fn type_id() -> &'static str {
        NthPowerClassGroup::<RationalField>::TYPE_ID
    }

    #[getter]
    fn field(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PowerGroupInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            PowerGroupInner::RationalFunction(v) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(v.field().clone()),
            )?
            .into_any()),
        }
    }

    #[getter]
    fn exponent(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.exponent_value())
    }

    fn identity(&self) -> PyNthPowerClass {
        let inner = match &self.inner {
            PowerGroupInner::Rational(v) => PowerClassInner::Rational(v.identity()),
            PowerGroupInner::RationalFunction(v) => PowerClassInner::RationalFunction(v.identity()),
        };
        PyNthPowerClass { inner }
    }

    fn class_of(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyNthPowerClass> {
        let inner = match &self.inner {
            PowerGroupInner::Rational(v) => PowerClassInner::Rational(
                v.class(extract_rational(py, value)?).map_err(power_error)?,
            ),
            PowerGroupInner::RationalFunction(v) => PowerClassInner::RationalFunction(
                v.class(coerce_rational_function(py, v.field(), value)?)
                    .map_err(power_error)?,
            ),
        };
        Ok(PyNthPowerClass { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PowerGroupInner::Rational(v) => v.verify(),
            PowerGroupInner::RationalFunction(v) => v.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PowerGroupInner::Rational(v) => v.to_json(),
            PowerGroupInner::RationalFunction(v) => v.to_json(),
        }
        .map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_power_context(value)? {
            EllipticContext::Rational => PowerGroupInner::Rational(
                NthPowerClassGroup::<RationalField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::RationalFunction => PowerGroupInner::RationalFunction(
                NthPowerClassGroup::<RationalFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "power classes are implemented only for Q and Q(t)",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PowerGroupInner::Rational(v) => canonical_hash(v),
            PowerGroupInner::RationalFunction(v) => canonical_hash(v),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "NthPowerClassGroup(exponent={})",
            self.exponent_value()
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", NthPowerClassGroup::<RationalField>::TYPE_ID)?;
        summary.set_item("exponent", integer_to_python(py, self.exponent_value())?)?;
        summary.set_item("effective_factorization", true)?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PowerClassInner {
    Rational(NthPowerClass<RationalField>),
    RationalFunction(NthPowerClass<RationalFunctions>),
}

/// One canonical class in an effective quotient `K*/K*^n`.
#[pyclass(
    module = "noether.numthy",
    name = "NthPowerClass",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNthPowerClass {
    inner: PowerClassInner,
}

impl PyNthPowerClass {
    fn exponent_value(&self) -> &Integer {
        match &self.inner {
            PowerClassInner::Rational(v) => v.group().exponent(),
            PowerClassInner::RationalFunction(v) => v.group().exponent(),
        }
    }

    fn binary(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (PowerClassInner::Rational(a), PowerClassInner::Rational(b)) => {
                PowerClassInner::Rational(a.checked_mul(b).map_err(power_error)?)
            }
            (PowerClassInner::RationalFunction(a), PowerClassInner::RationalFunction(b)) => {
                PowerClassInner::RationalFunction(a.checked_mul(b).map_err(power_error)?)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "power classes belong to different quotient contexts",
                ))
            }
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyNthPowerClass {
    #[classattr]
    fn type_id() -> &'static str {
        NthPowerClass::<RationalField>::TYPE_ID
    }

    #[getter]
    fn group(&self) -> PyNthPowerClassGroup {
        let inner = match &self.inner {
            PowerClassInner::Rational(v) => PowerGroupInner::Rational(v.group().clone()),
            PowerClassInner::RationalFunction(v) => {
                PowerGroupInner::RationalFunction(v.group().clone())
            }
        };
        PyNthPowerClassGroup { inner }
    }

    fn is_identity(&self) -> bool {
        match &self.inner {
            PowerClassInner::Rational(v) => v.is_identity(),
            PowerClassInner::RationalFunction(v) => v.is_identity(),
        }
    }

    fn representative(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PowerClassInner::Rational(v) => {
                rational_to_python(py, &v.representative().map_err(power_error)?)
            }
            PowerClassInner::RationalFunction(v) => Py::new(
                py,
                PyRationalFunction::from_rational(v.representative().map_err(power_error)?),
            )
            .map(Py::into_any),
        }
    }

    fn inverse(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            PowerClassInner::Rational(v) => {
                PowerClassInner::Rational(v.inverse().map_err(power_error)?)
            }
            PowerClassInner::RationalFunction(v) => {
                PowerClassInner::RationalFunction(v.inverse().map_err(power_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn pow(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exponent = extract_integer(exponent)?;
        let inner = match &self.inner {
            PowerClassInner::Rational(v) => {
                PowerClassInner::Rational(v.pow(&exponent).map_err(power_error)?)
            }
            PowerClassInner::RationalFunction(v) => {
                PowerClassInner::RationalFunction(v.pow(&exponent).map_err(power_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PowerClassInner::Rational(v) => v.verify(),
            PowerClassInner::RationalFunction(v) => v.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PowerClassInner::Rational(v) => v.to_json(),
            PowerClassInner::RationalFunction(v) => v.to_json(),
        }
        .map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_power_context(value)? {
            EllipticContext::Rational => PowerClassInner::Rational(
                NthPowerClass::<RationalField>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            EllipticContext::RationalFunction => PowerClassInner::RationalFunction(
                NthPowerClass::<RationalFunctions>::from_json(value)
                    .map_err(super::errors::noether_error)?,
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "power classes are implemented only for Q and Q(t)",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other)
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        self.binary(&other.inverse()?)
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        _modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        self.pow(exponent)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PowerClassInner::Rational(v) => canonical_hash(v),
            PowerClassInner::RationalFunction(v) => canonical_hash(v),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "NthPowerClass(exponent={}, identity={})",
            self.exponent_value(),
            self.is_identity()
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", NthPowerClass::<RationalField>::TYPE_ID)?;
        summary.set_item("exponent", integer_to_python(py, self.exponent_value())?)?;
        summary.set_item("identity", self.is_identity())?;
        summary.set_item("canonical_valuation_representative", true)?;
        Ok(summary.unbind())
    }
}
