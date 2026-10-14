// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of checked curves, places, divisors, and functions.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]
// These private context enums live inside heap-allocated PyO3 classes. Boxing
// individual mathematical variants would add another allocation and pointer
// chase to every boundary access.
#![allow(clippy::large_enum_variant)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    Curve, CurveClosedPoint, CurveError, CurveFunction, CurveFunctionEvaluation,
    CurveFunctionField, CurveFunctionOrder, CurveMorphism, CurvePoint, Polynomial,
    RationalFunctionField, RiemannRochSpace, WeilDivisor,
};
use noether::numthy::{FiniteField, NumberField};
use noether::{Field, Integer, RationalField, Ring, SemanticType, Q};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};

use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::number_field::{
    extract_number_field_element, PyIdeal, PyNumberField, PyNumberFieldElement,
};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::polynomial_facade::{extract_finite_field, PyPolynomial};
use super::rational_function::{PyRationalFunction, PyRationalFunctionField};
use super::root::PyRationalField;

type RationalFunctions = RationalFunctionField<RationalField>;
type FiniteFunctions = RationalFunctionField<FiniteField>;

fn curve_error(error: CurveError) -> PyErr {
    match error {
        CurveError::ContextMismatch(error) => super::errors::context_mismatch(error),
        CurveError::DivisionByZero => PyZeroDivisionError::new_err(error.to_string()),
        CurveError::LocalExpansionLimit { .. } | CurveError::ResourceLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
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

#[derive(Clone, Copy)]
pub(crate) enum SerializedCurveContext {
    Rational,
    Finite,
    NumberField,
    RationalFunction,
    FiniteFunction,
}

pub(crate) fn serialized_curve_context(value: &str) -> PyResult<SerializedCurveContext> {
    let value: serde_json::Value =
        serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let equation = find_equation(&value).ok_or_else(|| {
        PyValueError::new_err("serialized curve value contains no curve equation")
    })?;
    let coefficients = equation
        .pointer("/ring/coefficient_ring")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            PyValueError::new_err("serialized curve equation has no coefficient ring")
        })?;
    let type_id = coefficients
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("serialized coefficient ring has no type"))?;
    match type_id {
        RationalField::TYPE_ID => Ok(SerializedCurveContext::Rational),
        FiniteField::TYPE_ID => Ok(SerializedCurveContext::Finite),
        NumberField::TYPE_ID => Ok(SerializedCurveContext::NumberField),
        RationalFunctionField::<RationalField>::TYPE_ID => {
            let constant_type = coefficients
                .get("coefficient_ring")
                .and_then(serde_json::Value::as_object)
                .and_then(|value| value.get("type"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "serialized rational-function field has no constant field",
                    )
                })?;
            match constant_type {
                RationalField::TYPE_ID => Ok(SerializedCurveContext::RationalFunction),
                FiniteField::TYPE_ID => Ok(SerializedCurveContext::FiniteFunction),
                _ => Err(PyTypeError::new_err(
                    "unsupported constant field in serialized curve value",
                )),
            }
        }
        _ => Err(PyTypeError::new_err(
            "unsupported coefficient context in serialized curve value",
        )),
    }
}

fn find_equation(value: &serde_json::Value) -> Option<&serde_json::Value> {
    match value {
        serde_json::Value::Object(object) => object
            .get("equation")
            .filter(|equation| equation.get("ring").is_some())
            .or_else(|| object.values().find_map(find_equation)),
        serde_json::Value::Array(values) => values.iter().find_map(find_equation),
        _ => None,
    }
}

fn json_error(error: noether::NoetherError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[derive(Clone, PartialEq, Eq)]
enum CurveInner {
    Rational(Curve<RationalField>),
    Finite(Curve<FiniteField>),
    NumberField(Curve<NumberField>),
    RationalFunction(Curve<RationalFunctions>),
    FiniteFunction(Curve<FiniteFunctions>),
}

/// A checked smooth projective plane or Weierstrass curve.
#[pyclass(module = "noether.commalg", name = "Curve", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCurve {
    inner: CurveInner,
}

impl PyCurve {
    pub(crate) fn from_rational(inner: Curve<RationalField>) -> Self {
        Self {
            inner: CurveInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite(inner: Curve<FiniteField>) -> Self {
        Self {
            inner: CurveInner::Finite(inner),
        }
    }

    pub(crate) fn from_number_field(inner: Curve<NumberField>) -> Self {
        Self {
            inner: CurveInner::NumberField(inner),
        }
    }

    pub(crate) fn from_rational_function(inner: Curve<RationalFunctions>) -> Self {
        Self {
            inner: CurveInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_function(inner: Curve<FiniteFunctions>) -> Self {
        Self {
            inner: CurveInner::FiniteFunction(inner),
        }
    }

    fn context_name(&self) -> &'static str {
        match &self.inner {
            CurveInner::Rational(_) => RationalField::TYPE_ID,
            CurveInner::Finite(_) => FiniteField::TYPE_ID,
            CurveInner::NumberField(_) => NumberField::TYPE_ID,
            CurveInner::RationalFunction(_) | CurveInner::FiniteFunction(_) => {
                RationalFunctionField::<RationalField>::TYPE_ID
            }
        }
    }
}

#[pymethods]
impl PyCurve {
    #[classattr]
    fn type_id() -> &'static str {
        Curve::<RationalField>::TYPE_ID
    }

    #[staticmethod]
    fn smooth_projective_plane(equation: &PyPolynomial) -> PyResult<Self> {
        if let Some(equation) = equation.as_rational() {
            return Curve::smooth_projective_plane(equation.clone())
                .map(Self::from_rational)
                .map_err(curve_error);
        }
        if let Some(equation) = equation.as_finite_field() {
            return Curve::smooth_projective_plane(equation.clone())
                .map(Self::from_finite)
                .map_err(curve_error);
        }
        if let Some(equation) = equation.as_number_field() {
            return Curve::smooth_projective_plane(equation.clone())
                .map(Self::from_number_field)
                .map_err(curve_error);
        }
        if let Some(equation) = equation.as_rational_function() {
            return Curve::smooth_projective_plane(equation.clone())
                .map(Self::from_rational_function)
                .map_err(curve_error);
        }
        if let Some(equation) = equation.as_finite_rational_function() {
            return Curve::smooth_projective_plane(equation.clone())
                .map(Self::from_finite_function)
                .map_err(curve_error);
        }
        Err(PyTypeError::new_err(
            "curve equations currently support Q, finite-field, Q(t), and finite-field rational-function coefficients",
        ))
    }

    #[staticmethod]
    fn weierstrass(
        py: Python<'_>,
        base: &Bound<'_, PyAny>,
        coefficients: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let values = coefficients
            .try_iter()
            .map_err(|_| PyTypeError::new_err("Weierstrass coefficients must be iterable"))?
            .collect::<PyResult<Vec<_>>>()?;
        if values.len() != 5 {
            return Err(PyValueError::new_err(
                "a generalized Weierstrass equation needs five coefficients",
            ));
        }
        if base.is_instance_of::<PyRationalField>() {
            let coefficients = values
                .iter()
                .map(|value| extract_rational(py, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            return Curve::smooth_projective_weierstrass(Q, coefficients)
                .map(Self::from_rational)
                .map_err(curve_error);
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyFiniteField>>() {
            let coefficients = values
                .iter()
                .map(|value| extract_finite_field(&field.inner, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            return Curve::smooth_projective_weierstrass(field.inner.clone(), coefficients)
                .map(Self::from_finite)
                .map_err(curve_error);
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyNumberField>>() {
            let coefficients = values
                .iter()
                .map(|value| extract_number_field_element(py, &field.inner, value))
                .collect::<PyResult<Vec<_>>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
            return Curve::smooth_projective_weierstrass(field.inner.clone(), coefficients)
                .map(Self::from_number_field)
                .map_err(curve_error);
        }
        if let Ok(field) = base.extract::<PyRef<'_, PyRationalFunctionField>>() {
            if let Some(inner) = field.as_rational() {
                let coefficients = values
                    .iter()
                    .map(|value| {
                        field.coerce(py, value).and_then(|value| {
                            value.as_rational().cloned().ok_or_else(|| {
                                PyTypeError::new_err(
                                    "coefficient has the wrong rational-function field",
                                )
                            })
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
                return Curve::smooth_projective_weierstrass(inner.clone(), coefficients)
                    .map(Self::from_rational_function)
                    .map_err(curve_error);
            }
            if let Some(inner) = field.as_finite_field() {
                let coefficients = values
                    .iter()
                    .map(|value| {
                        field.coerce(py, value).and_then(|value| {
                            value.as_finite_field().cloned().ok_or_else(|| {
                                PyTypeError::new_err(
                                    "coefficient has the wrong rational-function field",
                                )
                            })
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| PyValueError::new_err("expected five coefficients"))?;
                return Curve::smooth_projective_weierstrass(inner.clone(), coefficients)
                    .map(Self::from_finite_function)
                    .map_err(curve_error);
            }
        }
        Err(PyTypeError::new_err(
            "Curve.weierstrass supports Q, FiniteField, NumberField, and RationalFunctionField bases",
        ))
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            CurveInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            CurveInner::Finite(curve) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: curve.base().clone(),
                },
            )?
            .into_any()),
            CurveInner::NumberField(curve) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: curve.base().clone(),
                },
            )?
            .into_any()),
            CurveInner::RationalFunction(curve) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(curve.base().clone()),
            )?
            .into_any()),
            CurveInner::FiniteFunction(curve) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_finite_field(curve.base().clone()),
            )?
            .into_any()),
        }
    }

    #[getter]
    fn equation(&self) -> PyPolynomial {
        match &self.inner {
            CurveInner::Rational(curve) => PyPolynomial::from_rational(curve.equation().clone()),
            CurveInner::Finite(curve) => PyPolynomial::from_finite_field(curve.equation().clone()),
            CurveInner::NumberField(curve) => {
                PyPolynomial::from_number_field(curve.equation().clone())
            }
            CurveInner::RationalFunction(curve) => {
                PyPolynomial::from_rational_function(curve.equation().clone())
            }
            CurveInner::FiniteFunction(curve) => {
                PyPolynomial::from_finite_rational_function(curve.equation().clone())
            }
        }
    }

    #[getter]
    fn presentation_kind(&self) -> &'static str {
        let kind = match &self.inner {
            CurveInner::Rational(curve) => curve.presentation_kind(),
            CurveInner::Finite(curve) => curve.presentation_kind(),
            CurveInner::NumberField(curve) => curve.presentation_kind(),
            CurveInner::RationalFunction(curve) => curve.presentation_kind(),
            CurveInner::FiniteFunction(curve) => curve.presentation_kind(),
        };
        match kind {
            noether::commalg::CurvePresentationKind::SmoothProjectivePlane => {
                "smooth_projective_plane"
            }
            noether::commalg::CurvePresentationKind::SmoothProjectiveWeierstrass => {
                "smooth_projective_weierstrass"
            }
        }
    }

    #[getter]
    fn quotient_dimension(&self) -> i64 {
        match &self.inner {
            CurveInner::Rational(curve) => curve.smoothness_evidence().quotient_dimension(),
            CurveInner::Finite(curve) => curve.smoothness_evidence().quotient_dimension(),
            CurveInner::NumberField(curve) => curve.smoothness_evidence().quotient_dimension(),
            CurveInner::RationalFunction(curve) => curve.smoothness_evidence().quotient_dimension(),
            CurveInner::FiniteFunction(curve) => curve.smoothness_evidence().quotient_dimension(),
        }
    }

    #[getter]
    fn singular_cone_dimension(&self) -> i64 {
        match &self.inner {
            CurveInner::Rational(curve) => curve.smoothness_evidence().singular_cone_dimension(),
            CurveInner::Finite(curve) => curve.smoothness_evidence().singular_cone_dimension(),
            CurveInner::NumberField(curve) => curve.smoothness_evidence().singular_cone_dimension(),
            CurveInner::RationalFunction(curve) => {
                curve.smoothness_evidence().singular_cone_dimension()
            }
            CurveInner::FiniteFunction(curve) => {
                curve.smoothness_evidence().singular_cone_dimension()
            }
        }
    }

    #[getter]
    fn weierstrass_coefficients(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        match &self.inner {
            CurveInner::Rational(curve) => curve
                .weierstrass_coefficients()
                .map(|values| {
                    values
                        .iter()
                        .map(|value| rational_to_python(py, value))
                        .collect()
                })
                .transpose(),
            CurveInner::Finite(curve) => curve
                .weierstrass_coefficients()
                .map(|values| {
                    values
                        .iter()
                        .cloned()
                        .map(|inner| {
                            Py::new(py, PyFiniteFieldElement { inner })
                                .map(|value| value.into_any())
                        })
                        .collect()
                })
                .transpose(),
            CurveInner::NumberField(curve) => curve
                .weierstrass_coefficients()
                .map(|values| {
                    values
                        .iter()
                        .cloned()
                        .map(|inner| {
                            Py::new(py, PyNumberFieldElement { inner })
                                .map(|value| value.into_any())
                        })
                        .collect()
                })
                .transpose(),
            CurveInner::RationalFunction(curve) => curve
                .weierstrass_coefficients()
                .map(|values| {
                    values
                        .iter()
                        .cloned()
                        .map(|inner| {
                            Py::new(py, PyRationalFunction::from_rational(inner))
                                .map(|value| value.into_any())
                        })
                        .collect()
                })
                .transpose(),
            CurveInner::FiniteFunction(curve) => curve
                .weierstrass_coefficients()
                .map(|values| {
                    values
                        .iter()
                        .cloned()
                        .map(|inner| {
                            Py::new(py, PyRationalFunction::from_finite_field(inner))
                                .map(|value| value.into_any())
                        })
                        .collect()
                })
                .transpose(),
        }
    }

    fn function_field(&self) -> PyResult<PyCurveFunctionField> {
        let inner = match &self.inner {
            CurveInner::Rational(curve) => {
                CurveFunctionFieldInner::Rational(curve.function_field().map_err(curve_error)?)
            }
            CurveInner::Finite(curve) => {
                CurveFunctionFieldInner::Finite(curve.function_field().map_err(curve_error)?)
            }
            CurveInner::NumberField(curve) => {
                CurveFunctionFieldInner::NumberField(curve.function_field().map_err(curve_error)?)
            }
            CurveInner::RationalFunction(curve) => CurveFunctionFieldInner::RationalFunction(
                curve.function_field().map_err(curve_error)?,
            ),
            CurveInner::FiniteFunction(curve) => CurveFunctionFieldInner::FiniteFunction(
                curve.function_field().map_err(curve_error)?,
            ),
        };
        Ok(PyCurveFunctionField { inner })
    }

    fn point(&self, py: Python<'_>, coordinates: &Bound<'_, PyAny>) -> PyResult<PyCurvePoint> {
        let values = coordinates
            .try_iter()
            .map_err(|_| PyTypeError::new_err("curve point coordinates must be iterable"))?
            .collect::<PyResult<Vec<_>>>()?;
        let inner = match &self.inner {
            CurveInner::Rational(curve) => CurvePointInner::Rational(
                curve
                    .point(
                        values
                            .iter()
                            .map(|value| extract_rational(py, value))
                            .collect::<PyResult<Vec<_>>>()?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveInner::Finite(curve) => CurvePointInner::Finite(
                curve
                    .point(
                        values
                            .iter()
                            .map(|value| extract_finite_field(curve.base(), value))
                            .collect::<PyResult<Vec<_>>>()?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveInner::NumberField(curve) => CurvePointInner::NumberField(
                curve
                    .point(
                        values
                            .iter()
                            .map(|value| extract_number_field_element(py, curve.base(), value))
                            .collect::<PyResult<Vec<_>>>()?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveInner::RationalFunction(curve) => CurvePointInner::RationalFunction(
                curve
                    .point(
                        values
                            .iter()
                            .map(|value| coerce_rational_function(py, curve.base(), value))
                            .collect::<PyResult<Vec<_>>>()?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveInner::FiniteFunction(curve) => CurvePointInner::FiniteFunction(
                curve
                    .point(
                        values
                            .iter()
                            .map(|value| coerce_finite_function(py, curve.base(), value))
                            .collect::<PyResult<Vec<_>>>()?,
                    )
                    .map_err(curve_error)?,
            ),
        };
        Ok(PyCurvePoint { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurveInner::Rational(curve) => curve.verify(),
            CurveInner::Finite(curve) => curve.verify(),
            CurveInner::NumberField(curve) => curve.verify(),
            CurveInner::RationalFunction(curve) => curve.verify(),
            CurveInner::FiniteFunction(curve) => curve.verify(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurveInner::Rational(value) => value.to_json(),
            CurveInner::Finite(value) => value.to_json(),
            CurveInner::NumberField(value) => value.to_json(),
            CurveInner::RationalFunction(value) => value.to_json(),
            CurveInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => Curve::<RationalField>::from_json(value)
                .map(Self::from_rational)
                .map_err(json_error),
            SerializedCurveContext::Finite => Curve::<FiniteField>::from_json(value)
                .map(Self::from_finite)
                .map_err(json_error),
            SerializedCurveContext::NumberField => Curve::<NumberField>::from_json(value)
                .map(Self::from_number_field)
                .map_err(json_error),
            SerializedCurveContext::RationalFunction => {
                Curve::<RationalFunctions>::from_json(value)
                    .map(Self::from_rational_function)
                    .map_err(json_error)
            }
            SerializedCurveContext::FiniteFunction => Curve::<FiniteFunctions>::from_json(value)
                .map(Self::from_finite_function)
                .map_err(json_error),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            CurveInner::Rational(value) => canonical_hash(value),
            CurveInner::Finite(value) => canonical_hash(value),
            CurveInner::NumberField(value) => canonical_hash(value),
            CurveInner::RationalFunction(value) => canonical_hash(value),
            CurveInner::FiniteFunction(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "Curve(base={}, presentation={})",
            self.context_name(),
            self.presentation_kind()
        ))
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", Curve::<RationalField>::TYPE_ID)?;
        summary.set_item("base_type", self.context_name())?;
        summary.set_item("presentation", self.presentation_kind())?;
        summary.set_item("quotient_dimension", self.quotient_dimension())?;
        summary.set_item("singular_cone_dimension", self.singular_cone_dimension())?;
        Ok(summary.unbind())
    }
}

fn coerce_rational_function(
    py: Python<'_>,
    field: &RationalFunctions,
    value: &Bound<'_, PyAny>,
) -> PyResult<noether::commalg::RationalFunction<RationalField>> {
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
    _py: Python<'_>,
    field: &FiniteFunctions,
    value: &Bound<'_, PyAny>,
) -> PyResult<noether::commalg::RationalFunction<FiniteField>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        if let Some(value) = value
            .as_finite_field()
            .filter(|value| value.field() == field)
        {
            return Ok(value.clone());
        }
    }
    field
        .constant(extract_finite_field(field.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[derive(Clone, PartialEq, Eq)]
enum CurvePointInner {
    Rational(CurvePoint<RationalField>),
    Finite(CurvePoint<FiniteField>),
    NumberField(CurvePoint<NumberField>),
    RationalFunction(CurvePoint<RationalFunctions>),
    FiniteFunction(CurvePoint<FiniteFunctions>),
}

/// A checked rational point on one curve.
#[pyclass(
    module = "noether.commalg",
    name = "CurvePoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurvePoint {
    inner: CurvePointInner,
}

impl PyCurvePoint {
    pub(crate) fn from_rational(inner: CurvePoint<RationalField>) -> Self {
        Self {
            inner: CurvePointInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite(inner: CurvePoint<FiniteField>) -> Self {
        Self {
            inner: CurvePointInner::Finite(inner),
        }
    }

    pub(crate) fn from_number_field(inner: CurvePoint<NumberField>) -> Self {
        Self {
            inner: CurvePointInner::NumberField(inner),
        }
    }

    pub(crate) fn from_rational_function(inner: CurvePoint<RationalFunctions>) -> Self {
        Self {
            inner: CurvePointInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_function(inner: CurvePoint<FiniteFunctions>) -> Self {
        Self {
            inner: CurvePointInner::FiniteFunction(inner),
        }
    }
}

#[pymethods]
impl PyCurvePoint {
    #[classattr]
    fn type_id() -> &'static str {
        CurvePoint::<RationalField>::TYPE_ID
    }

    #[getter]
    fn curve(&self) -> PyCurve {
        match &self.inner {
            CurvePointInner::Rational(value) => PyCurve::from_rational(value.curve().clone()),
            CurvePointInner::Finite(value) => PyCurve::from_finite(value.curve().clone()),
            CurvePointInner::NumberField(value) => {
                PyCurve::from_number_field(value.curve().clone())
            }
            CurvePointInner::RationalFunction(value) => {
                PyCurve::from_rational_function(value.curve().clone())
            }
            CurvePointInner::FiniteFunction(value) => {
                PyCurve::from_finite_function(value.curve().clone())
            }
        }
    }

    #[getter]
    fn coordinates(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            CurvePointInner::Rational(value) => value
                .coordinates()
                .iter()
                .map(|value| rational_to_python(py, value))
                .collect(),
            CurvePointInner::Finite(value) => value
                .coordinates()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyFiniteFieldElement { inner }).map(|value| value.into_any())
                })
                .collect(),
            CurvePointInner::NumberField(value) => value
                .coordinates()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyNumberFieldElement { inner }).map(|value| value.into_any())
                })
                .collect(),
            CurvePointInner::RationalFunction(value) => value
                .coordinates()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_rational(value))
                        .map(|value| value.into_any())
                })
                .collect(),
            CurvePointInner::FiniteFunction(value) => value
                .coordinates()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_finite_field(value))
                        .map(|value| value.into_any())
                })
                .collect(),
        }
    }

    fn as_closed_point(&self) -> PyResult<PyCurveClosedPoint> {
        let inner = match &self.inner {
            CurvePointInner::Rational(value) => {
                CurveClosedPointInner::Rational(value.as_closed_point().map_err(curve_error)?)
            }
            CurvePointInner::Finite(value) => {
                CurveClosedPointInner::Finite(value.as_closed_point().map_err(curve_error)?)
            }
            CurvePointInner::NumberField(value) => {
                CurveClosedPointInner::NumberField(value.as_closed_point().map_err(curve_error)?)
            }
            CurvePointInner::RationalFunction(value) => CurveClosedPointInner::RationalFunction(
                value.as_closed_point().map_err(curve_error)?,
            ),
            CurvePointInner::FiniteFunction(value) => {
                CurveClosedPointInner::FiniteFunction(value.as_closed_point().map_err(curve_error)?)
            }
        };
        Ok(PyCurveClosedPoint { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurvePointInner::Rational(value) => value.verify(),
            CurvePointInner::Finite(value) => value.verify(),
            CurvePointInner::NumberField(value) => value.verify(),
            CurvePointInner::RationalFunction(value) => value.verify(),
            CurvePointInner::FiniteFunction(value) => value.verify(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurvePointInner::Rational(value) => value.to_json(),
            CurvePointInner::Finite(value) => value.to_json(),
            CurvePointInner::NumberField(value) => value.to_json(),
            CurvePointInner::RationalFunction(value) => value.to_json(),
            CurvePointInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => CurvePointInner::Rational(
                CurvePoint::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => CurvePointInner::Finite(
                CurvePoint::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => CurvePointInner::NumberField(
                CurvePoint::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => CurvePointInner::RationalFunction(
                CurvePoint::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => CurvePointInner::FiniteFunction(
                CurvePoint::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            CurvePointInner::Rational(value) => canonical_hash(value),
            CurvePointInner::Finite(value) => canonical_hash(value),
            CurvePointInner::NumberField(value) => canonical_hash(value),
            CurvePointInner::RationalFunction(value) => canonical_hash(value),
            CurvePointInner::FiniteFunction(value) => canonical_hash(value),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text("CurvePoint(projective=True)".to_owned())
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", CurvePoint::<RationalField>::TYPE_ID)?;
        summary.set_item("projective", true)?;
        summary.set_item("coordinate_count", 3)?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CurveFunctionFieldInner {
    Rational(CurveFunctionField<RationalField>),
    Finite(CurveFunctionField<FiniteField>),
    NumberField(CurveFunctionField<NumberField>),
    RationalFunction(CurveFunctionField<RationalFunctions>),
    FiniteFunction(CurveFunctionField<FiniteFunctions>),
}

/// Canonical algebraic function field of a checked curve.
#[pyclass(
    module = "noether.commalg",
    name = "CurveFunctionField",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveFunctionField {
    inner: CurveFunctionFieldInner,
}

#[pymethods]
impl PyCurveFunctionField {
    #[classattr]
    fn type_id() -> &'static str {
        CurveFunctionField::<RationalField>::TYPE_ID
    }

    #[getter]
    fn curve(&self) -> PyCurve {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => {
                PyCurve::from_rational(value.curve().clone())
            }
            CurveFunctionFieldInner::Finite(value) => PyCurve::from_finite(value.curve().clone()),
            CurveFunctionFieldInner::NumberField(value) => {
                PyCurve::from_number_field(value.curve().clone())
            }
            CurveFunctionFieldInner::RationalFunction(value) => {
                PyCurve::from_rational_function(value.curve().clone())
            }
            CurveFunctionFieldInner::FiniteFunction(value) => {
                PyCurve::from_finite_function(value.curve().clone())
            }
        }
    }

    #[getter]
    fn extension_degree(&self) -> usize {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.extension_degree(),
            CurveFunctionFieldInner::Finite(value) => value.extension_degree(),
            CurveFunctionFieldInner::NumberField(value) => value.extension_degree(),
            CurveFunctionFieldInner::RationalFunction(value) => value.extension_degree(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.extension_degree(),
        }
    }

    #[getter]
    fn transcendental_coordinate(&self) -> usize {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.transcendental_coordinate(),
            CurveFunctionFieldInner::Finite(value) => value.transcendental_coordinate(),
            CurveFunctionFieldInner::NumberField(value) => value.transcendental_coordinate(),
            CurveFunctionFieldInner::RationalFunction(value) => value.transcendental_coordinate(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.transcendental_coordinate(),
        }
    }

    #[getter]
    fn algebraic_coordinate(&self) -> usize {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.algebraic_coordinate(),
            CurveFunctionFieldInner::Finite(value) => value.algebraic_coordinate(),
            CurveFunctionFieldInner::NumberField(value) => value.algebraic_coordinate(),
            CurveFunctionFieldInner::RationalFunction(value) => value.algebraic_coordinate(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.algebraic_coordinate(),
        }
    }

    #[getter]
    fn patch_coordinate(&self) -> usize {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.patch_coordinate(),
            CurveFunctionFieldInner::Finite(value) => value.patch_coordinate(),
            CurveFunctionFieldInner::NumberField(value) => value.patch_coordinate(),
            CurveFunctionFieldInner::RationalFunction(value) => value.patch_coordinate(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.patch_coordinate(),
        }
    }

    fn zero(&self) -> PyCurveFunction {
        self.wrap_zero_or_one(false)
    }
    fn one(&self) -> PyCurveFunction {
        self.wrap_zero_or_one(true)
    }

    fn x(&self) -> PyResult<PyCurveFunction> {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(value) => {
                CurveFunctionInner::Rational(value.x().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::Finite(value) => {
                CurveFunctionInner::Finite(value.x().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::NumberField(value) => {
                CurveFunctionInner::NumberField(value.x().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::RationalFunction(value) => {
                CurveFunctionInner::RationalFunction(value.x().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::FiniteFunction(value) => {
                CurveFunctionInner::FiniteFunction(value.x().map_err(curve_error)?)
            }
        };
        Ok(PyCurveFunction { inner })
    }

    fn y(&self) -> PyResult<PyCurveFunction> {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(value) => {
                CurveFunctionInner::Rational(value.y().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::Finite(value) => {
                CurveFunctionInner::Finite(value.y().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::NumberField(value) => {
                CurveFunctionInner::NumberField(value.y().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::RationalFunction(value) => {
                CurveFunctionInner::RationalFunction(value.y().map_err(curve_error)?)
            }
            CurveFunctionFieldInner::FiniteFunction(value) => {
                CurveFunctionInner::FiniteFunction(value.y().map_err(curve_error)?)
            }
        };
        Ok(PyCurveFunction { inner })
    }

    fn coordinate(&self, coordinate: usize) -> PyResult<PyCurveFunction> {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(value) => {
                CurveFunctionInner::Rational(value.coordinate(coordinate).map_err(curve_error)?)
            }
            CurveFunctionFieldInner::Finite(value) => {
                CurveFunctionInner::Finite(value.coordinate(coordinate).map_err(curve_error)?)
            }
            CurveFunctionFieldInner::NumberField(value) => {
                CurveFunctionInner::NumberField(value.coordinate(coordinate).map_err(curve_error)?)
            }
            CurveFunctionFieldInner::RationalFunction(value) => {
                CurveFunctionInner::RationalFunction(
                    value.coordinate(coordinate).map_err(curve_error)?,
                )
            }
            CurveFunctionFieldInner::FiniteFunction(value) => CurveFunctionInner::FiniteFunction(
                value.coordinate(coordinate).map_err(curve_error)?,
            ),
        };
        Ok(PyCurveFunction { inner })
    }

    fn constant(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyCurveFunction> {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(field) => CurveFunctionInner::Rational(
                field
                    .constant(extract_rational(py, value)?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::Finite(field) => CurveFunctionInner::Finite(
                field
                    .constant(extract_finite_field(field.curve().base(), value)?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::NumberField(field) => CurveFunctionInner::NumberField(
                field
                    .constant(extract_number_field_element(
                        py,
                        field.curve().base(),
                        value,
                    )?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::RationalFunction(field) => {
                CurveFunctionInner::RationalFunction(
                    field
                        .constant(coerce_rational_function(py, field.curve().base(), value)?)
                        .map_err(curve_error)?,
                )
            }
            CurveFunctionFieldInner::FiniteFunction(field) => CurveFunctionInner::FiniteFunction(
                field
                    .constant(coerce_finite_function(py, field.curve().base(), value)?)
                    .map_err(curve_error)?,
            ),
        };
        Ok(PyCurveFunction { inner })
    }

    fn from_homogeneous_polynomial(&self, value: &PyPolynomial) -> PyResult<PyCurveFunction> {
        let wrong_context = || PyTypeError::new_err("polynomial has the wrong coefficient context");
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(field) => CurveFunctionInner::Rational(
                field
                    .from_homogeneous_polynomial(value.as_rational().ok_or_else(wrong_context)?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::Finite(field) => CurveFunctionInner::Finite(
                field
                    .from_homogeneous_polynomial(value.as_finite_field().ok_or_else(wrong_context)?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::NumberField(field) => CurveFunctionInner::NumberField(
                field
                    .from_homogeneous_polynomial(value.as_number_field().ok_or_else(wrong_context)?)
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::RationalFunction(field) => {
                CurveFunctionInner::RationalFunction(
                    field
                        .from_homogeneous_polynomial(
                            value.as_rational_function().ok_or_else(wrong_context)?,
                        )
                        .map_err(curve_error)?,
                )
            }
            CurveFunctionFieldInner::FiniteFunction(field) => CurveFunctionInner::FiniteFunction(
                field
                    .from_homogeneous_polynomial(
                        value
                            .as_finite_rational_function()
                            .ok_or_else(wrong_context)?,
                    )
                    .map_err(curve_error)?,
            ),
        };
        Ok(PyCurveFunction { inner })
    }

    fn homogeneous_fraction(
        &self,
        numerator: &PyPolynomial,
        denominator: &PyPolynomial,
    ) -> PyResult<PyCurveFunction> {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(field) => CurveFunctionInner::Rational(
                field
                    .homogeneous_fraction(
                        numerator.as_rational().ok_or_else(|| {
                            PyTypeError::new_err("numerator has the wrong coefficient context")
                        })?,
                        denominator.as_rational().ok_or_else(|| {
                            PyTypeError::new_err("denominator has the wrong coefficient context")
                        })?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::Finite(field) => CurveFunctionInner::Finite(
                field
                    .homogeneous_fraction(
                        numerator.as_finite_field().ok_or_else(|| {
                            PyTypeError::new_err("numerator has the wrong coefficient context")
                        })?,
                        denominator.as_finite_field().ok_or_else(|| {
                            PyTypeError::new_err("denominator has the wrong coefficient context")
                        })?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::NumberField(field) => CurveFunctionInner::NumberField(
                field
                    .homogeneous_fraction(
                        numerator.as_number_field().ok_or_else(|| {
                            PyTypeError::new_err("numerator has the wrong coefficient context")
                        })?,
                        denominator.as_number_field().ok_or_else(|| {
                            PyTypeError::new_err("denominator has the wrong coefficient context")
                        })?,
                    )
                    .map_err(curve_error)?,
            ),
            CurveFunctionFieldInner::RationalFunction(field) => {
                CurveFunctionInner::RationalFunction(
                    field
                        .homogeneous_fraction(
                            numerator.as_rational_function().ok_or_else(|| {
                                PyTypeError::new_err("numerator has the wrong coefficient context")
                            })?,
                            denominator.as_rational_function().ok_or_else(|| {
                                PyTypeError::new_err(
                                    "denominator has the wrong coefficient context",
                                )
                            })?,
                        )
                        .map_err(curve_error)?,
                )
            }
            CurveFunctionFieldInner::FiniteFunction(field) => CurveFunctionInner::FiniteFunction(
                field
                    .homogeneous_fraction(
                        numerator.as_finite_rational_function().ok_or_else(|| {
                            PyTypeError::new_err("numerator has the wrong coefficient context")
                        })?,
                        denominator.as_finite_rational_function().ok_or_else(|| {
                            PyTypeError::new_err("denominator has the wrong coefficient context")
                        })?,
                    )
                    .map_err(curve_error)?,
            ),
        };
        Ok(PyCurveFunction { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.verify(),
            CurveFunctionFieldInner::Finite(value) => value.verify(),
            CurveFunctionFieldInner::NumberField(value) => value.verify(),
            CurveFunctionFieldInner::RationalFunction(value) => value.verify(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.verify(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => value.to_json(),
            CurveFunctionFieldInner::Finite(value) => value.to_json(),
            CurveFunctionFieldInner::NumberField(value) => value.to_json(),
            CurveFunctionFieldInner::RationalFunction(value) => value.to_json(),
            CurveFunctionFieldInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => CurveFunctionFieldInner::Rational(
                CurveFunctionField::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => CurveFunctionFieldInner::Finite(
                CurveFunctionField::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => CurveFunctionFieldInner::NumberField(
                CurveFunctionField::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => CurveFunctionFieldInner::RationalFunction(
                CurveFunctionField::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => CurveFunctionFieldInner::FiniteFunction(
                CurveFunctionField::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            CurveFunctionFieldInner::Rational(value) => canonical_hash(value),
            CurveFunctionFieldInner::Finite(value) => canonical_hash(value),
            CurveFunctionFieldInner::NumberField(value) => canonical_hash(value),
            CurveFunctionFieldInner::RationalFunction(value) => canonical_hash(value),
            CurveFunctionFieldInner::FiniteFunction(value) => canonical_hash(value),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CurveFunctionField(degree={})",
            self.extension_degree()
        ))
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", CurveFunctionField::<RationalField>::TYPE_ID)?;
        summary.set_item("extension_degree", self.extension_degree())?;
        summary.set_item(
            "transcendental_coordinate",
            self.transcendental_coordinate(),
        )?;
        summary.set_item("algebraic_coordinate", self.algebraic_coordinate())?;
        summary.set_item("patch_coordinate", self.patch_coordinate())?;
        Ok(summary.unbind())
    }
}

impl PyCurveFunctionField {
    fn wrap_zero_or_one(&self, one: bool) -> PyCurveFunction {
        let inner = match &self.inner {
            CurveFunctionFieldInner::Rational(value) => {
                CurveFunctionInner::Rational(if one { value.one() } else { value.zero() })
            }
            CurveFunctionFieldInner::Finite(value) => {
                CurveFunctionInner::Finite(if one { value.one() } else { value.zero() })
            }
            CurveFunctionFieldInner::NumberField(value) => {
                CurveFunctionInner::NumberField(if one { value.one() } else { value.zero() })
            }
            CurveFunctionFieldInner::RationalFunction(value) => {
                CurveFunctionInner::RationalFunction(if one { value.one() } else { value.zero() })
            }
            CurveFunctionFieldInner::FiniteFunction(value) => {
                CurveFunctionInner::FiniteFunction(if one { value.one() } else { value.zero() })
            }
        };
        PyCurveFunction { inner }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CurveFunctionInner {
    Rational(CurveFunction<RationalField>),
    Finite(CurveFunction<FiniteField>),
    NumberField(CurveFunction<NumberField>),
    RationalFunction(CurveFunction<RationalFunctions>),
    FiniteFunction(CurveFunction<FiniteFunctions>),
}

/// One canonical rational function on a checked curve.
#[pyclass(
    module = "noether.commalg",
    name = "CurveFunction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveFunction {
    inner: CurveFunctionInner,
}

impl PyCurveFunction {
    pub(crate) fn from_rational(inner: CurveFunction<RationalField>) -> Self {
        Self {
            inner: CurveFunctionInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite(inner: CurveFunction<FiniteField>) -> Self {
        Self {
            inner: CurveFunctionInner::Finite(inner),
        }
    }

    pub(crate) fn from_number_field(inner: CurveFunction<NumberField>) -> Self {
        Self {
            inner: CurveFunctionInner::NumberField(inner),
        }
    }

    pub(crate) fn from_rational_function(inner: CurveFunction<RationalFunctions>) -> Self {
        Self {
            inner: CurveFunctionInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_function(inner: CurveFunction<FiniteFunctions>) -> Self {
        Self {
            inner: CurveFunctionInner::FiniteFunction(inner),
        }
    }

    fn field_value(&self) -> PyCurveFunctionField {
        let inner = match &self.inner {
            CurveFunctionInner::Rational(value) => {
                CurveFunctionFieldInner::Rational(value.field().clone())
            }
            CurveFunctionInner::Finite(value) => {
                CurveFunctionFieldInner::Finite(value.field().clone())
            }
            CurveFunctionInner::NumberField(value) => {
                CurveFunctionFieldInner::NumberField(value.field().clone())
            }
            CurveFunctionInner::RationalFunction(value) => {
                CurveFunctionFieldInner::RationalFunction(value.field().clone())
            }
            CurveFunctionInner::FiniteFunction(value) => {
                CurveFunctionFieldInner::FiniteFunction(value.field().clone())
            }
        };
        PyCurveFunctionField { inner }
    }

    fn coerce(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = value.extract::<PyRef<'_, Self>>() {
            return Ok(value.clone());
        }
        self.field_value().constant(py, value)
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {{
                let value = match operation {
                    "add" => ($left + $right),
                    "sub" => ($left - $right),
                    "mul" => ($left * $right),
                    "div" => ($left / $right),
                    _ => return Err(PyValueError::new_err("unknown curve-function operation")),
                }
                .map_err(curve_error)?;
                CurveFunctionInner::$variant(value)
            }};
        }
        let inner = match (&self.inner, &other.inner) {
            (CurveFunctionInner::Rational(left), CurveFunctionInner::Rational(right)) => {
                apply!(left, right, Rational)
            }
            (CurveFunctionInner::Finite(left), CurveFunctionInner::Finite(right)) => {
                apply!(left, right, Finite)
            }
            (CurveFunctionInner::NumberField(left), CurveFunctionInner::NumberField(right)) => {
                apply!(left, right, NumberField)
            }
            (
                CurveFunctionInner::RationalFunction(left),
                CurveFunctionInner::RationalFunction(right),
            ) => apply!(left, right, RationalFunction),
            (
                CurveFunctionInner::FiniteFunction(left),
                CurveFunctionInner::FiniteFunction(right),
            ) => apply!(left, right, FiniteFunction),
            _ => {
                return Err(PyValueError::new_err(
                    "curve functions belong to different fields",
                ))
            }
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyCurveFunction {
    #[classattr]
    fn type_id() -> &'static str {
        CurveFunction::<RationalField>::TYPE_ID
    }

    #[getter]
    fn field(&self) -> PyCurveFunctionField {
        self.field_value()
    }
    fn is_zero(&self) -> bool {
        match &self.inner {
            CurveFunctionInner::Rational(v) => v.is_zero(),
            CurveFunctionInner::Finite(v) => v.is_zero(),
            CurveFunctionInner::NumberField(v) => v.is_zero(),
            CurveFunctionInner::RationalFunction(v) => v.is_zero(),
            CurveFunctionInner::FiniteFunction(v) => v.is_zero(),
        }
    }
    fn is_one(&self) -> bool {
        match &self.inner {
            CurveFunctionInner::Rational(v) => v.is_one(),
            CurveFunctionInner::Finite(v) => v.is_one(),
            CurveFunctionInner::NumberField(v) => v.is_one(),
            CurveFunctionInner::RationalFunction(v) => v.is_one(),
            CurveFunctionInner::FiniteFunction(v) => v.is_one(),
        }
    }

    fn as_constant(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            CurveFunctionInner::Rational(value) => value
                .as_constant()
                .map_err(curve_error)?
                .map(|value| rational_to_python(py, &value))
                .transpose(),
            CurveFunctionInner::Finite(value) => value
                .as_constant()
                .map_err(curve_error)?
                .map(|inner| {
                    Py::new(py, PyFiniteFieldElement { inner }).map(|value| value.into_any())
                })
                .transpose(),
            CurveFunctionInner::NumberField(value) => value
                .as_constant()
                .map_err(curve_error)?
                .map(|inner| {
                    Py::new(py, PyNumberFieldElement { inner }).map(|value| value.into_any())
                })
                .transpose(),
            CurveFunctionInner::RationalFunction(value) => value
                .as_constant()
                .map_err(curve_error)?
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_rational(value))
                        .map(|value| value.into_any())
                })
                .transpose(),
            CurveFunctionInner::FiniteFunction(value) => value
                .as_constant()
                .map_err(curve_error)?
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_finite_field(value))
                        .map(|value| value.into_any())
                })
                .transpose(),
        }
    }

    fn homogeneous_representation(&self) -> PyResult<(PyPolynomial, PyPolynomial)> {
        match &self.inner {
            CurveFunctionInner::Rational(value) => {
                let (numerator, denominator) =
                    value.homogeneous_representation().map_err(curve_error)?;
                Ok((
                    PyPolynomial::from_rational(numerator),
                    PyPolynomial::from_rational(denominator),
                ))
            }
            CurveFunctionInner::Finite(value) => {
                let (numerator, denominator) =
                    value.homogeneous_representation().map_err(curve_error)?;
                Ok((
                    PyPolynomial::from_finite_field(numerator),
                    PyPolynomial::from_finite_field(denominator),
                ))
            }
            CurveFunctionInner::NumberField(value) => {
                let (numerator, denominator) =
                    value.homogeneous_representation().map_err(curve_error)?;
                Ok((
                    PyPolynomial::from_number_field(numerator),
                    PyPolynomial::from_number_field(denominator),
                ))
            }
            CurveFunctionInner::RationalFunction(value) => {
                let (numerator, denominator) =
                    value.homogeneous_representation().map_err(curve_error)?;
                Ok((
                    PyPolynomial::from_rational_function(numerator),
                    PyPolynomial::from_rational_function(denominator),
                ))
            }
            CurveFunctionInner::FiniteFunction(value) => {
                let (numerator, denominator) =
                    value.homogeneous_representation().map_err(curve_error)?;
                Ok((
                    PyPolynomial::from_finite_rational_function(numerator),
                    PyPolynomial::from_finite_rational_function(denominator),
                ))
            }
        }
    }

    #[pyo3(signature = (point, *, limit=256))]
    fn order_at(&self, point: &PyCurvePoint, limit: usize) -> PyResult<PyCurveFunctionOrder> {
        let order = match (&self.inner, &point.inner) {
            (CurveFunctionInner::Rational(value), CurvePointInner::Rational(point)) => {
                value.order_at_with_limit(point, limit)
            }
            (CurveFunctionInner::Finite(value), CurvePointInner::Finite(point)) => {
                value.order_at_with_limit(point, limit)
            }
            (CurveFunctionInner::NumberField(value), CurvePointInner::NumberField(point)) => {
                value.order_at_with_limit(point, limit)
            }
            (
                CurveFunctionInner::RationalFunction(value),
                CurvePointInner::RationalFunction(point),
            ) => value.order_at_with_limit(point, limit),
            (CurveFunctionInner::FiniteFunction(value), CurvePointInner::FiniteFunction(point)) => {
                value.order_at_with_limit(point, limit)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "point belongs to another curve context",
                ))
            }
        }
        .map_err(curve_error)?;
        Ok(PyCurveFunctionOrder { inner: order })
    }

    #[pyo3(signature = (point, *, limit=256))]
    fn evaluate_at(
        &self,
        point: &PyCurvePoint,
        limit: usize,
    ) -> PyResult<PyCurveFunctionEvaluation> {
        let inner = match (&self.inner, &point.inner) {
            (CurveFunctionInner::Rational(value), CurvePointInner::Rational(point)) => {
                FunctionEvaluationInner::Rational(
                    value
                        .evaluate_at_with_limit(point, limit)
                        .map_err(curve_error)?,
                )
            }
            (CurveFunctionInner::Finite(value), CurvePointInner::Finite(point)) => {
                FunctionEvaluationInner::Finite(
                    value
                        .evaluate_at_with_limit(point, limit)
                        .map_err(curve_error)?,
                )
            }
            (CurveFunctionInner::NumberField(value), CurvePointInner::NumberField(point)) => {
                FunctionEvaluationInner::NumberField(
                    value
                        .evaluate_at_with_limit(point, limit)
                        .map_err(curve_error)?,
                )
            }
            (
                CurveFunctionInner::RationalFunction(value),
                CurvePointInner::RationalFunction(point),
            ) => FunctionEvaluationInner::RationalFunction(
                value
                    .evaluate_at_with_limit(point, limit)
                    .map_err(curve_error)?,
            ),
            (CurveFunctionInner::FiniteFunction(value), CurvePointInner::FiniteFunction(point)) => {
                FunctionEvaluationInner::FiniteFunction(
                    value
                        .evaluate_at_with_limit(point, limit)
                        .map_err(curve_error)?,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "point belongs to another curve context",
                ))
            }
        };
        Ok(PyCurveFunctionEvaluation { inner })
    }

    fn principal_divisor(&self) -> PyResult<PyWeilDivisor> {
        let inner = match &self.inner {
            CurveFunctionInner::Rational(value) => {
                WeilDivisorInner::Rational(value.principal_divisor().map_err(curve_error)?)
            }
            CurveFunctionInner::Finite(value) => {
                WeilDivisorInner::Finite(value.principal_divisor().map_err(curve_error)?)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "principal_divisor currently requires Q or finite-field constants",
                ))
            }
        };
        Ok(PyWeilDivisor { inner })
    }

    fn verify_principal_divisor(&self, divisor: &PyWeilDivisor) -> PyResult<()> {
        match (&self.inner, &divisor.inner) {
            (CurveFunctionInner::Rational(value), WeilDivisorInner::Rational(divisor)) => {
                value.verify_principal_divisor(divisor)
            }
            (CurveFunctionInner::Finite(value), WeilDivisorInner::Finite(divisor)) => {
                value.verify_principal_divisor(divisor)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "principal-divisor verification currently requires matching Q or finite-field contexts",
                ))
            }
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn inverse(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            CurveFunctionInner::Rational(value) => {
                CurveFunctionInner::Rational(value.checked_inverse().map_err(curve_error)?)
            }
            CurveFunctionInner::Finite(value) => {
                CurveFunctionInner::Finite(value.checked_inverse().map_err(curve_error)?)
            }
            CurveFunctionInner::NumberField(value) => {
                CurveFunctionInner::NumberField(value.checked_inverse().map_err(curve_error)?)
            }
            CurveFunctionInner::RationalFunction(value) => {
                CurveFunctionInner::RationalFunction(value.checked_inverse().map_err(curve_error)?)
            }
            CurveFunctionInner::FiniteFunction(value) => {
                CurveFunctionInner::FiniteFunction(value.checked_inverse().map_err(curve_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurveFunctionInner::Rational(v) => v.verify(),
            CurveFunctionInner::Finite(v) => v.verify(),
            CurveFunctionInner::NumberField(v) => v.verify(),
            CurveFunctionInner::RationalFunction(v) => v.verify(),
            CurveFunctionInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurveFunctionInner::Rational(value) => value.to_json(),
            CurveFunctionInner::Finite(value) => value.to_json(),
            CurveFunctionInner::NumberField(value) => value.to_json(),
            CurveFunctionInner::RationalFunction(value) => value.to_json(),
            CurveFunctionInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => CurveFunctionInner::Rational(
                CurveFunction::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => CurveFunctionInner::Finite(
                CurveFunction::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => CurveFunctionInner::NumberField(
                CurveFunction::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => CurveFunctionInner::RationalFunction(
                CurveFunction::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => CurveFunctionInner::FiniteFunction(
                CurveFunction::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce(py, other)?, "add")
    }
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce(py, other)?, "sub")
    }
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.coerce(py, other)?.binary(self, "sub")
    }
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce(py, other)?, "mul")
    }
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce(py, other)?, "div")
    }
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.coerce(py, other)?.binary(self, "div")
    }
    fn __neg__(&self) -> PyResult<Self> {
        self.field_value().zero().binary(self, "sub")
    }
    fn __pow__(&self, exponent: i64, _modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (base, exponent) = if exponent < 0 {
            (self.inverse()?, exponent.unsigned_abs())
        } else {
            (self.clone(), exponent as u64)
        };
        let inner = match base.inner {
            CurveFunctionInner::Rational(v) => {
                CurveFunctionInner::Rational(v.pow(exponent).map_err(curve_error)?)
            }
            CurveFunctionInner::Finite(v) => {
                CurveFunctionInner::Finite(v.pow(exponent).map_err(curve_error)?)
            }
            CurveFunctionInner::NumberField(v) => {
                CurveFunctionInner::NumberField(v.pow(exponent).map_err(curve_error)?)
            }
            CurveFunctionInner::RationalFunction(v) => {
                CurveFunctionInner::RationalFunction(v.pow(exponent).map_err(curve_error)?)
            }
            CurveFunctionInner::FiniteFunction(v) => {
                CurveFunctionInner::FiniteFunction(v.pow(exponent).map_err(curve_error)?)
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            CurveFunctionInner::Rational(v) => canonical_hash(v),
            CurveFunctionInner::Finite(v) => canonical_hash(v),
            CurveFunctionInner::NumberField(v) => canonical_hash(v),
            CurveFunctionInner::RationalFunction(v) => canonical_hash(v),
            CurveFunctionInner::FiniteFunction(v) => canonical_hash(v),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            CurveFunctionInner::Rational(v) => v.to_string(),
            CurveFunctionInner::Finite(v) => v.to_string(),
            CurveFunctionInner::NumberField(v) => v.to_string(),
            CurveFunctionInner::RationalFunction(v) => v.to_string(),
            CurveFunctionInner::FiniteFunction(v) => v.to_string(),
        })
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", CurveFunction::<RationalField>::TYPE_ID)?;
        let terms = match &self.inner {
            CurveFunctionInner::Rational(value) => value.representative().nonzero_term_count(),
            CurveFunctionInner::Finite(value) => value.representative().nonzero_term_count(),
            CurveFunctionInner::NumberField(value) => value.representative().nonzero_term_count(),
            CurveFunctionInner::RationalFunction(value) => {
                value.representative().nonzero_term_count()
            }
            CurveFunctionInner::FiniteFunction(value) => {
                value.representative().nonzero_term_count()
            }
        };
        summary.set_item("representative_nonzero_term_count", terms)?;
        summary.set_item("is_zero", self.is_zero())?;
        Ok(summary.unbind())
    }
}

/// Exact finite or infinite order of a curve function at a point.
#[pyclass(
    module = "noether.commalg",
    name = "CurveFunctionOrder",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveFunctionOrder {
    inner: CurveFunctionOrder,
}

#[pymethods]
impl PyCurveFunctionOrder {
    #[getter]
    fn is_infinite(&self) -> bool {
        matches!(self.inner, CurveFunctionOrder::Infinite)
    }
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            CurveFunctionOrder::Infinite => Ok(None),
            CurveFunctionOrder::Finite(value) => integer_to_python(py, value).map(Some),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            CurveFunctionOrder::Infinite => "CurveFunctionOrder(infinite=True)".to_owned(),
            CurveFunctionOrder::Finite(value) => format!("CurveFunctionOrder(value={value})"),
        })
    }
}

#[derive(Clone)]
enum FunctionEvaluationInner {
    Rational(CurveFunctionEvaluation<RationalField>),
    Finite(CurveFunctionEvaluation<FiniteField>),
    NumberField(CurveFunctionEvaluation<NumberField>),
    RationalFunction(CurveFunctionEvaluation<RationalFunctions>),
    FiniteFunction(CurveFunctionEvaluation<FiniteFunctions>),
}

/// Typed regular value, zero, or pole of a curve function.
#[pyclass(
    module = "noether.commalg",
    name = "CurveFunctionEvaluation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveFunctionEvaluation {
    inner: FunctionEvaluationInner,
}

#[pymethods]
impl PyCurveFunctionEvaluation {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            FunctionEvaluationInner::Rational(v) => evaluation_kind(v),
            FunctionEvaluationInner::Finite(v) => evaluation_kind(v),
            FunctionEvaluationInner::NumberField(v) => evaluation_kind(v),
            FunctionEvaluationInner::RationalFunction(v) => evaluation_kind(v),
            FunctionEvaluationInner::FiniteFunction(v) => evaluation_kind(v),
        }
    }
    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let order = match &self.inner {
            FunctionEvaluationInner::Rational(v) => evaluation_order(v),
            FunctionEvaluationInner::Finite(v) => evaluation_order(v),
            FunctionEvaluationInner::NumberField(v) => evaluation_order(v),
            FunctionEvaluationInner::RationalFunction(v) => evaluation_order(v),
            FunctionEvaluationInner::FiniteFunction(v) => evaluation_order(v),
        };
        order.map(|value| integer_to_python(py, value)).transpose()
    }
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            FunctionEvaluationInner::Rational(CurveFunctionEvaluation::Regular(value)) => {
                rational_to_python(py, value).map(Some)
            }
            FunctionEvaluationInner::Finite(CurveFunctionEvaluation::Regular(value)) => Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.clone(),
                },
            )
            .map(|value| Some(value.into_any())),
            FunctionEvaluationInner::NumberField(CurveFunctionEvaluation::Regular(value)) => {
                Py::new(
                    py,
                    PyNumberFieldElement {
                        inner: value.clone(),
                    },
                )
                .map(|value| Some(value.into_any()))
            }
            FunctionEvaluationInner::RationalFunction(CurveFunctionEvaluation::Regular(value)) => {
                Py::new(py, PyRationalFunction::from_rational(value.clone()))
                    .map(|value| Some(value.into_any()))
            }
            FunctionEvaluationInner::FiniteFunction(CurveFunctionEvaluation::Regular(value)) => {
                Py::new(py, PyRationalFunction::from_finite_field(value.clone()))
                    .map(|value| Some(value.into_any()))
            }
            _ => Ok(None),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("CurveFunctionEvaluation(kind={:?})", self.kind()))
    }
}

fn evaluation_kind<R: Field>(value: &CurveFunctionEvaluation<R>) -> &'static str {
    match value {
        CurveFunctionEvaluation::Regular(_) => "regular",
        CurveFunctionEvaluation::Zero { .. } => "zero",
        CurveFunctionEvaluation::Pole { .. } => "pole",
    }
}
fn evaluation_order<R: Field>(value: &CurveFunctionEvaluation<R>) -> Option<&Integer> {
    match value {
        CurveFunctionEvaluation::Regular(_) => None,
        CurveFunctionEvaluation::Zero { order } => order.as_ref(),
        CurveFunctionEvaluation::Pole { order } => Some(order),
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CurveClosedPointInner {
    Rational(CurveClosedPoint<RationalField>),
    Finite(CurveClosedPoint<FiniteField>),
    NumberField(CurveClosedPoint<NumberField>),
    RationalFunction(CurveClosedPoint<RationalFunctions>),
    FiniteFunction(CurveClosedPoint<FiniteFunctions>),
}

/// A closed point (place) on one curve.
#[pyclass(
    module = "noether.commalg",
    name = "CurveClosedPoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveClosedPoint {
    inner: CurveClosedPointInner,
}

#[pymethods]
impl PyCurveClosedPoint {
    #[classattr]
    fn type_id() -> &'static str {
        CurveClosedPoint::<RationalField>::TYPE_ID
    }
    #[getter]
    fn curve(&self) -> PyCurve {
        match &self.inner {
            CurveClosedPointInner::Rational(v) => PyCurve::from_rational(v.curve().clone()),
            CurveClosedPointInner::Finite(v) => PyCurve::from_finite(v.curve().clone()),
            CurveClosedPointInner::NumberField(v) => PyCurve::from_number_field(v.curve().clone()),
            CurveClosedPointInner::RationalFunction(v) => {
                PyCurve::from_rational_function(v.curve().clone())
            }
            CurveClosedPointInner::FiniteFunction(v) => {
                PyCurve::from_finite_function(v.curve().clone())
            }
        }
    }
    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            match &self.inner {
                CurveClosedPointInner::Rational(v) => v.degree(),
                CurveClosedPointInner::Finite(v) => v.degree(),
                CurveClosedPointInner::NumberField(v) => v.degree(),
                CurveClosedPointInner::RationalFunction(v) => v.degree(),
                CurveClosedPointInner::FiniteFunction(v) => v.degree(),
            },
        )
    }
    #[getter]
    fn defining_ideal(&self) -> PyResult<PyIdeal> {
        Ok(match &self.inner {
            CurveClosedPointInner::Rational(v) => {
                PyIdeal::from_rational_polynomial(v.prime_ideal().ideal().clone())
            }
            CurveClosedPointInner::Finite(v) => {
                PyIdeal::from_finite_field_polynomial(v.prime_ideal().ideal().clone())
            }
            CurveClosedPointInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "number-field polynomial ideals are not yet projected",
                ))
            }
            CurveClosedPointInner::RationalFunction(v) => {
                PyIdeal::from_rational_function_polynomial(v.prime_ideal().ideal().clone())
            }
            CurveClosedPointInner::FiniteFunction(v) => {
                PyIdeal::from_finite_rational_function_polynomial(v.prime_ideal().ideal().clone())
            }
        })
    }
    #[getter]
    fn rational_point(&self) -> Option<PyCurvePoint> {
        match &self.inner {
            CurveClosedPointInner::Rational(v) => {
                v.rational_point().cloned().map(|v| PyCurvePoint {
                    inner: CurvePointInner::Rational(v),
                })
            }
            CurveClosedPointInner::Finite(v) => v.rational_point().cloned().map(|v| PyCurvePoint {
                inner: CurvePointInner::Finite(v),
            }),
            CurveClosedPointInner::NumberField(v) => v
                .rational_point()
                .cloned()
                .map(PyCurvePoint::from_number_field),
            CurveClosedPointInner::RationalFunction(v) => {
                v.rational_point().cloned().map(|v| PyCurvePoint {
                    inner: CurvePointInner::RationalFunction(v),
                })
            }
            CurveClosedPointInner::FiniteFunction(v) => {
                v.rational_point().cloned().map(|v| PyCurvePoint {
                    inner: CurvePointInner::FiniteFunction(v),
                })
            }
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurveClosedPointInner::Rational(v) => v.verify(),
            CurveClosedPointInner::Finite(v) => v.verify(),
            CurveClosedPointInner::NumberField(v) => v.verify(),
            CurveClosedPointInner::RationalFunction(v) => v.verify(),
            CurveClosedPointInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurveClosedPointInner::Rational(value) => value.to_json(),
            CurveClosedPointInner::Finite(value) => value.to_json(),
            CurveClosedPointInner::NumberField(value) => value.to_json(),
            CurveClosedPointInner::RationalFunction(value) => value.to_json(),
            CurveClosedPointInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => CurveClosedPointInner::Rational(
                CurveClosedPoint::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => CurveClosedPointInner::Finite(
                CurveClosedPoint::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => CurveClosedPointInner::NumberField(
                CurveClosedPoint::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => CurveClosedPointInner::RationalFunction(
                CurveClosedPoint::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => CurveClosedPointInner::FiniteFunction(
                CurveClosedPoint::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            CurveClosedPointInner::Rational(v) => canonical_hash(v),
            CurveClosedPointInner::Finite(v) => canonical_hash(v),
            CurveClosedPointInner::NumberField(v) => canonical_hash(v),
            CurveClosedPointInner::RationalFunction(v) => canonical_hash(v),
            CurveClosedPointInner::FiniteFunction(v) => canonical_hash(v),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text("CurveClosedPoint()".to_owned())
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", CurveClosedPoint::<RationalField>::TYPE_ID)?;
        summary.set_item("degree", self.degree(py)?)?;
        summary.set_item("rational", self.rational_point().is_some())?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WeilDivisorInner {
    Rational(WeilDivisor<RationalField>),
    Finite(WeilDivisor<FiniteField>),
    NumberField(WeilDivisor<NumberField>),
    RationalFunction(WeilDivisor<RationalFunctions>),
    FiniteFunction(WeilDivisor<FiniteFunctions>),
}

/// A Weil divisor with arbitrary-precision integral coefficients.
#[pyclass(
    module = "noether.commalg",
    name = "WeilDivisor",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeilDivisor {
    inner: WeilDivisorInner,
}

impl PyWeilDivisor {
    pub(crate) fn from_rational(inner: WeilDivisor<RationalField>) -> Self {
        Self {
            inner: WeilDivisorInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite(inner: WeilDivisor<FiniteField>) -> Self {
        Self {
            inner: WeilDivisorInner::Finite(inner),
        }
    }

    pub(crate) fn from_number_field(inner: WeilDivisor<NumberField>) -> Self {
        Self {
            inner: WeilDivisorInner::NumberField(inner),
        }
    }

    pub(crate) fn from_rational_function(inner: WeilDivisor<RationalFunctions>) -> Self {
        Self {
            inner: WeilDivisorInner::RationalFunction(inner),
        }
    }

    pub(crate) fn from_finite_function(inner: WeilDivisor<FiniteFunctions>) -> Self {
        Self {
            inner: WeilDivisorInner::FiniteFunction(inner),
        }
    }
}

#[pymethods]
impl PyWeilDivisor {
    #[new]
    fn new(curve: &PyCurve, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let terms = terms
            .try_iter()
            .map_err(|_| PyTypeError::new_err("divisor terms must be iterable pairs"))?
            .map(|item| {
                let item = item?;
                let (point, coefficient): (PyRef<'_, PyCurveClosedPoint>, Bound<'_, PyAny>) =
                    item.extract()?;
                Ok((point.clone(), extract_integer(&coefficient)?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = match &curve.inner {
            CurveInner::Rational(curve) => WeilDivisorInner::Rational(
                WeilDivisor::new(
                    curve.clone(),
                    terms
                        .iter()
                        .map(|(p, c)| match &p.inner {
                            CurveClosedPointInner::Rational(p) => Ok((p.clone(), c.clone())),
                            _ => Err(PyTypeError::new_err("divisor point has another context")),
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(curve_error)?,
            ),
            CurveInner::NumberField(curve) => WeilDivisorInner::NumberField(
                WeilDivisor::new(
                    curve.clone(),
                    terms
                        .iter()
                        .map(|(p, c)| match &p.inner {
                            CurveClosedPointInner::NumberField(p) => Ok((p.clone(), c.clone())),
                            _ => Err(PyTypeError::new_err("divisor point has another context")),
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(curve_error)?,
            ),
            CurveInner::Finite(curve) => WeilDivisorInner::Finite(
                WeilDivisor::new(
                    curve.clone(),
                    terms
                        .iter()
                        .map(|(p, c)| match &p.inner {
                            CurveClosedPointInner::Finite(p) => Ok((p.clone(), c.clone())),
                            _ => Err(PyTypeError::new_err("divisor point has another context")),
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(curve_error)?,
            ),
            CurveInner::RationalFunction(curve) => WeilDivisorInner::RationalFunction(
                WeilDivisor::new(
                    curve.clone(),
                    terms
                        .iter()
                        .map(|(p, c)| match &p.inner {
                            CurveClosedPointInner::RationalFunction(p) => {
                                Ok((p.clone(), c.clone()))
                            }
                            _ => Err(PyTypeError::new_err("divisor point has another context")),
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(curve_error)?,
            ),
            CurveInner::FiniteFunction(curve) => WeilDivisorInner::FiniteFunction(
                WeilDivisor::new(
                    curve.clone(),
                    terms
                        .iter()
                        .map(|(p, c)| match &p.inner {
                            CurveClosedPointInner::FiniteFunction(p) => Ok((p.clone(), c.clone())),
                            _ => Err(PyTypeError::new_err("divisor point has another context")),
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map_err(curve_error)?,
            ),
        };
        Ok(Self { inner })
    }
    #[classattr]
    fn type_id() -> &'static str {
        WeilDivisor::<RationalField>::TYPE_ID
    }
    #[getter]
    fn curve(&self) -> PyCurve {
        match &self.inner {
            WeilDivisorInner::Rational(v) => PyCurve::from_rational(v.curve().clone()),
            WeilDivisorInner::Finite(v) => PyCurve::from_finite(v.curve().clone()),
            WeilDivisorInner::NumberField(v) => PyCurve::from_number_field(v.curve().clone()),
            WeilDivisorInner::RationalFunction(v) => {
                PyCurve::from_rational_function(v.curve().clone())
            }
            WeilDivisorInner::FiniteFunction(v) => PyCurve::from_finite_function(v.curve().clone()),
        }
    }
    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            &match &self.inner {
                WeilDivisorInner::Rational(v) => v.degree(),
                WeilDivisorInner::Finite(v) => v.degree(),
                WeilDivisorInner::NumberField(v) => v.degree(),
                WeilDivisorInner::RationalFunction(v) => v.degree(),
                WeilDivisorInner::FiniteFunction(v) => v.degree(),
            },
        )
    }
    #[getter]
    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(PyCurveClosedPoint, Py<PyAny>)>> {
        match &self.inner {
            WeilDivisorInner::Rational(v) => divisor_terms(py, v, CurveClosedPointInner::Rational),
            WeilDivisorInner::Finite(v) => divisor_terms(py, v, CurveClosedPointInner::Finite),
            WeilDivisorInner::NumberField(v) => {
                divisor_terms(py, v, CurveClosedPointInner::NumberField)
            }
            WeilDivisorInner::RationalFunction(v) => {
                divisor_terms(py, v, CurveClosedPointInner::RationalFunction)
            }
            WeilDivisorInner::FiniteFunction(v) => {
                divisor_terms(py, v, CurveClosedPointInner::FiniteFunction)
            }
        }
    }
    #[getter]
    fn support(&self) -> Vec<PyCurveClosedPoint> {
        match &self.inner {
            WeilDivisorInner::Rational(v) => v
                .support()
                .cloned()
                .map(|inner| PyCurveClosedPoint {
                    inner: CurveClosedPointInner::Rational(inner),
                })
                .collect(),
            WeilDivisorInner::Finite(v) => v
                .support()
                .cloned()
                .map(|inner| PyCurveClosedPoint {
                    inner: CurveClosedPointInner::Finite(inner),
                })
                .collect(),
            WeilDivisorInner::NumberField(v) => v
                .support()
                .cloned()
                .map(|inner| PyCurveClosedPoint {
                    inner: CurveClosedPointInner::NumberField(inner),
                })
                .collect(),
            WeilDivisorInner::RationalFunction(v) => v
                .support()
                .cloned()
                .map(|inner| PyCurveClosedPoint {
                    inner: CurveClosedPointInner::RationalFunction(inner),
                })
                .collect(),
            WeilDivisorInner::FiniteFunction(v) => v
                .support()
                .cloned()
                .map(|inner| PyCurveClosedPoint {
                    inner: CurveClosedPointInner::FiniteFunction(inner),
                })
                .collect(),
        }
    }
    fn coefficient(&self, py: Python<'_>, point: &PyCurveClosedPoint) -> PyResult<Py<PyAny>> {
        let value = match (&self.inner, &point.inner) {
            (WeilDivisorInner::Rational(v), CurveClosedPointInner::Rational(p)) => v.coefficient(p),
            (WeilDivisorInner::Finite(v), CurveClosedPointInner::Finite(p)) => v.coefficient(p),
            (WeilDivisorInner::NumberField(v), CurveClosedPointInner::NumberField(p)) => {
                v.coefficient(p)
            }
            (WeilDivisorInner::RationalFunction(v), CurveClosedPointInner::RationalFunction(p)) => {
                v.coefficient(p)
            }
            (WeilDivisorInner::FiniteFunction(v), CurveClosedPointInner::FiniteFunction(p)) => {
                v.coefficient(p)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "divisor and closed point have different contexts",
                ))
            }
        }
        .map_err(curve_error)?;
        integer_to_python(py, &value)
    }
    fn scale(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_integer(scalar)?;
        Ok(Self {
            inner: match &self.inner {
                WeilDivisorInner::Rational(v) => WeilDivisorInner::Rational(v.scale(&scalar)),
                WeilDivisorInner::Finite(v) => WeilDivisorInner::Finite(v.scale(&scalar)),
                WeilDivisorInner::NumberField(v) => WeilDivisorInner::NumberField(v.scale(&scalar)),
                WeilDivisorInner::RationalFunction(v) => {
                    WeilDivisorInner::RationalFunction(v.scale(&scalar))
                }
                WeilDivisorInner::FiniteFunction(v) => {
                    WeilDivisorInner::FiniteFunction(v.scale(&scalar))
                }
            },
        })
    }
    fn positive_part(&self) -> Self {
        Self {
            inner: match &self.inner {
                WeilDivisorInner::Rational(v) => WeilDivisorInner::Rational(v.positive_part()),
                WeilDivisorInner::Finite(v) => WeilDivisorInner::Finite(v.positive_part()),
                WeilDivisorInner::NumberField(v) => {
                    WeilDivisorInner::NumberField(v.positive_part())
                }
                WeilDivisorInner::RationalFunction(v) => {
                    WeilDivisorInner::RationalFunction(v.positive_part())
                }
                WeilDivisorInner::FiniteFunction(v) => {
                    WeilDivisorInner::FiniteFunction(v.positive_part())
                }
            },
        }
    }
    fn negative_part(&self) -> Self {
        Self {
            inner: match &self.inner {
                WeilDivisorInner::Rational(v) => WeilDivisorInner::Rational(v.negative_part()),
                WeilDivisorInner::Finite(v) => WeilDivisorInner::Finite(v.negative_part()),
                WeilDivisorInner::NumberField(v) => {
                    WeilDivisorInner::NumberField(v.negative_part())
                }
                WeilDivisorInner::RationalFunction(v) => {
                    WeilDivisorInner::RationalFunction(v.negative_part())
                }
                WeilDivisorInner::FiniteFunction(v) => {
                    WeilDivisorInner::FiniteFunction(v.negative_part())
                }
            },
        }
    }
    fn riemann_roch_space(&self) -> PyResult<PyRiemannRochSpace> {
        let inner = match &self.inner {
            WeilDivisorInner::Rational(v) => {
                RiemannRochInner::Rational(v.riemann_roch_space().map_err(curve_error)?)
            }
            WeilDivisorInner::Finite(v) => {
                RiemannRochInner::Finite(v.riemann_roch_space().map_err(curve_error)?)
            }
            WeilDivisorInner::NumberField(v) => {
                RiemannRochInner::NumberField(v.riemann_roch_space().map_err(curve_error)?)
            }
            WeilDivisorInner::RationalFunction(v) => {
                RiemannRochInner::RationalFunction(v.riemann_roch_space().map_err(curve_error)?)
            }
            WeilDivisorInner::FiniteFunction(v) => {
                RiemannRochInner::FiniteFunction(v.riemann_roch_space().map_err(curve_error)?)
            }
        };
        Ok(PyRiemannRochSpace { inner })
    }
    fn pullback(&self, morphism: &PyCurveMorphism) -> PyResult<Self> {
        let inner = match (&self.inner, &morphism.inner) {
            (WeilDivisorInner::Rational(v), CurveMorphismInner::Rational(m)) => {
                WeilDivisorInner::Rational(v.pullback(m).map_err(curve_error)?)
            }
            (WeilDivisorInner::Finite(v), CurveMorphismInner::Finite(m)) => {
                WeilDivisorInner::Finite(v.pullback(m).map_err(curve_error)?)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "pullback currently requires matching Q or finite-field contexts",
                ))
            }
        };
        Ok(Self { inner })
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            WeilDivisorInner::Rational(v) => v.verify(),
            WeilDivisorInner::Finite(v) => v.verify(),
            WeilDivisorInner::NumberField(v) => v.verify(),
            WeilDivisorInner::RationalFunction(v) => v.verify(),
            WeilDivisorInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            WeilDivisorInner::Rational(value) => value.to_json(),
            WeilDivisorInner::Finite(value) => value.to_json(),
            WeilDivisorInner::NumberField(value) => value.to_json(),
            WeilDivisorInner::RationalFunction(value) => value.to_json(),
            WeilDivisorInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => WeilDivisorInner::Rational(
                WeilDivisor::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => WeilDivisorInner::Finite(
                WeilDivisor::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => WeilDivisorInner::NumberField(
                WeilDivisor::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => WeilDivisorInner::RationalFunction(
                WeilDivisor::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => WeilDivisorInner::FiniteFunction(
                WeilDivisor::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __add__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (WeilDivisorInner::Rational(left), WeilDivisorInner::Rational(right)) => {
                WeilDivisorInner::Rational(left.checked_add(right).map_err(curve_error)?)
            }
            (WeilDivisorInner::Finite(left), WeilDivisorInner::Finite(right)) => {
                WeilDivisorInner::Finite(left.checked_add(right).map_err(curve_error)?)
            }
            (WeilDivisorInner::NumberField(left), WeilDivisorInner::NumberField(right)) => {
                WeilDivisorInner::NumberField(left.checked_add(right).map_err(curve_error)?)
            }
            (
                WeilDivisorInner::RationalFunction(left),
                WeilDivisorInner::RationalFunction(right),
            ) => WeilDivisorInner::RationalFunction(left.checked_add(right).map_err(curve_error)?),
            (WeilDivisorInner::FiniteFunction(left), WeilDivisorInner::FiniteFunction(right)) => {
                WeilDivisorInner::FiniteFunction(left.checked_add(right).map_err(curve_error)?)
            }
            _ => return Err(PyTypeError::new_err("divisors have different contexts")),
        };
        Ok(Self { inner })
    }
    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.__add__(&other.__neg__())
    }
    fn __neg__(&self) -> Self {
        Self {
            inner: match &self.inner {
                WeilDivisorInner::Rational(v) => WeilDivisorInner::Rational(v.negated()),
                WeilDivisorInner::Finite(v) => WeilDivisorInner::Finite(v.negated()),
                WeilDivisorInner::NumberField(v) => WeilDivisorInner::NumberField(v.negated()),
                WeilDivisorInner::RationalFunction(v) => {
                    WeilDivisorInner::RationalFunction(v.negated())
                }
                WeilDivisorInner::FiniteFunction(v) => {
                    WeilDivisorInner::FiniteFunction(v.negated())
                }
            },
        }
    }
    fn __mul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scale(scalar)
    }
    fn __rmul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scale(scalar)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            WeilDivisorInner::Rational(v) => canonical_hash(v),
            WeilDivisorInner::Finite(v) => canonical_hash(v),
            WeilDivisorInner::NumberField(v) => canonical_hash(v),
            WeilDivisorInner::RationalFunction(v) => canonical_hash(v),
            WeilDivisorInner::FiniteFunction(v) => canonical_hash(v),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text("WeilDivisor()".to_owned())
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", WeilDivisor::<RationalField>::TYPE_ID)?;
        summary.set_item("support_count", self.support().len())?;
        summary.set_item("degree", self.degree(py)?)?;
        Ok(summary.unbind())
    }
}

fn divisor_terms<R: Field>(
    py: Python<'_>,
    divisor: &WeilDivisor<R>,
    wrap: fn(CurveClosedPoint<R>) -> CurveClosedPointInner,
) -> PyResult<Vec<(PyCurveClosedPoint, Py<PyAny>)>> {
    divisor
        .terms()
        .iter()
        .map(|(p, c)| {
            Ok((
                PyCurveClosedPoint {
                    inner: wrap(p.clone()),
                },
                integer_to_python(py, c)?,
            ))
        })
        .collect()
}

#[derive(Clone, PartialEq, Eq)]
enum RiemannRochInner {
    Rational(RiemannRochSpace<RationalField>),
    Finite(RiemannRochSpace<FiniteField>),
    NumberField(RiemannRochSpace<NumberField>),
    RationalFunction(RiemannRochSpace<RationalFunctions>),
    FiniteFunction(RiemannRochSpace<FiniteFunctions>),
}
/// A canonical verified Riemann--Roch space.
#[pyclass(
    module = "noether.commalg",
    name = "RiemannRochSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRiemannRochSpace {
    inner: RiemannRochInner,
}
#[pymethods]
impl PyRiemannRochSpace {
    #[classattr]
    fn type_id() -> &'static str {
        RiemannRochSpace::<RationalField>::TYPE_ID
    }
    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            RiemannRochInner::Rational(v) => v.dimension(),
            RiemannRochInner::Finite(v) => v.dimension(),
            RiemannRochInner::NumberField(v) => v.dimension(),
            RiemannRochInner::RationalFunction(v) => v.dimension(),
            RiemannRochInner::FiniteFunction(v) => v.dimension(),
        }
    }
    #[getter]
    fn divisor(&self) -> PyWeilDivisor {
        PyWeilDivisor {
            inner: match &self.inner {
                RiemannRochInner::Rational(v) => WeilDivisorInner::Rational(v.divisor().clone()),
                RiemannRochInner::Finite(v) => WeilDivisorInner::Finite(v.divisor().clone()),
                RiemannRochInner::NumberField(v) => {
                    WeilDivisorInner::NumberField(v.divisor().clone())
                }
                RiemannRochInner::RationalFunction(v) => {
                    WeilDivisorInner::RationalFunction(v.divisor().clone())
                }
                RiemannRochInner::FiniteFunction(v) => {
                    WeilDivisorInner::FiniteFunction(v.divisor().clone())
                }
            },
        }
    }
    #[getter]
    fn curve(&self) -> PyCurve {
        self.divisor().curve()
    }
    #[getter]
    fn basis(&self) -> Vec<PyCurveFunction> {
        match &self.inner {
            RiemannRochInner::Rational(v) => v
                .basis()
                .iter()
                .cloned()
                .map(|v| PyCurveFunction {
                    inner: CurveFunctionInner::Rational(v),
                })
                .collect(),
            RiemannRochInner::Finite(v) => v
                .basis()
                .iter()
                .cloned()
                .map(|v| PyCurveFunction {
                    inner: CurveFunctionInner::Finite(v),
                })
                .collect(),
            RiemannRochInner::NumberField(v) => v
                .basis()
                .iter()
                .cloned()
                .map(PyCurveFunction::from_number_field)
                .collect(),
            RiemannRochInner::RationalFunction(v) => v
                .basis()
                .iter()
                .cloned()
                .map(|v| PyCurveFunction {
                    inner: CurveFunctionInner::RationalFunction(v),
                })
                .collect(),
            RiemannRochInner::FiniteFunction(v) => v
                .basis()
                .iter()
                .cloned()
                .map(|v| PyCurveFunction {
                    inner: CurveFunctionInner::FiniteFunction(v),
                })
                .collect(),
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            RiemannRochInner::Rational(v) => v.verify(),
            RiemannRochInner::Finite(v) => v.verify(),
            RiemannRochInner::NumberField(v) => v.verify(),
            RiemannRochInner::RationalFunction(v) => v.verify(),
            RiemannRochInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            RiemannRochInner::Rational(value) => value.to_json(),
            RiemannRochInner::Finite(value) => value.to_json(),
            RiemannRochInner::NumberField(value) => value.to_json(),
            RiemannRochInner::RationalFunction(value) => value.to_json(),
            RiemannRochInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => RiemannRochInner::Rational(
                RiemannRochSpace::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => RiemannRochInner::Finite(
                RiemannRochSpace::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => RiemannRochInner::NumberField(
                RiemannRochSpace::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => RiemannRochInner::RationalFunction(
                RiemannRochSpace::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => RiemannRochInner::FiniteFunction(
                RiemannRochSpace::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            RiemannRochInner::Rational(v) => canonical_hash(v),
            RiemannRochInner::Finite(v) => canonical_hash(v),
            RiemannRochInner::NumberField(v) => canonical_hash(v),
            RiemannRochInner::RationalFunction(v) => canonical_hash(v),
            RiemannRochInner::FiniteFunction(v) => canonical_hash(v),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("RiemannRochSpace(dimension={})", self.dimension()))
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RiemannRochSpace::<RationalField>::TYPE_ID)?;
        summary.set_item("dimension", self.dimension())?;
        summary.set_item("divisor_degree", self.divisor().degree(py)?)?;
        summary.set_item("evidence", "deterministic_recomputation")?;
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CurveMorphismInner {
    Rational(CurveMorphism<RationalField>),
    Finite(CurveMorphism<FiniteField>),
    NumberField(CurveMorphism<NumberField>),
    RationalFunction(CurveMorphism<RationalFunctions>),
    FiniteFunction(CurveMorphism<FiniteFunctions>),
}
/// A checked projective morphism between curves.
#[pyclass(
    module = "noether.commalg",
    name = "CurveMorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurveMorphism {
    inner: CurveMorphismInner,
}
#[pymethods]
impl PyCurveMorphism {
    #[new]
    fn new(source: &PyCurve, target: &PyCurve, images: &Bound<'_, PyAny>) -> PyResult<Self> {
        let images = images
            .try_iter()
            .map_err(|_| PyTypeError::new_err("coordinate images must be iterable"))?
            .map(|value| {
                let value = value?;
                let value = value.extract::<PyRef<'_, PyPolynomial>>()?;
                Ok((*value).clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        if images.len() != 3 {
            return Err(PyValueError::new_err(
                "a projective curve morphism needs three coordinate images",
            ));
        }
        let inner = match (&source.inner, &target.inner) {
            (CurveInner::Rational(s), CurveInner::Rational(t)) => CurveMorphismInner::Rational(
                CurveMorphism::new(
                    s.clone(),
                    t.clone(),
                    extract_polynomial_array(&images, PyPolynomial::as_rational)?,
                )
                .map_err(curve_error)?,
            ),
            (CurveInner::Finite(s), CurveInner::Finite(t)) => CurveMorphismInner::Finite(
                CurveMorphism::new(
                    s.clone(),
                    t.clone(),
                    extract_polynomial_array(&images, PyPolynomial::as_finite_field)?,
                )
                .map_err(curve_error)?,
            ),
            (CurveInner::NumberField(s), CurveInner::NumberField(t)) => {
                CurveMorphismInner::NumberField(
                    CurveMorphism::new(
                        s.clone(),
                        t.clone(),
                        extract_polynomial_array(&images, PyPolynomial::as_number_field)?,
                    )
                    .map_err(curve_error)?,
                )
            }
            (CurveInner::RationalFunction(s), CurveInner::RationalFunction(t)) => {
                CurveMorphismInner::RationalFunction(
                    CurveMorphism::new(
                        s.clone(),
                        t.clone(),
                        extract_polynomial_array(&images, PyPolynomial::as_rational_function)?,
                    )
                    .map_err(curve_error)?,
                )
            }
            (CurveInner::FiniteFunction(s), CurveInner::FiniteFunction(t)) => {
                CurveMorphismInner::FiniteFunction(
                    CurveMorphism::new(
                        s.clone(),
                        t.clone(),
                        extract_polynomial_array(
                            &images,
                            PyPolynomial::as_finite_rational_function,
                        )?,
                    )
                    .map_err(curve_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "curve morphism endpoints need the same coefficient context",
                ))
            }
        };
        Ok(Self { inner })
    }
    #[staticmethod]
    fn identity(curve: &PyCurve) -> PyResult<Self> {
        let inner = match &curve.inner {
            CurveInner::Rational(v) => CurveMorphismInner::Rational(
                CurveMorphism::identity(v.clone()).map_err(curve_error)?,
            ),
            CurveInner::Finite(v) => {
                CurveMorphismInner::Finite(CurveMorphism::identity(v.clone()).map_err(curve_error)?)
            }
            CurveInner::NumberField(v) => CurveMorphismInner::NumberField(
                CurveMorphism::identity(v.clone()).map_err(curve_error)?,
            ),
            CurveInner::RationalFunction(v) => CurveMorphismInner::RationalFunction(
                CurveMorphism::identity(v.clone()).map_err(curve_error)?,
            ),
            CurveInner::FiniteFunction(v) => CurveMorphismInner::FiniteFunction(
                CurveMorphism::identity(v.clone()).map_err(curve_error)?,
            ),
        };
        Ok(Self { inner })
    }
    #[getter]
    fn source(&self) -> PyCurve {
        match &self.inner {
            CurveMorphismInner::Rational(v) => PyCurve::from_rational(v.source().clone()),
            CurveMorphismInner::Finite(v) => PyCurve::from_finite(v.source().clone()),
            CurveMorphismInner::NumberField(v) => PyCurve::from_number_field(v.source().clone()),
            CurveMorphismInner::RationalFunction(v) => {
                PyCurve::from_rational_function(v.source().clone())
            }
            CurveMorphismInner::FiniteFunction(v) => {
                PyCurve::from_finite_function(v.source().clone())
            }
        }
    }
    #[getter]
    fn target(&self) -> PyCurve {
        match &self.inner {
            CurveMorphismInner::Rational(v) => PyCurve::from_rational(v.target().clone()),
            CurveMorphismInner::Finite(v) => PyCurve::from_finite(v.target().clone()),
            CurveMorphismInner::NumberField(v) => PyCurve::from_number_field(v.target().clone()),
            CurveMorphismInner::RationalFunction(v) => {
                PyCurve::from_rational_function(v.target().clone())
            }
            CurveMorphismInner::FiniteFunction(v) => {
                PyCurve::from_finite_function(v.target().clone())
            }
        }
    }
    #[getter]
    fn coordinate_images(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            CurveMorphismInner::Rational(v) => v
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            CurveMorphismInner::Finite(v) => v
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
            CurveMorphismInner::NumberField(v) => v
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_number_field)
                .collect(),
            CurveMorphismInner::RationalFunction(v) => v
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational_function)
                .collect(),
            CurveMorphismInner::FiniteFunction(v) => v
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_rational_function)
                .collect(),
        }
    }
    #[getter]
    fn coordinate_degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            match &self.inner {
                CurveMorphismInner::Rational(v) => v.coordinate_degree(),
                CurveMorphismInner::Finite(v) => v.coordinate_degree(),
                CurveMorphismInner::NumberField(v) => v.coordinate_degree(),
                CurveMorphismInner::RationalFunction(v) => v.coordinate_degree(),
                CurveMorphismInner::FiniteFunction(v) => v.coordinate_degree(),
            },
        )
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            CurveMorphismInner::Rational(v) => v.verify(),
            CurveMorphismInner::Finite(v) => v.verify(),
            CurveMorphismInner::NumberField(v) => v.verify(),
            CurveMorphismInner::RationalFunction(v) => v.verify(),
            CurveMorphismInner::FiniteFunction(v) => v.verify(),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CurveMorphismInner::Rational(value) => value.to_json(),
            CurveMorphismInner::Finite(value) => value.to_json(),
            CurveMorphismInner::NumberField(value) => value.to_json(),
            CurveMorphismInner::RationalFunction(value) => value.to_json(),
            CurveMorphismInner::FiniteFunction(value) => value.to_json(),
        }
        .map_err(json_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_curve_context(value)? {
            SerializedCurveContext::Rational => CurveMorphismInner::Rational(
                CurveMorphism::<RationalField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::Finite => CurveMorphismInner::Finite(
                CurveMorphism::<FiniteField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::NumberField => CurveMorphismInner::NumberField(
                CurveMorphism::<NumberField>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::RationalFunction => CurveMorphismInner::RationalFunction(
                CurveMorphism::<RationalFunctions>::from_json(value).map_err(json_error)?,
            ),
            SerializedCurveContext::FiniteFunction => CurveMorphismInner::FiniteFunction(
                CurveMorphism::<FiniteFunctions>::from_json(value).map_err(json_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            CurveMorphismInner::Rational(v) => canonical_hash(v),
            CurveMorphismInner::Finite(v) => canonical_hash(v),
            CurveMorphismInner::NumberField(v) => canonical_hash(v),
            CurveMorphismInner::RationalFunction(v) => canonical_hash(v),
            CurveMorphismInner::FiniteFunction(v) => canonical_hash(v),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text("CurveMorphism()".to_owned())
    }
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", CurveMorphism::<RationalField>::TYPE_ID)?;
        summary.set_item("coordinate_degree", self.coordinate_degree(py)?)?;
        summary.set_item("coordinate_count", 3)?;
        Ok(summary.unbind())
    }
}

fn extract_polynomial_array<R: Field>(
    values: &[PyPolynomial],
    extract: fn(&PyPolynomial) -> Option<&Polynomial<R>>,
) -> PyResult<[Polynomial<R>; 3]> {
    values
        .iter()
        .map(|v| {
            extract(v).cloned().ok_or_else(|| {
                PyTypeError::new_err("coordinate image has the wrong polynomial context")
            })
        })
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| PyValueError::new_err("expected three coordinate images"))
}
