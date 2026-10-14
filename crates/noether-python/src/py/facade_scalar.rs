// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared exact-scalar conversion at the typed Python facade boundary.

use noether::{Integer, Rational};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyInt};

pub(super) fn extract_integer(value: &Bound<'_, PyAny>) -> PyResult<Integer> {
    if !value.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "integer coefficient must be a Python int",
        ));
    }
    if let Ok(integer) = value.extract::<i64>() {
        return Ok(Integer::from(integer));
    }
    let hexadecimal = value
        .call_method1("__format__", ("x",))?
        .extract::<String>()?;
    Integer::from_str_radix(&hexadecimal, 16)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn fraction_type(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("fractions")?.getattr("Fraction")
}

pub(super) fn extract_rational(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if value.is_instance_of::<PyInt>() {
        return Ok(Rational::from(extract_integer(value)?));
    }
    if !value.is_instance(&fraction_type(py)?)? {
        return Err(PyTypeError::new_err(
            "rational coefficient must be a Python int or fractions.Fraction",
        ));
    }
    let numerator = extract_integer(&value.getattr("numerator")?)?;
    let denominator = extract_integer(&value.getattr("denominator")?)?;
    Rational::new(numerator, denominator).map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn integer_to_python(py: Python<'_>, value: &Integer) -> PyResult<Py<PyAny>> {
    let hexadecimal = value
        .to_string_radix(16)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(py
        .get_type::<PyInt>()
        .call1((hexadecimal, 16))?
        .unbind()
        .into_any())
}

pub(super) fn rational_to_python(py: Python<'_>, value: &Rational) -> PyResult<Py<PyAny>> {
    let numerator = integer_to_python(py, &value.numerator())?;
    let denominator = integer_to_python(py, &value.denominator())?;
    Ok(fraction_type(py)?.call1((numerator, denominator))?.unbind())
}
