// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed Python projections of the canonical certified-ball contexts.

use noether::{ComplexBall, ComplexBallField, RealBall, RealBallField, Ring, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::extract_rational;

fn context_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_real_operand(
    py: Python<'_>,
    field: RealBallField,
    value: &Bound<'_, PyAny>,
) -> PyResult<RealBall> {
    if let Ok(value) = value.extract::<PyRef<'_, PyRealBall>>() {
        Ok(value.inner.clone())
    } else {
        Ok(field.element(extract_rational(py, value)?))
    }
}

fn extract_complex_operand(
    py: Python<'_>,
    field: ComplexBallField,
    value: &Bound<'_, PyAny>,
) -> PyResult<ComplexBall> {
    if let Ok(value) = value.extract::<PyRef<'_, PyComplexBall>>() {
        Ok(value.inner.clone())
    } else {
        Ok(field.element_parts(extract_rational(py, value)?, 0))
    }
}

/// A fixed-working-precision certified real-ball context.
#[pyclass(module = "noether", name = "RealBallField", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyRealBallField {
    pub(crate) inner: RealBallField,
}

#[pymethods]
impl PyRealBallField {
    /// Construct a real-ball context with precision fixed in advance.
    #[new]
    #[pyo3(signature = (precision))]
    fn new(precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: RealBallField::new(precision).map_err(context_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RealBallField::TYPE_ID
    }

    /// Working precision in bits.
    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision_bits()
    }

    /// Additive identity.
    fn zero(&self) -> PyRealBall {
        PyRealBall {
            inner: self.inner.zero(),
        }
    }

    /// Multiplicative identity.
    fn one(&self) -> PyRealBall {
        PyRealBall {
            inner: self.inner.one(),
        }
    }

    /// Enclose a Python integer or exact fraction.
    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyRealBall> {
        Ok(PyRealBall {
            inner: self.inner.element(extract_rational(py, value)?),
        })
    }

    /// Enclose an exact rational interval.
    fn interval(
        &self,
        py: Python<'_>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<PyRealBall> {
        Ok(PyRealBall {
            inner: self
                .inner
                .interval(extract_rational(py, lower)?, extract_rational(py, upper)?)
                .map_err(context_error)?,
        })
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(context_error)
    }

    /// Restore and validate a canonical context representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RealBallField::from_json(value).map_err(context_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("RealBallField(precision={})", self.precision()))
    }
}

/// A certified real enclosure in one fixed precision context.
#[pyclass(module = "noether", name = "RealBall", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyRealBall {
    pub(crate) inner: RealBall,
}

#[pymethods]
impl PyRealBall {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RealBall::TYPE_ID
    }

    /// Concrete precision context.
    #[getter]
    fn field(&self) -> PyRealBallField {
        PyRealBallField {
            inner: self.inner.field(),
        }
    }

    /// Whether the enclosure has zero radius.
    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    /// Whether zero belongs to the enclosure.
    fn contains_zero(&self) -> bool {
        self.inner.contains_zero()
    }

    /// Whether two same-context enclosures intersect.
    fn overlaps(&self, other: &Self) -> PyResult<bool> {
        self.inner.overlaps(&other.inner).map_err(context_error)
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(context_error)
    }

    /// Restore and validate an exact enclosure representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RealBall::from_json(value).map_err(context_error)?,
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_real_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner + &other).map_err(context_error)?,
        })
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_real_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner - &other).map_err(context_error)?,
        })
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_real_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&other - &self.inner).map_err(context_error)?,
        })
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_real_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner * &other).map_err(context_error)?,
        })
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(context_error)?,
        })
    }

    fn __pow__(&self, exponent: i64, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "RealBall.__pow__ does not accept a modulus",
            ));
        }
        let exponent = u64::try_from(exponent)
            .map_err(|_| PyValueError::new_err("RealBall exponent must be nonnegative"))?;
        Ok(Self {
            inner: self.inner.pow(exponent),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A fixed-working-precision certified complex-ball context.
#[pyclass(module = "noether", name = "ComplexBallField", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyComplexBallField {
    pub(crate) inner: ComplexBallField,
}

#[pymethods]
impl PyComplexBallField {
    /// Construct a complex-ball context with precision fixed in advance.
    #[new]
    #[pyo3(signature = (precision))]
    fn new(precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: ComplexBallField::new(precision).map_err(context_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        ComplexBallField::TYPE_ID
    }

    /// Working precision in bits.
    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision_bits()
    }

    /// Additive identity.
    fn zero(&self) -> PyComplexBall {
        PyComplexBall {
            inner: self.inner.zero(),
        }
    }

    /// Multiplicative identity.
    fn one(&self) -> PyComplexBall {
        PyComplexBall {
            inner: self.inner.one(),
        }
    }

    /// Enclose exact rational real and optional imaginary parts.
    #[pyo3(signature = (real, imaginary=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        real: &Bound<'_, PyAny>,
        imaginary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyComplexBall> {
        let real = extract_rational(py, real)?;
        let imaginary = match imaginary {
            Some(value) => extract_rational(py, value)?,
            None => 0.into(),
        };
        Ok(PyComplexBall {
            inner: self.inner.element_parts(real, imaginary),
        })
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(context_error)
    }

    /// Restore and validate a canonical context representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ComplexBallField::from_json(value).map_err(context_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!("ComplexBallField(precision={})", self.precision())
        })
    }
}

/// A certified complex enclosure in one fixed precision context.
#[pyclass(module = "noether", name = "ComplexBall", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyComplexBall {
    pub(crate) inner: ComplexBall,
}

#[pymethods]
impl PyComplexBall {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        ComplexBall::TYPE_ID
    }

    /// Concrete precision context.
    #[getter]
    fn field(&self) -> PyComplexBallField {
        PyComplexBallField {
            inner: self.inner.field(),
        }
    }

    /// Certified real component.
    #[getter]
    fn real(&self) -> PyRealBall {
        PyRealBall {
            inner: self.inner.real(),
        }
    }

    /// Certified imaginary component.
    #[getter]
    fn imaginary(&self) -> PyRealBall {
        PyRealBall {
            inner: self.inner.imaginary(),
        }
    }

    /// Whether both components have zero radius.
    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    /// Whether the point zero belongs to the enclosure.
    fn contains_zero(&self) -> bool {
        self.inner.contains_zero()
    }

    /// Whether two same-context enclosures intersect.
    fn overlaps(&self, other: &Self) -> PyResult<bool> {
        self.inner.overlaps(&other.inner).map_err(context_error)
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(context_error)
    }

    /// Restore and validate an exact enclosure representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ComplexBall::from_json(value).map_err(context_error)?,
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_complex_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner + &other).map_err(context_error)?,
        })
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_complex_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner - &other).map_err(context_error)?,
        })
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_complex_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&other - &self.inner).map_err(context_error)?,
        })
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_complex_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: (&self.inner * &other).map_err(context_error)?,
        })
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(context_error)?,
        })
    }

    fn __pow__(&self, exponent: i64, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "ComplexBall.__pow__ does not accept a modulus",
            ));
        }
        let exponent = u64::try_from(exponent)
            .map_err(|_| PyValueError::new_err("ComplexBall exponent must be nonnegative"))?;
        Ok(Self {
            inner: self.inner.pow(exponent),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
