// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of exact complete-intersection theory.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::commalg::{
    ChowClass, ChowRing, CompleteIntersectionHypotheses, HirzebruchRiemannRochResult,
    IntersectionTheoryError, ProjectiveCompleteIntersection, RationalChowClass,
};
use noether::SemanticType;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::validity::PyMathematicalValidity;

fn intersection_error(error: IntersectionTheoryError) -> PyErr {
    match error {
        IntersectionTheoryError::ResourceLimit(_) | IntersectionTheoryError::ExponentOverflow => {
            PyOverflowError::new_err(error.to_string())
        }
        IntersectionTheoryError::ContextMismatch => PyTypeError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn exponent_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    value
        .try_iter()
        .map_err(|_| PyTypeError::new_err("a Chow exponent vector must be iterable"))?
        .map(|entry| {
            entry?.extract::<usize>().map_err(|_| {
                PyValueError::new_err("Chow exponents must be nonnegative machine-sized integers")
            })
        })
        .collect()
}

fn integral_terms(value: &Bound<'_, PyAny>) -> PyResult<Vec<(Vec<usize>, noether::Integer)>> {
    value
        .try_iter()
        .map_err(|_| PyTypeError::new_err("Chow terms must be iterable"))?
        .map(|term| {
            let term = term?;
            Ok((
                exponent_vector(&term.get_item(0)?)?,
                extract_integer(&term.get_item(1)?)?,
            ))
        })
        .collect()
}

fn rational_terms(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Vec<usize>, noether::Rational)>> {
    value
        .try_iter()
        .map_err(|_| PyTypeError::new_err("Chow terms must be iterable"))?
        .map(|term| {
            let term = term?;
            Ok((
                exponent_vector(&term.get_item(0)?)?,
                extract_rational(py, &term.get_item(1)?)?,
            ))
        })
        .collect()
}

/// Explicit hypothesis attached to a complete-intersection type.
#[pyclass(
    module = "noether.commalg",
    name = "CompleteIntersectionHypotheses",
    frozen,
    eq,
    eq_int,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyCompleteIntersectionHypotheses {
    RegularSequence = 0,
    Smooth = 1,
}

impl From<CompleteIntersectionHypotheses> for PyCompleteIntersectionHypotheses {
    fn from(value: CompleteIntersectionHypotheses) -> Self {
        match value {
            CompleteIntersectionHypotheses::RegularSequence => Self::RegularSequence,
            CompleteIntersectionHypotheses::Smooth => Self::Smooth,
        }
    }
}

/// The exact integral Chow ring of a product of projective spaces.
#[pyclass(module = "noether.commalg", name = "ChowRing", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyChowRing {
    inner: ChowRing,
}

#[pymethods]
impl PyChowRing {
    #[new]
    fn new(projective_dimensions: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: ChowRing::new(projective_dimensions).map_err(intersection_error)?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        ChowRing::TYPE_ID
    }

    #[getter]
    fn projective_dimensions(&self) -> Vec<usize> {
        self.inner.projective_dimensions().to_vec()
    }

    #[getter]
    fn factor_count(&self) -> usize {
        self.inner.factor_count()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn zero(&self) -> PyChowClass {
        PyChowClass {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyChowClass {
        PyChowClass {
            inner: self.inner.one(),
        }
    }

    fn generator(&self, factor: usize) -> PyResult<PyChowClass> {
        Ok(PyChowClass {
            inner: self.inner.generator(factor).map_err(intersection_error)?,
        })
    }

    fn generators(&self) -> Vec<PyChowClass> {
        self.inner
            .generators()
            .into_iter()
            .map(|inner| PyChowClass { inner })
            .collect()
    }

    fn from_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyChowClass> {
        let terms = integral_terms(terms)?;
        py.detach(|| {
            Ok(PyChowClass {
                inner: self.inner.from_terms(terms).map_err(intersection_error)?,
            })
        })
    }

    fn from_rational_terms(
        &self,
        py: Python<'_>,
        terms: &Bound<'_, PyAny>,
    ) -> PyResult<PyRationalChowClass> {
        let terms = rational_terms(py, terms)?;
        py.detach(|| {
            Ok(PyRationalChowClass {
                inner: self
                    .inner
                    .from_rational_terms(terms)
                    .map_err(intersection_error)?,
            })
        })
    }

    fn integrate(&self, py: Python<'_>, class: &PyChowClass) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            &self
                .inner
                .integrate(&class.inner)
                .map_err(intersection_error)?,
        )
    }

    fn integrate_rational(
        &self,
        py: Python<'_>,
        class: &PyRationalChowClass,
    ) -> PyResult<Py<PyAny>> {
        rational_to_python(
            py,
            &self
                .inner
                .integrate_rational(&class.inner)
                .map_err(intersection_error)?,
        )
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: ChowRing::from_json(value).map_err(value_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ChowRing(projective_dimensions={:?})",
            self.inner.projective_dimensions()
        ))
    }
}

/// A canonical sparse integral Chow class.
#[pyclass(module = "noether.commalg", name = "ChowClass", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyChowClass {
    inner: ChowClass,
}

#[pymethods]
impl PyChowClass {
    #[classattr]
    fn type_id() -> &'static str {
        ChowClass::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyChowRing {
        PyChowRing {
            inner: self.inner.ring().clone(),
        }
    }

    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(Vec<usize>, Py<PyAny>)>> {
        self.inner
            .terms()
            .iter()
            .map(|(exponent, coefficient)| {
                Ok((exponent.clone(), integer_to_python(py, coefficient)?))
            })
            .collect()
    }

    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    #[getter]
    fn codimension(&self) -> Option<usize> {
        self.inner.codimension()
    }

    fn scalar_mul(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.scalar_mul(&extract_integer(scalar)?),
        })
    }

    fn truncate(&self, maximum: usize) -> Self {
        Self {
            inner: self.inner.truncate(maximum),
        }
    }

    fn over_rationals(&self) -> PyRationalChowClass {
        PyRationalChowClass {
            inner: self.inner.over_rationals(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(intersection_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: ChowClass::from_json(value).map_err(value_error)?,
            })
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.add(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.sub(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn __mul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.mul(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: u64,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "modular exponentiation is not defined for Chow classes",
            ));
        }
        py.detach(|| {
            Ok(Self {
                inner: self.inner.pow(exponent).map_err(intersection_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("ChowClass(terms={:?})", self.inner.terms()))
    }
}

/// A canonical sparse rational Chow class.
#[pyclass(
    module = "noether.commalg",
    name = "RationalChowClass",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRationalChowClass {
    inner: RationalChowClass,
}

#[pymethods]
impl PyRationalChowClass {
    #[classattr]
    fn type_id() -> &'static str {
        RationalChowClass::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyChowRing {
        PyChowRing {
            inner: self.inner.ring().clone(),
        }
    }

    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(Vec<usize>, Py<PyAny>)>> {
        self.inner
            .terms()
            .iter()
            .map(|(exponent, coefficient)| {
                Ok((exponent.clone(), rational_to_python(py, coefficient)?))
            })
            .collect()
    }

    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    #[getter]
    fn codimension(&self) -> Option<usize> {
        self.inner.codimension()
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.scalar_mul(&extract_rational(py, scalar)?),
        })
    }

    fn truncate(&self, maximum: usize) -> Self {
        Self {
            inner: self.inner.truncate(maximum),
        }
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(intersection_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: RationalChowClass::from_json(value).map_err(value_error)?,
            })
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.add(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.sub(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn __mul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.mul(&other.inner).map_err(intersection_error)?,
            })
        })
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: u64,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "modular exponentiation is not defined for Chow classes",
            ));
        }
        py.detach(|| {
            Ok(Self {
                inner: self.inner.pow(exponent).map_err(intersection_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("RationalChowClass(terms={:?})", self.inner.terms()))
    }
}

/// A multidegree complete intersection with explicit geometric hypotheses.
#[pyclass(
    module = "noether.commalg",
    name = "ProjectiveCompleteIntersection",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyProjectiveCompleteIntersection {
    inner: ProjectiveCompleteIntersection,
}

#[pymethods]
impl PyProjectiveCompleteIntersection {
    #[classmethod]
    fn assuming_regular_sequence(
        _cls: &Bound<'_, PyType>,
        ring: &PyChowRing,
        multidegrees: Vec<Vec<u64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ProjectiveCompleteIntersection::assuming_regular_sequence(
                ring.inner.clone(),
                multidegrees,
            )
            .map_err(intersection_error)?,
        })
    }

    #[classmethod]
    /// Construct a projective complete intersection under an explicit smoothness hypothesis.
    fn assuming_smooth(
        _cls: &Bound<'_, PyType>,
        ring: &PyChowRing,
        multidegrees: Vec<Vec<u64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ProjectiveCompleteIntersection::assuming_smooth(
                ring.inner.clone(),
                multidegrees,
            )
            .map_err(intersection_error)?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        ProjectiveCompleteIntersection::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyChowRing {
        PyChowRing {
            inner: self.inner.ring().clone(),
        }
    }

    #[getter]
    fn multidegrees(&self) -> Vec<Vec<u64>> {
        self.inner.multidegrees().to_vec()
    }

    #[getter]
    fn hypotheses(&self) -> PyCompleteIntersectionHypotheses {
        self.inner.hypotheses().into()
    }

    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        PyMathematicalValidity {
            inner: self.inner.validity(),
        }
    }

    #[getter]
    fn codimension(&self) -> usize {
        self.inner.codimension()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn fundamental_class(&self) -> PyResult<PyChowClass> {
        Ok(PyChowClass {
            inner: self.inner.fundamental_class().map_err(intersection_error)?,
        })
    }

    fn pushforward_restriction(&self, class: &PyChowClass) -> PyResult<PyChowClass> {
        Ok(PyChowClass {
            inner: self
                .inner
                .pushforward_restriction(&class.inner)
                .map_err(intersection_error)?,
        })
    }

    fn integrate_restriction(&self, py: Python<'_>, class: &PyChowClass) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            &self
                .inner
                .integrate_restriction(&class.inner)
                .map_err(intersection_error)?,
        )
    }

    fn normal_total_chern_class(&self, py: Python<'_>) -> PyResult<PyChowClass> {
        py.detach(|| {
            Ok(PyChowClass {
                inner: self
                    .inner
                    .normal_total_chern_class()
                    .map_err(intersection_error)?,
            })
        })
    }

    fn tangent_total_chern_class(&self, py: Python<'_>) -> PyResult<PyChowClass> {
        py.detach(|| {
            Ok(PyChowClass {
                inner: self
                    .inner
                    .tangent_total_chern_class()
                    .map_err(intersection_error)?,
            })
        })
    }

    fn todd_class(&self, py: Python<'_>) -> PyResult<PyRationalChowClass> {
        py.detach(|| {
            Ok(PyRationalChowClass {
                inner: self.inner.todd_class().map_err(intersection_error)?,
            })
        })
    }

    fn line_bundle_chern_character(
        &self,
        py: Python<'_>,
        multidegree: Vec<i64>,
    ) -> PyResult<PyRationalChowClass> {
        py.detach(|| {
            Ok(PyRationalChowClass {
                inner: self
                    .inner
                    .line_bundle_chern_character(&multidegree)
                    .map_err(intersection_error)?,
            })
        })
    }

    fn line_bundle_euler_characteristic_koszul(
        &self,
        py: Python<'_>,
        multidegree: Vec<i64>,
    ) -> PyResult<Py<PyAny>> {
        let value = py.detach(|| {
            self.inner
                .line_bundle_euler_characteristic_koszul(multidegree)
                .map_err(intersection_error)
        })?;
        integer_to_python(py, &value)
    }

    fn line_bundle_euler_characteristic(
        &self,
        py: Python<'_>,
        multidegree: Vec<i64>,
    ) -> PyResult<PyHirzebruchRiemannRochResult> {
        py.detach(|| {
            Ok(PyHirzebruchRiemannRochResult {
                inner: self
                    .inner
                    .line_bundle_euler_characteristic(multidegree)
                    .map_err(intersection_error)?,
            })
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(intersection_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: ProjectiveCompleteIntersection::from_json(value).map_err(value_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ProjectiveCompleteIntersection(dimensions={:?}, multidegrees={:?}, hypotheses={:?})",
            self.inner.ring().projective_dimensions(),
            self.inner.multidegrees(),
            self.inner.hypotheses()
        ))
    }
}

/// Exact HRR result with independent Koszul evidence.
#[pyclass(
    module = "noether.commalg",
    name = "HirzebruchRiemannRochResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyHirzebruchRiemannRochResult {
    inner: HirzebruchRiemannRochResult,
}

#[pymethods]
impl PyHirzebruchRiemannRochResult {
    #[classattr]
    fn type_id() -> &'static str {
        HirzebruchRiemannRochResult::TYPE_ID
    }

    #[getter]
    fn complete_intersection(&self) -> PyProjectiveCompleteIntersection {
        PyProjectiveCompleteIntersection {
            inner: self.inner.complete_intersection().clone(),
        }
    }

    #[getter]
    fn line_bundle_multidegree(&self) -> Vec<i64> {
        self.inner.line_bundle_multidegree().to_vec()
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.value())
    }

    #[getter]
    fn hrr_integral(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, self.inner.hrr_integral())
    }

    #[getter]
    fn koszul_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.koszul_value())
    }

    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        PyMathematicalValidity {
            inner: self.inner.validity(),
        }
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(intersection_error))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: HirzebruchRiemannRochResult::from_json(value).map_err(value_error)?,
            })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "HirzebruchRiemannRochResult(multidegree={:?}, value={})",
            self.inner.line_bundle_multidegree(),
            self.inner.value()
        ))
    }
}
