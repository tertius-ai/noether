// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of generic fixed-truncation power series.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    Degree, FreeAbelianGrading, HilbertSeries, PositiveGrading, PowerSeries, PowerSeriesError,
    PowerSeriesRing, ResidueClass, ResidueRing, SeriesTruncation, SeriesTruncationMode,
    WeightedGrading,
};
use noether::numthy::{
    FiniteField, FiniteFieldElement, PadicElement, PadicField, PadicLocalField,
    PadicLocalFieldElement, PadicRing, QadicElement, QadicField, QadicRing,
};
use noether::{
    Integer, IntegerRing, RationalField, Ring, SemanticType, UnitError, UnsupportedOperationError,
    Q, Z,
};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyList, PyString, PyTuple, PyType};

use super::commalg::{PyResidueClass, PyResidueRing};
use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::linalg::{PyMatrix, PyVector};
use super::numthy::{
    PyFiniteField, PyFiniteFieldElement, PyPadicElement, PyPadicField, PyPadicFieldFrobenius,
    PyPadicLocalField, PyPadicLocalFieldCoefficientFrobenius, PyPadicLocalFieldElement,
    PyPadicRing, PyPadicRingFrobenius, PyQadicElement, PyQadicField, PyQadicFieldFrobenius,
    PyQadicRing, PyQadicRingFrobenius,
};
use super::polynomial_facade::PyMonomialMap;
use super::root::{PyIntegerRing, PyRationalField};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn unit_error(error: UnitError) -> PyErr {
    match error {
        UnitError::ContextMismatch(_) => value_error(error),
        UnitError::DivisionByZero(_) | UnitError::NotAUnit { .. } => {
            PyZeroDivisionError::new_err(error.to_string())
        }
    }
}

fn series_error(error: PowerSeriesError) -> PyErr {
    match error {
        PowerSeriesError::DivisionByZero => PyZeroDivisionError::new_err(error.to_string()),
        _ => value_error(error),
    }
}

fn unsupported_series_operation(operation: &'static str, context: &'static str) -> PyErr {
    super::errors::unsupported_operation(
        UnsupportedOperationError::new(
            format!("noether/commalg/PowerSeries/{operation}@1"),
            context,
        )
        .with_supported_contexts([
            PadicRing::TYPE_ID,
            PadicField::TYPE_ID,
            QadicRing::TYPE_ID,
            QadicField::TYPE_ID,
        ]),
    )
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

fn coefficient_type_from_json(value: &str, series: bool) -> PyResult<String> {
    let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
    let ring = if series {
        payload.get("ring")
    } else {
        Some(&payload)
    }
    .and_then(|ring| ring.get("coefficient_ring"))
    .ok_or_else(|| value_error("serialized power-series context has no coefficient ring"))?;
    ring.get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| value_error("serialized coefficient ring has no string type identity"))
}

fn extract_residue(ring: &ResidueRing, value: &Bound<'_, PyAny>) -> PyResult<ResidueClass> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyResidueClass>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, ring
        )));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or ResidueClass",
    ))
}

fn extract_finite_field_element(
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
        return Err(value_error(format!(
            "coefficient belongs to {}, expected {}",
            element.inner.field(),
            field
        )));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or FiniteFieldElement",
    ))
}

fn extract_padic(ring: &PadicRing, value: &Bound<'_, PyAny>) -> PyResult<PadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicElement>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, ring
        )));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or PadicElement",
    ))
}

fn extract_padic_field(field: &PadicField, value: &Bound<'_, PyAny>) -> PyResult<PadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(field.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error("p-adic coefficient belongs to another context"));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or PadicElement",
    ))
}

fn extract_qadic(ring: &QadicRing, value: &Bound<'_, PyAny>) -> PyResult<QadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyQadicElement>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error("q-adic coefficient belongs to another context"));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or QadicElement",
    ))
}

fn extract_qadic_field(field: &QadicField, value: &Bound<'_, PyAny>) -> PyResult<QadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(field.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyQadicElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error("q-adic coefficient belongs to another context"));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or QadicElement",
    ))
}

fn extract_padic_local_field(
    field: &PadicLocalField,
    value: &Bound<'_, PyAny>,
) -> PyResult<PadicLocalFieldElement> {
    if value.is_instance_of::<PyInt>() {
        let scalar = field
            .coefficient_field()
            .coefficient_field()
            .element(extract_integer(value)?);
        return field.element_from_padic(&scalar).map_err(value_error);
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicLocalFieldElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(
            "p-adic local coefficient belongs to another context",
        ));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or PadicLocalFieldElement",
    ))
}

fn collect_terms<E>(
    terms: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<(Vec<u64>, E)>> {
    terms
        .try_iter()?
        .map(|item| {
            let item = item?;
            let pair = item.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(
                    "each power-series term must be an (exponent, coefficient) tuple",
                )
            })?;
            if pair.len() != 2 {
                return Err(PyTypeError::new_err(
                    "each power-series term must contain exactly two items",
                ));
            }
            let exponent = pair.get_item(0)?.extract::<Vec<u64>>()?;
            let coefficient = extract(&pair.get_item(1)?)?;
            Ok((exponent, coefficient))
        })
        .collect()
}

/// One arbitrary-precision degree in a free abelian grading group.
#[pyclass(module = "noether.commalg", name = "Degree", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDegree {
    pub(crate) inner: Degree,
}

#[pymethods]
impl PyDegree {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Degree::TYPE_ID
    }

    #[new]
    fn new(coordinates: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Degree::new(
                coordinates
                    .try_iter()?
                    .map(|value| extract_integer(&value?))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        })
    }

    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>, rank: usize) -> Self {
        Self {
            inner: Degree::zero(rank),
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn coordinates(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .coordinates()
            .iter()
            .map(|value| integer_to_python(py, value))
            .collect()
    }

    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.add(&rhs.inner).map_err(value_error)?,
        })
    }

    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.sub(&rhs.inner).map_err(value_error)?,
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Degree::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("Degree{}", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Degrees of ordered variables in one free abelian group `Z^r`.
#[pyclass(
    module = "noether.commalg",
    name = "FreeAbelianGrading",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFreeAbelianGrading {
    pub(crate) inner: FreeAbelianGrading,
}

#[pymethods]
impl PyFreeAbelianGrading {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FreeAbelianGrading::TYPE_ID
    }

    #[new]
    fn new(rank: usize, variable_degrees: Vec<PyDegree>) -> PyResult<Self> {
        Ok(Self {
            inner: FreeAbelianGrading::new(
                rank,
                variable_degrees
                    .into_iter()
                    .map(|degree| degree.inner)
                    .collect(),
            )
            .map_err(value_error)?,
        })
    }

    #[classmethod]
    fn standard(_cls: &Bound<'_, PyType>, arity: usize) -> Self {
        Self {
            inner: FreeAbelianGrading::standard(arity),
        }
    }

    #[classmethod]
    fn trivial(_cls: &Bound<'_, PyType>, arity: usize) -> Self {
        Self {
            inner: FreeAbelianGrading::trivial(arity),
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    #[getter]
    fn variable_degrees(&self) -> Vec<PyDegree> {
        self.inner
            .variable_degrees()
            .iter()
            .cloned()
            .map(|inner| PyDegree { inner })
            .collect()
    }

    fn degree(&self, exponents: &Bound<'_, PyAny>) -> PyResult<PyDegree> {
        let exponents = exponents
            .try_iter()?
            .map(|value| extract_integer(&value?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyDegree {
            inner: self.inner.degree(&exponents).map_err(value_error)?,
        })
    }

    fn weighted_view(&self) -> Option<PyWeightedGrading> {
        self.inner
            .weighted_view()
            .map(|inner| PyWeightedGrading { inner })
    }

    fn certify_positive(&self, functional: &PyDegree) -> PyResult<PyPositiveGrading> {
        Ok(PyPositiveGrading {
            inner: self
                .inner
                .certify_positive(functional.inner.clone())
                .map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FreeAbelianGrading::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FreeAbelianGrading(rank={}, variable_degrees={:?})",
            self.inner.rank(),
            self.inner.variable_degrees()
        ))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Exact Hilbert series in a free-abelian grading group algebra.
#[pyclass(
    module = "noether.commalg",
    name = "HilbertSeries",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyHilbertSeries {
    pub(crate) inner: HilbertSeries,
}

#[pymethods]
impl PyHilbertSeries {
    #[classattr]
    fn type_id() -> &'static str {
        HilbertSeries::TYPE_ID
    }

    #[new]
    fn new(grading: &PyFreeAbelianGrading, numerator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut terms = Vec::new();
        for term in numerator.try_iter()? {
            let term = term?;
            let term = term.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err("a Hilbert numerator term must be (Degree, coefficient)")
            })?;
            if term.len() != 2 {
                return Err(PyTypeError::new_err(
                    "a Hilbert numerator term must be (Degree, coefficient)",
                ));
            }
            let degree = term
                .get_item(0)?
                .extract::<PyRef<'_, PyDegree>>()?
                .inner
                .clone();
            let coefficient = extract_integer(&term.get_item(1)?)?;
            terms.push((degree, coefficient));
        }
        Ok(Self {
            inner: HilbertSeries::new(grading.inner.clone(), terms).map_err(value_error)?,
        })
    }

    #[getter]
    fn grading(&self) -> PyFreeAbelianGrading {
        PyFreeAbelianGrading {
            inner: self.inner.grading().clone(),
        }
    }

    #[getter]
    fn numerator(&self, py: Python<'_>) -> PyResult<Vec<(PyDegree, Py<PyAny>)>> {
        self.inner
            .numerator()
            .iter()
            .map(|(degree, coefficient)| {
                Ok((
                    PyDegree {
                        inner: degree.clone(),
                    },
                    integer_to_python(py, coefficient)?,
                ))
            })
            .collect()
    }

    fn standard_numerator(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .standard_numerator()
            .map_err(value_error)?
            .iter()
            .map(|coefficient| integer_to_python(py, coefficient))
            .collect()
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
            inner: HilbertSeries::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "HilbertSeries(grading={}, numerator_terms={})",
            self.inner.grading(),
            self.inner.numerator().len()
        ))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Exact evidence that a free-abelian grading has finite graded pieces.
#[pyclass(
    module = "noether.commalg",
    name = "PositiveGrading",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPositiveGrading {
    pub(crate) inner: PositiveGrading,
}

#[pymethods]
impl PyPositiveGrading {
    #[classattr]
    fn type_id() -> &'static str {
        PositiveGrading::TYPE_ID
    }

    #[new]
    fn new(grading: &PyFreeAbelianGrading, functional: &PyDegree) -> PyResult<Self> {
        Ok(Self {
            inner: PositiveGrading::new(grading.inner.clone(), functional.inner.clone())
                .map_err(value_error)?,
        })
    }

    #[getter]
    fn grading(&self) -> PyFreeAbelianGrading {
        PyFreeAbelianGrading {
            inner: self.inner.grading().clone(),
        }
    }

    #[getter]
    fn functional(&self) -> PyDegree {
        PyDegree {
            inner: self.inner.functional().clone(),
        }
    }

    #[getter]
    fn weights(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .weights()
            .iter()
            .map(|value| integer_to_python(py, value))
            .collect()
    }

    fn project(&self, py: Python<'_>, degree: &PyDegree) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.project(&degree.inner).map_err(value_error)?)
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
            inner: PositiveGrading::from_json(value).map_err(value_error)?,
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

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A positive integer weight for each generator in an ordered context.
#[pyclass(
    module = "noether.commalg",
    name = "WeightedGrading",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyWeightedGrading {
    pub(crate) inner: WeightedGrading,
}

#[pymethods]
impl PyWeightedGrading {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        WeightedGrading::TYPE_ID
    }

    #[new]
    fn new(weights: Vec<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: WeightedGrading::new(weights).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn standard(_cls: &Bound<'_, PyType>, arity: usize) -> PyResult<Self> {
        Ok(Self {
            inner: WeightedGrading::standard(arity).map_err(value_error)?,
        })
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    #[getter]
    fn weights(&self) -> Vec<u64> {
        self.inner.weights().to_vec()
    }

    fn is_standard(&self) -> bool {
        self.inner.is_standard()
    }

    fn degree(&self, exponent: Vec<u64>) -> PyResult<u64> {
        self.inner.degree(&exponent).map_err(value_error)
    }

    fn as_free_abelian(&self) -> PyFreeAbelianGrading {
        PyFreeAbelianGrading {
            inner: self.inner.as_free_abelian(),
        }
    }

    /// Serialize the positive weights as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a positive grading.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WeightedGrading::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("WeightedGrading({:?})", self.inner.weights()))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// One valid finite monomial region, fixed before series allocation.
#[pyclass(
    module = "noether.commalg",
    name = "SeriesTruncation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySeriesTruncation {
    pub(crate) inner: SeriesTruncation,
}

#[pymethods]
impl PySeriesTruncation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        SeriesTruncation::TYPE_ID
    }

    #[classmethod]
    fn univariate(_cls: &Bound<'_, PyType>, precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::univariate(precision).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn total_degree(_cls: &Bound<'_, PyType>, bound: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::total_degree(bound).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn box_bounds(_cls: &Bound<'_, PyType>, bounds: Vec<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::box_bounds(bounds).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn weighted_degree(_cls: &Bound<'_, PyType>, weights: Vec<u64>, bound: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::weighted_degree(weights, bound).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn graded(_cls: &Bound<'_, PyType>, grading: &PyWeightedGrading, bound: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::graded(grading.inner.clone(), bound).map_err(value_error)?,
        })
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.inner.mode() {
            SeriesTruncationMode::Univariate => "univariate",
            SeriesTruncationMode::TotalDegree => "total_degree",
            SeriesTruncationMode::Box => "box",
            SeriesTruncationMode::WeightedDegree => "weighted_degree",
        }
    }

    #[getter]
    fn bound(&self) -> Option<u64> {
        self.inner.bound()
    }

    #[getter]
    fn bounds(&self) -> Option<Vec<u64>> {
        self.inner.bounds().map(<[u64]>::to_vec)
    }

    #[getter]
    fn weights(&self) -> Option<Vec<u64>> {
        self.inner.weights().map(<[u64]>::to_vec)
    }

    fn contains_exponent(&self, exponent: Vec<u64>) -> bool {
        self.inner.contains_exponent(&exponent)
    }

    fn refines(&self, reference: &Self) -> bool {
        self.inner.refines(&reference.inner)
    }

    /// Serialize the exact finite monomial region as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a finite monomial region.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesTruncation::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match self.inner.mode() {
            SeriesTruncationMode::Univariate => format!(
                "SeriesTruncation.univariate({})",
                self.inner.bound().unwrap_or_default()
            ),
            SeriesTruncationMode::TotalDegree => format!(
                "SeriesTruncation.total_degree({})",
                self.inner.bound().unwrap_or_default()
            ),
            SeriesTruncationMode::Box => {
                format!(
                    "SeriesTruncation.box_bounds({:?})",
                    self.inner.bounds().unwrap_or_default()
                )
            }
            SeriesTruncationMode::WeightedDegree => format!(
                "SeriesTruncation.weighted_degree({:?}, {})",
                self.inner.weights().unwrap_or_default(),
                self.inner.bound().unwrap_or_default()
            ),
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

#[derive(Clone)]
pub(crate) enum PowerSeriesRingInner {
    Integer(PowerSeriesRing<IntegerRing>),
    Rational(PowerSeriesRing<RationalField>),
    Residue(PowerSeriesRing<ResidueRing>),
    FiniteField(PowerSeriesRing<FiniteField>),
    Padic(PowerSeriesRing<PadicRing>),
    PadicField(PowerSeriesRing<PadicField>),
    Qadic(PowerSeriesRing<QadicRing>),
    QadicField(PowerSeriesRing<QadicField>),
    PadicLocalField(PowerSeriesRing<PadicLocalField>),
}

impl PowerSeriesRingInner {
    fn variables(&self) -> &[String] {
        match self {
            Self::Integer(ring) => ring.variables(),
            Self::Rational(ring) => ring.variables(),
            Self::Residue(ring) => ring.variables(),
            Self::FiniteField(ring) => ring.variables(),
            Self::Padic(ring) => ring.variables(),
            Self::PadicField(ring) => ring.variables(),
            Self::Qadic(ring) => ring.variables(),
            Self::QadicField(ring) => ring.variables(),
            Self::PadicLocalField(ring) => ring.variables(),
        }
    }

    fn truncation(&self) -> &SeriesTruncation {
        match self {
            Self::Integer(ring) => ring.truncation(),
            Self::Rational(ring) => ring.truncation(),
            Self::Residue(ring) => ring.truncation(),
            Self::FiniteField(ring) => ring.truncation(),
            Self::Padic(ring) => ring.truncation(),
            Self::PadicField(ring) => ring.truncation(),
            Self::Qadic(ring) => ring.truncation(),
            Self::QadicField(ring) => ring.truncation(),
            Self::PadicLocalField(ring) => ring.truncation(),
        }
    }

    fn grading(&self) -> Option<WeightedGrading> {
        match self {
            Self::Integer(ring) => ring.grading(),
            Self::Rational(ring) => ring.grading(),
            Self::Residue(ring) => ring.grading(),
            Self::FiniteField(ring) => ring.grading(),
            Self::Padic(ring) => ring.grading(),
            Self::PadicField(ring) => ring.grading(),
            Self::Qadic(ring) => ring.grading(),
            Self::QadicField(ring) => ring.grading(),
            Self::PadicLocalField(ring) => ring.grading(),
        }
    }

    fn coefficient_capacity_bound(&self) -> Integer {
        match self {
            Self::Integer(ring) => ring.coefficient_capacity_bound(),
            Self::Rational(ring) => ring.coefficient_capacity_bound(),
            Self::Residue(ring) => ring.coefficient_capacity_bound(),
            Self::FiniteField(ring) => ring.coefficient_capacity_bound(),
            Self::Padic(ring) => ring.coefficient_capacity_bound(),
            Self::PadicField(ring) => ring.coefficient_capacity_bound(),
            Self::Qadic(ring) => ring.coefficient_capacity_bound(),
            Self::QadicField(ring) => ring.coefficient_capacity_bound(),
            Self::PadicLocalField(ring) => ring.coefficient_capacity_bound(),
        }
    }

    fn canonical_hash(&self) -> isize {
        match self {
            Self::Integer(ring) => canonical_hash(ring),
            Self::Rational(ring) => canonical_hash(ring),
            Self::Residue(ring) => canonical_hash(ring),
            Self::FiniteField(ring) => canonical_hash(ring),
            Self::Padic(ring) => canonical_hash(ring),
            Self::PadicField(ring) => canonical_hash(ring),
            Self::Qadic(ring) => canonical_hash(ring),
            Self::QadicField(ring) => canonical_hash(ring),
            Self::PadicLocalField(ring) => canonical_hash(ring),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Integer(ring) => ring.to_json(),
            Self::Rational(ring) => ring.to_json(),
            Self::Residue(ring) => ring.to_json(),
            Self::FiniteField(ring) => ring.to_json(),
            Self::Padic(ring) => ring.to_json(),
            Self::PadicField(ring) => ring.to_json(),
            Self::Qadic(ring) => ring.to_json(),
            Self::QadicField(ring) => ring.to_json(),
            Self::PadicLocalField(ring) => ring.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        match coefficient_type_from_json(value, false)?.as_str() {
            IntegerRing::TYPE_ID => Ok(Self::Integer(
                PowerSeriesRing::<IntegerRing>::from_json(value).map_err(value_error)?,
            )),
            RationalField::TYPE_ID => Ok(Self::Rational(
                PowerSeriesRing::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            ResidueRing::TYPE_ID => Ok(Self::Residue(
                PowerSeriesRing::<ResidueRing>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::FiniteField(
                PowerSeriesRing::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            PadicRing::TYPE_ID => Ok(Self::Padic(
                PowerSeriesRing::<PadicRing>::from_json(value).map_err(value_error)?,
            )),
            PadicField::TYPE_ID => Ok(Self::PadicField(
                PowerSeriesRing::<PadicField>::from_json(value).map_err(value_error)?,
            )),
            QadicRing::TYPE_ID => Ok(Self::Qadic(
                PowerSeriesRing::<QadicRing>::from_json(value).map_err(value_error)?,
            )),
            QadicField::TYPE_ID => Ok(Self::QadicField(
                PowerSeriesRing::<QadicField>::from_json(value).map_err(value_error)?,
            )),
            PadicLocalField::TYPE_ID => Ok(Self::PadicLocalField(
                PowerSeriesRing::<PadicLocalField>::from_json(value).map_err(value_error)?,
            )),
            type_id => Err(value_error(format!(
                "unsupported power-series coefficient context {type_id:?}"
            ))),
        }
    }
}

impl PartialEq for PowerSeriesRingInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::Residue(left), Self::Residue(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            (Self::Padic(left), Self::Padic(right)) => left == right,
            (Self::PadicField(left), Self::PadicField(right)) => left == right,
            (Self::Qadic(left), Self::Qadic(right)) => left == right,
            (Self::QadicField(left), Self::QadicField(right)) => left == right,
            (Self::PadicLocalField(left), Self::PadicLocalField(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for PowerSeriesRingInner {}

/// A fixed-truncation power-series ring over one accepted coefficient context.
#[pyclass(
    module = "noether.commalg",
    name = "PowerSeriesRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPowerSeriesRing {
    pub(crate) inner: PowerSeriesRingInner,
}

impl PyPowerSeriesRing {
    fn coefficient_ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PowerSeriesRingInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            PowerSeriesRingInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            PowerSeriesRingInner::Residue(ring) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::FiniteField(ring) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::Padic(ring) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::PadicField(ring) => Ok(Py::new(
                py,
                PyPadicField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::Qadic(ring) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::QadicField(ring) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PowerSeriesRingInner::PadicLocalField(ring) => Ok(Py::new(
                py,
                PyPadicLocalField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
        }
    }

    fn resolve_variable(&self, variable: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(index) = variable.extract::<usize>() {
            if index < self.inner.variables().len() {
                return Ok(index);
            }
            return Err(PyValueError::new_err(format!(
                "series variable index {index} is outside arity {}",
                self.inner.variables().len()
            )));
        }
        if variable.is_instance_of::<PyString>() {
            let name = variable.extract::<String>()?;
            return self
                .inner
                .variables()
                .iter()
                .position(|candidate| candidate == &name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown series variable {name:?}")));
        }
        Err(PyTypeError::new_err(
            "series variable must be a nonnegative index or declared name",
        ))
    }

    fn build_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyPowerSeries> {
        let inner = match &self.inner {
            PowerSeriesRingInner::Integer(ring) => SeriesInner::Integer(
                ring.from_terms(collect_terms(terms, extract_integer)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Rational(ring) => SeriesInner::Rational(
                ring.from_terms(collect_terms(terms, |value| extract_rational(py, value))?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Residue(ring) => SeriesInner::Residue(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_residue(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::FiniteField(ring) => SeriesInner::FiniteField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_finite_field_element(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Padic(ring) => SeriesInner::Padic(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::PadicField(ring) => SeriesInner::PadicField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Qadic(ring) => SeriesInner::Qadic(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_qadic(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::QadicField(ring) => SeriesInner::QadicField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_qadic_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::PadicLocalField(ring) => SeriesInner::PadicLocalField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic_local_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
        };
        Ok(PyPowerSeries { inner })
    }

    fn constant_from_python(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyPowerSeries> {
        let inner = match &self.inner {
            PowerSeriesRingInner::Integer(ring) => SeriesInner::Integer(
                ring.constant(extract_integer(value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Rational(ring) => SeriesInner::Rational(
                ring.constant(extract_rational(py, value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Residue(ring) => SeriesInner::Residue(
                ring.constant(extract_residue(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::FiniteField(ring) => SeriesInner::FiniteField(
                ring.constant(extract_finite_field_element(
                    ring.coefficient_ring(),
                    value,
                )?)
                .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Padic(ring) => SeriesInner::Padic(
                ring.constant(extract_padic(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::PadicField(ring) => SeriesInner::PadicField(
                ring.constant(extract_padic_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::Qadic(ring) => SeriesInner::Qadic(
                ring.constant(extract_qadic(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::QadicField(ring) => SeriesInner::QadicField(
                ring.constant(extract_qadic_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PowerSeriesRingInner::PadicLocalField(ring) => SeriesInner::PadicLocalField(
                ring.constant(extract_padic_local_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
        };
        Ok(PyPowerSeries { inner })
    }
}

#[pymethods]
impl PyPowerSeriesRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PowerSeriesRing::<IntegerRing>::TYPE_ID
    }

    #[new]
    fn new(
        coefficient_ring: &Bound<'_, PyAny>,
        variables: Vec<String>,
        truncation: PySeriesTruncation,
    ) -> PyResult<Self> {
        let inner = if coefficient_ring.is_instance_of::<PyIntegerRing>() {
            PowerSeriesRingInner::Integer(
                PowerSeriesRing::new(Z, variables, truncation.inner).map_err(value_error)?,
            )
        } else if coefficient_ring.is_instance_of::<PyRationalField>() {
            PowerSeriesRingInner::Rational(
                PowerSeriesRing::new(Q, variables, truncation.inner).map_err(value_error)?,
            )
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyResidueRing>>() {
            PowerSeriesRingInner::Residue(
                PowerSeriesRing::new(ring.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyFiniteField>>() {
            PowerSeriesRingInner::FiniteField(
                PowerSeriesRing::new(field.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyPadicRing>>() {
            PowerSeriesRingInner::Padic(
                PowerSeriesRing::new(ring.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyPadicField>>() {
            PowerSeriesRingInner::PadicField(
                PowerSeriesRing::new(field.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyQadicRing>>() {
            PowerSeriesRingInner::Qadic(
                PowerSeriesRing::new(ring.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyQadicField>>() {
            PowerSeriesRingInner::QadicField(
                PowerSeriesRing::new(field.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            PowerSeriesRingInner::PadicLocalField(
                PowerSeriesRing::new(field.inner.clone(), variables, truncation.inner)
                    .map_err(value_error)?,
            )
        } else {
            return Err(PyTypeError::new_err(
                "PowerSeriesRing supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, and PadicLocalField coefficient contexts",
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
        self.coefficient_ring_to_python(py)
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.inner.variables().to_vec()
    }

    #[getter]
    fn truncation(&self) -> PySeriesTruncation {
        PySeriesTruncation {
            inner: self.inner.truncation().clone(),
        }
    }

    #[getter]
    fn grading(&self) -> Option<PyWeightedGrading> {
        self.inner
            .grading()
            .map(|inner| PyWeightedGrading { inner })
    }

    fn coefficient_capacity_bound(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.coefficient_capacity_bound())
    }

    fn zero(&self) -> PyPowerSeries {
        let inner = match &self.inner {
            PowerSeriesRingInner::Integer(ring) => SeriesInner::Integer(ring.zero()),
            PowerSeriesRingInner::Rational(ring) => SeriesInner::Rational(ring.zero()),
            PowerSeriesRingInner::Residue(ring) => SeriesInner::Residue(ring.zero()),
            PowerSeriesRingInner::FiniteField(ring) => SeriesInner::FiniteField(ring.zero()),
            PowerSeriesRingInner::Padic(ring) => SeriesInner::Padic(ring.zero()),
            PowerSeriesRingInner::PadicField(ring) => SeriesInner::PadicField(ring.zero()),
            PowerSeriesRingInner::Qadic(ring) => SeriesInner::Qadic(ring.zero()),
            PowerSeriesRingInner::QadicField(ring) => SeriesInner::QadicField(ring.zero()),
            PowerSeriesRingInner::PadicLocalField(ring) => {
                SeriesInner::PadicLocalField(ring.zero())
            }
        };
        PyPowerSeries { inner }
    }

    fn one(&self) -> PyPowerSeries {
        let inner = match &self.inner {
            PowerSeriesRingInner::Integer(ring) => SeriesInner::Integer(ring.one()),
            PowerSeriesRingInner::Rational(ring) => SeriesInner::Rational(ring.one()),
            PowerSeriesRingInner::Residue(ring) => SeriesInner::Residue(ring.one()),
            PowerSeriesRingInner::FiniteField(ring) => SeriesInner::FiniteField(ring.one()),
            PowerSeriesRingInner::Padic(ring) => SeriesInner::Padic(ring.one()),
            PowerSeriesRingInner::PadicField(ring) => SeriesInner::PadicField(ring.one()),
            PowerSeriesRingInner::Qadic(ring) => SeriesInner::Qadic(ring.one()),
            PowerSeriesRingInner::QadicField(ring) => SeriesInner::QadicField(ring.one()),
            PowerSeriesRingInner::PadicLocalField(ring) => SeriesInner::PadicLocalField(ring.one()),
        };
        PyPowerSeries { inner }
    }

    fn constant(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyPowerSeries> {
        self.constant_from_python(py, value)
    }

    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyPowerSeries> {
        self.constant_from_python(py, value)
    }

    fn from_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyPowerSeries> {
        self.build_terms(py, terms)
    }

    fn generator(&self, variable: &Bound<'_, PyAny>) -> PyResult<PyPowerSeries> {
        let variable = self.resolve_variable(variable)?;
        let inner = match &self.inner {
            PowerSeriesRingInner::Integer(ring) => {
                SeriesInner::Integer(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::Rational(ring) => {
                SeriesInner::Rational(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::Residue(ring) => {
                SeriesInner::Residue(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::FiniteField(ring) => {
                SeriesInner::FiniteField(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::Padic(ring) => {
                SeriesInner::Padic(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::PadicField(ring) => {
                SeriesInner::PadicField(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::Qadic(ring) => {
                SeriesInner::Qadic(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::QadicField(ring) => {
                SeriesInner::QadicField(ring.generator(variable).map_err(value_error)?)
            }
            PowerSeriesRingInner::PadicLocalField(ring) => {
                SeriesInner::PadicLocalField(ring.generator(variable).map_err(value_error)?)
            }
        };
        Ok(PyPowerSeries { inner })
    }

    fn generators(&self) -> PyResult<Vec<PyPowerSeries>> {
        let mut result = Vec::with_capacity(self.inner.variables().len());
        for variable in 0..self.inner.variables().len() {
            let inner = match &self.inner {
                PowerSeriesRingInner::Integer(ring) => {
                    SeriesInner::Integer(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::Rational(ring) => {
                    SeriesInner::Rational(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::Residue(ring) => {
                    SeriesInner::Residue(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::FiniteField(ring) => {
                    SeriesInner::FiniteField(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::Padic(ring) => {
                    SeriesInner::Padic(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::PadicField(ring) => {
                    SeriesInner::PadicField(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::Qadic(ring) => {
                    SeriesInner::Qadic(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::QadicField(ring) => {
                    SeriesInner::QadicField(ring.generator(variable).map_err(value_error)?)
                }
                PowerSeriesRingInner::PadicLocalField(ring) => {
                    SeriesInner::PadicLocalField(ring.generator(variable).map_err(value_error)?)
                }
            };
            result.push(PyPowerSeries { inner });
        }
        Ok(result)
    }

    /// Serialize the complete coefficient and fixed-precision context.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a complete fixed-precision context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PowerSeriesRingInner::from_json(value)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PowerSeriesRing(variables={:?}, truncation={})",
                self.inner.variables(),
                self.inner.truncation()
            )
        })
    }

    fn __str__(&self) -> String {
        match &self.inner {
            PowerSeriesRingInner::Integer(ring) => ring.to_string(),
            PowerSeriesRingInner::Rational(ring) => ring.to_string(),
            PowerSeriesRingInner::Residue(ring) => ring.to_string(),
            PowerSeriesRingInner::FiniteField(ring) => ring.to_string(),
            PowerSeriesRingInner::Padic(ring) => ring.to_string(),
            PowerSeriesRingInner::PadicField(ring) => ring.to_string(),
            PowerSeriesRingInner::Qadic(ring) => ring.to_string(),
            PowerSeriesRingInner::QadicField(ring) => ring.to_string(),
            PowerSeriesRingInner::PadicLocalField(ring) => ring.to_string(),
        }
    }
}

#[derive(Clone)]
pub(crate) enum SeriesInner {
    Integer(PowerSeries<IntegerRing>),
    Rational(PowerSeries<RationalField>),
    Residue(PowerSeries<ResidueRing>),
    FiniteField(PowerSeries<FiniteField>),
    Padic(PowerSeries<PadicRing>),
    PadicField(PowerSeries<PadicField>),
    Qadic(PowerSeries<QadicRing>),
    QadicField(PowerSeries<QadicField>),
    PadicLocalField(PowerSeries<PadicLocalField>),
}

impl SeriesInner {
    fn context_type_id(&self) -> &'static str {
        match self {
            Self::Integer(_) => IntegerRing::TYPE_ID,
            Self::Rational(_) => RationalField::TYPE_ID,
            Self::Residue(_) => ResidueRing::TYPE_ID,
            Self::FiniteField(_) => FiniteField::TYPE_ID,
            Self::Padic(_) => PadicRing::TYPE_ID,
            Self::PadicField(_) => PadicField::TYPE_ID,
            Self::Qadic(_) => QadicRing::TYPE_ID,
            Self::QadicField(_) => QadicField::TYPE_ID,
            Self::PadicLocalField(_) => PadicLocalField::TYPE_ID,
        }
    }

    fn ring(&self) -> PowerSeriesRingInner {
        match self {
            Self::Integer(value) => PowerSeriesRingInner::Integer(value.ring().clone()),
            Self::Rational(value) => PowerSeriesRingInner::Rational(value.ring().clone()),
            Self::Residue(value) => PowerSeriesRingInner::Residue(value.ring().clone()),
            Self::FiniteField(value) => PowerSeriesRingInner::FiniteField(value.ring().clone()),
            Self::Padic(value) => PowerSeriesRingInner::Padic(value.ring().clone()),
            Self::PadicField(value) => PowerSeriesRingInner::PadicField(value.ring().clone()),
            Self::Qadic(value) => PowerSeriesRingInner::Qadic(value.ring().clone()),
            Self::QadicField(value) => PowerSeriesRingInner::QadicField(value.ring().clone()),
            Self::PadicLocalField(value) => {
                PowerSeriesRingInner::PadicLocalField(value.ring().clone())
            }
        }
    }

    fn nonzero_term_count(&self) -> usize {
        match self {
            Self::Integer(value) => value.nonzero_term_count(),
            Self::Rational(value) => value.nonzero_term_count(),
            Self::Residue(value) => value.nonzero_term_count(),
            Self::FiniteField(value) => value.nonzero_term_count(),
            Self::Padic(value) => value.nonzero_term_count(),
            Self::PadicField(value) => value.nonzero_term_count(),
            Self::Qadic(value) => value.nonzero_term_count(),
            Self::QadicField(value) => value.nonzero_term_count(),
            Self::PadicLocalField(value) => value.nonzero_term_count(),
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Integer(value) => value.is_zero(),
            Self::Rational(value) => value.is_zero(),
            Self::Residue(value) => value.is_zero(),
            Self::FiniteField(value) => value.is_zero(),
            Self::Padic(value) => value.is_zero(),
            Self::PadicField(value) => value.is_zero(),
            Self::Qadic(value) => value.is_zero(),
            Self::QadicField(value) => value.is_zero(),
            Self::PadicLocalField(value) => value.is_zero(),
        }
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Integer(value) => value.is_one(),
            Self::Rational(value) => value.is_one(),
            Self::Residue(value) => value.is_one(),
            Self::FiniteField(value) => value.is_one(),
            Self::Padic(value) => value.is_one(),
            Self::PadicField(value) => value.is_one(),
            Self::Qadic(value) => value.is_one(),
            Self::QadicField(value) => value.is_one(),
            Self::PadicLocalField(value) => value.is_one(),
        }
    }

    fn canonical_hash(&self) -> isize {
        match self {
            Self::Integer(value) => canonical_hash(value),
            Self::Rational(value) => canonical_hash(value),
            Self::Residue(value) => canonical_hash(value),
            Self::FiniteField(value) => canonical_hash(value),
            Self::Padic(value) => canonical_hash(value),
            Self::PadicField(value) => canonical_hash(value),
            Self::Qadic(value) => canonical_hash(value),
            Self::QadicField(value) => canonical_hash(value),
            Self::PadicLocalField(value) => canonical_hash(value),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Integer(value) => value.to_json(),
            Self::Rational(value) => value.to_json(),
            Self::Residue(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
            Self::Padic(value) => value.to_json(),
            Self::PadicField(value) => value.to_json(),
            Self::Qadic(value) => value.to_json(),
            Self::QadicField(value) => value.to_json(),
            Self::PadicLocalField(value) => value.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        match coefficient_type_from_json(value, true)?.as_str() {
            IntegerRing::TYPE_ID => Ok(Self::Integer(
                PowerSeries::<IntegerRing>::from_json(value).map_err(value_error)?,
            )),
            RationalField::TYPE_ID => Ok(Self::Rational(
                PowerSeries::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            ResidueRing::TYPE_ID => Ok(Self::Residue(
                PowerSeries::<ResidueRing>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::FiniteField(
                PowerSeries::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            PadicRing::TYPE_ID => Ok(Self::Padic(
                PowerSeries::<PadicRing>::from_json(value).map_err(value_error)?,
            )),
            PadicField::TYPE_ID => Ok(Self::PadicField(
                PowerSeries::<PadicField>::from_json(value).map_err(value_error)?,
            )),
            QadicRing::TYPE_ID => Ok(Self::Qadic(
                PowerSeries::<QadicRing>::from_json(value).map_err(value_error)?,
            )),
            QadicField::TYPE_ID => Ok(Self::QadicField(
                PowerSeries::<QadicField>::from_json(value).map_err(value_error)?,
            )),
            PadicLocalField::TYPE_ID => Ok(Self::PadicLocalField(
                PowerSeries::<PadicLocalField>::from_json(value).map_err(value_error)?,
            )),
            type_id => Err(value_error(format!(
                "unsupported power-series coefficient context {type_id:?}"
            ))),
        }
    }
}

impl PartialEq for SeriesInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::Residue(left), Self::Residue(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            (Self::Padic(left), Self::Padic(right)) => left == right,
            (Self::PadicField(left), Self::PadicField(right)) => left == right,
            (Self::Qadic(left), Self::Qadic(right)) => left == right,
            (Self::QadicField(left), Self::QadicField(right)) => left == right,
            (Self::PadicLocalField(left), Self::PadicLocalField(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for SeriesInner {}

/// A sparse power series carrying its coefficient and truncation context.
#[pyclass(
    module = "noether.commalg",
    name = "PowerSeries",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPowerSeries {
    pub(crate) inner: SeriesInner,
}

impl PyPowerSeries {
    pub(crate) fn from_integer(value: PowerSeries<IntegerRing>) -> Self {
        Self {
            inner: SeriesInner::Integer(value),
        }
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner, operation) {
            (SeriesInner::Integer(left), SeriesInner::Integer(right), "add") => {
                SeriesInner::Integer((left + right).map_err(value_error)?)
            }
            (SeriesInner::Rational(left), SeriesInner::Rational(right), "add") => {
                SeriesInner::Rational((left + right).map_err(value_error)?)
            }
            (SeriesInner::Residue(left), SeriesInner::Residue(right), "add") => {
                SeriesInner::Residue((left + right).map_err(value_error)?)
            }
            (SeriesInner::FiniteField(left), SeriesInner::FiniteField(right), "add") => {
                SeriesInner::FiniteField((left + right).map_err(value_error)?)
            }
            (SeriesInner::Padic(left), SeriesInner::Padic(right), "add") => {
                SeriesInner::Padic((left + right).map_err(value_error)?)
            }
            (SeriesInner::PadicField(left), SeriesInner::PadicField(right), "add") => {
                SeriesInner::PadicField((left + right).map_err(value_error)?)
            }
            (SeriesInner::Qadic(left), SeriesInner::Qadic(right), "add") => {
                SeriesInner::Qadic((left + right).map_err(value_error)?)
            }
            (SeriesInner::QadicField(left), SeriesInner::QadicField(right), "add") => {
                SeriesInner::QadicField((left + right).map_err(value_error)?)
            }
            (SeriesInner::PadicLocalField(left), SeriesInner::PadicLocalField(right), "add") => {
                SeriesInner::PadicLocalField((left + right).map_err(value_error)?)
            }
            (SeriesInner::Integer(left), SeriesInner::Integer(right), "sub") => {
                SeriesInner::Integer((left - right).map_err(value_error)?)
            }
            (SeriesInner::Rational(left), SeriesInner::Rational(right), "sub") => {
                SeriesInner::Rational((left - right).map_err(value_error)?)
            }
            (SeriesInner::Residue(left), SeriesInner::Residue(right), "sub") => {
                SeriesInner::Residue((left - right).map_err(value_error)?)
            }
            (SeriesInner::FiniteField(left), SeriesInner::FiniteField(right), "sub") => {
                SeriesInner::FiniteField((left - right).map_err(value_error)?)
            }
            (SeriesInner::Padic(left), SeriesInner::Padic(right), "sub") => {
                SeriesInner::Padic((left - right).map_err(value_error)?)
            }
            (SeriesInner::PadicField(left), SeriesInner::PadicField(right), "sub") => {
                SeriesInner::PadicField((left - right).map_err(value_error)?)
            }
            (SeriesInner::Qadic(left), SeriesInner::Qadic(right), "sub") => {
                SeriesInner::Qadic((left - right).map_err(value_error)?)
            }
            (SeriesInner::QadicField(left), SeriesInner::QadicField(right), "sub") => {
                SeriesInner::QadicField((left - right).map_err(value_error)?)
            }
            (SeriesInner::PadicLocalField(left), SeriesInner::PadicLocalField(right), "sub") => {
                SeriesInner::PadicLocalField((left - right).map_err(value_error)?)
            }
            (SeriesInner::Integer(left), SeriesInner::Integer(right), "mul") => {
                SeriesInner::Integer((left * right).map_err(value_error)?)
            }
            (SeriesInner::Rational(left), SeriesInner::Rational(right), "mul") => {
                SeriesInner::Rational((left * right).map_err(value_error)?)
            }
            (SeriesInner::Residue(left), SeriesInner::Residue(right), "mul") => {
                SeriesInner::Residue((left * right).map_err(value_error)?)
            }
            (SeriesInner::FiniteField(left), SeriesInner::FiniteField(right), "mul") => {
                SeriesInner::FiniteField((left * right).map_err(value_error)?)
            }
            (SeriesInner::Padic(left), SeriesInner::Padic(right), "mul") => {
                SeriesInner::Padic((left * right).map_err(value_error)?)
            }
            (SeriesInner::PadicField(left), SeriesInner::PadicField(right), "mul") => {
                SeriesInner::PadicField((left * right).map_err(value_error)?)
            }
            (SeriesInner::Qadic(left), SeriesInner::Qadic(right), "mul") => {
                SeriesInner::Qadic((left * right).map_err(value_error)?)
            }
            (SeriesInner::QadicField(left), SeriesInner::QadicField(right), "mul") => {
                SeriesInner::QadicField((left * right).map_err(value_error)?)
            }
            (SeriesInner::PadicLocalField(left), SeriesInner::PadicLocalField(right), "mul") => {
                SeriesInner::PadicLocalField((left * right).map_err(value_error)?)
            }
            (SeriesInner::Integer(left), SeriesInner::Integer(right), "exact_div") => {
                SeriesInner::Integer(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::Rational(left), SeriesInner::Rational(right), "exact_div") => {
                SeriesInner::Rational(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::Residue(left), SeriesInner::Residue(right), "exact_div") => {
                SeriesInner::Residue(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::FiniteField(left), SeriesInner::FiniteField(right), "exact_div") => {
                SeriesInner::FiniteField(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::Padic(left), SeriesInner::Padic(right), "exact_div") => {
                SeriesInner::Padic(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::PadicField(left), SeriesInner::PadicField(right), "exact_div") => {
                SeriesInner::PadicField(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::Qadic(left), SeriesInner::Qadic(right), "exact_div") => {
                SeriesInner::Qadic(left.exact_div(right).map_err(series_error)?)
            }
            (SeriesInner::QadicField(left), SeriesInner::QadicField(right), "exact_div") => {
                SeriesInner::QadicField(left.exact_div(right).map_err(series_error)?)
            }
            (
                SeriesInner::PadicLocalField(left),
                SeriesInner::PadicLocalField(right),
                "exact_div",
            ) => SeriesInner::PadicLocalField(left.exact_div(right).map_err(series_error)?),
            _ => {
                return Err(PyValueError::new_err(
                    "power-series operands have different coefficient or truncation contexts",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn images_from_python(&self, images: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        if let Ok(mapping) = images.cast::<PyDict>() {
            if mapping.len() != self.inner.ring().variables().len() {
                return Err(PyValueError::new_err(
                    "composition mapping must contain exactly one image per variable",
                ));
            }
            return self
                .inner
                .ring()
                .variables()
                .iter()
                .map(|name| -> PyResult<Self> {
                    Ok(mapping
                        .get_item(name)?
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "composition mapping has no image for variable {name:?}"
                            ))
                        })?
                        .extract::<PyRef<'_, PyPowerSeries>>()?
                        .clone())
                })
                .collect();
        }
        images
            .try_iter()?
            .map(|image| -> PyResult<Self> {
                Ok(image?.extract::<PyRef<'_, PyPowerSeries>>()?.clone())
            })
            .collect()
    }

    fn coerce_operand(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
            return Ok(series.clone());
        }
        self.ring().constant_from_python(py, value)
    }
}

#[pymethods]
impl PyPowerSeries {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PowerSeries::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn ring(&self) -> PyPowerSeriesRing {
        PyPowerSeriesRing {
            inner: self.inner.ring(),
        }
    }

    #[getter]
    fn nonzero_term_count(&self) -> usize {
        self.inner.nonzero_term_count()
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Serialize this series with its complete fixed-precision context.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a fixed-precision power series.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SeriesInner::from_json(value)?,
        })
    }

    fn terms(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let terms = PyList::empty(py);
        match &self.inner {
            SeriesInner::Integer(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((exponent, integer_to_python(py, coefficient)?))?;
                }
            }
            SeriesInner::Rational(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((exponent, rational_to_python(py, coefficient)?))?;
                }
            }
            SeriesInner::Residue(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyResidueClass {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::FiniteField(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyFiniteFieldElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::Padic(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyPadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::PadicField(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyPadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::Qadic(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyQadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::QadicField(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyQadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            SeriesInner::PadicLocalField(value) => {
                for (exponent, coefficient) in value.terms() {
                    terms.append((
                        exponent,
                        Py::new(
                            py,
                            PyPadicLocalFieldElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
        }
        Ok(terms.unbind().into_any())
    }

    fn coefficient(&self, py: Python<'_>, exponent: Vec<u64>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SeriesInner::Integer(value) => {
                integer_to_python(py, &value.coefficient(&exponent).map_err(value_error)?)
            }
            SeriesInner::Rational(value) => {
                rational_to_python(py, &value.coefficient(&exponent).map_err(value_error)?)
            }
            SeriesInner::Residue(value) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::Padic(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::PadicField(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::Qadic(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::QadicField(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            SeriesInner::PadicLocalField(value) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
        }
    }

    fn constant_coefficient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SeriesInner::Integer(value) => integer_to_python(py, &value.constant_coefficient()),
            SeriesInner::Rational(value) => rational_to_python(py, &value.constant_coefficient()),
            SeriesInner::Residue(value) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::Padic(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::PadicField(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::Qadic(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::QadicField(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            SeriesInner::PadicLocalField(value) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
        }
    }

    fn pow(&self, py: Python<'_>, exponent: u64) -> PyResult<Self> {
        let inner = py.detach(|| -> PyResult<SeriesInner> {
            let inner = match &self.inner {
                SeriesInner::Integer(value) => {
                    SeriesInner::Integer(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::Rational(value) => {
                    SeriesInner::Rational(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::Residue(value) => {
                    SeriesInner::Residue(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::FiniteField(value) => {
                    SeriesInner::FiniteField(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::Padic(value) => {
                    SeriesInner::Padic(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::PadicField(value) => {
                    SeriesInner::PadicField(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::Qadic(value) => {
                    SeriesInner::Qadic(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::QadicField(value) => {
                    SeriesInner::QadicField(value.pow(exponent).map_err(value_error)?)
                }
                SeriesInner::PadicLocalField(value) => {
                    SeriesInner::PadicLocalField(value.pow(exponent).map_err(value_error)?)
                }
            };
            Ok(inner)
        })?;
        Ok(Self { inner })
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Integer(value) => SeriesInner::Integer(
                value
                    .scalar_mul(&extract_integer(scalar)?)
                    .map_err(value_error)?,
            ),
            SeriesInner::Rational(value) => SeriesInner::Rational(
                value
                    .scalar_mul(&extract_rational(py, scalar)?)
                    .map_err(value_error)?,
            ),
            SeriesInner::Residue(value) => SeriesInner::Residue(
                value
                    .scalar_mul(&extract_residue(value.ring().coefficient_ring(), scalar)?)
                    .map_err(value_error)?,
            ),
            SeriesInner::FiniteField(value) => SeriesInner::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_element(
                        value.ring().coefficient_ring(),
                        scalar,
                    )?)
                    .map_err(value_error)?,
            ),
            SeriesInner::Padic(value) => SeriesInner::Padic(
                value
                    .scalar_mul(&extract_padic(value.ring().coefficient_ring(), scalar)?)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicField(value) => SeriesInner::PadicField(
                value
                    .scalar_mul(&extract_padic_field(
                        value.ring().coefficient_ring(),
                        scalar,
                    )?)
                    .map_err(value_error)?,
            ),
            SeriesInner::Qadic(value) => SeriesInner::Qadic(
                value
                    .scalar_mul(&extract_qadic(value.ring().coefficient_ring(), scalar)?)
                    .map_err(value_error)?,
            ),
            SeriesInner::QadicField(value) => SeriesInner::QadicField(
                value
                    .scalar_mul(&extract_qadic_field(
                        value.ring().coefficient_ring(),
                        scalar,
                    )?)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicLocalField(value) => SeriesInner::PadicLocalField(
                value
                    .scalar_mul(&extract_padic_local_field(
                        value.ring().coefficient_ring(),
                        scalar,
                    )?)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn derivative(&self, variable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let variable = self.ring().resolve_variable(variable)?;
        let inner = match &self.inner {
            SeriesInner::Integer(value) => {
                SeriesInner::Integer(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::Rational(value) => {
                SeriesInner::Rational(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::Residue(value) => {
                SeriesInner::Residue(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::FiniteField(value) => {
                SeriesInner::FiniteField(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::Padic(value) => {
                SeriesInner::Padic(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::PadicField(value) => {
                SeriesInner::PadicField(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::Qadic(value) => {
                SeriesInner::Qadic(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::QadicField(value) => {
                SeriesInner::QadicField(value.derivative(variable).map_err(value_error)?)
            }
            SeriesInner::PadicLocalField(value) => {
                SeriesInner::PadicLocalField(value.derivative(variable).map_err(value_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn compose(&self, images: &Bound<'_, PyAny>) -> PyResult<Self> {
        let images = self.images_from_python(images)?;
        let inner = match &self.inner {
            SeriesInner::Integer(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::Integer(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::Integer(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::Rational(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::Rational(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::Rational(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::Residue(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::Residue(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::Residue(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::FiniteField(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::FiniteField(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::FiniteField(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::Padic(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::Padic(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::Padic(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::PadicField(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::PadicField(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::PadicField(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::Qadic(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::Qadic(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::Qadic(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::QadicField(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::QadicField(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::QadicField(value.compose(&typed).map_err(value_error)?)
            }
            SeriesInner::PadicLocalField(value) => {
                let typed = images
                    .iter()
                    .map(|image| match &image.inner {
                        SeriesInner::PadicLocalField(value) => Ok(value.clone()),
                        _ => Err(PyValueError::new_err(
                            "composition image has a different coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SeriesInner::PadicLocalField(value.compose(&typed).map_err(value_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn monomial_substitution(&self, monomial_map: &PyMonomialMap) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Integer(value) => SeriesInner::Integer(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Rational(value) => SeriesInner::Rational(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Residue(value) => SeriesInner::Residue(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::FiniteField(value) => SeriesInner::FiniteField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Padic(value) => SeriesInner::Padic(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicField(value) => SeriesInner::PadicField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Qadic(value) => SeriesInner::Qadic(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::QadicField(value) => SeriesInner::QadicField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicLocalField(value) => SeriesInner::PadicLocalField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn semilinear_substitution(
        &self,
        coefficient_map: &Bound<'_, PyAny>,
        monomial_map: &PyMonomialMap,
    ) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Padic(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicRingFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a power series over PadicRing requires PadicRingFrobenius",
                        )
                    })?;
                SeriesInner::Padic(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            SeriesInner::PadicField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicFieldFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a power series over PadicField requires PadicFieldFrobenius",
                        )
                    })?;
                SeriesInner::PadicField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            SeriesInner::Qadic(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyQadicRingFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a power series over QadicRing requires QadicRingFrobenius",
                        )
                    })?;
                SeriesInner::Qadic(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            SeriesInner::QadicField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyQadicFieldFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a power series over QadicField requires QadicFieldFrobenius",
                        )
                    })?;
                SeriesInner::QadicField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            SeriesInner::PadicLocalField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicLocalFieldCoefficientFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a power series over PadicLocalField requires PadicLocalFieldCoefficientFrobenius",
                        )
                    })?;
                SeriesInner::PadicLocalField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            SeriesInner::Integer(_)
            | SeriesInner::Rational(_)
            | SeriesInner::Residue(_)
            | SeriesInner::FiniteField(_) => {
                return Err(unsupported_series_operation(
                    "semilinear_substitution",
                    self.inner.context_type_id(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn frobenius_lift(&self, exponent: u64) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Padic(value) => {
                SeriesInner::Padic(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            SeriesInner::PadicField(value) => {
                SeriesInner::PadicField(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            SeriesInner::Qadic(value) => {
                SeriesInner::Qadic(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            SeriesInner::QadicField(value) => {
                SeriesInner::QadicField(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            SeriesInner::Integer(_)
            | SeriesInner::Rational(_)
            | SeriesInner::Residue(_)
            | SeriesInner::FiniteField(_)
            | SeriesInner::PadicLocalField(_) => {
                return Err(unsupported_series_operation(
                    "frobenius_lift",
                    self.inner.context_type_id(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn inverse(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| -> PyResult<SeriesInner> {
            let inner = match &self.inner {
                SeriesInner::Integer(value) => {
                    SeriesInner::Integer(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::Rational(value) => {
                    SeriesInner::Rational(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::Residue(value) => {
                    SeriesInner::Residue(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::FiniteField(value) => {
                    SeriesInner::FiniteField(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::Padic(value) => {
                    SeriesInner::Padic(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::PadicField(value) => {
                    SeriesInner::PadicField(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::Qadic(value) => {
                    SeriesInner::Qadic(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::QadicField(value) => {
                    SeriesInner::QadicField(value.inverse().map_err(unit_error)?)
                }
                SeriesInner::PadicLocalField(value) => {
                    SeriesInner::PadicLocalField(value.inverse().map_err(unit_error)?)
                }
            };
            Ok(inner)
        })?;
        Ok(Self { inner })
    }

    fn exact_div(&self, py: Python<'_>, divisor: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(&self.coerce_operand(py, divisor)?, "exact_div")
    }

    fn restrict_truncation(&self, target: PySeriesTruncation) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Integer(value) => SeriesInner::Integer(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Rational(value) => SeriesInner::Rational(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Residue(value) => SeriesInner::Residue(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::FiniteField(value) => SeriesInner::FiniteField(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::Padic(value) => SeriesInner::Padic(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicField(value) => SeriesInner::PadicField(
                value
                    .restrict_truncation(target.inner.clone())
                    .map_err(value_error)?,
            ),
            SeriesInner::Qadic(value) => SeriesInner::Qadic(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::QadicField(value) => SeriesInner::QadicField(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
            SeriesInner::PadicLocalField(value) => SeriesInner::PadicLocalField(
                value
                    .restrict_truncation(target.inner)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let right = self.coerce_operand(py, other)?;
        py.detach(|| self.binary(&right, "add"))
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let right = self.coerce_operand(py, other)?;
        py.detach(|| self.binary(&right, "sub"))
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let left = self.coerce_operand(py, other)?;
        py.detach(|| left.binary(self, "sub"))
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyMatrix>() || other.is_instance_of::<PyVector>() {
            return Ok(py.NotImplemented());
        }
        let right = self.coerce_operand(py, other)?;
        let product = py.detach(|| self.binary(&right, "mul"))?;
        Ok(Py::new(py, product)?.into_any())
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(py, other)
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            SeriesInner::Integer(value) => SeriesInner::Integer((-value).map_err(value_error)?),
            SeriesInner::Rational(value) => SeriesInner::Rational((-value).map_err(value_error)?),
            SeriesInner::Residue(value) => SeriesInner::Residue((-value).map_err(value_error)?),
            SeriesInner::FiniteField(value) => {
                SeriesInner::FiniteField((-value).map_err(value_error)?)
            }
            SeriesInner::Padic(value) => SeriesInner::Padic((-value).map_err(value_error)?),
            SeriesInner::PadicField(value) => {
                SeriesInner::PadicField((-value).map_err(value_error)?)
            }
            SeriesInner::Qadic(value) => SeriesInner::Qadic((-value).map_err(value_error)?),
            SeriesInner::QadicField(value) => {
                SeriesInner::QadicField((-value).map_err(value_error)?)
            }
            SeriesInner::PadicLocalField(value) => {
                SeriesInner::PadicLocalField((-value).map_err(value_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: u64,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "PowerSeries power does not accept Python's third modulus argument",
            ));
        }
        self.pow(py, exponent)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PowerSeries(nonzero_terms={}, variables={:?}, truncation={})",
                self.inner.nonzero_term_count(),
                self.inner.ring().variables(),
                self.inner.ring().truncation()
            )
        })
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let ring = self.inner.ring();
        let variables = ring.variables();
        summary.set_item("type", "noether/commalg/PowerSeries@1")?;
        summary.set_item(
            "variables",
            variables
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .collect::<Vec<_>>(),
        )?;
        if variables.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("variables_truncated", true)?;
        }
        summary.set_item(
            "truncation",
            crate::py::repr::bounded_text(self.inner.ring().truncation().to_string()),
        )?;
        summary.set_item("nonzero_term_count", self.inner.nonzero_term_count())?;
        Ok(summary.unbind())
    }
}
