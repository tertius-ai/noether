// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of polynomial local rings and Mora standard bases.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    LocalAlgebraError, LocalCompletionMap, LocalIdeal, LocalMonomialOrder, LocalPolynomial,
    LocalPolynomialRing, LocalReduction, LocalStandardBasis,
};
use noether::numthy::FiniteField;
use noether::{RationalField, SemanticType};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyType};

use super::facade_scalar::{extract_rational, integer_to_python, rational_to_python};
use super::number_field::PyIdeal;
use super::numthy::PyFiniteFieldElement;
use super::polynomial_facade::{
    extract_finite_field, PolynomialRingInner, PyPolynomial, PyPolynomialRing,
};
use super::quotient::PyQuotientRing;
use super::series::{PowerSeriesRingInner, PyPowerSeries, PyPowerSeriesRing, SeriesInner};

fn local_error(error: LocalAlgebraError) -> PyErr {
    match error {
        LocalAlgebraError::ContextMismatch(error) => super::errors::context_mismatch(error),
        LocalAlgebraError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        error => PyValueError::new_err(error.to_string()),
    }
}

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

/// One local monomial order, kept separate from global admissible orders.
#[pyclass(
    module = "noether.commalg",
    name = "LocalMonomialOrder",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyLocalMonomialOrder {
    inner: LocalMonomialOrder,
}

#[pymethods]
impl PyLocalMonomialOrder {
    #[classattr]
    fn type_id() -> &'static str {
        LocalMonomialOrder::TYPE_ID
    }

    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "local_lex" => LocalMonomialOrder::LocalLexicographic,
            "local_deglex" => LocalMonomialOrder::LocalDegreeLexicographic,
            "local_degrevlex" => LocalMonomialOrder::LocalDegreeReverseLexicographic,
            _ => {
                return Err(PyValueError::new_err(
                    "unknown local monomial order; expected 'local_lex', 'local_deglex', or 'local_degrevlex'",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn LOCAL_LEX() -> Self {
        Self {
            inner: LocalMonomialOrder::LocalLexicographic,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn LOCAL_DEGLEX() -> Self {
        Self {
            inner: LocalMonomialOrder::LocalDegreeLexicographic,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn LOCAL_DEGREVLEX() -> Self {
        Self {
            inner: LocalMonomialOrder::LocalDegreeReverseLexicographic,
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    fn is_degree_compatible(&self) -> bool {
        self.inner.is_degree_compatible()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: LocalMonomialOrder::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("LocalMonomialOrder({})", self.inner.name()))
    }

    fn __str__(&self) -> &'static str {
        self.inner.name()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum LocalRingInner {
    Rational(LocalPolynomialRing<RationalField>),
    FiniteField(LocalPolynomialRing<FiniteField>),
}

/// A polynomial ring localized at one rational maximal ideal.
#[pyclass(
    module = "noether.commalg",
    name = "LocalPolynomialRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalPolynomialRing {
    pub(crate) inner: LocalRingInner,
}

impl PyLocalPolynomialRing {
    pub(crate) fn from_rational(inner: LocalPolynomialRing<RationalField>) -> Self {
        Self {
            inner: LocalRingInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: LocalPolynomialRing<FiniteField>) -> Self {
        Self {
            inner: LocalRingInner::FiniteField(inner),
        }
    }

    fn canonical_hash(&self) -> isize {
        match &self.inner {
            LocalRingInner::Rational(value) => canonical_hash(value),
            LocalRingInner::FiniteField(value) => canonical_hash(value),
        }
    }
}

#[pymethods]
impl PyLocalPolynomialRing {
    #[classattr]
    fn type_id() -> &'static str {
        LocalPolynomialRing::<RationalField>::TYPE_ID
    }

    #[new]
    #[pyo3(signature = (ambient_ring, point, *, order=None))]
    fn new(
        py: Python<'_>,
        ambient_ring: &PyPolynomialRing,
        point: &Bound<'_, PyAny>,
        order: Option<&PyLocalMonomialOrder>,
    ) -> PyResult<Self> {
        let order = order.map_or_else(LocalMonomialOrder::default, |order| order.inner);
        let coordinates = point
            .try_iter()?
            .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?;
        match &ambient_ring.inner {
            PolynomialRingInner::Rational(ring) => {
                let point = coordinates
                    .iter()
                    .map(|coordinate| extract_rational(py, coordinate))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Self::from_rational(
                    LocalPolynomialRing::new(ring.clone(), point, order).map_err(local_error)?,
                ))
            }
            PolynomialRingInner::FiniteField(ring) => {
                let point = coordinates
                    .iter()
                    .map(|coordinate| extract_finite_field(ring.coefficient_ring(), coordinate))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Self::from_finite_field(
                    LocalPolynomialRing::new(ring.clone(), point, order).map_err(local_error)?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "LocalPolynomialRing currently accepts polynomial rings over Q or a finite field",
            )),
        }
    }

    #[classmethod]
    #[pyo3(signature = (ambient_ring, *, order=None))]
    fn at_origin(
        _cls: &Bound<'_, PyType>,
        ambient_ring: &PyPolynomialRing,
        order: Option<&PyLocalMonomialOrder>,
    ) -> PyResult<Self> {
        let order = order.map_or_else(LocalMonomialOrder::default, |order| order.inner);
        match &ambient_ring.inner {
            PolynomialRingInner::Rational(ring) => Ok(Self::from_rational(
                LocalPolynomialRing::at_origin(ring.clone(), order).map_err(local_error)?,
            )),
            PolynomialRingInner::FiniteField(ring) => Ok(Self::from_finite_field(
                LocalPolynomialRing::at_origin(ring.clone(), order).map_err(local_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "LocalPolynomialRing currently accepts polynomial rings over Q or a finite field",
            )),
        }
    }

    #[getter]
    fn ambient_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            LocalRingInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.ambient_ring().clone())
            }
            LocalRingInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.ambient_ring().clone())
            }
        }
    }

    #[getter]
    fn coordinate_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            LocalRingInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.coordinate_ring().clone())
            }
            LocalRingInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.coordinate_ring().clone())
            }
        }
    }

    #[getter]
    fn point(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            LocalRingInner::Rational(value) => {
                let values = value
                    .point()
                    .iter()
                    .map(|coordinate| rational_to_python(py, coordinate))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, values)?.unbind().into_any())
            }
            LocalRingInner::FiniteField(value) => {
                let values = value
                    .point()
                    .iter()
                    .cloned()
                    .map(|inner| Py::new(py, PyFiniteFieldElement { inner }))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, values)?.unbind().into_any())
            }
        }
    }

    #[getter]
    fn order(&self) -> PyLocalMonomialOrder {
        let inner = match &self.inner {
            LocalRingInner::Rational(value) => value.order(),
            LocalRingInner::FiniteField(value) => value.order(),
        };
        PyLocalMonomialOrder { inner }
    }

    fn from_polynomial(&self, value: &PyPolynomial) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalRingInner::Rational(ring) => Ok(PyLocalPolynomial::from_rational(
                ring.from_polynomial(value.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("localization over Q requires a polynomial over Q")
                })?)
                .map_err(local_error)?,
            )),
            LocalRingInner::FiniteField(ring) => Ok(PyLocalPolynomial::from_finite_field(
                ring.from_polynomial(value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "finite-field localization requires a polynomial over the same field",
                    )
                })?)
                .map_err(local_error)?,
            )),
        }
    }

    fn fraction(
        &self,
        numerator: &PyPolynomial,
        denominator: &PyPolynomial,
    ) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalRingInner::Rational(ring) => Ok(PyLocalPolynomial::from_rational(
                ring.fraction(
                    numerator
                        .as_rational()
                        .ok_or_else(|| PyTypeError::new_err("numerator must be over Q"))?
                        .clone(),
                    denominator
                        .as_rational()
                        .ok_or_else(|| PyTypeError::new_err("denominator must be over Q"))?
                        .clone(),
                )
                .map_err(local_error)?,
            )),
            LocalRingInner::FiniteField(ring) => Ok(PyLocalPolynomial::from_finite_field(
                ring.fraction(
                    numerator
                        .as_finite_field()
                        .ok_or_else(|| {
                            PyTypeError::new_err("numerator must be over the same finite field")
                        })?
                        .clone(),
                    denominator
                        .as_finite_field()
                        .ok_or_else(|| {
                            PyTypeError::new_err("denominator must be over the same finite field")
                        })?
                        .clone(),
                )
                .map_err(local_error)?,
            )),
        }
    }

    fn from_ambient_fraction(
        &self,
        numerator: &PyPolynomial,
        denominator: &PyPolynomial,
    ) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalRingInner::Rational(ring) => Ok(PyLocalPolynomial::from_rational(
                ring.from_ambient_fraction(
                    numerator
                        .as_rational()
                        .ok_or_else(|| PyTypeError::new_err("ambient numerator must be over Q"))?,
                    denominator.as_rational().ok_or_else(|| {
                        PyTypeError::new_err("ambient denominator must be over Q")
                    })?,
                )
                .map_err(local_error)?,
            )),
            LocalRingInner::FiniteField(ring) => Ok(PyLocalPolynomial::from_finite_field(
                ring.from_ambient_fraction(
                    numerator.as_finite_field().ok_or_else(|| {
                        PyTypeError::new_err("ambient numerator must be over the same finite field")
                    })?,
                    denominator.as_finite_field().ok_or_else(|| {
                        PyTypeError::new_err(
                            "ambient denominator must be over the same finite field",
                        )
                    })?,
                )
                .map_err(local_error)?,
            )),
        }
    }

    fn zero(&self) -> PyLocalPolynomial {
        match &self.inner {
            LocalRingInner::Rational(ring) => PyLocalPolynomial::from_rational(ring.zero()),
            LocalRingInner::FiniteField(ring) => PyLocalPolynomial::from_finite_field(ring.zero()),
        }
    }

    fn one(&self) -> PyLocalPolynomial {
        match &self.inner {
            LocalRingInner::Rational(ring) => PyLocalPolynomial::from_rational(ring.one()),
            LocalRingInner::FiniteField(ring) => PyLocalPolynomial::from_finite_field(ring.one()),
        }
    }

    fn generators(&self) -> PyResult<Vec<PyLocalPolynomial>> {
        match &self.inner {
            LocalRingInner::Rational(ring) => Ok(ring
                .generators()
                .map_err(local_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_rational)
                .collect()),
            LocalRingInner::FiniteField(ring) => Ok(ring
                .generators()
                .map_err(local_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_finite_field)
                .collect()),
        }
    }

    fn standard_basis(
        &self,
        generators: Vec<PyRef<'_, PyLocalPolynomial>>,
    ) -> PyResult<PyLocalStandardBasis> {
        match &self.inner {
            LocalRingInner::Rational(ring) => {
                let generators = generators
                    .iter()
                    .map(|generator| generator.as_rational().cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        PyTypeError::new_err("all local generators must belong to this Q-context")
                    })?;
                Ok(PyLocalStandardBasis::from_rational(
                    ring.standard_basis(&generators).map_err(local_error)?,
                ))
            }
            LocalRingInner::FiniteField(ring) => {
                let generators = generators
                    .iter()
                    .map(|generator| generator.as_finite_field().cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        PyTypeError::new_err(
                            "all local generators must belong to this finite-field context",
                        )
                    })?;
                Ok(PyLocalStandardBasis::from_finite_field(
                    ring.standard_basis(&generators).map_err(local_error)?,
                ))
            }
        }
    }

    fn ideal(&self, generators: Vec<PyRef<'_, PyLocalPolynomial>>) -> PyResult<PyLocalIdeal> {
        match &self.inner {
            LocalRingInner::Rational(ring) => {
                let generators = generators
                    .iter()
                    .map(|generator| generator.as_rational().cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        PyTypeError::new_err("all local generators must belong to this Q-context")
                    })?;
                Ok(PyLocalIdeal::from_rational(
                    ring.ideal(&generators).map_err(local_error)?,
                ))
            }
            LocalRingInner::FiniteField(ring) => {
                let generators = generators
                    .iter()
                    .map(|generator| generator.as_finite_field().cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        PyTypeError::new_err(
                            "all local generators must belong to this finite-field context",
                        )
                    })?;
                Ok(PyLocalIdeal::from_finite_field(
                    ring.ideal(&generators).map_err(local_error)?,
                ))
            }
        }
    }

    fn completion(&self, precision: u64) -> PyResult<PyLocalCompletionMap> {
        match &self.inner {
            LocalRingInner::Rational(ring) => Ok(PyLocalCompletionMap::from_rational(
                ring.completion(precision).map_err(local_error)?,
            )),
            LocalRingInner::FiniteField(ring) => Ok(PyLocalCompletionMap::from_finite_field(
                ring.completion(precision).map_err(local_error)?,
            )),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalRingInner::Rational(value) => value.to_json().map_err(value_error),
            LocalRingInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ambient_ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalPolynomialRing::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalPolynomialRing::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local polynomial coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalRingInner::Rational(value) => value.to_string(),
            LocalRingInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalCompletionMapInner {
    Rational(LocalCompletionMap<RationalField>),
    FiniteField(LocalCompletionMap<FiniteField>),
}

/// An explicit finite-precision map to a local power-series completion.
#[pyclass(
    module = "noether.commalg",
    name = "LocalCompletionMap",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalCompletionMap {
    inner: LocalCompletionMapInner,
}

impl PyLocalCompletionMap {
    fn from_rational(inner: LocalCompletionMap<RationalField>) -> Self {
        Self {
            inner: LocalCompletionMapInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalCompletionMap<FiniteField>) -> Self {
        Self {
            inner: LocalCompletionMapInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalCompletionMap {
    #[classattr]
    fn type_id() -> &'static str {
        LocalCompletionMap::<RationalField>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyLocalPolynomialRing {
        match &self.inner {
            LocalCompletionMapInner::Rational(value) => {
                PyLocalPolynomialRing::from_rational(value.source().clone())
            }
            LocalCompletionMapInner::FiniteField(value) => {
                PyLocalPolynomialRing::from_finite_field(value.source().clone())
            }
        }
    }

    #[getter]
    fn target(&self) -> PyPowerSeriesRing {
        let inner = match &self.inner {
            LocalCompletionMapInner::Rational(value) => {
                PowerSeriesRingInner::Rational(value.target().clone())
            }
            LocalCompletionMapInner::FiniteField(value) => {
                PowerSeriesRingInner::FiniteField(value.target().clone())
            }
        };
        PyPowerSeriesRing { inner }
    }

    #[getter]
    fn precision(&self) -> u64 {
        match &self.inner {
            LocalCompletionMapInner::Rational(value) => value.precision(),
            LocalCompletionMapInner::FiniteField(value) => value.precision(),
        }
    }

    fn apply(&self, value: &PyLocalPolynomial) -> PyResult<PyPowerSeries> {
        let inner = match (&self.inner, &value.inner) {
            (LocalCompletionMapInner::Rational(map), LocalPolynomialInner::Rational(value)) => {
                SeriesInner::Rational(map.apply(value).map_err(local_error)?)
            }
            (
                LocalCompletionMapInner::FiniteField(map),
                LocalPolynomialInner::FiniteField(value),
            ) => SeriesInner::FiniteField(map.apply(value).map_err(local_error)?),
            _ => {
                return Err(PyTypeError::new_err(
                    "completion map and local value contexts do not match",
                ))
            }
        };
        Ok(PyPowerSeries { inner })
    }

    fn restrict(&self, precision: u64) -> PyResult<Self> {
        match &self.inner {
            LocalCompletionMapInner::Rational(value) => Ok(Self::from_rational(
                value.restrict(precision).map_err(local_error)?,
            )),
            LocalCompletionMapInner::FiniteField(value) => Ok(Self::from_finite_field(
                value.restrict(precision).map_err(local_error)?,
            )),
        }
    }

    fn restrict_value(&self, value: &PyPowerSeries, precision: u64) -> PyResult<PyPowerSeries> {
        let inner = match (&self.inner, &value.inner) {
            (LocalCompletionMapInner::Rational(map), SeriesInner::Rational(value)) => {
                SeriesInner::Rational(map.restrict_value(value, precision).map_err(local_error)?)
            }
            (LocalCompletionMapInner::FiniteField(map), SeriesInner::FiniteField(value)) => {
                SeriesInner::FiniteField(map.restrict_value(value, precision).map_err(local_error)?)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "completion map and power-series contexts do not match",
                ))
            }
        };
        Ok(PyPowerSeries { inner })
    }

    fn verify_restriction(&self, coarser: &Self, value: &PyLocalPolynomial) -> PyResult<()> {
        match (&self.inner, &coarser.inner, &value.inner) {
            (
                LocalCompletionMapInner::Rational(fine),
                LocalCompletionMapInner::Rational(coarse),
                LocalPolynomialInner::Rational(value),
            ) => fine.verify_restriction(coarse, value).map_err(local_error),
            (
                LocalCompletionMapInner::FiniteField(fine),
                LocalCompletionMapInner::FiniteField(coarse),
                LocalPolynomialInner::FiniteField(value),
            ) => fine.verify_restriction(coarse, value).map_err(local_error),
            _ => Err(PyTypeError::new_err(
                "completion restriction data must share one exact context",
            )),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalCompletionMapInner::Rational(value) => value.to_json().map_err(value_error),
            LocalCompletionMapInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/source/ambient_ring/coefficient_ring/type")?.as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalCompletionMap::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalCompletionMap::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local completion coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalCompletionMapInner::Rational(value) => canonical_hash(value),
            LocalCompletionMapInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalCompletionMapInner::Rational(value) => value.to_string(),
            LocalCompletionMapInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum LocalPolynomialInner {
    Rational(LocalPolynomial<RationalField>),
    FiniteField(LocalPolynomial<FiniteField>),
}

/// One exact fraction in a polynomial local ring.
#[pyclass(
    module = "noether.commalg",
    name = "LocalPolynomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalPolynomial {
    pub(crate) inner: LocalPolynomialInner,
}

impl PyLocalPolynomial {
    pub(crate) fn from_rational(inner: LocalPolynomial<RationalField>) -> Self {
        Self {
            inner: LocalPolynomialInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: LocalPolynomial<FiniteField>) -> Self {
        Self {
            inner: LocalPolynomialInner::FiniteField(inner),
        }
    }

    pub(crate) fn as_rational(&self) -> Option<&LocalPolynomial<RationalField>> {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => Some(value),
            LocalPolynomialInner::FiniteField(_) => None,
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&LocalPolynomial<FiniteField>> {
        match &self.inner {
            LocalPolynomialInner::Rational(_) => None,
            LocalPolynomialInner::FiniteField(value) => Some(value),
        }
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        match (&self.inner, &other.inner, operation) {
            (
                LocalPolynomialInner::Rational(left),
                LocalPolynomialInner::Rational(right),
                "add",
            ) => Ok(Self::from_rational((left + right).map_err(local_error)?)),
            (
                LocalPolynomialInner::Rational(left),
                LocalPolynomialInner::Rational(right),
                "sub",
            ) => Ok(Self::from_rational((left - right).map_err(local_error)?)),
            (
                LocalPolynomialInner::Rational(left),
                LocalPolynomialInner::Rational(right),
                "mul",
            ) => Ok(Self::from_rational((left * right).map_err(local_error)?)),
            (
                LocalPolynomialInner::FiniteField(left),
                LocalPolynomialInner::FiniteField(right),
                "add",
            ) => Ok(Self::from_finite_field(
                (left + right).map_err(local_error)?,
            )),
            (
                LocalPolynomialInner::FiniteField(left),
                LocalPolynomialInner::FiniteField(right),
                "sub",
            ) => Ok(Self::from_finite_field(
                (left - right).map_err(local_error)?,
            )),
            (
                LocalPolynomialInner::FiniteField(left),
                LocalPolynomialInner::FiniteField(right),
                "mul",
            ) => Ok(Self::from_finite_field(
                (left * right).map_err(local_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "local polynomial operands must belong to the same exact context",
            )),
        }
    }
}

#[pymethods]
impl PyLocalPolynomial {
    #[classattr]
    fn type_id() -> &'static str {
        LocalPolynomial::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyLocalPolynomialRing {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => {
                PyLocalPolynomialRing::from_rational(value.ring().clone())
            }
            LocalPolynomialInner::FiniteField(value) => {
                PyLocalPolynomialRing::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn numerator(&self) -> PyPolynomial {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => {
                PyPolynomial::from_rational(value.numerator().clone())
            }
            LocalPolynomialInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.numerator().clone())
            }
        }
    }

    #[getter]
    fn denominator(&self) -> PyPolynomial {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => {
                PyPolynomial::from_rational(value.denominator().clone())
            }
            LocalPolynomialInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.denominator().clone())
            }
        }
    }

    #[getter]
    fn is_zero(&self) -> bool {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => value.is_zero(),
            LocalPolynomialInner::FiniteField(value) => value.is_zero(),
        }
    }

    #[getter]
    fn is_unit(&self) -> bool {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => value.is_unit(),
            LocalPolynomialInner::FiniteField(value) => value.is_unit(),
        }
    }

    fn inverse(&self) -> PyResult<Self> {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => {
                Ok(Self::from_rational(value.inverse().map_err(local_error)?))
            }
            LocalPolynomialInner::FiniteField(value) => Ok(Self::from_finite_field(
                value.inverse().map_err(local_error)?,
            )),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => value.to_json().map_err(value_error),
            LocalPolynomialInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ring/ambient_ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalPolynomial::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalPolynomial::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local polynomial coefficient context {coefficient}"
            ))),
        }
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other, "add")
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other, "sub")
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.binary(other, "mul")
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        self.binary(&other.inverse()?, "mul")
    }

    fn __neg__(&self) -> PyResult<Self> {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => {
                Ok(Self::from_rational((-value).map_err(local_error)?))
            }
            LocalPolynomialInner::FiniteField(value) => {
                Ok(Self::from_finite_field((-value).map_err(local_error)?))
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => canonical_hash(value),
            LocalPolynomialInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalPolynomialInner::Rational(value) => value.to_string(),
            LocalPolynomialInner::FiniteField(value) => value.to_string(),
        })
    }

    fn __str__(&self) -> String {
        match &self.inner {
            LocalPolynomialInner::Rational(value) => value.to_string(),
            LocalPolynomialInner::FiniteField(value) => value.to_string(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalIdealInner {
    Rational(LocalIdeal<RationalField>),
    FiniteField(LocalIdeal<FiniteField>),
}

/// A canonical ideal in one polynomial localization.
#[pyclass(
    module = "noether.commalg",
    name = "LocalIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalIdeal {
    inner: LocalIdealInner,
}

impl PyLocalIdeal {
    fn from_rational(inner: LocalIdeal<RationalField>) -> Self {
        Self {
            inner: LocalIdealInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalIdeal<FiniteField>) -> Self {
        Self {
            inner: LocalIdealInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalIdeal {
    #[classattr]
    fn type_id() -> &'static str {
        LocalIdeal::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyLocalPolynomialRing {
        match &self.inner {
            LocalIdealInner::Rational(value) => {
                PyLocalPolynomialRing::from_rational(value.ring().clone())
            }
            LocalIdealInner::FiniteField(value) => {
                PyLocalPolynomialRing::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn generators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            LocalIdealInner::Rational(value) => value
                .generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            LocalIdealInner::FiniteField(value) => value
                .generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    fn standard_basis(&self) -> PyResult<PyLocalStandardBasis> {
        match &self.inner {
            LocalIdealInner::Rational(value) => Ok(PyLocalStandardBasis::from_rational(
                value.standard_basis().map_err(local_error)?,
            )),
            LocalIdealInner::FiniteField(value) => Ok(PyLocalStandardBasis::from_finite_field(
                value.standard_basis().map_err(local_error)?,
            )),
        }
    }

    fn reduce(&self, value: &PyLocalPolynomial) -> PyResult<PyLocalReduction> {
        match &self.inner {
            LocalIdealInner::Rational(ideal) => Ok(PyLocalReduction::from_rational(
                ideal
                    .reduce(value.as_rational().ok_or_else(|| {
                        PyTypeError::new_err("reduction value belongs to another local context")
                    })?)
                    .map_err(local_error)?,
            )),
            LocalIdealInner::FiniteField(ideal) => Ok(PyLocalReduction::from_finite_field(
                ideal
                    .reduce(value.as_finite_field().ok_or_else(|| {
                        PyTypeError::new_err("reduction value belongs to another local context")
                    })?)
                    .map_err(local_error)?,
            )),
        }
    }

    fn contains(&self, value: &PyLocalPolynomial) -> PyResult<bool> {
        Ok(self.reduce(value)?.is_member())
    }

    fn tangent_cone_ideal(&self) -> PyResult<PyIdeal> {
        match &self.inner {
            LocalIdealInner::Rational(value) => Ok(PyIdeal::from_rational_polynomial(
                value.tangent_cone_ideal().map_err(local_error)?,
            )),
            LocalIdealInner::FiniteField(value) => Ok(PyIdeal::from_finite_field_polynomial(
                value.tangent_cone_ideal().map_err(local_error)?,
            )),
        }
    }

    fn associated_graded(&self) -> PyResult<PyQuotientRing> {
        match &self.inner {
            LocalIdealInner::Rational(value) => Ok(PyQuotientRing::from_rational(
                value.associated_graded().map_err(local_error)?,
            )),
            LocalIdealInner::FiniteField(value) => Ok(PyQuotientRing::from_finite_field(
                value.associated_graded().map_err(local_error)?,
            )),
        }
    }

    #[getter]
    fn dimension(&self) -> PyResult<i64> {
        match &self.inner {
            LocalIdealInner::Rational(value) => value.dimension().map_err(local_error),
            LocalIdealInner::FiniteField(value) => value.dimension().map_err(local_error),
        }
    }

    #[getter]
    fn multiplicity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = match &self.inner {
            LocalIdealInner::Rational(value) => value.multiplicity().map_err(local_error)?,
            LocalIdealInner::FiniteField(value) => value.multiplicity().map_err(local_error)?,
        };
        integer_to_python(py, &value)
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            LocalIdealInner::Rational(value) => value.verify().map_err(local_error),
            LocalIdealInner::FiniteField(value) => value.verify().map_err(local_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalIdealInner::Rational(value) => value.to_json().map_err(value_error),
            LocalIdealInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ring/ambient_ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalIdeal::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalIdeal::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local ideal coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalIdealInner::Rational(value) => canonical_hash(value),
            LocalIdealInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalIdealInner::Rational(value) => value.to_string(),
            LocalIdealInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalStandardBasisInner {
    Rational(LocalStandardBasis<RationalField>),
    FiniteField(LocalStandardBasis<FiniteField>),
}

/// A verified Mora standard basis and its local invariants.
#[pyclass(
    module = "noether.commalg",
    name = "LocalStandardBasis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalStandardBasis {
    inner: LocalStandardBasisInner,
}

impl PyLocalStandardBasis {
    fn from_rational(inner: LocalStandardBasis<RationalField>) -> Self {
        Self {
            inner: LocalStandardBasisInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalStandardBasis<FiniteField>) -> Self {
        Self {
            inner: LocalStandardBasisInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalStandardBasis {
    #[classattr]
    fn type_id() -> &'static str {
        LocalStandardBasis::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyLocalPolynomialRing {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => {
                PyLocalPolynomialRing::from_rational(value.ring().clone())
            }
            LocalStandardBasisInner::FiniteField(value) => {
                PyLocalPolynomialRing::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn generators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => value
                .generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            LocalStandardBasisInner::FiniteField(value) => value
                .generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn basis(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            LocalStandardBasisInner::FiniteField(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn ideal(&self) -> PyLocalIdeal {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => PyLocalIdeal::from_rational(value.ideal()),
            LocalStandardBasisInner::FiniteField(value) => {
                PyLocalIdeal::from_finite_field(value.ideal())
            }
        }
    }

    fn reduce(&self, value: &PyLocalPolynomial) -> PyResult<PyLocalReduction> {
        match &self.inner {
            LocalStandardBasisInner::Rational(basis) => Ok(PyLocalReduction::from_rational(
                basis
                    .reduce(value.as_rational().ok_or_else(|| {
                        PyTypeError::new_err("reduction value belongs to another local context")
                    })?)
                    .map_err(local_error)?,
            )),
            LocalStandardBasisInner::FiniteField(basis) => Ok(PyLocalReduction::from_finite_field(
                basis
                    .reduce(value.as_finite_field().ok_or_else(|| {
                        PyTypeError::new_err("reduction value belongs to another local context")
                    })?)
                    .map_err(local_error)?,
            )),
        }
    }

    fn contains(&self, value: &PyLocalPolynomial) -> PyResult<bool> {
        match &self.inner {
            LocalStandardBasisInner::Rational(basis) => basis
                .contains(value.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("membership value belongs to another local context")
                })?)
                .map_err(local_error),
            LocalStandardBasisInner::FiniteField(basis) => basis
                .contains(value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err("membership value belongs to another local context")
                })?)
                .map_err(local_error),
        }
    }

    fn tangent_cone_ideal(&self) -> PyResult<PyIdeal> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => Ok(PyIdeal::from_rational_polynomial(
                value.tangent_cone_ideal().map_err(local_error)?,
            )),
            LocalStandardBasisInner::FiniteField(value) => {
                Ok(PyIdeal::from_finite_field_polynomial(
                    value.tangent_cone_ideal().map_err(local_error)?,
                ))
            }
        }
    }

    fn associated_graded(&self) -> PyResult<PyQuotientRing> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => Ok(PyQuotientRing::from_rational(
                value.associated_graded().map_err(local_error)?,
            )),
            LocalStandardBasisInner::FiniteField(value) => Ok(PyQuotientRing::from_finite_field(
                value.associated_graded().map_err(local_error)?,
            )),
        }
    }

    #[getter]
    fn dimension(&self) -> PyResult<i64> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => value.dimension().map_err(local_error),
            LocalStandardBasisInner::FiniteField(value) => value.dimension().map_err(local_error),
        }
    }

    #[getter]
    fn multiplicity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = match &self.inner {
            LocalStandardBasisInner::Rational(value) => {
                value.multiplicity().map_err(local_error)?
            }
            LocalStandardBasisInner::FiniteField(value) => {
                value.multiplicity().map_err(local_error)?
            }
        };
        integer_to_python(py, &value)
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => value.verify().map_err(local_error),
            LocalStandardBasisInner::FiniteField(value) => value.verify().map_err(local_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => value.to_json().map_err(value_error),
            LocalStandardBasisInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ring/ambient_ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalStandardBasis::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalStandardBasis::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local standard-basis coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalStandardBasisInner::Rational(value) => canonical_hash(value),
            LocalStandardBasisInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalStandardBasisInner::Rational(value) => value.to_string(),
            LocalStandardBasisInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalReductionInner {
    Rational(LocalReduction<RationalField>),
    FiniteField(LocalReduction<FiniteField>),
}

/// Replayable weak-normal-form evidence for one Mora reduction.
#[pyclass(
    module = "noether.commalg",
    name = "LocalReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalReduction {
    inner: LocalReductionInner,
}

impl PyLocalReduction {
    fn from_rational(inner: LocalReduction<RationalField>) -> Self {
        Self {
            inner: LocalReductionInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalReduction<FiniteField>) -> Self {
        Self {
            inner: LocalReductionInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalReduction {
    #[classattr]
    fn type_id() -> &'static str {
        LocalReduction::<RationalField>::TYPE_ID
    }

    #[getter]
    fn basis(&self) -> PyLocalStandardBasis {
        match &self.inner {
            LocalReductionInner::Rational(value) => {
                PyLocalStandardBasis::from_rational(value.basis().clone())
            }
            LocalReductionInner::FiniteField(value) => {
                PyLocalStandardBasis::from_finite_field(value.basis().clone())
            }
        }
    }

    #[getter]
    fn value(&self) -> PyLocalPolynomial {
        match &self.inner {
            LocalReductionInner::Rational(value) => {
                PyLocalPolynomial::from_rational(value.value().clone())
            }
            LocalReductionInner::FiniteField(value) => {
                PyLocalPolynomial::from_finite_field(value.value().clone())
            }
        }
    }

    #[getter]
    fn multiplier(&self) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalReductionInner::Rational(value) => Ok(PyLocalPolynomial::from_rational(
                value.multiplier().map_err(local_error)?,
            )),
            LocalReductionInner::FiniteField(value) => Ok(PyLocalPolynomial::from_finite_field(
                value.multiplier().map_err(local_error)?,
            )),
        }
    }

    #[getter]
    fn quotients(&self) -> PyResult<Vec<PyLocalPolynomial>> {
        match &self.inner {
            LocalReductionInner::Rational(value) => Ok(value
                .quotients()
                .map_err(local_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_rational)
                .collect()),
            LocalReductionInner::FiniteField(value) => Ok(value
                .quotients()
                .map_err(local_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_finite_field)
                .collect()),
        }
    }

    #[getter]
    fn remainder(&self) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalReductionInner::Rational(value) => Ok(PyLocalPolynomial::from_rational(
                value.remainder().map_err(local_error)?,
            )),
            LocalReductionInner::FiniteField(value) => Ok(PyLocalPolynomial::from_finite_field(
                value.remainder().map_err(local_error)?,
            )),
        }
    }

    #[getter]
    fn is_member(&self) -> bool {
        match &self.inner {
            LocalReductionInner::Rational(value) => value.is_member(),
            LocalReductionInner::FiniteField(value) => value.is_member(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            LocalReductionInner::Rational(value) => value.verify().map_err(local_error),
            LocalReductionInner::FiniteField(value) => value.verify().map_err(local_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalReductionInner::Rational(value) => value.to_json().map_err(value_error),
            LocalReductionInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/basis/ring/ambient_ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalReduction::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalReduction::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local reduction coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalReductionInner::Rational(value) => canonical_hash(value),
            LocalReductionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalReductionInner::Rational(value) => value.to_string(),
            LocalReductionInner::FiniteField(value) => value.to_string(),
        })
    }
}

fn serialized_coefficient(value: &str, pointer: &str) -> PyResult<String> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    document
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| PyValueError::new_err("serialized local object has no coefficient identity"))
}
