// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of polynomial quotient rings.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    PolynomialRing, QuotientElement, QuotientError, QuotientRing, QuotientRingMap,
};
use noether::numthy::FiniteField;
use noether::{RationalField, Ring, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::integer_to_python;
use super::number_field::{IdealInner, PyFiniteQuotientAlgebra, PyIdeal};
use super::polynomial_facade::{
    FiniteEmbeddingPolynomialHomomorphism, IdentityPolynomialHomomorphism,
    PolynomialHomomorphismInner, PrimeFinitePolynomialHomomorphism, PyPolynomial,
    PyPolynomialHomomorphism, PyPolynomialRing,
};
use super::series::{PyDegree, PyFreeAbelianGrading, PyHilbertSeries};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn quotient_error(error: QuotientError) -> PyErr {
    match error {
        QuotientError::NotAUnit => PyZeroDivisionError::new_err(error.to_string()),
        _ => value_error(error),
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

#[derive(Clone, PartialEq, Eq)]
pub(super) enum QuotientRingInner {
    Rational(QuotientRing<PolynomialRing<RationalField>>),
    FiniteField(QuotientRing<PolynomialRing<FiniteField>>),
}

impl QuotientRingInner {
    fn canonical_hash(&self) -> isize {
        match self {
            Self::Rational(value) => canonical_hash(value),
            Self::FiniteField(value) => canonical_hash(value),
        }
    }
}

/// An exact polynomial quotient ring with presentation-exact identity.
#[pyclass(
    module = "noether.commalg",
    name = "QuotientRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQuotientRing {
    pub(super) inner: QuotientRingInner,
}

impl PyQuotientRing {
    pub(crate) fn from_rational(inner: QuotientRing<PolynomialRing<RationalField>>) -> Self {
        Self {
            inner: QuotientRingInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: QuotientRing<PolynomialRing<FiniteField>>) -> Self {
        Self {
            inner: QuotientRingInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyQuotientRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Ambient polynomial ring before taking the quotient.
    #[getter]
    fn base_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            QuotientRingInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.base_ring().clone())
            }
            QuotientRingInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.base_ring().clone())
            }
        }
    }

    /// Canonical defining ideal in the ambient polynomial ring.
    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            QuotientRingInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.defining_ideal().clone())
            }
            QuotientRingInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.defining_ideal().clone())
            }
        }
    }

    /// Project one ambient polynomial to its canonical quotient normal form.
    fn element(&self, value: &PyPolynomial) -> PyResult<PyQuotientElement> {
        match &self.inner {
            QuotientRingInner::Rational(ring) => Ok(PyQuotientElement::from_rational(
                ring.element(value.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("quotient element requires a polynomial over Q")
                })?)
                .map_err(value_error)?,
            )),
            QuotientRingInner::FiniteField(ring) => Ok(PyQuotientElement::from_finite_field(
                ring.element(value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "quotient element requires a polynomial over the quotient's finite field",
                    )
                })?)
                .map_err(value_error)?,
            )),
        }
    }

    fn __call__(&self, value: &PyPolynomial) -> PyResult<PyQuotientElement> {
        self.element(value)
    }

    fn zero(&self) -> PyQuotientElement {
        match &self.inner {
            QuotientRingInner::Rational(value) => PyQuotientElement::from_rational(value.zero()),
            QuotientRingInner::FiniteField(value) => {
                PyQuotientElement::from_finite_field(value.zero())
            }
        }
    }

    fn one(&self) -> PyQuotientElement {
        match &self.inner {
            QuotientRingInner::Rational(value) => PyQuotientElement::from_rational(value.one()),
            QuotientRingInner::FiniteField(value) => {
                PyQuotientElement::from_finite_field(value.one())
            }
        }
    }

    #[getter]
    fn is_graded(&self) -> bool {
        match &self.inner {
            QuotientRingInner::Rational(value) => value.is_graded(),
            QuotientRingInner::FiniteField(value) => value.is_graded(),
        }
    }

    #[getter]
    fn grading(&self) -> Option<PyFreeAbelianGrading> {
        match &self.inner {
            QuotientRingInner::Rational(value) => value.grading(),
            QuotientRingInner::FiniteField(value) => value.grading(),
        }
        .cloned()
        .map(|inner| PyFreeAbelianGrading { inner })
    }

    fn hilbert_series(&self) -> PyResult<PyHilbertSeries> {
        let inner = match &self.inner {
            QuotientRingInner::Rational(value) => value.hilbert_series(),
            QuotientRingInner::FiniteField(value) => value.hilbert_series(),
        }
        .map_err(value_error)?;
        Ok(PyHilbertSeries { inner })
    }

    fn hilbert_numerator(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            QuotientRingInner::Rational(value) => value.hilbert_numerator(),
            QuotientRingInner::FiniteField(value) => value.hilbert_numerator(),
        }
        .map_err(value_error)?
        .iter()
        .map(|coefficient| integer_to_python(py, coefficient))
        .collect()
    }

    fn dimension(&self) -> PyResult<i64> {
        match &self.inner {
            QuotientRingInner::Rational(value) => value.dimension(),
            QuotientRingInner::FiniteField(value) => value.dimension(),
        }
        .map_err(value_error)
    }

    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = match &self.inner {
            QuotientRingInner::Rational(value) => value.degree(),
            QuotientRingInner::FiniteField(value) => value.degree(),
        }
        .map_err(value_error)?;
        integer_to_python(py, &value)
    }

    fn hilbert_polynomial(&self) -> PyResult<PyPolynomial> {
        let value = match &self.inner {
            QuotientRingInner::Rational(value) => value.hilbert_polynomial(),
            QuotientRingInner::FiniteField(value) => value.hilbert_polynomial(),
        }
        .map_err(value_error)?;
        Ok(PyPolynomial::from_rational(value))
    }

    fn finite_algebra(&self) -> PyResult<PyFiniteQuotientAlgebra> {
        Ok(match &self.inner {
            QuotientRingInner::Rational(value) => {
                PyFiniteQuotientAlgebra::from_rational(value.finite_algebra().map_err(value_error)?)
            }
            QuotientRingInner::FiniteField(value) => PyFiniteQuotientAlgebra::from_finite_field(
                value.finite_algebra().map_err(value_error)?,
            ),
        })
    }

    /// Exact Kähler differentials over the coefficient field.
    fn kahler_differentials(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::differentials::PyKahlerDifferentials> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact module construction.
        super::differentials::kahler_differentials_from_quotient_ring(py, self)
    }

    /// Cotangent fiber at an explicit coefficient-field point.
    fn cotangent_space_at(
        &self,
        py: Python<'_>,
        point: Vec<Py<PyAny>>,
    ) -> PyResult<super::differentials::PyCotangentSpace> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact fiber construction.
        super::differentials::cotangent_space_from_quotient_ring(py, self, point)
    }

    /// Proved two-term cotangent complex for the canonical equations.
    fn cotangent_complex(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::differentials::PyCotangentComplexPresentation> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact complex construction.
        super::differentials::cotangent_complex_from_quotient_ring(py, self)
    }

    /// Proved two-term cotangent complex for an explicit equation list.
    fn cotangent_complex_from_generators(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPolynomial>>,
    ) -> PyResult<super::differentials::PyCotangentComplexPresentation> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact complex construction.
        super::differentials::cotangent_complex_from_generators(py, self, generators)
    }

    /// Exact conormal sequence for a possibly redundant equation list.
    fn conormal_sequence_from_generators(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPolynomial>>,
    ) -> PyResult<super::differentials::PyConormalSequenceEvidence> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact evidence construction.
        super::differentials::conormal_sequence_from_generators(py, self, generators)
    }

    /// Descend an ambient polynomial map after checking every source relation.
    fn induced_map(
        &self,
        target: &PyQuotientRing,
        base_map: &PyPolynomialHomomorphism,
    ) -> PyResult<PyQuotientRingMap> {
        PyQuotientRingMap::new(self, target, base_map)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            QuotientRingInner::Rational(value) => value.to_json(),
            QuotientRingInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        if let Ok(inner) = QuotientRing::<PolynomialRing<RationalField>>::from_json(value) {
            return Ok(Self::from_rational(inner));
        }
        if let Ok(inner) = QuotientRing::<PolynomialRing<FiniteField>>::from_json(value) {
            return Ok(Self::from_finite_field(inner));
        }
        Err(value_error(
            "serialized quotient ring is not a supported quotient over Q or a finite field",
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "QuotientRing(base_ring={}, generators={})",
            match &self.inner {
                QuotientRingInner::Rational(value) => value.base_ring().to_string(),
                QuotientRingInner::FiniteField(value) => value.base_ring().to_string(),
            },
            match &self.inner {
                QuotientRingInner::Rational(value) => value.defining_ideal().generators().len(),
                QuotientRingInner::FiniteField(value) => {
                    value.defining_ideal().generators().len()
                }
            }
        ))
    }
}

pub(crate) type RationalQuotientRingMap = QuotientRingMap<
    PolynomialRing<RationalField>,
    PolynomialRing<RationalField>,
    IdentityPolynomialHomomorphism<RationalField>,
>;
pub(crate) type FiniteIdentityQuotientRingMap = QuotientRingMap<
    PolynomialRing<FiniteField>,
    PolynomialRing<FiniteField>,
    IdentityPolynomialHomomorphism<FiniteField>,
>;
type PrimeFiniteQuotientRingMap = QuotientRingMap<
    PolynomialRing<FiniteField>,
    PolynomialRing<FiniteField>,
    PrimeFinitePolynomialHomomorphism,
>;
type FiniteEmbeddingQuotientRingMap = QuotientRingMap<
    PolynomialRing<FiniteField>,
    PolynomialRing<FiniteField>,
    FiniteEmbeddingPolynomialHomomorphism,
>;

#[derive(Clone, PartialEq, Eq)]
enum QuotientRingMapInner {
    Rational(RationalQuotientRingMap),
    FiniteIdentity(FiniteIdentityQuotientRingMap),
    PrimeFinite(PrimeFiniteQuotientRingMap),
    FiniteEmbedding(FiniteEmbeddingQuotientRingMap),
}

impl QuotientRingMapInner {
    fn canonical_hash(&self) -> isize {
        match self {
            Self::Rational(value) => canonical_hash(value),
            Self::FiniteIdentity(value) => canonical_hash(value),
            Self::PrimeFinite(value) => canonical_hash(value),
            Self::FiniteEmbedding(value) => canonical_hash(value),
        }
    }
}

/// A checked map between explicitly presented polynomial quotient rings.
#[pyclass(
    module = "noether.commalg",
    name = "QuotientRingMap",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQuotientRingMap {
    inner: QuotientRingMapInner,
}

impl PyQuotientRingMap {
    pub(crate) fn from_rational_identity(inner: RationalQuotientRingMap) -> Self {
        Self {
            inner: QuotientRingMapInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_identity(inner: FiniteIdentityQuotientRingMap) -> Self {
        Self {
            inner: QuotientRingMapInner::FiniteIdentity(inner),
        }
    }

    fn new(
        source: &PyQuotientRing,
        target: &PyQuotientRing,
        base_map: &PyPolynomialHomomorphism,
    ) -> PyResult<Self> {
        let inner = match (&source.inner, &target.inner, &base_map.inner) {
            (
                QuotientRingInner::Rational(source),
                QuotientRingInner::Rational(target),
                PolynomialHomomorphismInner::RationalIdentity(base_map),
            ) => QuotientRingMapInner::Rational(
                source
                    .induced_map(target, base_map.clone())
                    .map_err(value_error)?,
            ),
            (
                QuotientRingInner::FiniteField(source),
                QuotientRingInner::FiniteField(target),
                PolynomialHomomorphismInner::FiniteIdentity(base_map),
            ) => QuotientRingMapInner::FiniteIdentity(
                source
                    .induced_map(target, base_map.clone())
                    .map_err(value_error)?,
            ),
            (
                QuotientRingInner::FiniteField(source),
                QuotientRingInner::FiniteField(target),
                PolynomialHomomorphismInner::PrimeFinitePolynomial(base_map),
            ) => QuotientRingMapInner::PrimeFinite(
                source
                    .induced_map(target, base_map.clone())
                    .map_err(value_error)?,
            ),
            (
                QuotientRingInner::FiniteField(source),
                QuotientRingInner::FiniteField(target),
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(base_map),
            ) => QuotientRingMapInner::FiniteEmbedding(
                source
                    .induced_map(target, base_map.clone())
                    .map_err(value_error)?,
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "quotient map requires an ambient polynomial map with the quotient base rings as exact endpoints",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn from_supported_json(value: &str) -> PyResult<Self> {
        if let Ok(inner) = RationalQuotientRingMap::from_json(value) {
            return Ok(Self {
                inner: QuotientRingMapInner::Rational(inner),
            });
        }
        if let Ok(inner) = FiniteIdentityQuotientRingMap::from_json(value) {
            return Ok(Self {
                inner: QuotientRingMapInner::FiniteIdentity(inner),
            });
        }
        if let Ok(inner) = PrimeFiniteQuotientRingMap::from_json(value) {
            return Ok(Self {
                inner: QuotientRingMapInner::PrimeFinite(inner),
            });
        }
        if let Ok(inner) = FiniteEmbeddingQuotientRingMap::from_json(value) {
            return Ok(Self {
                inner: QuotientRingMapInner::FiniteEmbedding(inner),
            });
        }
        Err(value_error(
            "serialized quotient map is not supported over Q or finite fields",
        ))
    }
}

#[pymethods]
impl PyQuotientRingMap {
    #[classattr]
    fn type_id() -> &'static str {
        RationalQuotientRingMap::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyQuotientRing {
        match &self.inner {
            QuotientRingMapInner::Rational(value) => {
                PyQuotientRing::from_rational(value.source().clone())
            }
            QuotientRingMapInner::FiniteIdentity(value) => {
                PyQuotientRing::from_finite_field(value.source().clone())
            }
            QuotientRingMapInner::PrimeFinite(value) => {
                PyQuotientRing::from_finite_field(value.source().clone())
            }
            QuotientRingMapInner::FiniteEmbedding(value) => {
                PyQuotientRing::from_finite_field(value.source().clone())
            }
        }
    }

    #[getter]
    fn target(&self) -> PyQuotientRing {
        match &self.inner {
            QuotientRingMapInner::Rational(value) => {
                PyQuotientRing::from_rational(value.target().clone())
            }
            QuotientRingMapInner::FiniteIdentity(value) => {
                PyQuotientRing::from_finite_field(value.target().clone())
            }
            QuotientRingMapInner::PrimeFinite(value) => {
                PyQuotientRing::from_finite_field(value.target().clone())
            }
            QuotientRingMapInner::FiniteEmbedding(value) => {
                PyQuotientRing::from_finite_field(value.target().clone())
            }
        }
    }

    #[getter]
    fn base_map(&self) -> PyPolynomialHomomorphism {
        let inner = match &self.inner {
            QuotientRingMapInner::Rational(value) => {
                PolynomialHomomorphismInner::RationalIdentity(value.base_map().clone())
            }
            QuotientRingMapInner::FiniteIdentity(value) => {
                PolynomialHomomorphismInner::FiniteIdentity(value.base_map().clone())
            }
            QuotientRingMapInner::PrimeFinite(value) => {
                PolynomialHomomorphismInner::PrimeFinitePolynomial(value.base_map().clone())
            }
            QuotientRingMapInner::FiniteEmbedding(value) => {
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value.base_map().clone())
            }
        };
        PyPolynomialHomomorphism { inner }
    }

    fn apply(&self, value: &PyQuotientElement) -> PyResult<PyQuotientElement> {
        match (&self.inner, &value.inner) {
            (QuotientRingMapInner::Rational(map), QuotientElementInner::Rational(value)) => map
                .apply(value)
                .map(PyQuotientElement::from_rational)
                .map_err(value_error),
            (
                QuotientRingMapInner::FiniteIdentity(map),
                QuotientElementInner::FiniteField(value),
            ) => map
                .apply(value)
                .map(PyQuotientElement::from_finite_field)
                .map_err(value_error),
            (QuotientRingMapInner::PrimeFinite(map), QuotientElementInner::FiniteField(value)) => {
                map.apply(value)
                    .map(PyQuotientElement::from_finite_field)
                    .map_err(value_error)
            }
            (
                QuotientRingMapInner::FiniteEmbedding(map),
                QuotientElementInner::FiniteField(value),
            ) => map
                .apply(value)
                .map(PyQuotientElement::from_finite_field)
                .map_err(value_error),
            _ => Err(PyTypeError::new_err(
                "quotient element does not belong to this quotient map's source context",
            )),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            QuotientRingMapInner::Rational(value) => value.verify(),
            QuotientRingMapInner::FiniteIdentity(value) => value.verify(),
            QuotientRingMapInner::PrimeFinite(value) => value.verify(),
            QuotientRingMapInner::FiniteEmbedding(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            QuotientRingMapInner::Rational(value) => value.to_json(),
            QuotientRingMapInner::FiniteIdentity(value) => value.to_json(),
            QuotientRingMapInner::PrimeFinite(value) => value.to_json(),
            QuotientRingMapInner::FiniteEmbedding(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Self::from_supported_json(value)
    }

    fn __call__(&self, value: &PyQuotientElement) -> PyResult<PyQuotientElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "QuotientRingMap(source={}, target={})",
            self.source().__repr__(),
            self.target().__repr__()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum QuotientElementInner {
    Rational(QuotientElement<PolynomialRing<RationalField>>),
    FiniteField(QuotientElement<PolynomialRing<FiniteField>>),
}

impl QuotientElementInner {
    fn canonical_hash(&self) -> isize {
        match self {
            Self::Rational(value) => canonical_hash(value),
            Self::FiniteField(value) => canonical_hash(value),
        }
    }
}

/// An exact quotient element retaining its canonical ambient lift.
#[pyclass(
    module = "noether.commalg",
    name = "QuotientElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQuotientElement {
    pub(super) inner: QuotientElementInner,
}

impl PyQuotientElement {
    pub(crate) fn from_rational(inner: QuotientElement<PolynomialRing<RationalField>>) -> Self {
        Self {
            inner: QuotientElementInner::Rational(inner),
        }
    }

    pub(crate) fn from_finite_field(inner: QuotientElement<PolynomialRing<FiniteField>>) -> Self {
        Self {
            inner: QuotientElementInner::FiniteField(inner),
        }
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        match (&self.inner, &other.inner) {
            (QuotientElementInner::Rational(left), QuotientElementInner::Rational(right)) => {
                let inner = match operation {
                    "add" => left + right,
                    "sub" => left - right,
                    "mul" => left * right,
                    _ => return Err(value_error("unknown quotient operation")),
                }
                .map_err(value_error)?;
                Ok(Self::from_rational(inner))
            }
            (QuotientElementInner::FiniteField(left), QuotientElementInner::FiniteField(right)) => {
                let inner = match operation {
                    "add" => left + right,
                    "sub" => left - right,
                    "mul" => left * right,
                    _ => return Err(value_error("unknown quotient operation")),
                }
                .map_err(value_error)?;
                Ok(Self::from_finite_field(inner))
            }
            _ => Err(PyTypeError::new_err(
                "quotient elements belong to incompatible coefficient contexts",
            )),
        }
    }
}

#[pymethods]
impl PyQuotientElement {
    #[classattr]
    fn type_id() -> &'static str {
        QuotientElement::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn ring(&self) -> PyQuotientRing {
        match &self.inner {
            QuotientElementInner::Rational(value) => {
                PyQuotientRing::from_rational(value.ring().clone())
            }
            QuotientElementInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn lift(&self) -> PyPolynomial {
        match &self.inner {
            QuotientElementInner::Rational(value) => {
                PyPolynomial::from_rational(value.lift().clone())
            }
            QuotientElementInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.lift().clone())
            }
        }
    }

    fn is_zero(&self) -> bool {
        match &self.inner {
            QuotientElementInner::Rational(value) => value.is_zero(),
            QuotientElementInner::FiniteField(value) => value.is_zero(),
        }
    }

    fn is_one(&self) -> bool {
        match &self.inner {
            QuotientElementInner::Rational(value) => value.is_one(),
            QuotientElementInner::FiniteField(value) => value.is_one(),
        }
    }

    fn multihomogeneous_degree(&self) -> PyResult<Option<PyDegree>> {
        match &self.inner {
            QuotientElementInner::Rational(value) => value.multihomogeneous_degree(),
            QuotientElementInner::FiniteField(value) => value.multihomogeneous_degree(),
        }
        .map_err(quotient_error)
        .map(|degree| degree.map(|inner| PyDegree { inner }))
    }

    fn is_unit(&self) -> PyResult<bool> {
        match &self.inner {
            QuotientElementInner::Rational(value) => value.is_unit(),
            QuotientElementInner::FiniteField(value) => value.is_unit(),
        }
        .map_err(quotient_error)
    }

    fn inverse(&self) -> PyResult<Self> {
        match &self.inner {
            QuotientElementInner::Rational(value) => value
                .inverse()
                .map(Self::from_rational)
                .map_err(quotient_error),
            QuotientElementInner::FiniteField(value) => value
                .inverse()
                .map(Self::from_finite_field)
                .map_err(quotient_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            QuotientElementInner::Rational(value) => value.to_json(),
            QuotientElementInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        if let Ok(inner) = QuotientElement::<PolynomialRing<RationalField>>::from_json(value) {
            return Ok(Self::from_rational(inner));
        }
        if let Ok(inner) = QuotientElement::<PolynomialRing<FiniteField>>::from_json(value) {
            return Ok(Self::from_finite_field(inner));
        }
        Err(value_error(
            "serialized quotient element is not supported over Q or a finite field",
        ))
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
            QuotientElementInner::Rational(value) => {
                Ok(Self::from_rational((-value).map_err(value_error)?))
            }
            QuotientElementInner::FiniteField(value) => {
                Ok(Self::from_finite_field((-value).map_err(value_error)?))
            }
        }
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "QuotientElement power does not accept Python's third modulus argument",
            ));
        }
        Ok(match &self.inner {
            QuotientElementInner::Rational(value) => Self::from_rational(value.pow(exponent)),
            QuotientElementInner::FiniteField(value) => {
                Self::from_finite_field(value.pow(exponent))
            }
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "QuotientElement(nonzero_terms={})",
            self.lift().term_count()
        ))
    }
}

pub(crate) fn quotient_from_ideal(
    ring: &PyPolynomialRing,
    ideal: &PyIdeal,
) -> PyResult<PyQuotientRing> {
    match (&ring.inner, &ideal.inner) {
        (
            super::polynomial_facade::PolynomialRingInner::Rational(ring),
            IdealInner::RationalPolynomial(ideal),
        ) => Ok(PyQuotientRing::from_rational(
            ring.quotient(ideal).map_err(value_error)?,
        )),
        (
            super::polynomial_facade::PolynomialRingInner::FiniteField(ring),
            IdealInner::FiniteFieldPolynomial(ideal),
        ) => Ok(PyQuotientRing::from_finite_field(
            ring.quotient(ideal).map_err(value_error)?,
        )),
        _ => Err(PyTypeError::new_err(
            "quotient rings currently require a polynomial ring and ideal over Q or a finite field",
        )),
    }
}
