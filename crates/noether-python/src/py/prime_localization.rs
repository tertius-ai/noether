// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of polynomial prime localizations.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    PolynomialPrimeLocalElement, PolynomialPrimeLocalIdeal, PolynomialPrimeLocalMembership,
    PolynomialPrimeLocalization, PrimeLocalizationError,
};
use noether::numthy::FiniteField;
use noether::{RationalField, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::number_field::{PyIdeal, PyPolynomialPrimeCertificate};
use super::polynomial_facade::PyPolynomial;

fn localization_error(error: PrimeLocalizationError) -> PyErr {
    match error {
        PrimeLocalizationError::ContextMismatch(error) => super::errors::context_mismatch(error),
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

#[derive(Clone, PartialEq, Eq)]
enum PrimeLocalizationInner {
    Rational(PolynomialPrimeLocalization<RationalField>),
    FiniteField(PolynomialPrimeLocalization<FiniteField>),
}

/// A polynomial ring localized at one certified prime ideal.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimeLocalization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimeLocalization {
    inner: PrimeLocalizationInner,
}

impl PyPolynomialPrimeLocalization {
    fn from_rational(inner: PolynomialPrimeLocalization<RationalField>) -> Self {
        Self {
            inner: PrimeLocalizationInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: PolynomialPrimeLocalization<FiniteField>) -> Self {
        Self {
            inner: PrimeLocalizationInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyPolynomialPrimeLocalization {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialPrimeLocalization::<RationalField>::TYPE_ID
    }

    #[new]
    fn new(prime: &PyPolynomialPrimeCertificate) -> PyResult<Self> {
        if let Some(prime) = prime.as_rational() {
            return Ok(Self::from_rational(PolynomialPrimeLocalization::new(
                prime.clone(),
            )));
        }
        if let Some(prime) = prime.as_finite_field() {
            return Ok(Self::from_finite_field(PolynomialPrimeLocalization::new(
                prime.clone(),
            )));
        }
        Err(PyTypeError::new_err(
            "unsupported polynomial-prime coefficient context",
        ))
    }

    #[getter]
    fn prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.prime().clone())
            }
            PrimeLocalizationInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.prime().clone())
            }
        }
    }

    fn from_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomialPrimeLocalElement> {
        match &self.inner {
            PrimeLocalizationInner::Rational(ring) => {
                Ok(PyPolynomialPrimeLocalElement::from_rational(
                    ring.from_polynomial(
                        value
                            .as_rational()
                            .ok_or_else(|| PyTypeError::new_err("polynomial must be over Q"))?
                            .clone(),
                    )
                    .map_err(localization_error)?,
                ))
            }
            PrimeLocalizationInner::FiniteField(ring) => {
                Ok(PyPolynomialPrimeLocalElement::from_finite_field(
                    ring.from_polynomial(
                        value
                            .as_finite_field()
                            .ok_or_else(|| {
                                PyTypeError::new_err(
                                    "polynomial must be over the localization finite field",
                                )
                            })?
                            .clone(),
                    )
                    .map_err(localization_error)?,
                ))
            }
        }
    }

    fn fraction(
        &self,
        numerator: &PyPolynomial,
        denominator: &PyPolynomial,
    ) -> PyResult<PyPolynomialPrimeLocalElement> {
        match &self.inner {
            PrimeLocalizationInner::Rational(ring) => {
                Ok(PyPolynomialPrimeLocalElement::from_rational(
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
                    .map_err(localization_error)?,
                ))
            }
            PrimeLocalizationInner::FiniteField(ring) => {
                Ok(PyPolynomialPrimeLocalElement::from_finite_field(
                    ring.fraction(
                        numerator
                            .as_finite_field()
                            .ok_or_else(|| {
                                PyTypeError::new_err(
                                    "numerator must be over the localization finite field",
                                )
                            })?
                            .clone(),
                        denominator
                            .as_finite_field()
                            .ok_or_else(|| {
                                PyTypeError::new_err(
                                    "denominator must be over the localization finite field",
                                )
                            })?
                            .clone(),
                    )
                    .map_err(localization_error)?,
                ))
            }
        }
    }

    fn zero(&self) -> PyPolynomialPrimeLocalElement {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => {
                PyPolynomialPrimeLocalElement::from_rational(value.zero())
            }
            PrimeLocalizationInner::FiniteField(value) => {
                PyPolynomialPrimeLocalElement::from_finite_field(value.zero())
            }
        }
    }

    fn one(&self) -> PyPolynomialPrimeLocalElement {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => {
                PyPolynomialPrimeLocalElement::from_rational(value.one())
            }
            PrimeLocalizationInner::FiniteField(value) => {
                PyPolynomialPrimeLocalElement::from_finite_field(value.one())
            }
        }
    }

    fn generators(&self) -> PyResult<Vec<PyPolynomialPrimeLocalElement>> {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => Ok(value
                .generators()
                .map_err(localization_error)?
                .into_iter()
                .map(PyPolynomialPrimeLocalElement::from_rational)
                .collect()),
            PrimeLocalizationInner::FiniteField(value) => Ok(value
                .generators()
                .map_err(localization_error)?
                .into_iter()
                .map(PyPolynomialPrimeLocalElement::from_finite_field)
                .collect()),
        }
    }

    fn ideal(&self, generators: &Bound<'_, PyAny>) -> PyResult<PyPolynomialPrimeLocalIdeal> {
        match &self.inner {
            PrimeLocalizationInner::Rational(ring) => {
                let generators = generators
                    .try_iter()?
                    .map(|value| {
                        let value = value?.extract::<PyRef<'_, PyPolynomialPrimeLocalElement>>()?;
                        match &value.inner {
                            PrimeLocalElementInner::Rational(value) => Ok(value.clone()),
                            PrimeLocalElementInner::FiniteField(_) => Err(PyTypeError::new_err(
                                "ideal generators must belong to the rational prime-local ring",
                            )),
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyPolynomialPrimeLocalIdeal::from_rational(
                    ring.ideal(&generators).map_err(localization_error)?,
                ))
            }
            PrimeLocalizationInner::FiniteField(ring) => {
                let generators = generators
                    .try_iter()?
                    .map(|value| {
                        let value = value?.extract::<PyRef<'_, PyPolynomialPrimeLocalElement>>()?;
                        match &value.inner {
                            PrimeLocalElementInner::Rational(_) => Err(PyTypeError::new_err(
                                "ideal generators must belong to the finite-field prime-local ring",
                            )),
                            PrimeLocalElementInner::FiniteField(value) => Ok(value.clone()),
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyPolynomialPrimeLocalIdeal::from_finite_field(
                    ring.ideal(&generators).map_err(localization_error)?,
                ))
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => value.to_json().map_err(value_error),
            PrimeLocalizationInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/prime/ideal/ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                PolynomialPrimeLocalization::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                PolynomialPrimeLocalization::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported polynomial prime-local coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PrimeLocalizationInner::Rational(value) => canonical_hash(value),
            PrimeLocalizationInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            PrimeLocalizationInner::Rational(value) => value.to_string(),
            PrimeLocalizationInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PrimeLocalElementInner {
    Rational(PolynomialPrimeLocalElement<RationalField>),
    FiniteField(PolynomialPrimeLocalElement<FiniteField>),
}

/// One exact fraction in a polynomial prime localization.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimeLocalElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimeLocalElement {
    inner: PrimeLocalElementInner,
}

impl PyPolynomialPrimeLocalElement {
    fn from_rational(inner: PolynomialPrimeLocalElement<RationalField>) -> Self {
        Self {
            inner: PrimeLocalElementInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: PolynomialPrimeLocalElement<FiniteField>) -> Self {
        Self {
            inner: PrimeLocalElementInner::FiniteField(inner),
        }
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        match (&self.inner, &other.inner, operation) {
            (
                PrimeLocalElementInner::Rational(left),
                PrimeLocalElementInner::Rational(right),
                "add",
            ) => Ok(Self::from_rational(
                (left + right).map_err(localization_error)?,
            )),
            (
                PrimeLocalElementInner::Rational(left),
                PrimeLocalElementInner::Rational(right),
                "sub",
            ) => Ok(Self::from_rational(
                (left - right).map_err(localization_error)?,
            )),
            (
                PrimeLocalElementInner::Rational(left),
                PrimeLocalElementInner::Rational(right),
                "mul",
            ) => Ok(Self::from_rational(
                (left * right).map_err(localization_error)?,
            )),
            (
                PrimeLocalElementInner::FiniteField(left),
                PrimeLocalElementInner::FiniteField(right),
                "add",
            ) => Ok(Self::from_finite_field(
                (left + right).map_err(localization_error)?,
            )),
            (
                PrimeLocalElementInner::FiniteField(left),
                PrimeLocalElementInner::FiniteField(right),
                "sub",
            ) => Ok(Self::from_finite_field(
                (left - right).map_err(localization_error)?,
            )),
            (
                PrimeLocalElementInner::FiniteField(left),
                PrimeLocalElementInner::FiniteField(right),
                "mul",
            ) => Ok(Self::from_finite_field(
                (left * right).map_err(localization_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "prime-local operands must belong to the same exact context",
            )),
        }
    }
}

#[pymethods]
impl PyPolynomialPrimeLocalElement {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialPrimeLocalElement::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyPolynomialPrimeLocalization {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => {
                PyPolynomialPrimeLocalization::from_rational(value.ring().clone())
            }
            PrimeLocalElementInner::FiniteField(value) => {
                PyPolynomialPrimeLocalization::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn numerator(&self) -> PyPolynomial {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => {
                PyPolynomial::from_rational(value.numerator().clone())
            }
            PrimeLocalElementInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.numerator().clone())
            }
        }
    }

    #[getter]
    fn denominator(&self) -> PyPolynomial {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => {
                PyPolynomial::from_rational(value.denominator().clone())
            }
            PrimeLocalElementInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.denominator().clone())
            }
        }
    }

    #[getter]
    fn is_zero(&self) -> bool {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => value.is_zero(),
            PrimeLocalElementInner::FiniteField(value) => value.is_zero(),
        }
    }

    #[getter]
    fn is_unit(&self) -> PyResult<bool> {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => value.is_unit().map_err(localization_error),
            PrimeLocalElementInner::FiniteField(value) => {
                value.is_unit().map_err(localization_error)
            }
        }
    }

    fn inverse(&self) -> PyResult<Self> {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => Ok(Self::from_rational(
                value.inverse().map_err(localization_error)?,
            )),
            PrimeLocalElementInner::FiniteField(value) => Ok(Self::from_finite_field(
                value.inverse().map_err(localization_error)?,
            )),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => value.to_json().map_err(value_error),
            PrimeLocalElementInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ring/prime/ideal/ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                PolynomialPrimeLocalElement::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                PolynomialPrimeLocalElement::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported polynomial prime-local coefficient context {coefficient}"
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
            PrimeLocalElementInner::Rational(value) => {
                Ok(Self::from_rational((-value).map_err(localization_error)?))
            }
            PrimeLocalElementInner::FiniteField(value) => Ok(Self::from_finite_field(
                (-value).map_err(localization_error)?,
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => canonical_hash(value),
            PrimeLocalElementInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            PrimeLocalElementInner::Rational(value) => value.to_string(),
            PrimeLocalElementInner::FiniteField(value) => value.to_string(),
        })
    }

    fn __str__(&self) -> String {
        match &self.inner {
            PrimeLocalElementInner::Rational(value) => value.to_string(),
            PrimeLocalElementInner::FiniteField(value) => value.to_string(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PrimeLocalIdealInner {
    Rational(PolynomialPrimeLocalIdeal<RationalField>),
    FiniteField(PolynomialPrimeLocalIdeal<FiniteField>),
}

/// A canonical finitely generated ideal in a polynomial prime localization.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimeLocalIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimeLocalIdeal {
    inner: PrimeLocalIdealInner,
}

impl PyPolynomialPrimeLocalIdeal {
    fn from_rational(inner: PolynomialPrimeLocalIdeal<RationalField>) -> Self {
        Self {
            inner: PrimeLocalIdealInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: PolynomialPrimeLocalIdeal<FiniteField>) -> Self {
        Self {
            inner: PrimeLocalIdealInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyPolynomialPrimeLocalIdeal {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialPrimeLocalIdeal::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> PyPolynomialPrimeLocalization {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => {
                PyPolynomialPrimeLocalization::from_rational(value.ring().clone())
            }
            PrimeLocalIdealInner::FiniteField(value) => {
                PyPolynomialPrimeLocalization::from_finite_field(value.ring().clone())
            }
        }
    }

    #[getter]
    fn ambient_ideal(&self) -> PyIdeal {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ambient_ideal().clone())
            }
            PrimeLocalIdealInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ambient_ideal().clone())
            }
        }
    }

    #[getter]
    fn generators(&self) -> PyResult<Vec<PyPolynomialPrimeLocalElement>> {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => Ok(value
                .generators()
                .map_err(localization_error)?
                .into_iter()
                .map(PyPolynomialPrimeLocalElement::from_rational)
                .collect()),
            PrimeLocalIdealInner::FiniteField(value) => Ok(value
                .generators()
                .map_err(localization_error)?
                .into_iter()
                .map(PyPolynomialPrimeLocalElement::from_finite_field)
                .collect()),
        }
    }

    fn certify_membership(
        &self,
        value: &PyPolynomialPrimeLocalElement,
    ) -> PyResult<PyPolynomialPrimeLocalMembership> {
        match (&self.inner, &value.inner) {
            (PrimeLocalIdealInner::Rational(ideal), PrimeLocalElementInner::Rational(value)) => {
                Ok(PyPolynomialPrimeLocalMembership::from_rational(
                    ideal
                        .certify_membership(value)
                        .map_err(localization_error)?,
                ))
            }
            (
                PrimeLocalIdealInner::FiniteField(ideal),
                PrimeLocalElementInner::FiniteField(value),
            ) => Ok(PyPolynomialPrimeLocalMembership::from_finite_field(
                ideal
                    .certify_membership(value)
                    .map_err(localization_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "membership value must belong to the ideal's exact prime-local context",
            )),
        }
    }

    fn contains(&self, value: &PyPolynomialPrimeLocalElement) -> PyResult<bool> {
        Ok(self.certify_membership(value)?.is_member())
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => value.verify().map_err(localization_error),
            PrimeLocalIdealInner::FiniteField(value) => value.verify().map_err(localization_error),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => value.to_json().map_err(value_error),
            PrimeLocalIdealInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ring/prime/ideal/ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                PolynomialPrimeLocalIdeal::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                PolynomialPrimeLocalIdeal::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported polynomial prime-local ideal coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PrimeLocalIdealInner::Rational(value) => canonical_hash(value),
            PrimeLocalIdealInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            PrimeLocalIdealInner::Rational(value) => value.to_string(),
            PrimeLocalIdealInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PrimeLocalMembershipInner {
    Rational(Box<PolynomialPrimeLocalMembership<RationalField>>),
    FiniteField(Box<PolynomialPrimeLocalMembership<FiniteField>>),
}

/// Replayable colon-witness evidence for prime-local ideal membership.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimeLocalMembership",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimeLocalMembership {
    inner: PrimeLocalMembershipInner,
}

impl PyPolynomialPrimeLocalMembership {
    fn from_rational(inner: PolynomialPrimeLocalMembership<RationalField>) -> Self {
        Self {
            inner: PrimeLocalMembershipInner::Rational(Box::new(inner)),
        }
    }

    fn from_finite_field(inner: PolynomialPrimeLocalMembership<FiniteField>) -> Self {
        Self {
            inner: PrimeLocalMembershipInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyPolynomialPrimeLocalMembership {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialPrimeLocalMembership::<RationalField>::TYPE_ID
    }

    #[getter]
    fn ideal(&self) -> PyPolynomialPrimeLocalIdeal {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => {
                PyPolynomialPrimeLocalIdeal::from_rational(value.ideal().clone())
            }
            PrimeLocalMembershipInner::FiniteField(value) => {
                PyPolynomialPrimeLocalIdeal::from_finite_field(value.ideal().clone())
            }
        }
    }

    #[getter]
    fn value(&self) -> PyPolynomialPrimeLocalElement {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => {
                PyPolynomialPrimeLocalElement::from_rational(value.value().clone())
            }
            PrimeLocalMembershipInner::FiniteField(value) => {
                PyPolynomialPrimeLocalElement::from_finite_field(value.value().clone())
            }
        }
    }

    #[getter]
    fn colon_ideal(&self) -> PyIdeal {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.colon_ideal().clone())
            }
            PrimeLocalMembershipInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.colon_ideal().clone())
            }
        }
    }

    #[getter]
    fn multiplier(&self) -> Option<PyPolynomialPrimeLocalElement> {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => value
                .multiplier()
                .cloned()
                .map(PyPolynomialPrimeLocalElement::from_rational),
            PrimeLocalMembershipInner::FiniteField(value) => value
                .multiplier()
                .cloned()
                .map(PyPolynomialPrimeLocalElement::from_finite_field),
        }
    }

    #[getter]
    fn quotients(&self) -> Vec<PyPolynomialPrimeLocalElement> {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => value
                .quotients()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeLocalElement::from_rational)
                .collect(),
            PrimeLocalMembershipInner::FiniteField(value) => value
                .quotients()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeLocalElement::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn is_member(&self) -> bool {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => value.is_member(),
            PrimeLocalMembershipInner::FiniteField(value) => value.is_member(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => {
                value.verify().map_err(localization_error)
            }
            PrimeLocalMembershipInner::FiniteField(value) => {
                value.verify().map_err(localization_error)
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => value.to_json().map_err(value_error),
            PrimeLocalMembershipInner::FiniteField(value) => value.to_json().map_err(value_error),
        }
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ideal/ring/prime/ideal/ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                PolynomialPrimeLocalMembership::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                PolynomialPrimeLocalMembership::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported polynomial prime-local membership coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => canonical_hash(value.as_ref()),
            PrimeLocalMembershipInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            PrimeLocalMembershipInner::Rational(value) => value.to_string(),
            PrimeLocalMembershipInner::FiniteField(value) => value.to_string(),
        })
    }
}

fn serialized_coefficient(value: &str, pointer: &str) -> PyResult<String> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    document
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PyValueError::new_err("serialized prime-local object has no coefficient identity")
        })
}
