// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of canonical monomial ideals.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{MonomialIdeal, MonomialIdealError, RationalFunctionField, ResidueRing};
use noether::numthy::{
    FiniteField, NumberField, PadicField, PadicLocalField, PadicRing, QadicField, QadicRing,
};
use noether::{Integer, IntegerRing, RationalField, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};

use super::facade_scalar::{extract_integer, integer_to_python};
use super::number_field::PyIdeal;
use super::polynomial_facade::{PolynomialRingInner, PyPolynomial, PyPolynomialRing};
use super::series::{PyDegree, PyHilbertSeries};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn monomial_error(error: MonomialIdealError) -> PyErr {
    match error {
        MonomialIdealError::ContextMismatch(error) => super::errors::context_mismatch(error),
        error => value_error(error),
    }
}

#[derive(Clone)]
pub(super) enum MonomialIdealInner {
    Integer(MonomialIdeal<IntegerRing>),
    Rational(MonomialIdeal<RationalField>),
    Residue(MonomialIdeal<ResidueRing>),
    FiniteField(MonomialIdeal<FiniteField>),
    Padic(MonomialIdeal<PadicRing>),
    PadicField(MonomialIdeal<PadicField>),
    Qadic(MonomialIdeal<QadicRing>),
    QadicField(MonomialIdeal<QadicField>),
    PadicLocalField(MonomialIdeal<PadicLocalField>),
    NumberField(MonomialIdeal<NumberField>),
    RationalFunction(MonomialIdeal<RationalFunctionField<RationalField>>),
    FiniteRationalFunction(MonomialIdeal<RationalFunctionField<FiniteField>>),
}

macro_rules! map_monomial {
    ($value:expr, |$inner:ident| $body:expr) => {
        match $value {
            MonomialIdealInner::Integer($inner) => $body,
            MonomialIdealInner::Rational($inner) => $body,
            MonomialIdealInner::Residue($inner) => $body,
            MonomialIdealInner::FiniteField($inner) => $body,
            MonomialIdealInner::Padic($inner) => $body,
            MonomialIdealInner::PadicField($inner) => $body,
            MonomialIdealInner::Qadic($inner) => $body,
            MonomialIdealInner::QadicField($inner) => $body,
            MonomialIdealInner::PadicLocalField($inner) => $body,
            MonomialIdealInner::NumberField($inner) => $body,
            MonomialIdealInner::RationalFunction($inner) => $body,
            MonomialIdealInner::FiniteRationalFunction($inner) => $body,
        }
    };
}

macro_rules! unary_operation {
    ($value:expr, $method:ident $(, $argument:expr)*) => {
        match $value {
            MonomialIdealInner::Integer(value) => value.$method($($argument),*).map(MonomialIdealInner::Integer),
            MonomialIdealInner::Rational(value) => value.$method($($argument),*).map(MonomialIdealInner::Rational),
            MonomialIdealInner::Residue(value) => value.$method($($argument),*).map(MonomialIdealInner::Residue),
            MonomialIdealInner::FiniteField(value) => value.$method($($argument),*).map(MonomialIdealInner::FiniteField),
            MonomialIdealInner::Padic(value) => value.$method($($argument),*).map(MonomialIdealInner::Padic),
            MonomialIdealInner::PadicField(value) => value.$method($($argument),*).map(MonomialIdealInner::PadicField),
            MonomialIdealInner::Qadic(value) => value.$method($($argument),*).map(MonomialIdealInner::Qadic),
            MonomialIdealInner::QadicField(value) => value.$method($($argument),*).map(MonomialIdealInner::QadicField),
            MonomialIdealInner::PadicLocalField(value) => value.$method($($argument),*).map(MonomialIdealInner::PadicLocalField),
            MonomialIdealInner::NumberField(value) => value.$method($($argument),*).map(MonomialIdealInner::NumberField),
            MonomialIdealInner::RationalFunction(value) => value.$method($($argument),*).map(MonomialIdealInner::RationalFunction),
            MonomialIdealInner::FiniteRationalFunction(value) => value.$method($($argument),*).map(MonomialIdealInner::FiniteRationalFunction),
        }
    };
}

macro_rules! binary_operation {
    ($left:expr, $right:expr, $method:ident) => {
        match ($left, $right) {
            (MonomialIdealInner::Integer(left), MonomialIdealInner::Integer(right)) => {
                left.$method(right).map(MonomialIdealInner::Integer)
            }
            (MonomialIdealInner::Rational(left), MonomialIdealInner::Rational(right)) => {
                left.$method(right).map(MonomialIdealInner::Rational)
            }
            (MonomialIdealInner::Residue(left), MonomialIdealInner::Residue(right)) => {
                left.$method(right).map(MonomialIdealInner::Residue)
            }
            (MonomialIdealInner::FiniteField(left), MonomialIdealInner::FiniteField(right)) => {
                left.$method(right).map(MonomialIdealInner::FiniteField)
            }
            (MonomialIdealInner::Padic(left), MonomialIdealInner::Padic(right)) => {
                left.$method(right).map(MonomialIdealInner::Padic)
            }
            (MonomialIdealInner::PadicField(left), MonomialIdealInner::PadicField(right)) => {
                left.$method(right).map(MonomialIdealInner::PadicField)
            }
            (MonomialIdealInner::Qadic(left), MonomialIdealInner::Qadic(right)) => {
                left.$method(right).map(MonomialIdealInner::Qadic)
            }
            (MonomialIdealInner::QadicField(left), MonomialIdealInner::QadicField(right)) => {
                left.$method(right).map(MonomialIdealInner::QadicField)
            }
            (
                MonomialIdealInner::PadicLocalField(left),
                MonomialIdealInner::PadicLocalField(right),
            ) => left.$method(right).map(MonomialIdealInner::PadicLocalField),
            (MonomialIdealInner::NumberField(left), MonomialIdealInner::NumberField(right)) => {
                left.$method(right).map(MonomialIdealInner::NumberField)
            }
            (
                MonomialIdealInner::RationalFunction(left),
                MonomialIdealInner::RationalFunction(right),
            ) => left
                .$method(right)
                .map(MonomialIdealInner::RationalFunction),
            (
                MonomialIdealInner::FiniteRationalFunction(left),
                MonomialIdealInner::FiniteRationalFunction(right),
            ) => left
                .$method(right)
                .map(MonomialIdealInner::FiniteRationalFunction),
            _ => {
                return Err(value_error(
                    "monomial ideals have different polynomial contexts",
                ))
            }
        }
    };
}

impl MonomialIdealInner {
    pub(super) fn from_ring(
        ring: &PolynomialRingInner,
        generators: Vec<Vec<Integer>>,
    ) -> PyResult<Self> {
        match ring {
            PolynomialRingInner::Integer(ring) => ring
                .monomial_ideal(generators)
                .map(Self::Integer)
                .map_err(monomial_error),
            PolynomialRingInner::Rational(ring) => ring
                .monomial_ideal(generators)
                .map(Self::Rational)
                .map_err(monomial_error),
            PolynomialRingInner::Residue(ring) => ring
                .monomial_ideal(generators)
                .map(Self::Residue)
                .map_err(monomial_error),
            PolynomialRingInner::FiniteField(ring) => ring
                .monomial_ideal(generators)
                .map(Self::FiniteField)
                .map_err(monomial_error),
            PolynomialRingInner::Padic(ring) => ring
                .monomial_ideal(generators)
                .map(Self::Padic)
                .map_err(monomial_error),
            PolynomialRingInner::PadicField(ring) => ring
                .monomial_ideal(generators)
                .map(Self::PadicField)
                .map_err(monomial_error),
            PolynomialRingInner::Qadic(ring) => ring
                .monomial_ideal(generators)
                .map(Self::Qadic)
                .map_err(monomial_error),
            PolynomialRingInner::QadicField(ring) => ring
                .monomial_ideal(generators)
                .map(Self::QadicField)
                .map_err(monomial_error),
            PolynomialRingInner::PadicLocalField(ring) => ring
                .monomial_ideal(generators)
                .map(Self::PadicLocalField)
                .map_err(monomial_error),
            PolynomialRingInner::NumberField(ring) => ring
                .monomial_ideal(generators)
                .map(Self::NumberField)
                .map_err(monomial_error),
            PolynomialRingInner::RationalFunction(ring) => ring
                .monomial_ideal(generators)
                .map(Self::RationalFunction)
                .map_err(monomial_error),
            PolynomialRingInner::FiniteRationalFunction(ring) => ring
                .monomial_ideal(generators)
                .map(Self::FiniteRationalFunction)
                .map_err(monomial_error),
        }
    }

    fn ring(&self) -> PolynomialRingInner {
        match self {
            Self::Integer(value) => PolynomialRingInner::Integer(value.ring().clone()),
            Self::Rational(value) => PolynomialRingInner::Rational(value.ring().clone()),
            Self::Residue(value) => PolynomialRingInner::Residue(value.ring().clone()),
            Self::FiniteField(value) => PolynomialRingInner::FiniteField(value.ring().clone()),
            Self::Padic(value) => PolynomialRingInner::Padic(value.ring().clone()),
            Self::PadicField(value) => PolynomialRingInner::PadicField(value.ring().clone()),
            Self::Qadic(value) => PolynomialRingInner::Qadic(value.ring().clone()),
            Self::QadicField(value) => PolynomialRingInner::QadicField(value.ring().clone()),
            Self::PadicLocalField(value) => {
                PolynomialRingInner::PadicLocalField(value.ring().clone())
            }
            Self::NumberField(value) => PolynomialRingInner::NumberField(value.ring().clone()),
            Self::RationalFunction(value) => {
                PolynomialRingInner::RationalFunction(value.ring().clone())
            }
            Self::FiniteRationalFunction(value) => {
                PolynomialRingInner::FiniteRationalFunction(value.ring().clone())
            }
        }
    }
}

impl PartialEq for MonomialIdealInner {
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
            (Self::NumberField(left), Self::NumberField(right)) => left == right,
            (Self::RationalFunction(left), Self::RationalFunction(right)) => left == right,
            (Self::FiniteRationalFunction(left), Self::FiniteRationalFunction(right)) => {
                left == right
            }
            _ => false,
        }
    }
}

impl Eq for MonomialIdealInner {}

pub(super) fn extract_exponent_vectors(
    generators: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<Integer>>> {
    generators
        .try_iter()
        .map_err(|_| PyTypeError::new_err("monomial generators must be an iterable"))?
        .map(|generator| {
            generator?
                .try_iter()
                .map_err(|_| PyTypeError::new_err("each monomial generator must be iterable"))?
                .map(|exponent| extract_integer(&exponent?))
                .collect()
        })
        .collect()
}

fn integer_vectors_to_python(
    py: Python<'_>,
    values: &[Vec<Integer>],
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    values
        .iter()
        .map(|value| {
            value
                .iter()
                .map(|entry| integer_to_python(py, entry))
                .collect()
        })
        .collect()
}

/// A canonical monomial ideal in one explicit polynomial-ring context.
#[pyclass(
    module = "noether.commalg",
    name = "MonomialIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMonomialIdeal {
    pub(super) inner: MonomialIdealInner,
}

impl PyMonomialIdeal {
    pub(super) fn from_inner(inner: MonomialIdealInner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMonomialIdeal {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        MonomialIdeal::<IntegerRing>::TYPE_ID
    }

    /// Ambient polynomial-ring context.
    #[getter]
    fn ring(&self) -> PyPolynomialRing {
        PyPolynomialRing {
            inner: self.inner.ring(),
        }
    }

    /// Canonical minimal exponent antichain.
    #[getter]
    fn minimal_generators(&self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        integer_vectors_to_python(
            py,
            map_monomial!(&self.inner, |value| value.minimal_generators()),
        )
    }

    fn is_zero(&self) -> bool {
        map_monomial!(&self.inner, |value| value.is_zero())
    }

    fn is_unit(&self) -> bool {
        map_monomial!(&self.inner, |value| value.is_unit())
    }

    fn contains_monomial(&self, exponent: &Bound<'_, PyAny>) -> PyResult<bool> {
        let exponent = exponent
            .try_iter()
            .map_err(|_| PyTypeError::new_err("a monomial exponent vector must be iterable"))?
            .map(|value| extract_integer(&value?))
            .collect::<PyResult<Vec<_>>>()?;
        map_monomial!(&self.inner, |value| value.contains_monomial(&exponent))
            .map_err(monomial_error)
    }

    fn sum(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: binary_operation!(&self.inner, &other.inner, sum).map_err(monomial_error)?,
        })
    }

    fn product(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: binary_operation!(&self.inner, &other.inner, product).map_err(monomial_error)?,
        })
    }

    fn intersection(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: binary_operation!(&self.inner, &other.inner, intersection)
                .map_err(monomial_error)?,
        })
    }

    fn colon_monomial(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exponent = exponent
            .try_iter()
            .map_err(|_| PyTypeError::new_err("a monomial exponent vector must be iterable"))?
            .map(|value| extract_integer(&value?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: unary_operation!(&self.inner, colon_monomial, &exponent)
                .map_err(monomial_error)?,
        })
    }

    fn colon(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: binary_operation!(&self.inner, &other.inner, colon).map_err(monomial_error)?,
        })
    }

    fn radical(&self) -> PyResult<Self> {
        Ok(Self {
            inner: unary_operation!(&self.inner, radical).map_err(monomial_error)?,
        })
    }

    fn pow(&self, exponent: u64) -> PyResult<Self> {
        Ok(Self {
            inner: unary_operation!(&self.inner, pow, exponent).map_err(monomial_error)?,
        })
    }

    fn saturate_monomial(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exponent = exponent
            .try_iter()
            .map_err(|_| PyTypeError::new_err("a monomial exponent vector must be iterable"))?
            .map(|value| extract_integer(&value?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: unary_operation!(&self.inner, saturate_monomial, &exponent)
                .map_err(monomial_error)?,
        })
    }

    fn saturate(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: binary_operation!(&self.inner, &other.inner, saturate)
                .map_err(monomial_error)?,
        })
    }

    fn support(&self) -> Vec<usize> {
        map_monomial!(&self.inner, |value| value.support())
    }

    #[pyo3(signature = (degree, *, max_count))]
    fn standard_monomials_of_degree(
        &self,
        py: Python<'_>,
        degree: usize,
        max_count: usize,
    ) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let values = map_monomial!(&self.inner, |value| value
            .standard_monomials_of_degree(degree, max_count))
        .map_err(monomial_error)?;
        integer_vectors_to_python(py, &values)
    }

    fn finite_standard_monomials(&self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let values = map_monomial!(&self.inner, |value| value.finite_standard_monomials())
            .map_err(monomial_error)?;
        integer_vectors_to_python(py, &values)
    }

    fn hilbert_numerator(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let values =
            map_monomial!(&self.inner, |value| value.hilbert_numerator()).map_err(value_error)?;
        values
            .iter()
            .map(|value| integer_to_python(py, value))
            .collect()
    }

    fn multigraded_hilbert_numerator(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(PyDegree, Py<PyAny>)>> {
        let values = map_monomial!(&self.inner, |value| value.multigraded_hilbert_numerator())
            .map_err(value_error)?;
        values
            .into_iter()
            .map(|(degree, coefficient)| {
                Ok((
                    PyDegree { inner: degree },
                    integer_to_python(py, &coefficient)?,
                ))
            })
            .collect()
    }

    fn hilbert_series(&self) -> PyResult<PyHilbertSeries> {
        let inner =
            map_monomial!(&self.inner, |value| value.hilbert_series()).map_err(value_error)?;
        Ok(PyHilbertSeries { inner })
    }

    fn dimension(&self) -> PyResult<i64> {
        map_monomial!(&self.inner, |value| value.dimension()).map_err(value_error)
    }

    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = map_monomial!(&self.inner, |value| value.degree()).map_err(value_error)?;
        integer_to_python(py, &value)
    }

    fn hilbert_polynomial(&self) -> PyResult<PyPolynomial> {
        let value =
            map_monomial!(&self.inner, |value| value.hilbert_polynomial()).map_err(value_error)?;
        Ok(PyPolynomial::from_rational(value))
    }

    /// Convert to a polynomial ideal in a currently projected Groebner field.
    fn to_ideal(&self) -> PyResult<PyIdeal> {
        match &self.inner {
            MonomialIdealInner::Rational(value) => Ok(PyIdeal::from_rational_polynomial(
                value.to_ideal().map_err(value_error)?,
            )),
            MonomialIdealInner::FiniteField(value) => Ok(PyIdeal::from_finite_field_polynomial(
                value.to_ideal().map_err(value_error)?,
            )),
            MonomialIdealInner::RationalFunction(value) => {
                Ok(PyIdeal::from_rational_function_polynomial(
                    value.to_ideal().map_err(value_error)?,
                ))
            }
            MonomialIdealInner::FiniteRationalFunction(value) => {
                Ok(PyIdeal::from_finite_rational_function_polynomial(
                    value.to_ideal().map_err(value_error)?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "to_ideal currently supports Q, finite fields, and rational-function fields over them",
            )),
        }
    }

    fn verify(&self) -> PyResult<()> {
        map_monomial!(&self.inner, |value| value.verify()).map_err(monomial_error)
    }

    fn to_json(&self) -> PyResult<String> {
        map_monomial!(&self.inner, |value| value.to_json()).map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient = payload
            .get("ring")
            .and_then(|ring| ring.get("coefficient_ring"))
            .ok_or_else(|| value_error("serialized monomial ideal has no coefficient context"))?;
        let coefficient_type = coefficient
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| value_error("serialized coefficient context has no type identity"))?;
        let inner = match coefficient_type {
            IntegerRing::TYPE_ID => MonomialIdealInner::Integer(
                MonomialIdeal::<IntegerRing>::from_json(value).map_err(value_error)?,
            ),
            RationalField::TYPE_ID => MonomialIdealInner::Rational(
                MonomialIdeal::<RationalField>::from_json(value).map_err(value_error)?,
            ),
            ResidueRing::TYPE_ID => MonomialIdealInner::Residue(
                MonomialIdeal::<ResidueRing>::from_json(value).map_err(value_error)?,
            ),
            FiniteField::TYPE_ID => MonomialIdealInner::FiniteField(
                MonomialIdeal::<FiniteField>::from_json(value).map_err(value_error)?,
            ),
            PadicRing::TYPE_ID => MonomialIdealInner::Padic(
                MonomialIdeal::<PadicRing>::from_json(value).map_err(value_error)?,
            ),
            PadicField::TYPE_ID => MonomialIdealInner::PadicField(
                MonomialIdeal::<PadicField>::from_json(value).map_err(value_error)?,
            ),
            QadicRing::TYPE_ID => MonomialIdealInner::Qadic(
                MonomialIdeal::<QadicRing>::from_json(value).map_err(value_error)?,
            ),
            QadicField::TYPE_ID => MonomialIdealInner::QadicField(
                MonomialIdeal::<QadicField>::from_json(value).map_err(value_error)?,
            ),
            PadicLocalField::TYPE_ID => MonomialIdealInner::PadicLocalField(
                MonomialIdeal::<PadicLocalField>::from_json(value).map_err(value_error)?,
            ),
            NumberField::TYPE_ID => MonomialIdealInner::NumberField(
                MonomialIdeal::<NumberField>::from_json(value).map_err(value_error)?,
            ),
            RationalFunctionField::<FiniteField>::TYPE_ID => {
                let base_type = coefficient
                    .get("coefficient_ring")
                    .and_then(|base| base.get("type"))
                    .and_then(serde_json::Value::as_str);
                if base_type == Some(RationalField::TYPE_ID) {
                    MonomialIdealInner::RationalFunction(
                        MonomialIdeal::<RationalFunctionField<RationalField>>::from_json(value)
                            .map_err(value_error)?,
                    )
                } else if base_type == Some(FiniteField::TYPE_ID) {
                    MonomialIdealInner::FiniteRationalFunction(
                        MonomialIdeal::<RationalFunctionField<FiniteField>>::from_json(value)
                            .map_err(value_error)?,
                    )
                } else {
                    return Err(PyTypeError::new_err(
                        "unsupported rational-function base field in serialized monomial ideal",
                    ));
                }
            }
            type_id => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported monomial-ideal coefficient context {type_id:?}"
                )));
            }
        };
        Ok(Self { inner })
    }

    fn __contains__(&self, exponent: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains_monomial(exponent)
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.sum(other)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.product(other)
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "MonomialIdeal power does not accept Python's third modulus argument",
            ));
        }
        self.pow(exponent)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        match self.to_json() {
            Ok(value) => value.hash(&mut hasher),
            Err(error) => error.to_string().hash(&mut hasher),
        }
        let hash = hasher.finish() as isize;
        if hash == -1 {
            -2
        } else {
            hash
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "MonomialIdeal(ring={}, generators={})",
            map_monomial!(&self.inner, |value| value.ring().variables().join(",")),
            map_monomial!(&self.inner, |value| value.minimal_generators().len())
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", MonomialIdeal::<IntegerRing>::TYPE_ID)?;
        summary.set_item(
            "variable_count",
            map_monomial!(&self.inner, |value| value.ring().variables().len()),
        )?;
        summary.set_item(
            "generator_count",
            map_monomial!(&self.inner, |value| value.minimal_generators().len()),
        )?;
        Ok(summary.unbind())
    }
}
