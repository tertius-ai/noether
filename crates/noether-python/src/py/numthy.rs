// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public `noether::numthy` facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::numthy::{
    cyclotomic_polynomial, factor, is_prime, partial_factor, CofactorClassification, FiniteField,
    FiniteFieldElement, FiniteFieldEmbedding, FiniteFieldIsomorphism, HenselLiftError,
    IntegerFactorization, PadicElement, PadicField, PadicFieldFrobenius, PadicFieldRestriction,
    PadicIntegralEmbedding, PadicLocalField, PadicLocalFieldCoefficientFrobenius,
    PadicLocalFieldElement, PadicLocalFieldOrder, PadicLocalFieldRestriction,
    PadicLocalFractionalIdeal, PadicLocalPrecision, PadicRing, PadicRingFrobenius,
    PadicRingRestriction, PadicValuation, PartialIntegerFactorization, QadicElement, QadicField,
    QadicFieldFrobenius, QadicFieldRestriction, QadicIntegralEmbedding, QadicRing,
    QadicRingFrobenius, QadicRingRestriction,
};
use noether::{Field, InversionError, Ring, SemanticType, UnitInversion};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};
use serde::{Deserialize, Serialize};

use super::coefficient_maps::PyPrimeSubfieldEmbedding;
use super::errors::context_mismatch;
use super::facade_scalar::{extract_integer, integer_to_python};
use super::homalg::{ChainComplexInner, PyChainComplex};
use super::linalg::{
    MatrixInner, PyMatrix, PySparseMatrix, PyVector, SparseMatrixInner, VectorInner,
};
use super::polynomial_facade::PyPolynomial;
use super::series::{PyPowerSeries, SeriesInner};

fn inversion_error(error: InversionError) -> PyErr {
    match error {
        InversionError::DivisionByZero(_) => PyZeroDivisionError::new_err(error.to_string()),
        InversionError::ContextMismatch(_) => PyValueError::new_err(error.to_string()),
    }
}

fn hensel_error(error: HenselLiftError) -> PyErr {
    match error {
        HenselLiftError::ContextMismatch(error) => context_mismatch(error),
        HenselLiftError::NotUnivariate { .. }
        | HenselLiftError::Polynomial(_)
        | HenselLiftError::NotAResidueRoot
        | HenselLiftError::SingularDerivative
        | HenselLiftError::DidNotConverge { .. } => PyValueError::new_err(error.to_string()),
    }
}

fn bounded_integer(mut rendered: String) -> String {
    const CHARACTER_LIMIT: usize = 64;
    if rendered.len() > CHARACTER_LIMIT {
        rendered.truncate(CHARACTER_LIMIT);
        rendered.push_str("...");
    }
    rendered
}

/// A complete proved prime factorization that reconstructs its input.
#[pyclass(
    module = "noether.numthy",
    name = "IntegerFactorization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegerFactorization {
    inner: IntegerFactorization,
}

#[pymethods]
impl PyIntegerFactorization {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        IntegerFactorization::TYPE_ID
    }

    /// Multiplicative unit, either `-1` or `1`.
    #[getter]
    fn unit(&self) -> i8 {
        self.inner.unit()
    }

    /// Strictly increasing proved prime factors with positive exponents.
    #[getter]
    fn factors(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u64)>> {
        self.inner
            .factors()
            .iter()
            .map(|(prime, exponent)| Ok((integer_to_python(py, prime)?, *exponent)))
            .collect()
    }

    /// Reconstruct the exact factored integer.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.value())
    }

    /// Verify reconstruction against an exact Python integer.
    fn verify(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.verify(&extract_integer(value)?))
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and validate a canonical factorization from JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: IntegerFactorization::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("IntegerFactorization({})", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Return the complete proved factorization of a nonzero Python integer.
#[pyfunction(name = "_facade_integer_factor")]
pub fn integer_factor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyIntegerFactorization> {
    let value = extract_integer(value)?;
    py.detach(|| factor(&value))
        .map(|inner| PyIntegerFactorization { inner })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A deterministic bounded factorization with an honestly classified cofactor.
#[pyclass(
    module = "noether.numthy",
    name = "PartialIntegerFactorization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPartialIntegerFactorization {
    pub(crate) inner: PartialIntegerFactorization,
}

#[pymethods]
impl PyPartialIntegerFactorization {
    #[classattr]
    fn type_id() -> &'static str {
        PartialIntegerFactorization::TYPE_ID
    }

    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.source())
    }

    #[getter]
    fn unit(&self) -> i8 {
        self.inner.unit()
    }

    #[getter]
    fn factors(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u64)>> {
        self.inner
            .factors()
            .iter()
            .map(|(prime, exponent)| Ok((integer_to_python(py, prime)?, *exponent)))
            .collect()
    }

    #[getter]
    fn cofactor(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.cofactor())
    }

    #[getter]
    fn classification(&self) -> &'static str {
        match self.inner.classification() {
            CofactorClassification::Unit => "unit",
            CofactorClassification::ProvedPrime => "proved_prime",
            CofactorClassification::ProvedComposite => "proved_composite",
            CofactorClassification::Unclassified => "unclassified",
        }
    }

    #[getter]
    fn trial_bound(&self) -> u64 {
        self.inner.trial_bound()
    }

    #[getter]
    fn primality_proof_bits(&self) -> u64 {
        self.inner.primality_proof_bits()
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.value())
    }

    fn verify(&self) -> bool {
        self.inner.verify()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PartialIntegerFactorization::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PartialIntegerFactorization({})", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Return deterministic partial factorization within explicit work bounds.
#[pyfunction(name = "_facade_integer_partial_factor", signature = (value, *, trial_bound=10_000, primality_proof_bits=256))]
pub fn integer_partial_factor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    trial_bound: u64,
    primality_proof_bits: u64,
) -> PyResult<PyPartialIntegerFactorization> {
    let value = extract_integer(value)?;
    py.detach(|| partial_factor(&value, trial_bound, primality_proof_bits))
        .map(|inner| PyPartialIntegerFactorization { inner })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Test primality with FLINT's deterministic primality prover.
#[pyfunction(name = "_facade_integer_is_prime")]
pub fn integer_is_prime(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let value = extract_integer(value)?;
    Ok(py.detach(|| is_prime(&value)))
}

/// Return the positive-index cyclotomic polynomial in `Z[x]`.
#[pyfunction(name = "_facade_cyclotomic_polynomial")]
pub fn facade_cyclotomic_polynomial(py: Python<'_>, index: u64) -> PyResult<PyPolynomial> {
    py.detach(|| cyclotomic_polynomial(index))
        .map(PyPolynomial::from_integer)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A finite field with one explicit, verified presentation.
#[pyclass(
    module = "noether.numthy",
    name = "FiniteField",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteField {
    pub(crate) inner: FiniteField,
}

#[pymethods]
impl PyFiniteField {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FiniteField::TYPE_ID
    }

    #[classmethod]
    fn prime(_cls: &Bound<'_, PyType>, characteristic: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteField::prime(extract_integer(characteristic)?)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[classmethod]
    fn extension(_cls: &Bound<'_, PyType>, characteristic: u64, degree: usize) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteField::extension(characteristic, degree)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[classmethod]
    fn with_modulus(
        _cls: &Bound<'_, PyType>,
        characteristic: u64,
        modulus: Vec<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteField::with_modulus(characteristic, modulus)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn characteristic(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.characteristic())
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.order())
    }

    #[getter]
    fn defining_modulus(&self) -> Option<Vec<u64>> {
        self.inner.defining_modulus().map(<[u64]>::to_vec)
    }

    fn zero(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.one(),
        }
    }

    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.order())
    }

    fn inverse(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self.inner.inverse(&value.inner).map_err(inversion_error)?,
        })
    }

    fn generator(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.generator(),
        }
    }

    fn element_from_coefficients(&self, coefficients: Vec<u64>) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .element_from_coefficients(&coefficients)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Return a deterministic checked map to an equal-order presentation.
    fn isomorphism_to(&self, target: &Self) -> PyResult<PyFiniteFieldIsomorphism> {
        Ok(PyFiniteFieldIsomorphism {
            inner: self
                .inner
                .isomorphism_to(&target.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Build a checked embedding by specifying the image of this field's
    /// distinguished generator.
    fn embedding_to(
        &self,
        target: &Self,
        generator_image: &PyFiniteFieldElement,
    ) -> PyResult<PyFiniteFieldEmbedding> {
        Ok(PyFiniteFieldEmbedding {
            inner: FiniteFieldEmbedding::new(
                self.inner.clone(),
                target.inner.clone(),
                generator_image.inner.clone(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Return the canonical embedding of this field's prime subfield.
    fn prime_subfield_embedding(&self) -> PyPrimeSubfieldEmbedding {
        PyPrimeSubfieldEmbedding {
            inner: self.inner.prime_subfield_embedding(),
        }
    }

    /// Canonical JSON containing the complete verified presentation.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a finite-field presentation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteField::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            if let Some(modulus) = self.inner.defining_modulus() {
                format!(
                    "FiniteField.with_modulus({}, {})",
                    bounded_integer(self.inner.characteristic().to_string()),
                    bounded_integer(format!("{modulus:?}"))
                )
            } else {
                format!(
                    "FiniteField.prime({})",
                    bounded_integer(self.inner.characteristic().to_string())
                )
            }
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A checked isomorphism between two explicit finite-field presentations.
#[pyclass(
    module = "noether.numthy",
    name = "FiniteFieldIsomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteFieldIsomorphism {
    inner: FiniteFieldIsomorphism,
}

#[pymethods]
impl PyFiniteFieldIsomorphism {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FiniteFieldIsomorphism::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.target().clone(),
        }
    }

    #[getter]
    fn generator_image(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.generator_image().clone(),
        }
    }

    #[getter]
    fn inverse_generator_image(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.inverse_generator_image().clone(),
        }
    }

    fn apply(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_inverse(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .apply_inverse(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteFieldIsomorphism::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        self.apply(value)
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
}

/// An explicitly presented embedding between compatible finite fields.
#[pyclass(
    module = "noether.numthy",
    name = "FiniteFieldEmbedding",
    frozen,
    from_py_object
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyFiniteFieldEmbedding {
    pub(crate) inner: FiniteFieldEmbedding,
}

#[pymethods]
impl PyFiniteFieldEmbedding {
    #[new]
    fn new(
        source: &PyFiniteField,
        target: &PyFiniteField,
        generator_image: &PyFiniteFieldElement,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteFieldEmbedding::new(
                source.inner.clone(),
                target.inner.clone(),
                generator_image.inner.clone(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        FiniteFieldEmbedding::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.target().clone(),
        }
    }

    #[getter]
    fn generator_image(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.generator_image().clone(),
        }
    }

    fn apply(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteFieldEmbedding::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyFiniteFieldElement) -> PyResult<PyFiniteFieldElement> {
        self.apply(value)
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
}

/// An exact finite-field value carrying its concrete presentation.
#[pyclass(
    module = "noether.numthy",
    name = "FiniteFieldElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteFieldElement {
    pub(crate) inner: FiniteFieldElement,
}

/// A p-adic valuation that preserves a finite-precision lower bound.
#[pyclass(
    module = "noether.numthy",
    name = "PadicValuation",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyPadicValuation {
    inner: PadicValuation,
}

#[pymethods]
impl PyPadicValuation {
    #[getter]
    fn is_exact(&self) -> bool {
        matches!(self.inner, PadicValuation::Exact(_))
    }

    #[getter]
    fn value(&self) -> Option<i64> {
        match self.inner {
            PadicValuation::Exact(value) => Some(value),
            PadicValuation::AtLeast(_) => None,
        }
    }

    #[getter]
    fn lower_bound(&self) -> i64 {
        match self.inner {
            PadicValuation::Exact(value) | PadicValuation::AtLeast(value) => value,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match self.inner {
            PadicValuation::Exact(value) => value as isize,
            PadicValuation::AtLeast(value) => !(value as isize),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match self.inner {
            PadicValuation::Exact(value) => format!("PadicValuation.exact({value})"),
            PadicValuation::AtLeast(value) => format!("PadicValuation.at_least({value})"),
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

macro_rules! define_coefficient_frobenius_wrapper {
    (
        $python:ident,
        $rust:ty,
        $python_name:literal,
        $context_python:ident,
        $element_python:ident,
        $repr_name:literal
    ) => {
        #[doc = concat!("Python projection of `", $repr_name, "`.")]
        #[pyclass(module = "noether.numthy", name = $python_name, frozen, from_py_object)]
        #[derive(Clone)]
        pub struct $python {
            pub(crate) inner: $rust,
        }

        #[pymethods]
        impl $python {
            #[classattr]
            fn type_id() -> &'static str {
                <$rust>::TYPE_ID
            }

            #[getter]
            fn context(&self) -> $context_python {
                $context_python {
                    inner: self.inner.context().clone(),
                }
            }

            #[getter]
            fn exponent(&self) -> u64 {
                self.inner.exponent()
            }

            fn apply(&self, value: &$element_python) -> PyResult<$element_python> {
                Ok($element_python {
                    inner: self
                        .inner
                        .apply(&value.inner)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                })
            }

            fn compose(&self, inner: &Self) -> PyResult<Self> {
                Ok(Self {
                    inner: self
                        .inner
                        .compose(&inner.inner)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                })
            }

            fn to_json(&self) -> PyResult<String> {
                self.inner
                    .to_json()
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            }

            #[classmethod]
            fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
                Ok(Self {
                    inner: <$rust>::from_json(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                })
            }

            fn __call__(&self, value: &$element_python) -> PyResult<$element_python> {
                self.apply(value)
            }

            fn __eq__(&self, other: &Self) -> bool {
                self.inner == other.inner
            }

            fn __hash__(&self) -> isize {
                canonical_hash(&self.inner)
            }

            fn __repr__(&self) -> String {
                crate::py::repr::bounded_text(format!(
                    concat!($repr_name, "({}, exponent={})"),
                    self.inner.context(),
                    self.inner.exponent()
                ))
            }
        }
    };
}

define_coefficient_frobenius_wrapper!(
    PyPadicRingFrobenius,
    PadicRingFrobenius,
    "PadicRingFrobenius",
    PyPadicRing,
    PyPadicElement,
    "PadicRingFrobenius"
);
define_coefficient_frobenius_wrapper!(
    PyPadicFieldFrobenius,
    PadicFieldFrobenius,
    "PadicFieldFrobenius",
    PyPadicField,
    PyPadicElement,
    "PadicFieldFrobenius"
);
define_coefficient_frobenius_wrapper!(
    PyQadicRingFrobenius,
    QadicRingFrobenius,
    "QadicRingFrobenius",
    PyQadicRing,
    PyQadicElement,
    "QadicRingFrobenius"
);
define_coefficient_frobenius_wrapper!(
    PyQadicFieldFrobenius,
    QadicFieldFrobenius,
    "QadicFieldFrobenius",
    PyQadicField,
    PyQadicElement,
    "QadicFieldFrobenius"
);

/// An explicit digit-discarding map between fixed-cap p-adic rings.
#[pyclass(
    module = "noether.numthy",
    name = "PadicRingRestriction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicRingRestriction {
    inner: PadicRingRestriction,
}

#[pymethods]
impl PyPadicRingRestriction {
    #[classattr]
    fn type_id() -> &'static str {
        PadicRingRestriction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPadicRing {
        PyPadicRing {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyPadicRing {
        PyPadicRing {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_padic()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a p-adic ring"))?;
        Ok(PyPolynomial::from_padic(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic ring",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::Padic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic ring",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::Padic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic ring",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::Padic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_sparse_matrix(&self, value: &PySparseMatrix) -> PyResult<PySparseMatrix> {
        let SparseMatrixInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a sparse matrix over a p-adic ring",
            ));
        };
        Ok(PySparseMatrix {
            inner: SparseMatrixInner::Padic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_chain_complex(&self, value: &PyChainComplex) -> PyResult<PyChainComplex> {
        let ChainComplexInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a chain complex over a p-adic ring",
            ));
        };
        Ok(PyChainComplex {
            inner: ChainComplexInner::Padic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicRingRestriction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicRingRestriction({})", self.inner))
    }
}

/// An explicit digit-discarding map between fixed-cap p-adic fields.
#[pyclass(
    module = "noether.numthy",
    name = "PadicFieldRestriction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicFieldRestriction {
    inner: PadicFieldRestriction,
}

#[pymethods]
impl PyPadicFieldRestriction {
    #[classattr]
    fn type_id() -> &'static str {
        PadicFieldRestriction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPadicField {
        PyPadicField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyPadicField {
        PyPadicField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_padic_field()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a p-adic field"))?;
        Ok(PyPolynomial::from_padic_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicFieldRestriction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicFieldRestriction({})", self.inner))
    }
}

/// The canonical fixed-cap embedding of p-adic integers in their field.
#[pyclass(
    module = "noether.numthy",
    name = "PadicIntegralEmbedding",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicIntegralEmbedding {
    inner: PadicIntegralEmbedding,
}

#[pymethods]
impl PyPadicIntegralEmbedding {
    #[classattr]
    fn type_id() -> &'static str {
        PadicIntegralEmbedding::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPadicRing {
        PyPadicRing {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyPadicField {
        PyPadicField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn retract(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .retract(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_padic()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a p-adic ring"))?;
        Ok(PyPolynomial::from_padic_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn retract_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_padic_field()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a p-adic field"))?;
        Ok(PyPolynomial::from_padic(
            self.inner
                .retract_polynomial(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic ring",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::Padic(
                self.inner
                    .retract_power_series(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic ring",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::Padic(
                self.inner
                    .retract_vector(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::Padic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic ring",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::PadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::PadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::Padic(
                self.inner
                    .retract_matrix(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicIntegralEmbedding::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicIntegralEmbedding({})", self.inner))
    }
}

/// Fixed-cap p-adic integers known modulo `p^N`.
#[pyclass(module = "noether.numthy", name = "PadicRing", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyPadicRing {
    pub(crate) inner: PadicRing,
}

#[pymethods]
impl PyPadicRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PadicRing::TYPE_ID
    }

    #[new]
    fn new(prime: &Bound<'_, PyAny>, absolute_precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: PadicRing::new(extract_integer(prime)?, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn absolute_precision(&self) -> u64 {
        self.inner.absolute_precision()
    }

    #[getter]
    fn modulus(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.modulus())
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    fn zero(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.one(),
        }
    }

    fn coefficient_frobenius(&self, exponent: u64) -> PyPadicRingFrobenius {
        PyPadicRingFrobenius {
            inner: self.inner.coefficient_frobenius(exponent),
        }
    }

    /// Serialize the fixed prime and absolute precision as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision p-adic ring context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn inverse(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        if value.inner.ring() != Some(&self.inner) {
            return Err(PyValueError::new_err(format!(
                "context mismatch: expected {}",
                self.inner
            )));
        }
        value.inverse()
    }

    fn residue(&self, value: &PyPadicElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .residue(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn teichmuller(&self, value: &PyFiniteFieldElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .teichmuller(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Lift a verified simple residue root to the full fixed precision.
    fn hensel_lift(
        &self,
        polynomial: &PyPolynomial,
        residue_root: &PyFiniteFieldElement,
    ) -> PyResult<PyPadicElement> {
        let polynomial = polynomial.as_padic().ok_or_else(|| {
            PyTypeError::new_err("Hensel lifting requires a polynomial over PadicRing")
        })?;
        Ok(PyPadicElement {
            inner: self
                .inner
                .hensel_lift(polynomial, &residue_root.inner)
                .map_err(hensel_error)?,
        })
    }

    fn restrict(
        &self,
        value: &PyPadicElement,
        absolute_precision: u64,
    ) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .restrict(&value.inner, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn precision_restriction(&self, absolute_precision: u64) -> PyResult<PyPadicRingRestriction> {
        Ok(PyPadicRingRestriction {
            inner: self
                .inner
                .precision_restriction(absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn divide_uniformizer_exact(
        &self,
        value: &PyPadicElement,
        exponent: u64,
    ) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .divide_uniformizer_exact(&value.inner, exponent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PadicRing({}, absolute_precision={})",
                bounded_integer(self.inner.prime().to_string()),
                self.inner.absolute_precision()
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A fixed-precision model of `Q_p`, with inversion but without exact-field claims.
#[pyclass(module = "noether.numthy", name = "PadicField", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyPadicField {
    pub(crate) inner: PadicField,
}

#[pymethods]
impl PyPadicField {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PadicField::TYPE_ID
    }

    #[new]
    fn new(prime: &Bound<'_, PyAny>, absolute_precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: PadicField::new(extract_integer(prime)?, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn absolute_precision(&self) -> u64 {
        self.inner.absolute_precision()
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    #[getter]
    fn integral_ring(&self) -> PyPadicRing {
        PyPadicRing {
            inner: self.inner.integral_ring().clone(),
        }
    }

    fn zero(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.one(),
        }
    }

    fn coefficient_frobenius(&self, exponent: u64) -> PyPadicFieldFrobenius {
        PyPadicFieldFrobenius {
            inner: self.inner.coefficient_frobenius(exponent),
        }
    }

    /// Serialize the fixed prime and absolute precision as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision p-adic field context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicField::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn inverse(&self, value: &PyPadicElement) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self.inner.inverse(&value.inner).map_err(inversion_error)?,
        })
    }

    fn restrict(
        &self,
        value: &PyPadicElement,
        absolute_precision: u64,
    ) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .restrict(&value.inner, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn precision_restriction(&self, absolute_precision: u64) -> PyResult<PyPadicFieldRestriction> {
        Ok(PyPadicFieldRestriction {
            inner: self
                .inner
                .precision_restriction(absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn integral_embedding(&self) -> PyPadicIntegralEmbedding {
        PyPadicIntegralEmbedding {
            inner: self.inner.integral_embedding(),
        }
    }

    fn divide_uniformizer_exact(
        &self,
        value: &PyPadicElement,
        exponent: u64,
    ) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .divide_uniformizer_exact(&value.inner, exponent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PadicField({}, absolute_precision={})",
                bounded_integer(self.inner.prime().to_string()),
                self.inner.absolute_precision()
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A p-adic integer carrying its fixed `(p, N)` context.
#[pyclass(
    module = "noether.numthy",
    name = "PadicElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicElement {
    pub(crate) inner: PadicElement,
}

#[pymethods]
impl PyPadicElement {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PadicElement::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> Option<PyPadicRing> {
        self.inner.ring().map(|ring| PyPadicRing {
            inner: ring.clone(),
        })
    }

    #[getter]
    fn field(&self) -> Option<PyPadicField> {
        self.inner.field().map(|field| PyPadicField {
            inner: field.clone(),
        })
    }

    #[getter]
    fn representative(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .representative()
            .as_ref()
            .map(|value| integer_to_python(py, value))
            .transpose()
    }

    #[getter]
    fn known_absolute_precision(&self) -> u64 {
        self.inner.known_absolute_precision()
    }

    fn valuation(&self) -> PyPadicValuation {
        PyPadicValuation {
            inner: self.inner.valuation(),
        }
    }

    fn residue(&self) -> PyResult<PyFiniteFieldElement> {
        self.inner
            .residue()
            .map(|inner| PyFiniteFieldElement { inner })
            .ok_or_else(|| {
                PyValueError::new_err("a p-adic value of negative valuation has no residue")
            })
    }

    fn is_unit(&self) -> bool {
        self.inner.is_unit()
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .inverse()
                .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))?,
        })
    }

    fn pow(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    /// Serialize the concrete parent and canonical FLINT coordinates as typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a p-adic value and its fixed-precision parent.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicElement::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.representative(py)?.ok_or_else(|| {
            PyTypeError::new_err("a p-adic value of negative valuation is not an integer")
        })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner + &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner - &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner * &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "PadicElement power does not accept Python's third modulus argument",
            ));
        }
        Ok(self.pow(exponent))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match (self.inner.ring(), self.inner.field()) {
            (Some(ring), None) => format!(
                "PadicElement({}, prime={}, absolute_precision={})",
                self.inner
                    .representative()
                    .map(|value| bounded_integer(value.to_string()))
                    .unwrap_or_else(|| "nonintegral".to_owned()),
                bounded_integer(ring.prime().to_string()),
                self.inner.known_absolute_precision()
            ),
            (None, Some(field)) => format!(
                "PadicElement({}, field=PadicField({}, absolute_precision={}))",
                self.inner,
                bounded_integer(field.prime().to_string()),
                self.inner.known_absolute_precision()
            ),
            _ => "PadicElement(invalid parent)".to_owned(),
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", "noether/numthy/PadicElement@1")?;
        summary.set_item("representative", self.representative(py)?)?;
        let prime = self
            .inner
            .ring()
            .map(PadicRing::prime)
            .or_else(|| self.inner.field().map(PadicField::prime));
        if let Some(prime) = prime {
            summary.set_item("prime", integer_to_python(py, prime)?)?;
        }
        summary.set_item("absolute_precision", self.known_absolute_precision())?;
        summary.set_item("valuation", self.valuation().__str__())?;
        Ok(summary.unbind())
    }
}

/// An explicit digit-discarding map between fixed-cap Witt rings.
#[pyclass(
    module = "noether.numthy",
    name = "QadicRingRestriction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQadicRingRestriction {
    inner: QadicRingRestriction,
}

#[pymethods]
impl PyQadicRingRestriction {
    #[classattr]
    fn type_id() -> &'static str {
        QadicRingRestriction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyQadicRing {
        PyQadicRing {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyQadicRing {
        PyQadicRing {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_qadic()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a q-adic ring"))?;
        Ok(PyPolynomial::from_qadic(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a q-adic ring",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::Qadic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a q-adic ring",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::Qadic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a q-adic ring",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::Qadic(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicRingRestriction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("QadicRingRestriction({})", self.inner))
    }
}

/// An explicit digit-discarding map between fixed-cap unramified fields.
#[pyclass(
    module = "noether.numthy",
    name = "QadicFieldRestriction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQadicFieldRestriction {
    inner: QadicFieldRestriction,
}

#[pymethods]
impl PyQadicFieldRestriction {
    #[classattr]
    fn type_id() -> &'static str {
        QadicFieldRestriction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyQadicField {
        PyQadicField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyQadicField {
        PyQadicField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_qadic_field()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a q-adic field"))?;
        Ok(PyPolynomial::from_qadic_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a q-adic field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a q-adic field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a q-adic field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicFieldRestriction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("QadicFieldRestriction({})", self.inner))
    }
}

/// The canonical fixed-cap embedding of a Witt ring in its fraction field.
#[pyclass(
    module = "noether.numthy",
    name = "QadicIntegralEmbedding",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQadicIntegralEmbedding {
    inner: QadicIntegralEmbedding,
}

#[pymethods]
impl PyQadicIntegralEmbedding {
    #[classattr]
    fn type_id() -> &'static str {
        QadicIntegralEmbedding::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyQadicRing {
        PyQadicRing {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyQadicField {
        PyQadicField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn retract(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .retract(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_qadic()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a q-adic ring"))?;
        Ok(PyPolynomial::from_qadic_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn retract_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_qadic_field()
            .ok_or_else(|| PyValueError::new_err("expected a polynomial over a q-adic field"))?;
        Ok(PyPolynomial::from_qadic(
            self.inner
                .retract_polynomial(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a q-adic ring",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a q-adic field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::Qadic(
                self.inner
                    .retract_power_series(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a q-adic ring",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a q-adic field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::Qadic(
                self.inner
                    .retract_vector(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::Qadic(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a q-adic ring",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::QadicField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn retract_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::QadicField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a q-adic field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::Qadic(
                self.inner
                    .retract_matrix(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicIntegralEmbedding::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("QadicIntegralEmbedding({})", self.inner))
    }
}

/// The fixed-cap Witt ring `W(F_q) / p^N`.
#[pyclass(module = "noether.numthy", name = "QadicRing", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyQadicRing {
    pub(crate) inner: QadicRing,
}

#[pymethods]
impl PyQadicRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        QadicRing::TYPE_ID
    }

    #[new]
    fn new(residue_field: &PyFiniteField, absolute_precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: QadicRing::new(residue_field.inner.clone(), absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn residue_degree(&self) -> usize {
        self.inner.residue_degree()
    }

    #[getter]
    fn coefficient_ring(&self) -> PyPadicRing {
        PyPadicRing {
            inner: self.inner.coefficient_ring().clone(),
        }
    }

    #[getter]
    fn absolute_precision(&self) -> u64 {
        self.inner.absolute_precision()
    }

    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.cardinality())
    }

    fn zero(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.one(),
        }
    }

    fn coefficient_frobenius(&self, exponent: u64) -> PyQadicRingFrobenius {
        PyQadicRingFrobenius {
            inner: self.inner.coefficient_frobenius(exponent),
        }
    }

    fn generator(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.generator(),
        }
    }

    fn element_from_coefficients(
        &self,
        py: Python<'_>,
        coefficients: Vec<Py<PyPadicElement>>,
    ) -> PyResult<PyQadicElement> {
        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| coefficient.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(PyQadicElement {
            inner: self
                .inner
                .element_from_coefficients(&coefficients)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Serialize the verified residue-field presentation and precision.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision Witt-ring context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn residue(&self, value: &PyQadicElement) -> PyResult<PyFiniteFieldElement> {
        Ok(PyFiniteFieldElement {
            inner: self
                .inner
                .residue(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn teichmuller(&self, value: &PyFiniteFieldElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .teichmuller(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn restrict(
        &self,
        value: &PyQadicElement,
        absolute_precision: u64,
    ) -> PyResult<PyQadicElement> {
        if value.inner.ring() != Some(&self.inner) {
            return Err(PyValueError::new_err(format!(
                "context mismatch: expected {}",
                self.inner
            )));
        }
        Ok(PyQadicElement {
            inner: self
                .inner
                .restrict(&value.inner, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn precision_restriction(&self, absolute_precision: u64) -> PyResult<PyQadicRingRestriction> {
        Ok(PyQadicRingRestriction {
            inner: self
                .inner
                .precision_restriction(absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn divide_uniformizer_exact(
        &self,
        value: &PyQadicElement,
        exponent: u64,
    ) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .divide_uniformizer_exact(&value.inner, exponent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "QadicRing({}, absolute_precision={})",
                PyFiniteField {
                    inner: self.inner.residue_field().clone()
                }
                .__repr__(),
                self.inner.absolute_precision()
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A fixed-precision unramified p-adic model, with inversion but without exact-field claims.
#[pyclass(module = "noether.numthy", name = "QadicField", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyQadicField {
    pub(crate) inner: QadicField,
}

#[pymethods]
impl PyQadicField {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        QadicField::TYPE_ID
    }

    #[new]
    fn new(residue_field: &PyFiniteField, absolute_precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: QadicField::new(residue_field.inner.clone(), absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    #[getter]
    fn coefficient_field(&self) -> PyPadicField {
        PyPadicField {
            inner: self.inner.coefficient_field().clone(),
        }
    }

    #[getter]
    fn integral_ring(&self) -> PyQadicRing {
        PyQadicRing {
            inner: self.inner.integral_ring().clone(),
        }
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn residue_degree(&self) -> usize {
        self.inner.residue_degree()
    }

    #[getter]
    fn absolute_precision(&self) -> u64 {
        self.inner.absolute_precision()
    }

    fn zero(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.one(),
        }
    }

    fn coefficient_frobenius(&self, exponent: u64) -> PyQadicFieldFrobenius {
        PyQadicFieldFrobenius {
            inner: self.inner.coefficient_frobenius(exponent),
        }
    }

    fn generator(&self) -> PyQadicElement {
        PyQadicElement {
            inner: self.inner.generator(),
        }
    }

    fn element_from_coefficients(
        &self,
        py: Python<'_>,
        coefficients: Vec<Py<PyPadicElement>>,
    ) -> PyResult<PyQadicElement> {
        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| coefficient.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(PyQadicElement {
            inner: self
                .inner
                .element_from_coefficients(&coefficients)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Serialize the verified residue-field presentation and precision.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate an unramified fixed-precision field context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicField::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn residue(&self, value: &PyQadicElement) -> PyResult<PyFiniteFieldElement> {
        self.inner
            .residue(&value.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .map(|inner| PyFiniteFieldElement { inner })
            .ok_or_else(|| {
                PyValueError::new_err("a q-adic value of negative valuation has no residue")
            })
    }

    fn teichmuller(&self, value: &PyFiniteFieldElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .teichmuller(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn inverse(&self, value: &PyQadicElement) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self.inner.inverse(&value.inner).map_err(inversion_error)?,
        })
    }

    fn restrict(
        &self,
        value: &PyQadicElement,
        absolute_precision: u64,
    ) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .restrict(&value.inner, absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn precision_restriction(&self, absolute_precision: u64) -> PyResult<PyQadicFieldRestriction> {
        Ok(PyQadicFieldRestriction {
            inner: self
                .inner
                .precision_restriction(absolute_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn integral_embedding(&self) -> PyQadicIntegralEmbedding {
        PyQadicIntegralEmbedding {
            inner: self.inner.integral_embedding(),
        }
    }

    fn divide_uniformizer_exact(
        &self,
        value: &PyQadicElement,
        exponent: u64,
    ) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self
                .inner
                .divide_uniformizer_exact(&value.inner, exponent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &Bound<'_, PyAny>) -> PyResult<PyQadicElement> {
        Ok(PyQadicElement {
            inner: self.inner.element(extract_integer(value)?),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "QadicField({}, absolute_precision={})",
                PyFiniteField {
                    inner: self.inner.residue_field().clone()
                }
                .__repr__(),
                self.inner.absolute_precision()
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A fixed-cap q-adic value carrying its concrete ring or field context.
#[pyclass(
    module = "noether.numthy",
    name = "QadicElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyQadicElement {
    pub(crate) inner: QadicElement,
}

#[pymethods]
impl PyQadicElement {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        QadicElement::TYPE_ID
    }

    #[getter]
    fn ring(&self) -> Option<PyQadicRing> {
        self.inner.ring().map(|ring| PyQadicRing {
            inner: ring.clone(),
        })
    }

    #[getter]
    fn field(&self) -> Option<PyQadicField> {
        self.inner.field().map(|field| PyQadicField {
            inner: field.clone(),
        })
    }

    #[getter]
    fn coefficients(&self) -> Vec<PyPadicElement> {
        self.inner
            .coefficients()
            .into_iter()
            .map(|inner| PyPadicElement { inner })
            .collect()
    }

    #[getter]
    fn known_absolute_precision(&self) -> u64 {
        self.inner.known_absolute_precision()
    }

    fn valuation(&self) -> PyPadicValuation {
        PyPadicValuation {
            inner: self.inner.valuation(),
        }
    }

    fn residue(&self) -> PyResult<PyFiniteFieldElement> {
        self.inner
            .residue()
            .map(|inner| PyFiniteFieldElement { inner })
            .ok_or_else(|| {
                PyValueError::new_err("a q-adic value of negative valuation has no residue")
            })
    }

    fn is_unit(&self) -> bool {
        self.inner.is_unit()
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .inverse()
                .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))?,
        })
    }

    fn pow(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    fn frobenius(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.frobenius(exponent),
        }
    }

    fn trace(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.trace(),
        }
    }

    fn norm(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.norm(),
        }
    }

    /// Serialize the complete parent and canonical power-basis coordinates.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a q-adic value and its parent context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: QadicElement::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner + &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner - &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner * &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "QadicElement power does not accept Python's third modulus argument",
            ));
        }
        Ok(self.pow(exponent))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match (self.inner.ring(), self.inner.field()) {
            (Some(ring), None) => format!(
                "QadicElement({}, ring=QadicRing({}, absolute_precision={}))",
                self.inner,
                ring.residue_field(),
                self.inner.known_absolute_precision()
            ),
            (None, Some(field)) => format!(
                "QadicElement({}, field=QadicField({}, absolute_precision={}))",
                self.inner,
                field.residue_field(),
                self.inner.known_absolute_precision()
            ),
            _ => "QadicElement(invalid parent)".to_owned(),
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

#[pymethods]
impl PyFiniteFieldElement {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FiniteFieldElement::TYPE_ID
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn representative(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.representative())
    }

    #[getter]
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .coefficients()
            .iter()
            .map(|coefficient| integer_to_python(py, coefficient))
            .collect()
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.inverse().map_err(inversion_error)?,
        })
    }

    fn pow(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    fn frobenius(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.frobenius(exponent),
        }
    }

    fn trace(&self) -> Self {
        Self {
            inner: self.inner.trace(),
        }
    }

    fn norm(&self) -> Self {
        Self {
            inner: self.inner.norm(),
        }
    }

    /// Canonical JSON containing the field presentation and coordinates.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a finite-field element.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FiniteFieldElement::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.representative())
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner + &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner - &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: (&self.inner * &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: (-&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "FiniteFieldElement power does not accept Python's third modulus argument",
            ));
        }
        Ok(self.pow(exponent))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "FiniteFieldElement({}, characteristic={})",
                bounded_integer(self.inner.representative().to_string()),
                bounded_integer(self.inner.field().characteristic().to_string())
            )
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let coefficients = self.inner.coefficients();
        let displayed_coefficients = coefficients
            .iter()
            .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
            .map(|coefficient| integer_to_python(py, coefficient))
            .collect::<PyResult<Vec<_>>>()?;
        summary.set_item("type", "noether/numthy/FiniteFieldElement@1")?;
        summary.set_item("representative", self.representative(py)?)?;
        summary.set_item("coefficients", displayed_coefficients)?;
        if coefficients.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("coefficients_truncated", true)?;
        }
        summary.set_item(
            "characteristic",
            integer_to_python(py, self.inner.field().characteristic())?,
        )?;
        summary.set_item("degree", self.inner.field().degree())?;
        Ok(summary.unbind())
    }
}

/// Absolute precision measured in powers of a ramified uniformizer.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalPrecision",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalPrecision {
    inner: PadicLocalPrecision,
}

#[pymethods]
impl PyPadicLocalPrecision {
    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalPrecision::TYPE_ID
    }

    #[new]
    fn new(absolute_uniformizer_precision: u64) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalPrecision::new(absolute_uniformizer_precision)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn absolute_uniformizer_precision(&self) -> u64 {
        self.inner.absolute_uniformizer_precision()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalPrecision::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
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
            "PadicLocalPrecision({})",
            self.inner.absolute_uniformizer_precision()
        ))
    }
}

/// A finite extension of `Q_p` in explicit unramified-plus-Eisenstein form.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalField",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalField {
    pub(crate) inner: PadicLocalField,
}

#[pymethods]
impl PyPadicLocalField {
    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalField::TYPE_ID
    }

    #[new]
    #[pyo3(signature = (coefficient_ring, eisenstein_polynomial, precision, *, uniformizer_name=None))]
    fn new(
        coefficient_ring: &PyQadicRing,
        eisenstein_polynomial: &PyPolynomial,
        precision: &PyPadicLocalPrecision,
        uniformizer_name: Option<String>,
    ) -> PyResult<Self> {
        let polynomial = eisenstein_polynomial.as_qadic().ok_or_else(|| {
            PyValueError::new_err("expected an Eisenstein polynomial over a QadicRing")
        })?;
        Ok(Self {
            inner: PadicLocalField::from_eisenstein(
                coefficient_ring.inner.clone(),
                polynomial.clone(),
                precision.inner,
                uniformizer_name,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn coefficient_ring(&self) -> PyQadicRing {
        PyQadicRing {
            inner: self.inner.coefficient_ring().clone(),
        }
    }

    #[getter]
    fn coefficient_field(&self) -> PyQadicField {
        PyQadicField {
            inner: self.inner.coefficient_field().clone(),
        }
    }

    #[getter]
    fn residue_field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.residue_field().clone(),
        }
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn local_precision(&self) -> PyPadicLocalPrecision {
        PyPadicLocalPrecision {
            inner: self.inner.local_precision(),
        }
    }

    #[getter]
    fn ramification_index(&self) -> usize {
        self.inner.ramification_index()
    }

    #[getter]
    fn residue_degree(&self) -> usize {
        self.inner.residue_degree()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn defining_polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_qadic(self.inner.defining_polynomial().clone())
    }

    #[getter]
    fn defining_coefficients(&self) -> Vec<PyQadicElement> {
        self.inner
            .defining_coefficients()
            .iter()
            .cloned()
            .map(|inner| PyQadicElement { inner })
            .collect()
    }

    #[getter]
    fn uniformizer_name(&self) -> &str {
        self.inner.uniformizer_name()
    }

    #[getter]
    fn coordinate_precisions(&self) -> Vec<u64> {
        self.inner.coordinate_precisions().to_vec()
    }

    fn integral_element(
        &self,
        py: Python<'_>,
        coefficients: Vec<Py<PyQadicElement>>,
    ) -> PyResult<PyPadicLocalFieldElement> {
        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| coefficient.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .integral_element(&coefficients)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn element_from_qadic(&self, value: &PyQadicElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .element_from_qadic(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn element_from_padic(&self, value: &PyPadicElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .element_from_padic(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn zero(&self) -> PyPadicLocalFieldElement {
        PyPadicLocalFieldElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyPadicLocalFieldElement {
        PyPadicLocalFieldElement {
            inner: self.inner.one(),
        }
    }

    fn uniformizer(&self) -> PyPadicLocalFieldElement {
        PyPadicLocalFieldElement {
            inner: self.inner.uniformizer(),
        }
    }

    fn residue(&self, value: &PyPadicLocalFieldElement) -> PyResult<Option<PyFiniteFieldElement>> {
        Ok(self
            .inner
            .residue(&value.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .map(|inner| PyFiniteFieldElement { inner }))
    }

    fn teichmuller(&self, value: &PyFiniteFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .teichmuller(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn precision_restriction(
        &self,
        target_precision: &PyPadicLocalPrecision,
    ) -> PyResult<PyPadicLocalFieldRestriction> {
        Ok(PyPadicLocalFieldRestriction {
            inner: self
                .inner
                .precision_restriction(target_precision.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn restrict(
        &self,
        value: &PyPadicLocalFieldElement,
        target_precision: &PyPadicLocalPrecision,
    ) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .restrict(&value.inner, target_precision.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn coefficient_frobenius_conjugate(
        &self,
        exponent: u64,
    ) -> PyResult<PyPadicLocalFieldCoefficientFrobenius> {
        Ok(PyPadicLocalFieldCoefficientFrobenius {
            inner: self
                .inner
                .coefficient_frobenius_conjugate(exponent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn order(
        &self,
        py: Python<'_>,
        basis: Vec<Py<PyPadicLocalFieldElement>>,
    ) -> PyResult<PyPadicLocalFieldOrder> {
        let basis = basis
            .into_iter()
            .map(|element| element.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(PyPadicLocalFieldOrder {
            inner: self
                .inner
                .order(&basis)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn equation_order(&self) -> PyResult<PyPadicLocalFieldOrder> {
        Ok(PyPadicLocalFieldOrder {
            inner: self
                .inner
                .equation_order()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Compute and certify the maximal order of this finite p-adic local field.
    fn maximal_order(&self) -> PyResult<PyPadicLocalFieldOrder> {
        Ok(PyPadicLocalFieldOrder {
            inner: self
                .inner
                .maximal_order()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalField::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicLocalField({})", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// One normalized finite-precision value in a `PadicLocalField`.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalFieldElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalFieldElement {
    pub(crate) inner: PadicLocalFieldElement,
}

#[pymethods]
impl PyPadicLocalFieldElement {
    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalFieldElement::TYPE_ID
    }

    #[getter]
    fn field(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn normalized_coordinates(&self) -> Vec<Option<PyQadicElement>> {
        self.inner
            .normalized_coordinates()
            .iter()
            .map(|coordinate| {
                coordinate
                    .as_ref()
                    .cloned()
                    .map(|inner| PyQadicElement { inner })
            })
            .collect()
    }

    #[getter]
    fn known_absolute_precision(&self) -> i64 {
        self.inner.known_absolute_precision()
    }

    #[getter]
    fn relative_precision(&self) -> Option<u64> {
        self.inner.relative_precision()
    }

    fn valuation(&self) -> PyPadicValuation {
        PyPadicValuation {
            inner: self.inner.valuation(),
        }
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .inverse()
                .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))?,
        })
    }

    fn trace(&self) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .trace()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn norm(&self) -> PyResult<PyPadicElement> {
        Ok(PyPadicElement {
            inner: self
                .inner
                .norm()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn characteristic_polynomial(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_padic_field(
            self.inner
                .characteristic_polynomial()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn minimal_polynomial(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_padic_field(
            self.inner
                .minimal_polynomial()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn pow(&self, exponent: u64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalFieldElement::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .field()
                .add(&self.inner, &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .field()
                .sub(&self.inner, &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .field()
                .mul(&self.inner, &other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .field()
                .neg(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "PadicLocalFieldElement power does not accept Python's third modulus argument",
            ));
        }
        Ok(self.pow(exponent))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicLocalFieldElement({})", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A full-rank `Z_p`-order certified in one p-adic local field presentation.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalFieldOrder",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalFieldOrder {
    pub(crate) inner: PadicLocalFieldOrder,
}

#[pymethods]
impl PyPadicLocalFieldOrder {
    #[new]
    fn new(
        py: Python<'_>,
        field: &PyPadicLocalField,
        basis: Vec<Py<PyPadicLocalFieldElement>>,
    ) -> PyResult<Self> {
        let basis = basis
            .into_iter()
            .map(|element| element.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            inner: field
                .inner
                .order(&basis)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalFieldOrder::TYPE_ID
    }

    #[getter]
    fn field(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn basis(&self) -> Vec<PyPadicLocalFieldElement> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(|inner| PyPadicLocalFieldElement { inner })
            .collect()
    }

    #[getter]
    fn certified_precision(&self) -> u64 {
        self.inner.certified_precision()
    }

    #[getter]
    fn index_in_maximal_order_valuation(&self) -> u64 {
        self.inner.index_in_maximal_order_valuation()
    }

    fn zero(&self) -> PyPadicLocalFieldElement {
        PyPadicLocalFieldElement {
            inner: self.inner.zero(),
        }
    }

    fn one(&self) -> PyPadicLocalFieldElement {
        PyPadicLocalFieldElement {
            inner: self.inner.one(),
        }
    }

    fn __call__(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        if !self.inner.contains(&value.inner) {
            return Err(PyValueError::new_err(
                "p-adic local field element does not belong to this order",
            ));
        }
        Ok(value.clone())
    }

    fn __contains__(&self, value: &PyPadicLocalFieldElement) -> bool {
        self.inner.contains(&value.inner)
    }

    fn index_valuation(&self, super_order: &Self) -> PyResult<u64> {
        self.inner
            .index_valuation(&super_order.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn is_maximal(&self) -> bool {
        self.inner.is_maximal()
    }

    fn maximal_overorder(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .maximal_overorder()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn discriminant_valuation(&self) -> PyResult<u64> {
        self.inner
            .discriminant_valuation()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn different_valuation(&self) -> PyResult<u64> {
        self.inner
            .different_valuation()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn inverse_unit(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .inverse_unit(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn fractional_ideal(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPadicLocalFieldElement>>,
    ) -> PyResult<PyPadicLocalFractionalIdeal> {
        let generators = generators
            .into_iter()
            .map(|element| element.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(PyPadicLocalFractionalIdeal {
            inner: self
                .inner
                .fractional_ideal(&generators)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn maximal_ideal(&self) -> PyResult<PyPadicLocalFractionalIdeal> {
        Ok(PyPadicLocalFractionalIdeal {
            inner: self
                .inner
                .maximal_ideal()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalFieldOrder::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("{}", self.inner))
    }
}

/// A nonzero certified fractional lattice over a p-adic local order.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalFractionalIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalFractionalIdeal {
    inner: PadicLocalFractionalIdeal,
}

#[pymethods]
impl PyPadicLocalFractionalIdeal {
    #[new]
    fn new(
        py: Python<'_>,
        order: &PyPadicLocalFieldOrder,
        generators: Vec<Py<PyPadicLocalFieldElement>>,
    ) -> PyResult<Self> {
        let generators = generators
            .into_iter()
            .map(|element| element.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            inner: order
                .inner
                .fractional_ideal(&generators)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalFractionalIdeal::TYPE_ID
    }

    #[getter]
    fn order(&self) -> PyPadicLocalFieldOrder {
        PyPadicLocalFieldOrder {
            inner: self.inner.order().clone(),
        }
    }

    #[getter]
    fn basis(&self) -> Vec<PyPadicLocalFieldElement> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(|inner| PyPadicLocalFieldElement { inner })
            .collect()
    }

    #[getter]
    fn certified_precision(&self) -> u64 {
        self.inner.certified_precision()
    }

    fn __contains__(&self, value: &PyPadicLocalFieldElement) -> bool {
        self.inner.contains(&value.inner)
    }

    fn is_integral(&self) -> bool {
        self.inner.is_integral()
    }

    fn add(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .add(&other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn mul(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .mul(&other.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn scale(&self, scalar: &PyPadicLocalFieldElement) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .scale(&scalar.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn is_subset_of(&self, other: &Self) -> PyResult<bool> {
        self.inner
            .is_subset_of(&other.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn norm_valuation(&self) -> PyResult<i64> {
        self.inner
            .norm_valuation()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn maximal_ideal_exponent(&self) -> PyResult<i64> {
        self.inner
            .maximal_ideal_exponent()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .inverse()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalFractionalIdeal::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.add(other)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.mul(other)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("{}", self.inner))
    }
}

/// A checked digit-discarding map between ramified local precision contexts.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalFieldRestriction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalFieldRestriction {
    pub(crate) inner: PadicLocalFieldRestriction,
}

#[pymethods]
impl PyPadicLocalFieldRestriction {
    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalFieldRestriction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.target().clone(),
        }
    }

    fn apply(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value.as_padic_local_field().ok_or_else(|| {
            PyValueError::new_err("expected a polynomial over a p-adic local field")
        })?;
        Ok(PyPolynomial::from_padic_local_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic local field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic local field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic local field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_sparse_matrix(&self, value: &PySparseMatrix) -> PyResult<PySparseMatrix> {
        let SparseMatrixInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a sparse matrix over a p-adic local field",
            ));
        };
        Ok(PySparseMatrix {
            inner: SparseMatrixInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalFieldRestriction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("PadicLocalFieldRestriction({:?})", self.inner))
    }
}

/// Coefficient Frobenius from a local field to its conjugate presentation.
#[pyclass(
    module = "noether.numthy",
    name = "PadicLocalFieldCoefficientFrobenius",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicLocalFieldCoefficientFrobenius {
    pub(crate) inner: PadicLocalFieldCoefficientFrobenius,
}

#[pymethods]
impl PyPadicLocalFieldCoefficientFrobenius {
    #[classattr]
    fn type_id() -> &'static str {
        PadicLocalFieldCoefficientFrobenius::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn target(&self) -> PyPadicLocalField {
        PyPadicLocalField {
            inner: self.inner.target().clone(),
        }
    }

    #[getter]
    fn exponent(&self) -> u64 {
        self.inner.exponent()
    }

    fn is_endomorphism(&self) -> bool {
        self.inner.is_endomorphism()
    }

    fn apply(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        Ok(PyPadicLocalFieldElement {
            inner: self
                .inner
                .apply(&value.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn apply_polynomial(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value.as_padic_local_field().ok_or_else(|| {
            PyValueError::new_err("expected a polynomial over a p-adic local field")
        })?;
        Ok(PyPolynomial::from_padic_local_field(
            value
                .map_coefficients(&self.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        ))
    }

    fn apply_power_series(&self, value: &PyPowerSeries) -> PyResult<PyPowerSeries> {
        let SeriesInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a power series over a p-adic local field",
            ));
        };
        Ok(PyPowerSeries {
            inner: SeriesInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_vector(&self, value: &PyVector) -> PyResult<PyVector> {
        let VectorInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a vector over a p-adic local field",
            ));
        };
        Ok(PyVector {
            inner: VectorInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_matrix(&self, value: &PyMatrix) -> PyResult<PyMatrix> {
        let MatrixInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a matrix over a p-adic local field",
            ));
        };
        Ok(PyMatrix {
            inner: MatrixInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn apply_sparse_matrix(&self, value: &PySparseMatrix) -> PyResult<PySparseMatrix> {
        let SparseMatrixInner::PadicLocalField(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "expected a sparse matrix over a p-adic local field",
            ));
        };
        Ok(PySparseMatrix {
            inner: SparseMatrixInner::PadicLocalField(
                value
                    .map_coefficients(&self.inner)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn compose(&self, inner: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .compose(&inner.inner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PadicLocalFieldCoefficientFrobenius::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __call__(&self, value: &PyPadicLocalFieldElement) -> PyResult<PyPadicLocalFieldElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PadicLocalFieldCoefficientFrobenius({})",
            self.inner
        ))
    }
}

fn canonical_hash(value: &impl SemanticType) -> isize {
    use std::hash::{DefaultHasher, Hash, Hasher};

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
