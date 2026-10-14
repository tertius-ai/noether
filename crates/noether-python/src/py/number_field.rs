// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of canonical Antic-backed number fields.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    AffineNormalization, AffineNormalizationError, AssociatedPrimeError, AssociatedPrimeExtStage,
    AssociatedPrimeSet, DecompositionError, EquidimensionalHull, EquidimensionalRadicalComponent,
    EquidimensionalRadicalDecomposition, FiniteQuotientAlgebra, GrauertRemmertStep, Ideal,
    IdealDecomposition, MinimalPrimeDecomposition, MonomialIdealError, NoetherNormalization,
    NoetherNormalizationError, PolynomialPrimaryCertificate, PolynomialPrimeCertificate,
    PolynomialRing, PrimaryDecomposition, PrimeIdeal, RadicalIdeal, RationalFunctionField,
    RationalUnivariateRepresentation, RealSolutionBox, ReducedAffineNormalization,
    ReducedNormalizationError,
};
use noether::numthy::{
    FiniteField, FractionalIdeal, NumberField, NumberFieldElement, NumberFieldOrder,
};
use noether::{
    ComplexBallField, IntegerRing, RationalField, Ring, SemanticType, UnitInversion,
    UnsupportedOperationError,
};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyString, PyType};

use super::ball::{PyComplexBall, PyRealBall};
use super::coefficient_maps::{
    PyIntegerFiniteFieldMap, PyIntegerRationalEmbedding, PyPrimeSubfieldEmbedding,
};
use super::construction::{construction_error, PyAssociatedGradedAlgebra, PyReesAlgebra};
use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::ideal_evidence::{
    IdealEliminationInner, PyArithmeticFiberSupport, PyArithmeticNullstellensatzCertificate,
    PyIdealElimination, PyIdealReduction,
};
use super::module_facade::{PyFreeResolution, PyPresentedModule};
use super::monomial_ideal::{MonomialIdealInner, PyMonomialIdeal};
use super::numthy::{PyFiniteField, PyFiniteFieldElement, PyFiniteFieldEmbedding};
use super::polynomial_facade::{
    base_change_error, extract_finite_field, extract_finite_rational_function, extract_order,
    extract_rational_function, PyPolynomial, PyPolynomialRing, PyRealRootIsolation,
};
use super::quotient::{PyQuotientElement, PyQuotientRing, PyQuotientRingMap};
use super::rational_function::PyRationalFunction;
use super::root::PyIntegerRing;
use super::series::PyHilbertSeries;

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn unsupported_ideal_operation(operation: &'static str, context: &'static str) -> PyErr {
    super::errors::unsupported_operation(
        UnsupportedOperationError::new(format!("noether/commalg/Ideal/{operation}@1"), context)
            .with_supported_contexts([RationalFunctionField::<FiniteField>::TYPE_ID]),
    )
}

fn decomposition_error(operation: &'static str, error: DecompositionError) -> PyErr {
    match error {
        DecompositionError::ContextMismatch(error) => super::errors::context_mismatch(error),
        DecompositionError::Unsupported(detail) => super::errors::unsupported_operation(
            UnsupportedOperationError::new(
                format!("noether/commalg/Ideal/{operation}@1"),
                RationalFunctionField::<FiniteField>::TYPE_ID,
            )
            .with_detail(detail),
        ),
        DecompositionError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        _ => value_error(error),
    }
}

fn field_decomposition_error(
    operation: &'static str,
    context: &'static str,
    error: DecompositionError,
) -> PyErr {
    match error {
        DecompositionError::ContextMismatch(error) => super::errors::context_mismatch(error),
        DecompositionError::MonomialIdeal(MonomialIdealError::ContextMismatch(error)) => {
            super::errors::context_mismatch(error)
        }
        DecompositionError::Unsupported(detail) => super::errors::unsupported_operation(
            UnsupportedOperationError::new(format!("noether/commalg/Ideal/{operation}@1"), context)
                .with_detail(detail),
        ),
        DecompositionError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        DecompositionError::MonomialIdeal(MonomialIdealError::ResourceLimit { .. }) => {
            PyOverflowError::new_err(error.to_string())
        }
        _ => value_error(error),
    }
}

fn associated_prime_error(context: &'static str, error: AssociatedPrimeError) -> PyErr {
    match error {
        AssociatedPrimeError::Decomposition(error) => {
            field_decomposition_error("associated_primes", context, error)
        }
        _ => value_error(error),
    }
}

fn normalization_error(context: &'static str, error: NoetherNormalizationError) -> PyErr {
    match error {
        NoetherNormalizationError::ContextMismatch(error) => super::errors::context_mismatch(error),
        NoetherNormalizationError::ResourceLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        NoetherNormalizationError::SearchExhausted { .. } => super::errors::unsupported_operation(
            UnsupportedOperationError::new(
                "noether/commalg/Ideal/noether_normalization@1",
                context,
            )
            .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID])
            .with_detail(error.to_string()),
        ),
        _ => value_error(error),
    }
}

fn affine_normalization_error(context: &'static str, error: AffineNormalizationError) -> PyErr {
    match error {
        AffineNormalizationError::ContextMismatch(error) => super::errors::context_mismatch(error),
        AffineNormalizationError::ResourceLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        AffineNormalizationError::Unsupported(_) => super::errors::unsupported_operation(
            UnsupportedOperationError::new("noether/commalg/Ideal/normalization@1", context)
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID])
                .with_detail(error.to_string()),
        ),
        _ => value_error(error),
    }
}

fn reduced_normalization_error(context: &'static str, error: ReducedNormalizationError) -> PyErr {
    match error {
        ReducedNormalizationError::ContextMismatch(error) => super::errors::context_mismatch(error),
        ReducedNormalizationError::Component(AffineNormalizationError::ContextMismatch(error)) => {
            super::errors::context_mismatch(error)
        }
        ReducedNormalizationError::Component(AffineNormalizationError::ResourceLimit {
            ..
        }) => PyOverflowError::new_err(error.to_string()),
        ReducedNormalizationError::Component(AffineNormalizationError::Unsupported(_)) => {
            super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/reduced_normalization@1",
                    context,
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID])
                .with_detail(error.to_string()),
            )
        }
        _ => value_error(error),
    }
}

fn resolve_polynomial_variables(
    value: &Bound<'_, PyAny>,
    names: &[String],
) -> PyResult<Vec<usize>> {
    if value.cast::<PyString>().is_ok() {
        return Err(PyTypeError::new_err(
            "elimination variables must be an iterable of names or indices, not one string",
        ));
    }
    value
        .try_iter()
        .map_err(|_| {
            PyTypeError::new_err("elimination variables must be an iterable of names or indices")
        })?
        .map(|item| {
            let item = item?;
            if let Ok(index) = item.extract::<usize>() {
                return (index < names.len()).then_some(index).ok_or_else(|| {
                    value_error(format!(
                        "polynomial variable index {index} is outside arity {}",
                        names.len()
                    ))
                });
            }
            if let Ok(name) = item.extract::<String>() {
                return names
                    .iter()
                    .position(|candidate| candidate == &name)
                    .ok_or_else(|| value_error(format!("unknown polynomial variable {name:?}")));
            }
            Err(PyTypeError::new_err(
                "each elimination variable must be a nonnegative index or declared name",
            ))
        })
        .collect()
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

fn extract_quotient_coordinates<T>(
    values: &Bound<'_, PyAny>,
    dimension: usize,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    if values.cast::<PyString>().is_ok() {
        return Err(PyTypeError::new_err(
            "quotient coordinates must be an iterable, not a string",
        ));
    }
    let values = values
        .try_iter()
        .map_err(|_| PyTypeError::new_err("quotient coordinates must be an iterable"))?
        .map(|item| extract(&item?))
        .collect::<PyResult<Vec<_>>>()?;
    if values.len() != dimension {
        return Err(value_error(format!(
            "quotient coordinates have dimension {}, expected {dimension}",
            values.len()
        )));
    }
    Ok(values)
}

fn extract_number_field_operand(
    py: Python<'_>,
    field: &NumberField,
    value: &Bound<'_, PyAny>,
) -> PyResult<NumberFieldElement> {
    if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(
            "number-field element belongs to another context",
        ));
    }
    extract_rational(py, value)
        .map(|scalar| field.scalar(scalar))
        .map_err(|_| {
            PyTypeError::new_err(
                "number-field arithmetic requires an int, Fraction, or same-field element",
            )
        })
}

fn division_error(error: noether::numthy::NumberFieldError) -> PyErr {
    match error {
        noether::numthy::NumberFieldError::DivisionByZero(_) => {
            PyZeroDivisionError::new_err(error.to_string())
        }
        _ => value_error(error),
    }
}

pub(crate) fn extract_number_field_element(
    py: Python<'_>,
    field: &NumberField,
    value: &Bound<'_, PyAny>,
) -> PyResult<NumberFieldElement> {
    if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(
            "number-field element belongs to another context",
        ));
    }
    if let Ok(scalar) = extract_rational(py, value) {
        return Ok(field.scalar(scalar));
    }
    let coefficients = value
        .try_iter()
        .map_err(|_| {
            PyTypeError::new_err(
                "number-field value must be an int, Fraction, same-field element, or coefficient sequence",
            )
        })?
        .map(|coefficient| extract_rational(py, &coefficient?))
        .collect::<PyResult<Vec<_>>>()?;
    field.element(&coefficients).map_err(value_error)
}

fn extract_order_basis(
    py: Python<'_>,
    field: &NumberField,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<NumberFieldElement>> {
    value
        .try_iter()
        .map_err(|_| PyTypeError::new_err("number-field order basis must be iterable"))?
        .map(|element| extract_number_field_element(py, field, &element?))
        .collect()
}

pub(crate) fn extract_number_field_order_element(
    py: Python<'_>,
    order: &NumberFieldOrder,
    value: &Bound<'_, PyAny>,
) -> PyResult<NumberFieldElement> {
    let element = extract_number_field_element(py, order.field(), value)?;
    if order.contains(&element) {
        Ok(element)
    } else {
        Err(value_error("number-field element is not in this order"))
    }
}

/// An exact absolute number-field context.
#[pyclass(
    module = "noether.numthy",
    name = "NumberField",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNumberField {
    pub(crate) inner: NumberField,
}

#[pymethods]
impl PyNumberField {
    /// Construct `Q[a]/(f(a))` from a monic irreducible `Polynomial` over `Q`.
    #[new]
    #[pyo3(signature = (defining_polynomial, *, name=None))]
    fn new(defining_polynomial: &PyPolynomial, name: Option<String>) -> PyResult<Self> {
        let polynomial = defining_polynomial.as_rational().ok_or_else(|| {
            PyTypeError::new_err("NumberField defining polynomial must have coefficient ring Q")
        })?;
        Ok(Self {
            inner: NumberField::new(polynomial.clone(), name).map_err(value_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        NumberField::TYPE_ID
    }

    /// Normalized defining polynomial over `Q`.
    #[getter]
    fn defining_polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_rational(self.inner.defining_polynomial().clone())
    }

    /// Distinguished power-basis generator.
    #[getter]
    fn generator(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.generator(),
        }
    }

    /// Display name of the distinguished generator.
    #[getter]
    fn generator_name(&self) -> &str {
        self.inner.generator_name()
    }

    /// Degree over `Q`.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Signature `(r1, r2)`.
    fn signature(&self) -> PyResult<(usize, usize)> {
        self.inner.signature().map_err(value_error)
    }

    /// Additive identity.
    fn zero(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.zero(),
        }
    }

    /// Multiplicative identity.
    fn one(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.one(),
        }
    }

    /// Construct an element from an exact scalar or power-basis coefficients.
    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyNumberFieldElement> {
        Ok(PyNumberFieldElement {
            inner: extract_number_field_element(py, &self.inner, value)?,
        })
    }

    /// Validate the order spanned by an explicit integral basis.
    fn order(&self, py: Python<'_>, basis: &Bound<'_, PyAny>) -> PyResult<PyNumberFieldOrder> {
        let basis = extract_order_basis(py, &self.inner, basis)?;
        py.detach(|| {
            Ok(PyNumberFieldOrder {
                inner: self.inner.order(&basis).map_err(value_error)?,
            })
        })
    }

    /// The equation order `Z[a]`, when the distinguished generator is integral.
    fn equation_order(&self, py: Python<'_>) -> PyResult<PyNumberFieldOrder> {
        py.detach(|| {
            Ok(PyNumberFieldOrder {
                inner: self.inner.equation_order().map_err(value_error)?,
            })
        })
    }

    /// The maximal order (ring of integers), computed unconditionally.
    fn maximal_order(&self, py: Python<'_>) -> PyResult<PyNumberFieldOrder> {
        py.detach(|| {
            Ok(PyNumberFieldOrder {
                inner: self.inner.maximal_order().map_err(value_error)?,
            })
        })
    }

    /// Exact field discriminant, computed from the maximal order.
    fn discriminant(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let discriminant = py.detach(|| self.inner.discriminant().map_err(value_error))?;
        integer_to_python(py, &discriminant)
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Restore and validate a canonical context representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: NumberField::from_json(value).map_err(value_error)?,
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
                "NumberField(degree={}, name={:?})",
                self.inner.degree(),
                self.inner.generator_name()
            )
        })
    }
}

/// A validated exact order in an absolute number field.
#[pyclass(
    module = "noether.numthy",
    name = "NumberFieldOrder",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNumberFieldOrder {
    pub(crate) inner: NumberFieldOrder,
}

#[pymethods]
impl PyNumberFieldOrder {
    /// Validate an explicit integral basis.
    #[new]
    fn new(py: Python<'_>, field: &PyNumberField, basis: &Bound<'_, PyAny>) -> PyResult<Self> {
        let basis = extract_order_basis(py, &field.inner, basis)?;
        Ok(Self {
            inner: field.inner.order(&basis).map_err(value_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        NumberFieldOrder::TYPE_ID
    }

    /// Ambient exact number field.
    #[getter]
    fn field(&self) -> PyNumberField {
        PyNumberField {
            inner: self.inner.field().clone(),
        }
    }

    /// Canonical integral basis.
    #[getter]
    fn basis(&self) -> Vec<PyNumberFieldElement> {
        self.inner
            .basis()
            .into_iter()
            .map(|inner| PyNumberFieldElement { inner })
            .collect()
    }

    /// Exact order discriminant.
    #[getter]
    fn discriminant(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.discriminant())
    }

    /// Additive identity.
    fn zero(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.zero(),
        }
    }

    /// Multiplicative identity.
    fn one(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.one(),
        }
    }

    /// Construct an element, rejecting values outside this order.
    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyNumberFieldElement> {
        let element = extract_number_field_order_element(py, &self.inner, value)?;
        Ok(PyNumberFieldElement { inner: element })
    }

    /// Whether an exact value belongs to this order.
    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
            return Ok(self.inner.contains(&element.inner));
        }
        let element = extract_number_field_element(py, self.inner.field(), value)?;
        Ok(self.inner.contains(&element))
    }

    /// Index `[super_order : self]`, requiring containment.
    fn index(&self, super_order: &Self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(
            py,
            &self.inner.index(&super_order.inner).map_err(value_error)?,
        )
    }

    /// Whether this is the maximal order of its ambient field.
    fn is_maximal(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.is_maximal().map_err(value_error))
    }

    /// Construct the nonzero fractional ideal generated by exact field elements.
    fn fractional_ideal(
        &self,
        py: Python<'_>,
        generators: &Bound<'_, PyAny>,
    ) -> PyResult<PyFractionalIdeal> {
        let generators = extract_order_basis(py, self.inner.field(), generators)?;
        Ok(PyFractionalIdeal {
            inner: self
                .inner
                .fractional_ideal(&generators)
                .map_err(value_error)?,
        })
    }

    /// Checked prime ideals above a positive rational prime.
    fn prime_ideals_above(
        &self,
        py: Python<'_>,
        prime: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyPrimeIdeal>> {
        let prime = extract_integer(prime)?;
        py.detach(|| {
            Ok(self
                .inner
                .prime_ideals_above(&prime)
                .map_err(value_error)?
                .into_iter()
                .map(|inner| PyPrimeIdeal { inner })
                .collect())
        })
    }

    /// Invert a unit of this order.
    fn inverse_unit(&self, value: &PyNumberFieldElement) -> PyResult<PyNumberFieldElement> {
        Ok(PyNumberFieldElement {
            inner: self.inner.inverse_unit(&value.inner).map_err(value_error)?,
        })
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Restore and revalidate a canonical order representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: NumberFieldOrder::from_json(value).map_err(value_error)?,
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

/// A canonical nonzero fractional ideal of a number-field order.
#[pyclass(
    module = "noether.numthy",
    name = "FractionalIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFractionalIdeal {
    pub(crate) inner: FractionalIdeal,
}

#[pymethods]
impl PyFractionalIdeal {
    /// Construct the ideal generated over `order` by exact field elements.
    #[new]
    fn new(
        py: Python<'_>,
        order: &PyNumberFieldOrder,
        generators: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let generators = extract_order_basis(py, order.inner.field(), generators)?;
        Ok(Self {
            inner: order
                .inner
                .fractional_ideal(&generators)
                .map_err(value_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FractionalIdeal::TYPE_ID
    }

    /// Coefficient order of this ideal.
    #[getter]
    fn order(&self) -> PyNumberFieldOrder {
        PyNumberFieldOrder {
            inner: self.inner.order().clone(),
        }
    }

    /// Canonical `Z`-basis of this fractional ideal.
    #[getter]
    fn basis(&self) -> Vec<PyNumberFieldElement> {
        self.inner
            .basis()
            .into_iter()
            .map(|inner| PyNumberFieldElement { inner })
            .collect()
    }

    /// Exact absolute ideal norm.
    fn norm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, &self.inner.norm())
    }

    /// Whether the ideal is contained in its coefficient order.
    fn is_integral(&self) -> bool {
        self.inner.is_integral()
    }

    /// Whether an exact element lies in this fractional ideal.
    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
            return Ok(self.inner.contains(&element.inner));
        }
        let element = extract_number_field_element(py, self.inner.order().field(), value)?;
        Ok(self.inner.contains(&element))
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.add(&other.inner).map_err(value_error)?,
        })
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.mul(&other.inner).map_err(value_error)?,
        })
    }

    /// Signed valuation at a checked prime ideal of the same maximal order.
    fn valuation(&self, prime: &PyPrimeIdeal) -> PyResult<i64> {
        self.inner.valuation(&prime.inner).map_err(value_error)
    }

    /// Complete signed prime-ideal factorization.
    fn factor(&self, py: Python<'_>) -> PyResult<Vec<(PyPrimeIdeal, i64)>> {
        py.detach(|| {
            Ok(self
                .inner
                .factor()
                .map_err(value_error)?
                .into_iter()
                .map(|(inner, exponent)| (PyPrimeIdeal { inner }, exponent))
                .collect())
        })
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Restore and revalidate a canonical fractional ideal.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FractionalIdeal::from_json(value).map_err(value_error)?,
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

/// A checked prime ideal of a number-field order.
#[pyclass(
    module = "noether.commalg",
    name = "PrimeIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPrimeIdeal {
    pub(crate) inner: PrimeIdeal<NumberFieldOrder>,
}

#[pymethods]
impl PyPrimeIdeal {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PrimeIdeal::<NumberFieldOrder>::TYPE_ID
    }

    /// Coefficient order of this prime ideal.
    #[getter]
    fn order(&self) -> PyNumberFieldOrder {
        PyNumberFieldOrder {
            inner: self.inner.order().clone(),
        }
    }

    /// Canonical integral basis.
    #[getter]
    fn basis(&self) -> Vec<PyNumberFieldElement> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(|inner| PyNumberFieldElement { inner })
            .collect()
    }

    /// Underlying ideal in number-field fractional-ideal arithmetic.
    #[getter]
    fn ideal(&self) -> PyIdeal {
        PyIdeal {
            inner: IdealInner::NumberField(self.inner.ideal().clone()),
        }
    }

    /// Positive rational prime below this ideal.
    #[getter]
    fn rational_prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.rational_prime())
    }

    /// Ramification index over the rational prime.
    #[getter]
    fn ramification_index(&self) -> usize {
        self.inner.ramification_index()
    }

    /// Residue-field degree over the rational prime field.
    #[getter]
    fn residue_degree(&self) -> usize {
        self.inner.residue_degree()
    }

    /// Absolute norm `p^f`.
    fn norm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.norm())
    }

    /// Recheck primality and the ramification/residue witness.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    /// Whether an exact element belongs to this prime ideal.
    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
            return Ok(self.inner.contains(&element.inner));
        }
        let element = extract_number_field_element(py, self.inner.order().field(), value)?;
        Ok(self.inner.contains(&element))
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Restore and reverify a canonical number-field prime ideal.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PrimeIdeal::<NumberFieldOrder>::from_json(value).map_err(value_error)?,
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
                "PrimeIdeal(p={}, e={}, f={})",
                self.inner.rational_prime(),
                self.inner.ramification_index(),
                self.inner.residue_degree()
            )
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum IdealInner {
    Integer(Ideal<IntegerRing>),
    NumberField(Ideal<NumberFieldOrder>),
    IntegerPolynomial(Ideal<PolynomialRing<IntegerRing>>),
    RationalPolynomial(Ideal<PolynomialRing<RationalField>>),
    FiniteFieldPolynomial(Ideal<PolynomialRing<FiniteField>>),
    RationalFunctionPolynomial(Ideal<PolynomialRing<RationalFunctionField<RationalField>>>),
    FiniteRationalFunctionPolynomial(Ideal<PolynomialRing<RationalFunctionField<FiniteField>>>),
}

impl IdealInner {
    fn coefficient_context_type_id(&self) -> &'static str {
        match self {
            Self::Integer(_) | Self::IntegerPolynomial(_) => IntegerRing::TYPE_ID,
            Self::NumberField(_) => NumberFieldOrder::TYPE_ID,
            Self::RationalPolynomial(_) => RationalField::TYPE_ID,
            Self::FiniteFieldPolynomial(_) => FiniteField::TYPE_ID,
            Self::RationalFunctionPolynomial(_) => RationalFunctionField::<RationalField>::TYPE_ID,
            Self::FiniteRationalFunctionPolynomial(_) => {
                RationalFunctionField::<FiniteField>::TYPE_ID
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum FiniteQuotientAlgebraInner {
    Rational(Box<FiniteQuotientAlgebra<RationalField>>),
    FiniteField(Box<FiniteQuotientAlgebra<FiniteField>>),
    RationalFunction(Box<FiniteQuotientAlgebra<RationalFunctionField<RationalField>>>),
    FiniteRationalFunction(Box<FiniteQuotientAlgebra<RationalFunctionField<FiniteField>>>),
}

/// A verified standard-basis description of a finite-dimensional quotient.
#[pyclass(
    module = "noether.commalg",
    name = "FiniteQuotientAlgebra",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteQuotientAlgebra {
    inner: FiniteQuotientAlgebraInner,
}

impl PyFiniteQuotientAlgebra {
    pub(crate) fn from_rational(inner: FiniteQuotientAlgebra<RationalField>) -> Self {
        Self {
            inner: FiniteQuotientAlgebraInner::Rational(Box::new(inner)),
        }
    }

    pub(crate) fn from_finite_field(inner: FiniteQuotientAlgebra<FiniteField>) -> Self {
        Self {
            inner: FiniteQuotientAlgebraInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyFiniteQuotientAlgebra {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FiniteQuotientAlgebra::<RationalField>::TYPE_ID
    }

    /// Source ideal whose quotient is represented.
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            FiniteQuotientAlgebraInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
            FiniteQuotientAlgebraInner::RationalFunction(value) => {
                PyIdeal::from_rational_function_polynomial(value.source().clone())
            }
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => {
                PyIdeal::from_finite_rational_function_polynomial(value.source().clone())
            }
        }
    }

    /// Canonically ordered standard-monomial basis.
    #[getter]
    fn basis(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational_function)
                .collect(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value
                .basis()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_rational_function)
                .collect(),
        }
    }

    /// Vector-space dimension of the quotient.
    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value.dimension(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value.dimension(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value.dimension(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value.dimension(),
        }
    }

    /// Whether the source ideal is the unit ideal and this is the zero algebra.
    #[getter]
    fn is_zero(&self) -> bool {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value.is_zero(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value.is_zero(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value.is_zero(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value.is_zero(),
        }
    }

    /// Nonzero counts of the sparse variable-multiplication operators.
    #[getter]
    fn multiplication_operator_nonzeros(&self) -> Vec<usize> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value.multiplication_operator_nonzeros(),
            FiniteQuotientAlgebraInner::FiniteField(value) => {
                value.multiplication_operator_nonzeros()
            }
            FiniteQuotientAlgebraInner::RationalFunction(value) => {
                value.multiplication_operator_nonzeros()
            }
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => {
                value.multiplication_operator_nonzeros()
            }
        }
    }

    /// Sparse `(row, column, coefficient)` entries for one variable operator.
    fn multiplication_operator(
        &self,
        py: Python<'_>,
        variable: usize,
    ) -> PyResult<Vec<(usize, usize, Py<PyAny>)>> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value
                .multiplication_operator_entries(variable)
                .map_err(value_error)?
                .into_iter()
                .map(|(row, column, coefficient)| {
                    Ok((row, column, rational_to_python(py, &coefficient)?))
                })
                .collect(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value
                .multiplication_operator_entries(variable)
                .map_err(value_error)?
                .into_iter()
                .map(|(row, column, inner)| {
                    Ok((
                        row,
                        column,
                        Py::new(py, PyFiniteFieldElement { inner })?.into_any(),
                    ))
                })
                .collect(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value
                .multiplication_operator_entries(variable)
                .map_err(value_error)?
                .into_iter()
                .map(|(row, column, value)| {
                    Ok((
                        row,
                        column,
                        Py::new(py, PyRationalFunction::from_rational(value))?.into_any(),
                    ))
                })
                .collect(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value
                .multiplication_operator_entries(variable)
                .map_err(value_error)?
                .into_iter()
                .map(|(row, column, value)| {
                    Ok((
                        row,
                        column,
                        Py::new(py, PyRationalFunction::from_finite_field(value))?.into_any(),
                    ))
                })
                .collect(),
        }
    }

    /// Coordinates of the multiplicative identity.
    fn one_coordinates(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value
                .one_coordinates()
                .iter()
                .map(|coefficient| rational_to_python(py, coefficient))
                .collect(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value
                .one_coordinates()
                .into_iter()
                .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                .collect(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value
                .one_coordinates()
                .into_iter()
                .map(|value| Ok(Py::new(py, PyRationalFunction::from_rational(value))?.into_any()))
                .collect(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value
                .one_coordinates()
                .into_iter()
                .map(|value| {
                    Ok(Py::new(py, PyRationalFunction::from_finite_field(value))?.into_any())
                })
                .collect(),
        }
    }

    /// Standard-basis coordinates of a source polynomial's normal form.
    fn normal_form_coordinates(
        &self,
        py: Python<'_>,
        polynomial: &PyPolynomial,
    ) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value
                .normal_form_coordinates(polynomial.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("polynomial has the wrong coefficient context")
                })?)
                .map_err(value_error)?
                .iter()
                .map(|coefficient| rational_to_python(py, coefficient))
                .collect(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value
                .normal_form_coordinates(polynomial.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err("polynomial has the wrong coefficient context")
                })?)
                .map_err(value_error)?
                .into_iter()
                .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                .collect(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value
                .normal_form_coordinates(polynomial.as_rational_function().ok_or_else(|| {
                    PyTypeError::new_err("polynomial has the wrong coefficient context")
                })?)
                .map_err(value_error)?
                .into_iter()
                .map(|value| Ok(Py::new(py, PyRationalFunction::from_rational(value))?.into_any()))
                .collect(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value
                .normal_form_coordinates(polynomial.as_finite_rational_function().ok_or_else(
                    || PyTypeError::new_err("polynomial has the wrong coefficient context"),
                )?)
                .map_err(value_error)?
                .into_iter()
                .map(|value| {
                    Ok(Py::new(py, PyRationalFunction::from_finite_field(value))?.into_any())
                })
                .collect(),
        }
    }

    /// Multiply two standard-basis coordinate vectors.
    fn multiply(
        &self,
        py: Python<'_>,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => {
                let left = extract_quotient_coordinates(left, value.dimension(), |item| {
                    extract_rational(py, item)
                })?;
                let right = extract_quotient_coordinates(right, value.dimension(), |item| {
                    extract_rational(py, item)
                })?;
                value
                    .multiply(&left, &right)
                    .map_err(value_error)?
                    .iter()
                    .map(|coefficient| rational_to_python(py, coefficient))
                    .collect()
            }
            FiniteQuotientAlgebraInner::FiniteField(value) => {
                let field = value.source().ring().coefficient_ring();
                let left = extract_quotient_coordinates(left, value.dimension(), |item| {
                    extract_finite_field(field, item)
                })?;
                let right = extract_quotient_coordinates(right, value.dimension(), |item| {
                    extract_finite_field(field, item)
                })?;
                value
                    .multiply(&left, &right)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                    .collect()
            }
            FiniteQuotientAlgebraInner::RationalFunction(value) => {
                let field = value.source().ring().coefficient_ring();
                let left = extract_quotient_coordinates(left, value.dimension(), |item| {
                    extract_rational_function(py, field, item)
                })?;
                let right = extract_quotient_coordinates(right, value.dimension(), |item| {
                    extract_rational_function(py, field, item)
                })?;
                value
                    .multiply(&left, &right)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|value| {
                        Ok(Py::new(py, PyRationalFunction::from_rational(value))?.into_any())
                    })
                    .collect()
            }
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => {
                let field = value.source().ring().coefficient_ring();
                let left = extract_quotient_coordinates(left, value.dimension(), |item| {
                    extract_finite_rational_function(field, item)
                })?;
                let right = extract_quotient_coordinates(right, value.dimension(), |item| {
                    extract_finite_rational_function(field, item)
                })?;
                value
                    .multiply(&left, &right)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|value| {
                        Ok(Py::new(py, PyRationalFunction::from_finite_field(value))?.into_any())
                    })
                    .collect()
            }
        }
    }

    /// Apply multiplication by one source variable to coordinates.
    fn multiply_by_variable(
        &self,
        py: Python<'_>,
        variable: usize,
        coordinates: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => {
                let coordinates =
                    extract_quotient_coordinates(coordinates, value.dimension(), |item| {
                        extract_rational(py, item)
                    })?;
                value
                    .multiply_by_variable(variable, &coordinates)
                    .map_err(value_error)?
                    .iter()
                    .map(|coefficient| rational_to_python(py, coefficient))
                    .collect()
            }
            FiniteQuotientAlgebraInner::FiniteField(value) => {
                let field = value.source().ring().coefficient_ring();
                let coordinates =
                    extract_quotient_coordinates(coordinates, value.dimension(), |item| {
                        extract_finite_field(field, item)
                    })?;
                value
                    .multiply_by_variable(variable, &coordinates)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                    .collect()
            }
            FiniteQuotientAlgebraInner::RationalFunction(value) => {
                let field = value.source().ring().coefficient_ring();
                let coordinates =
                    extract_quotient_coordinates(coordinates, value.dimension(), |item| {
                        extract_rational_function(py, field, item)
                    })?;
                value
                    .multiply_by_variable(variable, &coordinates)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|value| {
                        Ok(Py::new(py, PyRationalFunction::from_rational(value))?.into_any())
                    })
                    .collect()
            }
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => {
                let field = value.source().ring().coefficient_ring();
                let coordinates =
                    extract_quotient_coordinates(coordinates, value.dimension(), |item| {
                        extract_finite_rational_function(field, item)
                    })?;
                value
                    .multiply_by_variable(variable, &coordinates)
                    .map_err(value_error)?
                    .into_iter()
                    .map(|value| {
                        Ok(Py::new(py, PyRationalFunction::from_finite_field(value))?.into_any())
                    })
                    .collect()
            }
        }
    }

    /// Recompute and exactly verify this algebra description.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value.verify(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value.verify(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value.verify(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Canonical typed JSON representation.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => value.to_json(),
            FiniteQuotientAlgebraInner::FiniteField(value) => value.to_json(),
            FiniteQuotientAlgebraInner::RationalFunction(value) => value.to_json(),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Restore and exactly reverify a typed quotient algebra.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient_ring| coefficient_ring.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self {
                inner: FiniteQuotientAlgebraInner::Rational(Box::new(
                    FiniteQuotientAlgebra::<RationalField>::from_json(value)
                        .map_err(value_error)?,
                )),
            }),
            Some(FiniteField::TYPE_ID) => Ok(Self {
                inner: FiniteQuotientAlgebraInner::FiniteField(Box::new(
                    FiniteQuotientAlgebra::<FiniteField>::from_json(value).map_err(value_error)?,
                )),
            }),
            Some(RationalFunctionField::<FiniteField>::TYPE_ID) => {
                let base_type = payload
                    .get("source")
                    .and_then(|source| source.get("ring"))
                    .and_then(|ring| ring.get("coefficient_ring"))
                    .and_then(|field| field.get("coefficient_ring"))
                    .and_then(|base| base.get("type"))
                    .and_then(serde_json::Value::as_str);
                match base_type {
                    Some(RationalField::TYPE_ID) => Ok(Self {
                        inner: FiniteQuotientAlgebraInner::RationalFunction(Box::new(
                            FiniteQuotientAlgebra::<RationalFunctionField<RationalField>>::from_json(
                                value,
                            )
                            .map_err(value_error)?,
                        )),
                    }),
                    Some(FiniteField::TYPE_ID) => Ok(Self {
                        inner: FiniteQuotientAlgebraInner::FiniteRationalFunction(Box::new(
                            FiniteQuotientAlgebra::<RationalFunctionField<FiniteField>>::from_json(
                                value,
                            )
                            .map_err(value_error)?,
                        )),
                    }),
                    _ => Err(PyTypeError::new_err(
                        "unsupported rational-function base field in serialized finite quotient algebra",
                    )),
                }
            }
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported finite-quotient coefficient context {type_id:?}"
            ))),
            None => Err(value_error(
                "serialized finite quotient algebra has no source coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            FiniteQuotientAlgebraInner::Rational(value) => canonical_hash(value.as_ref()),
            FiniteQuotientAlgebraInner::FiniteField(value) => canonical_hash(value.as_ref()),
            FiniteQuotientAlgebraInner::RationalFunction(value) => canonical_hash(value.as_ref()),
            FiniteQuotientAlgebraInner::FiniteRationalFunction(value) => {
                canonical_hash(value.as_ref())
            }
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteQuotientAlgebra(dimension={}, variables={})",
            self.dimension(),
            self.multiplication_operator_nonzeros().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum RationalUnivariateRepresentationInner {
    Rational(Box<RationalUnivariateRepresentation<RationalField>>),
    FiniteField(Box<RationalUnivariateRepresentation<FiniteField>>),
    RationalFunction(Box<RationalUnivariateRepresentation<RationalFunctionField<RationalField>>>),
    FiniteRationalFunction(
        Box<RationalUnivariateRepresentation<RationalFunctionField<FiniteField>>>,
    ),
}

/// A verified one-parameter representation of a zero-dimensional ideal.
#[pyclass(
    module = "noether.commalg",
    name = "RationalUnivariateRepresentation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRationalUnivariateRepresentation {
    inner: RationalUnivariateRepresentationInner,
}

#[pymethods]
impl PyRationalUnivariateRepresentation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RationalUnivariateRepresentation::<RationalField>::TYPE_ID
    }

    /// Source ideal represented by this value.
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            RationalUnivariateRepresentationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                PyIdeal::from_rational_function_polynomial(value.source().clone())
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                PyIdeal::from_finite_rational_function_polynomial(value.source().clone())
            }
        }
    }

    /// Canonical univariate parameter context.
    #[getter]
    fn parameter_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.parameter_ring().clone())
            }
            RationalUnivariateRepresentationInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.parameter_ring().clone())
            }
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                PyPolynomialRing::from_rational_function(value.parameter_ring().clone())
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                PyPolynomialRing::from_finite_rational_function(value.parameter_ring().clone())
            }
        }
    }

    /// Monic equation for the selected parameter.
    #[getter]
    fn equation(&self) -> PyPolynomial {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => {
                PyPolynomial::from_rational(value.equation().clone())
            }
            RationalUnivariateRepresentationInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.equation().clone())
            }
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                PyPolynomial::from_rational_function(value.equation().clone())
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                PyPolynomial::from_finite_rational_function(value.equation().clone())
            }
        }
    }

    /// Coordinate numerators in source-variable order.
    #[getter]
    fn coordinate_numerators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value
                .coordinate_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value
                .coordinate_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => value
                .coordinate_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational_function)
                .collect(),
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => value
                .coordinate_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_rational_function)
                .collect(),
        }
    }

    /// Common coordinate denominator.
    #[getter]
    fn denominator(&self) -> PyPolynomial {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => {
                PyPolynomial::from_rational(value.denominator().clone())
            }
            RationalUnivariateRepresentationInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.denominator().clone())
            }
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                PyPolynomial::from_rational_function(value.denominator().clone())
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                PyPolynomial::from_finite_rational_function(value.denominator().clone())
            }
        }
    }

    /// Coefficients of the selected source-coordinate linear form.
    #[getter]
    fn linear_form(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value
                .linear_form()
                .iter()
                .map(|coefficient| rational_to_python(py, coefficient))
                .collect(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value
                .linear_form()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyFiniteFieldElement { inner }).map(|value| value.into_any())
                })
                .collect(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => value
                .linear_form()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_rational(value))
                        .map(|value| value.into_any())
                })
                .collect(),
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => value
                .linear_form()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyRationalFunction::from_finite_field(value))
                        .map(|value| value.into_any())
                })
                .collect(),
        }
    }

    /// Vector-space dimension of the quotient algebra.
    #[getter]
    fn quotient_dimension(&self) -> usize {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value.quotient_dimension(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value.quotient_dimension(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                value.quotient_dimension()
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                value.quotient_dimension()
            }
        }
    }

    /// Whether the represented quotient is reduced.
    #[getter]
    fn is_reduced(&self) -> bool {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value.is_reduced(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value.is_reduced(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => value.is_reduced(),
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                value.is_reduced()
            }
        }
    }

    /// Certified real solution boxes for a rational RUR.
    fn real_solutions(&self, precision: u64) -> PyResult<Vec<PyRealSolutionBox>> {
        let RationalUnivariateRepresentationInner::Rational(value) = &self.inner else {
            return Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/RationalUnivariateRepresentation/real_solutions@1",
                    match &self.inner {
                        RationalUnivariateRepresentationInner::Rational(_) => {
                            RationalField::TYPE_ID
                        }
                        RationalUnivariateRepresentationInner::FiniteField(_) => {
                            FiniteField::TYPE_ID
                        }
                        RationalUnivariateRepresentationInner::RationalFunction(_) => {
                            RationalFunctionField::<RationalField>::TYPE_ID
                        }
                        RationalUnivariateRepresentationInner::FiniteRationalFunction(_) => {
                            RationalFunctionField::<FiniteField>::TYPE_ID
                        }
                    },
                )
                .with_supported_contexts([RationalField::TYPE_ID]),
            ));
        };
        Ok(value
            .real_solutions(precision)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyRealSolutionBox { inner })
            .collect())
    }

    /// Recompute and exactly verify the representation.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value.verify(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value.verify(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => value.verify(),
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Canonical typed JSON representation.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => value.to_json(),
            RationalUnivariateRepresentationInner::FiniteField(value) => value.to_json(),
            RationalUnivariateRepresentationInner::RationalFunction(value) => value.to_json(),
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Restore and exactly reverify a typed representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient_ring| coefficient_ring.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self {
                inner: RationalUnivariateRepresentationInner::Rational(Box::new(
                    RationalUnivariateRepresentation::<RationalField>::from_json(value)
                        .map_err(value_error)?,
                )),
            }),
            Some(FiniteField::TYPE_ID) => Ok(Self {
                inner: RationalUnivariateRepresentationInner::FiniteField(Box::new(
                    RationalUnivariateRepresentation::<FiniteField>::from_json(value)
                        .map_err(value_error)?,
                )),
            }),
            Some(RationalFunctionField::<FiniteField>::TYPE_ID) => {
                let base_type = payload
                    .get("source")
                    .and_then(|source| source.get("ring"))
                    .and_then(|ring| ring.get("coefficient_ring"))
                    .and_then(|field| field.get("coefficient_ring"))
                    .and_then(|base| base.get("type"))
                    .and_then(serde_json::Value::as_str);
                match base_type {
                    Some(RationalField::TYPE_ID) => Ok(Self {
                        inner: RationalUnivariateRepresentationInner::RationalFunction(
                            Box::new(
                                RationalUnivariateRepresentation::<
                                    RationalFunctionField<RationalField>,
                                >::from_json(value)
                                .map_err(value_error)?,
                            ),
                        ),
                    }),
                    Some(FiniteField::TYPE_ID) => Ok(Self {
                        inner: RationalUnivariateRepresentationInner::FiniteRationalFunction(
                            Box::new(
                                RationalUnivariateRepresentation::<
                                    RationalFunctionField<FiniteField>,
                                >::from_json(value)
                                .map_err(value_error)?,
                            ),
                        ),
                    }),
                    _ => Err(PyTypeError::new_err(
                        "unsupported rational-function base field in serialized RUR",
                    )),
                }
            }
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported RUR coefficient context {type_id:?}"
            ))),
            None => Err(value_error(
                "serialized RUR has no source coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            RationalUnivariateRepresentationInner::Rational(value) => {
                canonical_hash(value.as_ref())
            }
            RationalUnivariateRepresentationInner::FiniteField(value) => {
                canonical_hash(value.as_ref())
            }
            RationalUnivariateRepresentationInner::RationalFunction(value) => {
                canonical_hash(value.as_ref())
            }
            RationalUnivariateRepresentationInner::FiniteRationalFunction(value) => {
                canonical_hash(value.as_ref())
            }
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "RationalUnivariateRepresentation(dimension={}, reduced={})",
            self.quotient_dimension(),
            self.is_reduced()
        ))
    }
}

/// One certified real solution box produced from a rational RUR.
#[pyclass(
    module = "noether.commalg",
    name = "RealSolutionBox",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRealSolutionBox {
    inner: RealSolutionBox,
}

#[pymethods]
impl PyRealSolutionBox {
    #[classattr]
    fn type_id() -> &'static str {
        RealSolutionBox::TYPE_ID
    }

    #[getter]
    fn representation(&self) -> PyRationalUnivariateRepresentation {
        PyRationalUnivariateRepresentation {
            inner: RationalUnivariateRepresentationInner::Rational(Box::new(
                self.inner.representation().clone(),
            )),
        }
    }

    #[getter]
    fn parameter(&self) -> PyRealRootIsolation {
        PyRealRootIsolation {
            inner: self.inner.parameter().clone(),
        }
    }

    #[getter]
    fn parameter_ball(&self) -> PyRealBall {
        PyRealBall {
            inner: self.inner.parameter_ball().clone(),
        }
    }

    #[getter]
    fn coordinates(&self) -> Vec<PyRealBall> {
        self.inner
            .coordinates()
            .iter()
            .cloned()
            .map(|inner| PyRealBall { inner })
            .collect()
    }

    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision()
    }

    #[getter]
    fn multiplicity(&self) -> usize {
        self.inner.multiplicity()
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
            inner: RealSolutionBox::from_json(value).map_err(value_error)?,
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
}

/// A canonical ideal of one accepted commutative-ring context.
#[pyclass(module = "noether.commalg", name = "Ideal", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyIdeal {
    pub(crate) inner: IdealInner,
}

impl PyIdeal {
    pub(crate) fn from_integer(inner: Ideal<IntegerRing>) -> Self {
        Self {
            inner: IdealInner::Integer(inner),
        }
    }

    pub(crate) fn from_integer_polynomial(inner: Ideal<PolynomialRing<IntegerRing>>) -> Self {
        Self {
            inner: IdealInner::IntegerPolynomial(inner),
        }
    }

    pub(crate) fn from_rational_polynomial(inner: Ideal<PolynomialRing<RationalField>>) -> Self {
        Self {
            inner: IdealInner::RationalPolynomial(inner),
        }
    }

    pub(crate) fn from_finite_field_polynomial(inner: Ideal<PolynomialRing<FiniteField>>) -> Self {
        Self {
            inner: IdealInner::FiniteFieldPolynomial(inner),
        }
    }

    pub(crate) fn from_rational_function_polynomial(
        inner: Ideal<PolynomialRing<RationalFunctionField<RationalField>>>,
    ) -> Self {
        Self {
            inner: IdealInner::RationalFunctionPolynomial(inner),
        }
    }

    pub(crate) fn from_finite_rational_function_polynomial(
        inner: Ideal<PolynomialRing<RationalFunctionField<FiniteField>>>,
    ) -> Self {
        Self {
            inner: IdealInner::FiniteRationalFunctionPolynomial(inner),
        }
    }
}

impl PyRationalUnivariateRepresentation {
    pub(crate) fn from_finite_field(inner: RationalUnivariateRepresentation<FiniteField>) -> Self {
        Self {
            inner: RationalUnivariateRepresentationInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyIdeal {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Ideal::<NumberFieldOrder>::TYPE_ID
    }

    /// Coefficient ring of this ideal.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            IdealInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            IdealInner::NumberField(ideal) => Ok(Py::new(
                py,
                PyNumberFieldOrder {
                    inner: ideal.ring().clone(),
                },
            )?
            .into_any()),
            IdealInner::IntegerPolynomial(ideal) => {
                Ok(Py::new(py, PyPolynomialRing::from_integer(ideal.ring().clone()))?.into_any())
            }
            IdealInner::RationalPolynomial(ideal) => {
                Ok(Py::new(py, PyPolynomialRing::from_rational(ideal.ring().clone()))?.into_any())
            }
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(ideal.ring().clone()),
            )?
            .into_any()),
            IdealInner::RationalFunctionPolynomial(ideal) => Ok(Py::new(
                py,
                PyPolynomialRing::from_rational_function(ideal.ring().clone()),
            )?
            .into_any()),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_rational_function(ideal.ring().clone()),
            )?
            .into_any()),
        }
    }

    /// Extend a finite-field polynomial ideal and recompute its canonical
    /// reduced Groebner basis over the target field.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        match &self.inner {
            IdealInner::IntegerPolynomial(ideal)
                if map.is_instance_of::<PyIntegerRationalEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerRationalEmbedding>>()?;
                Ok(Self::from_rational_polynomial(
                    ideal.base_change(&map.inner).map_err(base_change_error)?,
                ))
            }
            IdealInner::IntegerPolynomial(ideal)
                if map.is_instance_of::<PyIntegerFiniteFieldMap>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerFiniteFieldMap>>()?;
                Ok(Self::from_finite_field_polynomial(
                    ideal.base_change(&map.inner).map_err(base_change_error)?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal)
                if map.is_instance_of::<PyPrimeSubfieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>()?;
                Ok(Self::from_finite_field_polynomial(
                    ideal.base_change(&map.inner).map_err(base_change_error)?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal)
                if map.is_instance_of::<PyFiniteFieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>()?;
                Ok(Self::from_finite_field_polynomial(
                    ideal.base_change(&map.inner).map_err(base_change_error)?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "unsupported ideal base-change context or coefficient map",
            )),
        }
    }

    /// Canonical generating family.
    #[getter]
    fn generators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            IdealInner::Integer(ideal) => ideal
                .generators()
                .iter()
                .map(|value| integer_to_python(py, value))
                .collect(),
            IdealInner::NumberField(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|inner| {
                    Py::new(py, PyNumberFieldElement { inner }).map(|value| value.into_any())
                })
                .collect(),
            IdealInner::IntegerPolynomial(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyPolynomial::from_integer(value)).map(|value| value.into_any())
                })
                .collect(),
            IdealInner::RationalPolynomial(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyPolynomial::from_rational(value)).map(|value| value.into_any())
                })
                .collect(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyPolynomial::from_finite_field(value))
                        .map(|value| value.into_any())
                })
                .collect(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyPolynomial::from_rational_function(value))
                        .map(|value| value.into_any())
                })
                .collect(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal
                .generators()
                .iter()
                .cloned()
                .map(|value| {
                    Py::new(py, PyPolynomial::from_finite_rational_function(value))
                        .map(|value| value.into_any())
                })
                .collect(),
        }
    }

    /// Canonical nonnegative generator of a principal ideal of `Z`.
    fn principal_generator(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let IdealInner::Integer(ideal) = &self.inner else {
            return Err(PyTypeError::new_err(
                "principal_generator is defined for ideals of Z",
            ));
        };
        integer_to_python(py, &ideal.principal_generator())
    }

    /// Canonical reduced Groebner basis for polynomial ideals.
    fn groebner_basis(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            IdealInner::IntegerPolynomial(_)
            | IdealInner::RationalPolynomial(_)
            | IdealInner::FiniteFieldPolynomial(_)
            | IdealInner::RationalFunctionPolynomial(_)
            | IdealInner::FiniteRationalFunctionPolynomial(_) => self.generators(py),
            IdealInner::Integer(_) | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "groebner_basis is defined for polynomial ideals",
            )),
        }
    }

    /// Canonical normal form modulo a polynomial ideal.
    fn normal_form(&self, py: Python<'_>, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        py.detach(|| match &self.inner {
            IdealInner::IntegerPolynomial(ideal) => {
                let value = value.as_integer().ok_or_else(|| {
                    PyTypeError::new_err("normal_form requires a polynomial over Z")
                })?;
                Ok(PyPolynomial::from_integer(
                    ideal.normal_form(value).map_err(value_error)?,
                ))
            }
            IdealInner::RationalPolynomial(ideal) => {
                let value = value.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("normal_form requires a polynomial over Q")
                })?;
                Ok(PyPolynomial::from_rational(
                    ideal.normal_form(value).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                let value = value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "normal_form requires a polynomial over the ideal's finite field",
                    )
                })?;
                Ok(PyPolynomial::from_finite_field(
                    ideal.normal_form(value).map_err(value_error)?,
                ))
            }
            IdealInner::RationalFunctionPolynomial(ideal) => {
                let value = value.as_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "normal_form requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                Ok(PyPolynomial::from_rational_function(
                    ideal.normal_form(value).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                let value = value.as_finite_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "normal_form requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                Ok(PyPolynomial::from_finite_rational_function(
                    ideal.normal_form(value).map_err(value_error)?,
                ))
            }
            IdealInner::Integer(_) | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "normal_form is defined for polynomial ideals",
            )),
        })
    }

    /// Canonical normal forms of several values modulo a polynomial ideal.
    fn normal_forms(
        &self,
        py: Python<'_>,
        values: Vec<PyPolynomial>,
    ) -> PyResult<Vec<PyPolynomial>> {
        py.detach(|| match &self.inner {
            IdealInner::IntegerPolynomial(ideal) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value.as_integer().cloned().ok_or_else(|| {
                            PyTypeError::new_err("normal_forms requires polynomials over Z")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ideal
                    .normal_forms(&values)
                    .map(|values| values.into_iter().map(PyPolynomial::from_integer).collect())
                    .map_err(value_error)
            }
            IdealInner::RationalPolynomial(ideal) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value.as_rational().cloned().ok_or_else(|| {
                            PyTypeError::new_err("normal_forms requires polynomials over Q")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ideal
                    .normal_forms(&values)
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_rational)
                            .collect()
                    })
                    .map_err(value_error)
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value.as_finite_field().cloned().ok_or_else(|| {
                            PyTypeError::new_err(
                                "normal_forms requires polynomials over the ideal's finite field",
                            )
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ideal
                    .normal_forms(&values)
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_finite_field)
                            .collect()
                    })
                    .map_err(value_error)
            }
            IdealInner::RationalFunctionPolynomial(ideal) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value.as_rational_function().cloned().ok_or_else(|| {
                            PyTypeError::new_err(
                                "normal_forms requires polynomials over the ideal's rational-function field",
                            )
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ideal
                    .normal_forms(&values)
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_rational_function)
                            .collect()
                    })
                    .map_err(value_error)
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                let values = values
                    .iter()
                    .map(|value| {
                        value
                            .as_finite_rational_function()
                            .cloned()
                            .ok_or_else(|| {
                                PyTypeError::new_err(
                                    "normal_forms requires polynomials over the ideal's rational-function field",
                                )
                            })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ideal
                    .normal_forms(&values)
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_finite_rational_function)
                            .collect()
                    })
                    .map_err(value_error)
            }
            IdealInner::Integer(_) | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "normal_forms is defined for polynomial ideals",
            )),
        })
    }

    /// Canonical normal form with exact multiplier evidence.
    fn reduce(&self, py: Python<'_>, value: &PyPolynomial) -> PyResult<PyIdealReduction> {
        py.detach(|| match &self.inner {
            IdealInner::IntegerPolynomial(ideal) => {
                let value = value
                    .as_integer()
                    .ok_or_else(|| PyTypeError::new_err("reduce requires a polynomial over Z"))?;
                Ok(PyIdealReduction::from_integer(
                    ideal.reduce(value).map_err(value_error)?,
                ))
            }
            IdealInner::RationalPolynomial(ideal) => {
                let value = value
                    .as_rational()
                    .ok_or_else(|| PyTypeError::new_err("reduce requires a polynomial over Q"))?;
                Ok(PyIdealReduction::from_rational(
                    ideal.reduce(value).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                let value = value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "reduce requires a polynomial over the ideal's finite field",
                    )
                })?;
                Ok(PyIdealReduction::from_finite_field(
                    ideal.reduce(value).map_err(value_error)?,
                ))
            }
            IdealInner::RationalFunctionPolynomial(ideal) => {
                let value = value.as_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "reduce requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                Ok(PyIdealReduction::from_rational_function(
                    ideal.reduce(value).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                let value = value.as_finite_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "reduce requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                Ok(PyIdealReduction::from_finite_rational_function(
                    ideal.reduce(value).map_err(value_error)?,
                ))
            }
            IdealInner::Integer(_) | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "reduce is defined for polynomial ideals",
            )),
        })
    }

    /// Return a polynomial ideal in an otherwise identical context with the
    /// requested monomial order.
    fn change_order(&self, order: &Bound<'_, PyAny>) -> PyResult<Self> {
        let order = extract_order(Some(order))?;
        match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(Self::from_rational_polynomial(
                ideal.change_order(order).map_err(value_error)?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(Self::from_finite_field_polynomial(
                ideal.change_order(order).map_err(value_error)?,
            )),
            IdealInner::RationalFunctionPolynomial(ideal) => {
                Ok(Self::from_rational_function_polynomial(
                    ideal.change_order(order).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                Ok(Self::from_finite_rational_function_polynomial(
                    ideal.change_order(order).map_err(value_error)?,
                ))
            }
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "change_order is defined for polynomial ideals",
            )),
        }
    }

    /// Construct the verified finite-dimensional quotient algebra.
    fn finite_quotient_algebra(&self, py: Python<'_>) -> PyResult<PyFiniteQuotientAlgebra> {
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match &self.inner {
                IdealInner::RationalPolynomial(ideal) => FiniteQuotientAlgebraInner::Rational(
                    Box::new(ideal.finite_quotient_algebra().map_err(value_error)?),
                ),
                IdealInner::FiniteFieldPolynomial(ideal) => {
                    FiniteQuotientAlgebraInner::FiniteField(Box::new(
                        ideal.finite_quotient_algebra().map_err(value_error)?,
                    ))
                }
                IdealInner::RationalFunctionPolynomial(ideal) => {
                    FiniteQuotientAlgebraInner::RationalFunction(Box::new(
                        ideal.finite_quotient_algebra().map_err(value_error)?,
                    ))
                }
                IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                    FiniteQuotientAlgebraInner::FiniteRationalFunction(Box::new(
                        ideal.finite_quotient_algebra().map_err(value_error)?,
                    ))
                }
                IdealInner::Integer(_)
                | IdealInner::IntegerPolynomial(_)
                | IdealInner::NumberField(_) => {
                    return Err(PyTypeError::new_err(
                        "finite_quotient_algebra is defined for polynomial ideals over fields",
                    ));
                }
            })
        })?;
        Ok(PyFiniteQuotientAlgebra { inner })
    }

    /// Construct a verified rational univariate representation.
    fn rational_univariate_representation(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyRationalUnivariateRepresentation> {
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match &self.inner {
                IdealInner::RationalPolynomial(ideal) => {
                    RationalUnivariateRepresentationInner::Rational(Box::new(
                        ideal
                            .rational_univariate_representation()
                            .map_err(value_error)?,
                    ))
                }
                IdealInner::FiniteFieldPolynomial(ideal) => {
                    RationalUnivariateRepresentationInner::FiniteField(Box::new(
                        ideal
                            .rational_univariate_representation()
                            .map_err(value_error)?,
                    ))
                }
                IdealInner::RationalFunctionPolynomial(ideal) => {
                    RationalUnivariateRepresentationInner::RationalFunction(Box::new(
                        ideal
                            .rational_univariate_representation()
                            .map_err(value_error)?,
                    ))
                }
                IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                    RationalUnivariateRepresentationInner::FiniteRationalFunction(Box::new(
                        ideal
                            .rational_univariate_representation()
                            .map_err(value_error)?,
                    ))
                }
                IdealInner::Integer(_)
                | IdealInner::IntegerPolynomial(_)
                | IdealInner::NumberField(_) => {
                    return Err(PyTypeError::new_err(
                        "rational_univariate_representation is defined for polynomial ideals",
                    ));
                }
            })
        })?;
        Ok(PyRationalUnivariateRepresentation { inner })
    }

    /// Eliminate named or indexed variables from a polynomial ideal.
    fn eliminate(&self, py: Python<'_>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let available = match &self.inner {
            IdealInner::IntegerPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::RationalPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::Integer(_) | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "eliminate is defined for polynomial ideals",
                ));
            }
        };
        let variables = resolve_polynomial_variables(variables, available)?;
        py.detach(|| match &self.inner {
            IdealInner::IntegerPolynomial(ideal) => Ok(Self::from_integer_polynomial(
                ideal.eliminate(&variables).map_err(value_error)?,
            )),
            IdealInner::RationalPolynomial(ideal) => Ok(Self::from_rational_polynomial(
                ideal.eliminate(&variables).map_err(value_error)?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(Self::from_finite_field_polynomial(
                ideal.eliminate(&variables).map_err(value_error)?,
            )),
            IdealInner::RationalFunctionPolynomial(ideal) => {
                Ok(Self::from_rational_function_polynomial(
                    ideal.eliminate(&variables).map_err(value_error)?,
                ))
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                Ok(Self::from_finite_rational_function_polynomial(
                    ideal.eliminate(&variables).map_err(value_error)?,
                ))
            }
            IdealInner::Integer(_) | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "eliminate is defined for polynomial ideals",
            )),
        })
    }

    /// Eliminate selected variables and retain a replayable field result.
    fn eliminate_with_evidence(
        &self,
        py: Python<'_>,
        variables: &Bound<'_, PyAny>,
    ) -> PyResult<PyIdealElimination> {
        let available = match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal.ring().variables(),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "eliminate_with_evidence requires a polynomial ideal over a field",
                ));
            }
        };
        let variables = resolve_polynomial_variables(variables, available)?;
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match &self.inner {
                IdealInner::RationalPolynomial(ideal) => IdealEliminationInner::Rational(
                    ideal
                        .eliminate_with_evidence(&variables)
                        .map_err(value_error)?,
                ),
                IdealInner::FiniteFieldPolynomial(ideal) => IdealEliminationInner::FiniteField(
                    ideal
                        .eliminate_with_evidence(&variables)
                        .map_err(value_error)?,
                ),
                IdealInner::RationalFunctionPolynomial(ideal) => {
                    IdealEliminationInner::RationalFunction(
                        ideal
                            .eliminate_with_evidence(&variables)
                            .map_err(value_error)?,
                    )
                }
                IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                    IdealEliminationInner::FiniteRationalFunction(
                        ideal
                            .eliminate_with_evidence(&variables)
                            .map_err(value_error)?,
                    )
                }
                IdealInner::Integer(_)
                | IdealInner::IntegerPolynomial(_)
                | IdealInner::NumberField(_) => {
                    return Err(PyTypeError::new_err(
                        "eliminate_with_evidence requires a polynomial ideal over a field",
                    ));
                }
            })
        })?;
        Ok(PyIdealElimination { inner })
    }

    /// Contract an integral polynomial ideal into the polynomial ring on the
    /// remaining variables.
    fn contract(&self, py: Python<'_>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let IdealInner::IntegerPolynomial(ideal) = &self.inner else {
            return Err(PyTypeError::new_err(
                "contract currently requires a polynomial ideal over Z",
            ));
        };
        let variables = resolve_polynomial_variables(variables, ideal.ring().variables())?;
        Ok(Self::from_integer_polynomial(
            py.detach(|| ideal.contract(&variables))
                .map_err(value_error)?,
        ))
    }

    /// Contract an integral polynomial ideal all the way to `Z`.
    fn contract_to_coefficients(&self, py: Python<'_>) -> PyResult<Self> {
        let IdealInner::IntegerPolynomial(ideal) = &self.inner else {
            return Err(PyTypeError::new_err(
                "contract_to_coefficients requires a polynomial ideal over Z",
            ));
        };
        Ok(Self::from_integer(
            py.detach(|| ideal.contract_to_coefficients())
                .map_err(value_error)?,
        ))
    }

    /// Exact coefficient-contraction Nullstellensatz identity over `Z`.
    fn nullstellensatz_certificate(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyArithmeticNullstellensatzCertificate> {
        let IdealInner::IntegerPolynomial(ideal) = &self.inner else {
            return Err(PyTypeError::new_err(
                "nullstellensatz_certificate requires a polynomial ideal over Z",
            ));
        };
        Ok(PyArithmeticNullstellensatzCertificate {
            inner: py
                .detach(|| ideal.nullstellensatz_certificate())
                .map_err(value_error)?,
        })
    }

    /// Exact arithmetic fiber support with bounded integer factorization.
    fn arithmetic_fiber_support(
        &self,
        py: Python<'_>,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> PyResult<PyArithmeticFiberSupport> {
        let IdealInner::IntegerPolynomial(ideal) = &self.inner else {
            return Err(PyTypeError::new_err(
                "arithmetic_fiber_support requires a polynomial ideal over Z",
            ));
        };
        Ok(PyArithmeticFiberSupport {
            inner: py
                .detach(|| ideal.arithmetic_fiber_support(trial_bound, primality_proof_bits))
                .map_err(value_error)?,
        })
    }

    /// Sum of two ideals in the same supported exact context.
    fn sum(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| match (&self.inner, &other.inner) {
            (IdealInner::Integer(left), IdealInner::Integer(right)) => {
                Ok(Self::from_integer(left.sum(right)))
            }
            (IdealInner::IntegerPolynomial(left), IdealInner::IntegerPolynomial(right)) => Ok(
                Self::from_integer_polynomial(left.sum(right).map_err(value_error)?),
            ),
            (IdealInner::RationalPolynomial(left), IdealInner::RationalPolynomial(right)) => Ok(
                Self::from_rational_polynomial(left.sum(right).map_err(value_error)?),
            ),
            (IdealInner::FiniteFieldPolynomial(left), IdealInner::FiniteFieldPolynomial(right)) => {
                Ok(Self::from_finite_field_polynomial(
                    left.sum(right).map_err(value_error)?,
                ))
            }
            (
                IdealInner::RationalFunctionPolynomial(left),
                IdealInner::RationalFunctionPolynomial(right),
            ) => Ok(Self::from_rational_function_polynomial(
                left.sum(right).map_err(value_error)?,
            )),
            (
                IdealInner::FiniteRationalFunctionPolynomial(left),
                IdealInner::FiniteRationalFunctionPolynomial(right),
            ) => Ok(Self::from_finite_rational_function_polynomial(
                left.sum(right).map_err(value_error)?,
            )),
            (IdealInner::NumberField(_), _) | (_, IdealInner::NumberField(_)) => Err(
                PyTypeError::new_err("sum is not defined for number-field-order ideals"),
            ),
            _ => Err(value_error("ideals have different exact contexts")),
        })
    }

    /// Product of two ideals in the same supported exact context.
    fn product(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| match (&self.inner, &other.inner) {
            (IdealInner::Integer(left), IdealInner::Integer(right)) => {
                Ok(Self::from_integer(left.product(right)))
            }
            (IdealInner::IntegerPolynomial(left), IdealInner::IntegerPolynomial(right)) => Ok(
                Self::from_integer_polynomial(left.product(right).map_err(value_error)?),
            ),
            (IdealInner::RationalPolynomial(left), IdealInner::RationalPolynomial(right)) => Ok(
                Self::from_rational_polynomial(left.product(right).map_err(value_error)?),
            ),
            (IdealInner::FiniteFieldPolynomial(left), IdealInner::FiniteFieldPolynomial(right)) => {
                Ok(Self::from_finite_field_polynomial(
                    left.product(right).map_err(value_error)?,
                ))
            }
            (
                IdealInner::RationalFunctionPolynomial(left),
                IdealInner::RationalFunctionPolynomial(right),
            ) => Ok(Self::from_rational_function_polynomial(
                left.product(right).map_err(value_error)?,
            )),
            (
                IdealInner::FiniteRationalFunctionPolynomial(left),
                IdealInner::FiniteRationalFunctionPolynomial(right),
            ) => Ok(Self::from_finite_rational_function_polynomial(
                left.product(right).map_err(value_error)?,
            )),
            (IdealInner::NumberField(_), _) | (_, IdealInner::NumberField(_)) => Err(
                PyTypeError::new_err("product is not defined for number-field-order ideals"),
            ),
            _ => Err(value_error("ideals have different exact contexts")),
        })
    }

    /// Nonnegative ideal power, with `I^0` equal to the unit ideal.
    fn pow(&self, py: Python<'_>, exponent: u64) -> PyResult<Self> {
        py.detach(|| match &self.inner {
            IdealInner::Integer(ideal) => Ok(Self::from_integer(ideal.pow(exponent))),
            IdealInner::IntegerPolynomial(ideal) => Ok(Self::from_integer_polynomial(
                ideal.pow(exponent).map_err(value_error)?,
            )),
            IdealInner::RationalPolynomial(ideal) => Ok(Self::from_rational_polynomial(
                ideal.pow(exponent).map_err(value_error)?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(Self::from_finite_field_polynomial(
                ideal.pow(exponent).map_err(value_error)?,
            )),
            IdealInner::RationalFunctionPolynomial(ideal) => Ok(
                Self::from_rational_function_polynomial(ideal.pow(exponent).map_err(value_error)?),
            ),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                Ok(Self::from_finite_rational_function_polynomial(
                    ideal.pow(exponent).map_err(value_error)?,
                ))
            }
            IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "pow is not defined for number-field-order ideals",
            )),
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.sum(py, other)
    }

    fn __mul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.product(py, other)
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: u64,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "Ideal power does not accept Python's third modulus argument",
            ));
        }
        self.pow(py, exponent)
    }

    /// Exact intersection with another polynomial ideal in the same context.
    fn intersection(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| match (&self.inner, &other.inner) {
            (IdealInner::RationalPolynomial(left), IdealInner::RationalPolynomial(right)) => Ok(
                Self::from_rational_polynomial(left.intersection(right).map_err(value_error)?),
            ),
            (IdealInner::FiniteFieldPolynomial(left), IdealInner::FiniteFieldPolynomial(right)) => {
                Ok(Self::from_finite_field_polynomial(
                    left.intersection(right).map_err(value_error)?,
                ))
            }
            (
                IdealInner::RationalFunctionPolynomial(left),
                IdealInner::RationalFunctionPolynomial(right),
            ) => Ok(Self::from_rational_function_polynomial(
                left.intersection(right).map_err(value_error)?,
            )),
            (
                IdealInner::FiniteRationalFunctionPolynomial(left),
                IdealInner::FiniteRationalFunctionPolynomial(right),
            ) => Ok(Self::from_finite_rational_function_polynomial(
                left.intersection(right).map_err(value_error)?,
            )),
            (IdealInner::NumberField(_), _) | (_, IdealInner::NumberField(_)) => Err(
                PyTypeError::new_err("intersection is currently defined for polynomial ideals"),
            ),
            _ => Err(value_error("ideals have different polynomial contexts")),
        })
    }

    /// Exact ideal quotient `self : other` in one polynomial context.
    fn colon(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| match (&self.inner, &other.inner) {
            (IdealInner::RationalPolynomial(left), IdealInner::RationalPolynomial(right)) => Ok(
                Self::from_rational_polynomial(left.colon(right).map_err(value_error)?),
            ),
            (IdealInner::FiniteFieldPolynomial(left), IdealInner::FiniteFieldPolynomial(right)) => {
                Ok(Self::from_finite_field_polynomial(
                    left.colon(right).map_err(value_error)?,
                ))
            }
            (
                IdealInner::RationalFunctionPolynomial(left),
                IdealInner::RationalFunctionPolynomial(right),
            ) => Ok(Self::from_rational_function_polynomial(
                left.colon(right).map_err(value_error)?,
            )),
            (
                IdealInner::FiniteRationalFunctionPolynomial(left),
                IdealInner::FiniteRationalFunctionPolynomial(right),
            ) => Ok(Self::from_finite_rational_function_polynomial(
                left.colon(right).map_err(value_error)?,
            )),
            (IdealInner::NumberField(_), _) | (_, IdealInner::NumberField(_)) => Err(
                PyTypeError::new_err("colon is currently defined for polynomial ideals"),
            ),
            _ => Err(value_error("ideals have different polynomial contexts")),
        })
    }

    /// Exact saturation `self : other^infinity` in one polynomial context.
    fn saturate(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.detach(|| match (&self.inner, &other.inner) {
            (IdealInner::RationalPolynomial(left), IdealInner::RationalPolynomial(right)) => Ok(
                Self::from_rational_polynomial(left.saturate(right).map_err(value_error)?),
            ),
            (IdealInner::FiniteFieldPolynomial(left), IdealInner::FiniteFieldPolynomial(right)) => {
                Ok(Self::from_finite_field_polynomial(
                    left.saturate(right).map_err(value_error)?,
                ))
            }
            (
                IdealInner::RationalFunctionPolynomial(left),
                IdealInner::RationalFunctionPolynomial(right),
            ) => Ok(Self::from_rational_function_polynomial(
                left.saturate(right).map_err(value_error)?,
            )),
            (
                IdealInner::FiniteRationalFunctionPolynomial(left),
                IdealInner::FiniteRationalFunctionPolynomial(right),
            ) => Ok(Self::from_finite_rational_function_polynomial(
                left.saturate(right).map_err(value_error)?,
            )),
            (IdealInner::NumberField(_), _) | (_, IdealInner::NumberField(_)) => Err(
                PyTypeError::new_err("saturate is currently defined for polynomial ideals"),
            ),
            _ => Err(value_error("ideals have different polynomial contexts")),
        })
    }

    /// Change the finite constant field of an ideal over `F_q(c)`.
    fn base_change_constant_field(&self, target: &PyFiniteField) -> PyResult<Self> {
        match &self.inner {
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                Ok(Self::from_finite_rational_function_polynomial(
                    ideal
                        .base_change_constant_field(target.inner.clone())
                        .map_err(|error| {
                            decomposition_error("base_change_constant_field", error)
                        })?,
                ))
            }
            _ => Err(unsupported_ideal_operation(
                "base_change_constant_field",
                self.inner.coefficient_context_type_id(),
            )),
        }
    }

    /// Complete associated primes from verified Ext-annihilator evidence.
    fn associated_primes(&self, py: Python<'_>) -> PyResult<PyAssociatedPrimeSet> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyAssociatedPrimeSet::from_rational(
                ideal
                    .associated_primes()
                    .map_err(|error| associated_prime_error(RationalField::TYPE_ID, error))?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyAssociatedPrimeSet::from_finite_field(
                    ideal
                        .associated_primes()
                        .map_err(|error| associated_prime_error(FiniteField::TYPE_ID, error))?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/associated_primes@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Verified scheme-theoretic top-dimensional hull from Ext evidence.
    fn equidimensional_hull(&self, py: Python<'_>) -> PyResult<PyEquidimensionalHull> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyEquidimensionalHull::from_rational(
                ideal
                    .equidimensional_hull()
                    .map_err(|error| associated_prime_error(RationalField::TYPE_ID, error))?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyEquidimensionalHull::from_finite_field(
                    ideal
                        .equidimensional_hull()
                        .map_err(|error| associated_prime_error(FiniteField::TYPE_ID, error))?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/equidimensional_hull@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Deterministic finite polynomial-subring normalization with exact evidence.
    fn noether_normalization(&self, py: Python<'_>) -> PyResult<PyNoetherNormalization> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyNoetherNormalization::from_rational(
                ideal
                    .noether_normalization()
                    .map_err(|error| normalization_error(RationalField::TYPE_ID, error))?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyNoetherNormalization::from_finite_field(
                    ideal
                        .noether_normalization()
                        .map_err(|error| normalization_error(FiniteField::TYPE_ID, error))?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/noether_normalization@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    fn rees_algebra(&self, py: Python<'_>) -> PyResult<PyReesAlgebra> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyReesAlgebra::from_rational(
                ideal.rees_algebra().map_err(construction_error)?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(PyReesAlgebra::from_finite_field(
                ideal.rees_algebra().map_err(construction_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "Rees algebras currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }

    fn associated_graded_algebra(&self, py: Python<'_>) -> PyResult<PyAssociatedGradedAlgebra> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => {
                Ok(PyAssociatedGradedAlgebra::from_rational(
                    ideal
                        .associated_graded_algebra()
                        .map_err(construction_error)?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyAssociatedGradedAlgebra::from_finite_field(
                    ideal
                        .associated_graded_algebra()
                        .map_err(construction_error)?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "associated graded algebras currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }

    /// One verified finite birational Grauert--Remmert end-ring extension.
    fn grauert_remmert_step(&self, py: Python<'_>) -> PyResult<Option<PyGrauertRemmertStep>> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal
                .grauert_remmert_step()
                .map(|step| step.map(PyGrauertRemmertStep::from_rational))
                .map_err(|error| affine_normalization_error(RationalField::TYPE_ID, error)),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal
                .grauert_remmert_step()
                .map(|step| step.map(PyGrauertRemmertStep::from_finite_field))
                .map_err(|error| affine_normalization_error(FiniteField::TYPE_ID, error)),
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/grauert_remmert_step@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Verified integral closure and conductor of an affine domain.
    fn normalization(&self, py: Python<'_>) -> PyResult<PyAffineNormalization> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyAffineNormalization::from_rational(
                ideal
                    .normalization()
                    .map_err(|error| affine_normalization_error(RationalField::TYPE_ID, error))?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyAffineNormalization::from_finite_field(
                    ideal
                        .normalization()
                        .map_err(|error| affine_normalization_error(FiniteField::TYPE_ID, error))?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/normalization@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Admit an explicit smooth affine complete-intersection normal model.
    fn normalization_from_smooth_model(
        &self,
        py: Python<'_>,
        normal_ideal: &PyIdeal,
        coordinate_images: Vec<PyPolynomial>,
        integral_equations: Vec<Vec<PyPolynomial>>,
        fraction_numerators: Vec<PyPolynomial>,
        fraction_denominators: Vec<PyPolynomial>,
    ) -> PyResult<PyAffineNormalization> {
        py.detach(|| match (&self.inner, &normal_ideal.inner) {
            (IdealInner::RationalPolynomial(source), IdealInner::RationalPolynomial(normal)) => {
                let convert = |values: &[PyPolynomial], role: &str| {
                    values
                        .iter()
                        .map(|value| {
                            value.as_rational().cloned().ok_or_else(|| {
                                PyTypeError::new_err(format!(
                                    "{role} must contain only polynomials over Q"
                                ))
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()
                };
                let equations = integral_equations
                    .iter()
                    .map(|equation| convert(equation, "integral_equations"))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyAffineNormalization::from_rational(
                    source
                        .normalization_from_smooth_model(
                            normal.clone(),
                            convert(&coordinate_images, "coordinate_images")?,
                            equations,
                            convert(&fraction_numerators, "fraction_numerators")?,
                            convert(&fraction_denominators, "fraction_denominators")?,
                        )
                        .map_err(|error| {
                            affine_normalization_error(RationalField::TYPE_ID, error)
                        })?,
                ))
            }
            (
                IdealInner::FiniteFieldPolynomial(source),
                IdealInner::FiniteFieldPolynomial(normal),
            ) => {
                let convert = |values: &[PyPolynomial], role: &str| {
                    values
                        .iter()
                        .map(|value| {
                            value.as_finite_field().cloned().ok_or_else(|| {
                                PyTypeError::new_err(format!(
                                    "{role} must contain only finite-field polynomials"
                                ))
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()
                };
                let equations = integral_equations
                    .iter()
                    .map(|equation| convert(equation, "integral_equations"))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyAffineNormalization::from_finite_field(
                    source
                        .normalization_from_smooth_model(
                            normal.clone(),
                            convert(&coordinate_images, "coordinate_images")?,
                            equations,
                            convert(&fraction_numerators, "fraction_numerators")?,
                            convert(&fraction_denominators, "fraction_denominators")?,
                        )
                        .map_err(|error| affine_normalization_error(FiniteField::TYPE_ID, error))?,
                ))
            }
            _ => Err(PyTypeError::new_err(
                "source and normal model must be polynomial ideals over the same supported field",
            )),
        })
    }

    /// Normalize a reduced affine algebra as a product of normal domains.
    fn reduced_normalization(&self, py: Python<'_>) -> PyResult<PyReducedAffineNormalization> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => {
                Ok(PyReducedAffineNormalization::from_rational(
                    ideal.reduced_normalization().map_err(|error| {
                        reduced_normalization_error(RationalField::TYPE_ID, error)
                    })?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyReducedAffineNormalization::from_finite_field(
                    ideal.reduced_normalization().map_err(|error| {
                        reduced_normalization_error(FiniteField::TYPE_ID, error)
                    })?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/reduced_normalization@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Verified minimal components in the admitted exact coefficient context.
    fn minimal_primes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(Py::new(
                py,
                PyMinimalPrimeDecomposition {
                    inner: MinimalPrimeDecompositionInner::Rational(
                        py.detach(|| ideal.minimal_primes()).map_err(|error| {
                            field_decomposition_error(
                                "minimal_primes",
                                RationalField::TYPE_ID,
                                error,
                            )
                        })?,
                    ),
                },
            )?
            .into_any()),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(Py::new(
                py,
                PyMinimalPrimeDecomposition {
                    inner: MinimalPrimeDecompositionInner::FiniteField(
                        py.detach(|| ideal.minimal_primes()).map_err(|error| {
                            field_decomposition_error("minimal_primes", FiniteField::TYPE_ID, error)
                        })?,
                    ),
                },
            )?
            .into_any()),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => Ok(Py::new(
                py,
                PyIdealDecomposition {
                    inner: py
                        .detach(|| ideal.minimal_primes())
                        .map_err(|error| decomposition_error("minimal_primes", error))?,
                },
            )?
            .into_any()),
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/minimal_primes@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([
                    RationalField::TYPE_ID,
                    FiniteField::TYPE_ID,
                    RationalFunctionField::<FiniteField>::TYPE_ID,
                ]),
            )),
        }
    }

    /// Verified decomposition of the radical by component dimension.
    fn equidimensional_radical_decomposition(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyEquidimensionalRadicalDecomposition> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => {
                Ok(PyEquidimensionalRadicalDecomposition::from_rational(
                    ideal
                        .equidimensional_radical_decomposition()
                        .map_err(|error| {
                            field_decomposition_error(
                                "equidimensional_radical_decomposition",
                                RationalField::TYPE_ID,
                                error,
                            )
                        })?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyEquidimensionalRadicalDecomposition::from_finite_field(
                    ideal
                        .equidimensional_radical_decomposition()
                        .map_err(|error| {
                            field_decomposition_error(
                                "equidimensional_radical_decomposition",
                                FiniteField::TYPE_ID,
                                error,
                            )
                        })?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/equidimensional_radical_decomposition@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Complete verified primary decomposition in the admitted exact regimes.
    fn primary_decomposition(&self, py: Python<'_>) -> PyResult<PyPrimaryDecomposition> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyPrimaryDecomposition::from_rational(
                ideal.primary_decomposition().map_err(|error| {
                    field_decomposition_error(
                        "primary_decomposition",
                        RationalField::TYPE_ID,
                        error,
                    )
                })?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyPrimaryDecomposition::from_finite_field(
                    ideal.primary_decomposition().map_err(|error| {
                        field_decomposition_error(
                            "primary_decomposition",
                            FiniteField::TYPE_ID,
                            error,
                        )
                    })?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/primary_decomposition@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Independently prove that this ideal is primary.
    fn certify_primary(&self, py: Python<'_>) -> PyResult<PyPolynomialPrimaryCertificate> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => {
                Ok(PyPolynomialPrimaryCertificate::from_rational(
                    ideal.certify_primary().map_err(|error| {
                        field_decomposition_error("certify_primary", RationalField::TYPE_ID, error)
                    })?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                Ok(PyPolynomialPrimaryCertificate::from_finite_field(
                    ideal.certify_primary().map_err(|error| {
                        field_decomposition_error("certify_primary", FiniteField::TYPE_ID, error)
                    })?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/certify_primary@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Admit independently discovered primary components after exact verification.
    fn admit_primary_decomposition(
        &self,
        py: Python<'_>,
        components: Vec<PyIdeal>,
    ) -> PyResult<PyPrimaryDecomposition> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(source) => {
                let components = components
                    .into_iter()
                    .map(|component| match component.inner {
                        IdealInner::RationalPolynomial(component) => Ok(component),
                        _ => Err(PyTypeError::new_err(
                            "every primary component must belong to the source polynomial ring over Q",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyPrimaryDecomposition::from_rational(
                    source
                        .admit_primary_decomposition(&components)
                        .map_err(|error| {
                            field_decomposition_error(
                                "admit_primary_decomposition",
                                RationalField::TYPE_ID,
                                error,
                            )
                        })?,
                ))
            }
            IdealInner::FiniteFieldPolynomial(source) => {
                let components = components
                    .into_iter()
                    .map(|component| match component.inner {
                        IdealInner::FiniteFieldPolynomial(component) => Ok(component),
                        _ => Err(PyTypeError::new_err(
                            "every primary component must belong to the source finite-field polynomial ring",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyPrimaryDecomposition::from_finite_field(
                    source
                        .admit_primary_decomposition(&components)
                        .map_err(|error| {
                            field_decomposition_error(
                                "admit_primary_decomposition",
                                FiniteField::TYPE_ID,
                                error,
                            )
                        })?,
                ))
            }
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/admit_primary_decomposition@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Proved radical of an admitted ideal over Q or a finite field.
    fn radical(&self, py: Python<'_>) -> PyResult<PyRadicalIdeal> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyRadicalIdeal {
                inner: RadicalIdealInner::Rational(ideal.radical().map_err(|error| {
                    field_decomposition_error("radical", RationalField::TYPE_ID, error)
                })?),
            }),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(PyRadicalIdeal {
                inner: RadicalIdealInner::FiniteField(ideal.radical().map_err(|error| {
                    field_decomposition_error("radical", FiniteField::TYPE_ID, error)
                })?),
            }),
            _ => Err(super::errors::unsupported_operation(
                UnsupportedOperationError::new(
                    "noether/commalg/Ideal/radical@1",
                    self.inner.coefficient_context_type_id(),
                )
                .with_supported_contexts([RationalField::TYPE_ID, FiniteField::TYPE_ID]),
            )),
        })
    }

    /// Leading-monomial ideal of this canonical reduced Groebner basis.
    fn initial_ideal(&self) -> PyResult<PyMonomialIdeal> {
        let inner = match &self.inner {
            IdealInner::RationalPolynomial(ideal) => {
                MonomialIdealInner::Rational(ideal.initial_ideal())
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                MonomialIdealInner::FiniteField(ideal.initial_ideal())
            }
            IdealInner::RationalFunctionPolynomial(ideal) => {
                MonomialIdealInner::RationalFunction(ideal.initial_ideal())
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                MonomialIdealInner::FiniteRationalFunction(ideal.initial_ideal())
            }
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "initial_ideal is defined for polynomial ideals over projected Groebner fields",
                ));
            }
        };
        Ok(PyMonomialIdeal::from_inner(inner))
    }

    /// Standard Hilbert-series numerator of a polynomial quotient.
    fn hilbert_numerator(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let coefficients = match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal.hilbert_numerator(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.hilbert_numerator(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.hilbert_numerator(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal.hilbert_numerator(),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "hilbert_numerator is defined for polynomial ideals",
                ));
            }
        }
        .map_err(value_error)?;
        coefficients
            .iter()
            .map(|coefficient| integer_to_python(py, coefficient))
            .collect()
    }

    /// Exact Hilbert series in the polynomial-ring grading.
    fn hilbert_series(&self) -> PyResult<PyHilbertSeries> {
        let inner = match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal.hilbert_series(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.hilbert_series(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.hilbert_series(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal.hilbert_series(),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "hilbert_series is defined for polynomial ideals over fields",
                ));
            }
        }
        .map_err(value_error)?;
        Ok(PyHilbertSeries { inner })
    }

    /// Krull dimension of a polynomial quotient (`-1` for the unit ideal).
    fn dimension(&self, py: Python<'_>) -> PyResult<i64> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal.dimension().map_err(value_error),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.dimension().map_err(value_error),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.dimension().map_err(value_error),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                ideal.dimension().map_err(value_error)
            }
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "dimension is defined for polynomial ideals",
            )),
        })
    }

    /// Standard-graded degree (multiplicity) of a polynomial quotient.
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let degree = match &self.inner {
            IdealInner::RationalPolynomial(ideal) => ideal.degree(),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.degree(),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.degree(),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => ideal.degree(),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => {
                return Err(PyTypeError::new_err(
                    "degree is defined for polynomial ideals",
                ));
            }
        }
        .map_err(value_error)?;
        integer_to_python(py, &degree)
    }

    /// Standard Hilbert polynomial in the canonical variable `m`.
    fn hilbert_polynomial(&self, py: Python<'_>) -> PyResult<PyPolynomial> {
        py.detach(|| match &self.inner {
            IdealInner::RationalPolynomial(ideal) => Ok(PyPolynomial::from_rational(
                ideal.hilbert_polynomial().map_err(value_error)?,
            )),
            IdealInner::FiniteFieldPolynomial(ideal) => Ok(PyPolynomial::from_rational(
                ideal.hilbert_polynomial().map_err(value_error)?,
            )),
            IdealInner::RationalFunctionPolynomial(ideal) => Ok(PyPolynomial::from_rational(
                ideal.hilbert_polynomial().map_err(value_error)?,
            )),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => Ok(PyPolynomial::from_rational(
                ideal.hilbert_polynomial().map_err(value_error)?,
            )),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::NumberField(_) => Err(PyTypeError::new_err(
                "hilbert_polynomial is defined for polynomial ideals",
            )),
        })
    }

    /// Recheck the context-specific canonical ideal representation.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| match &self.inner {
            IdealInner::Integer(ideal) => ideal.verify().map_err(value_error),
            IdealInner::IntegerPolynomial(ideal) => ideal.verify().map_err(value_error),
            IdealInner::RationalPolynomial(ideal) => ideal.verify().map_err(value_error),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.verify().map_err(value_error),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.verify().map_err(value_error),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                ideal.verify().map_err(value_error)
            }
            IdealInner::NumberField(ideal) => ideal.verify().map_err(value_error),
        })
    }

    /// Absolute integral-ideal norm.
    fn norm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            IdealInner::NumberField(ideal) => integer_to_python(py, &ideal.norm()),
            IdealInner::Integer(_)
            | IdealInner::IntegerPolynomial(_)
            | IdealInner::RationalPolynomial(_)
            | IdealInner::FiniteFieldPolynomial(_)
            | IdealInner::RationalFunctionPolynomial(_)
            | IdealInner::FiniteRationalFunctionPolynomial(_) => Err(PyTypeError::new_err(
                "norm is defined for number-field-order ideals",
            )),
        }
    }

    /// Whether an exact element lies in this ideal.
    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        match &self.inner {
            IdealInner::Integer(ideal) => Ok(ideal.contains(&extract_integer(value)?)),
            IdealInner::NumberField(ideal) => {
                if let Ok(element) = value.extract::<PyRef<'_, PyNumberFieldElement>>() {
                    return Ok(ideal.contains(&element.inner));
                }
                let element = extract_number_field_element(py, ideal.ring().field(), value)?;
                Ok(ideal.contains(&element))
            }
            IdealInner::IntegerPolynomial(ideal) => {
                let polynomial = value.extract::<PyRef<'_, PyPolynomial>>()?;
                let polynomial = polynomial.as_integer().ok_or_else(|| {
                    PyTypeError::new_err("membership requires a polynomial over Z")
                })?;
                ideal.contains(polynomial).map_err(value_error)
            }
            IdealInner::RationalPolynomial(ideal) => {
                let polynomial = value.extract::<PyRef<'_, PyPolynomial>>()?;
                let polynomial = polynomial.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("membership requires a polynomial over Q")
                })?;
                ideal.contains(polynomial).map_err(value_error)
            }
            IdealInner::FiniteFieldPolynomial(ideal) => {
                let polynomial = value.extract::<PyRef<'_, PyPolynomial>>()?;
                let polynomial = polynomial.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "membership requires a polynomial over the ideal's finite field",
                    )
                })?;
                ideal.contains(polynomial).map_err(value_error)
            }
            IdealInner::RationalFunctionPolynomial(ideal) => {
                let polynomial = value.extract::<PyRef<'_, PyPolynomial>>()?;
                let polynomial = polynomial.as_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "membership requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                ideal.contains(polynomial).map_err(value_error)
            }
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                let polynomial = value.extract::<PyRef<'_, PyPolynomial>>()?;
                let polynomial = polynomial.as_finite_rational_function().ok_or_else(|| {
                    PyTypeError::new_err(
                        "membership requires a polynomial over the ideal's rational-function field",
                    )
                })?;
                ideal.contains(polynomial).map_err(value_error)
            }
        }
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            IdealInner::Integer(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::NumberField(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::IntegerPolynomial(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::RationalPolynomial(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::FiniteFieldPolynomial(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::RationalFunctionPolynomial(ideal) => ideal.to_json().map_err(value_error),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => {
                ideal.to_json().map_err(value_error)
            }
        }
    }

    /// Restore and revalidate a canonical ideal.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("ring")
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient_ring| coefficient_ring.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(IntegerRing::TYPE_ID) => Ok(Self::from_integer_polynomial(
                Ideal::<PolynomialRing<IntegerRing>>::from_json(value).map_err(value_error)?,
            )),
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational_polynomial(
                Ideal::<PolynomialRing<RationalField>>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field_polynomial(
                Ideal::<PolynomialRing<FiniteField>>::from_json(value).map_err(value_error)?,
            )),
            Some(RationalFunctionField::<FiniteField>::TYPE_ID) => {
                let base_type = payload
                    .get("ring")
                    .and_then(|ring| ring.get("coefficient_ring"))
                    .and_then(|field| field.get("coefficient_ring"))
                    .and_then(|base| base.get("type"))
                    .and_then(serde_json::Value::as_str);
                if base_type == Some(RationalField::TYPE_ID) {
                    return Ok(Self::from_rational_function_polynomial(
                        Ideal::<PolynomialRing<RationalFunctionField<RationalField>>>::from_json(
                            value,
                        )
                        .map_err(value_error)?,
                    ));
                }
                if base_type == Some(FiniteField::TYPE_ID) {
                    return Ok(Self::from_finite_rational_function_polynomial(
                        Ideal::<PolynomialRing<RationalFunctionField<FiniteField>>>::from_json(
                            value,
                        )
                        .map_err(value_error)?,
                    ));
                }
                Err(PyTypeError::new_err(
                    "unsupported rational-function base field in serialized polynomial ideal",
                ))
            }
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported polynomial ideal coefficient context {type_id:?}"
            ))),
            None => {
                let ring_type = payload
                    .get("ring")
                    .and_then(|ring| ring.get("type"))
                    .and_then(serde_json::Value::as_str);
                if ring_type == Some(IntegerRing::TYPE_ID) {
                    return Ok(Self::from_integer(
                        Ideal::<IntegerRing>::from_json(value).map_err(value_error)?,
                    ));
                }
                Ok(Self {
                    inner: IdealInner::NumberField(
                        Ideal::<NumberFieldOrder>::from_json(value).map_err(value_error)?,
                    ),
                })
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            IdealInner::Integer(ideal) => canonical_hash(ideal),
            IdealInner::NumberField(ideal) => canonical_hash(ideal),
            IdealInner::IntegerPolynomial(ideal) => canonical_hash(ideal),
            IdealInner::RationalPolynomial(ideal) => canonical_hash(ideal),
            IdealInner::FiniteFieldPolynomial(ideal) => canonical_hash(ideal),
            IdealInner::RationalFunctionPolynomial(ideal) => canonical_hash(ideal),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => canonical_hash(ideal),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            IdealInner::Integer(ideal) => {
                format!("Ideal(ring=Z, generator={})", ideal.principal_generator())
            }
            IdealInner::NumberField(ideal) => {
                format!("Ideal(ring=NumberFieldOrder, norm={})", ideal.norm())
            }
            IdealInner::IntegerPolynomial(ideal) => format!(
                "Ideal(ring=PolynomialRing<Z>, generators={})",
                ideal.generators().len()
            ),
            IdealInner::RationalPolynomial(ideal) => format!(
                "Ideal(ring=PolynomialRing<Q>, generators={})",
                ideal.generators().len()
            ),
            IdealInner::FiniteFieldPolynomial(ideal) => format!(
                "Ideal(ring=PolynomialRing<FiniteField>, generators={})",
                ideal.generators().len()
            ),
            IdealInner::RationalFunctionPolynomial(ideal) => format!(
                "Ideal(ring=PolynomialRing<Q(c)>, generators={})",
                ideal.generators().len()
            ),
            IdealInner::FiniteRationalFunctionPolynomial(ideal) => format!(
                "Ideal(ring=PolynomialRing<FiniteField(c)>, generators={})",
                ideal.generators().len()
            ),
        })
    }
}

/// A verified decomposition into top-dimensional minimal components.
#[pyclass(
    module = "noether.commalg",
    name = "IdealDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIdealDecomposition {
    inner: IdealDecomposition<RationalFunctionField<FiniteField>>,
}

#[derive(Clone, PartialEq, Eq)]
enum RadicalIdealInner {
    Rational(RadicalIdeal<RationalField>),
    FiniteField(RadicalIdeal<FiniteField>),
}

/// A proved polynomial-ideal radical with exact route-specific evidence.
#[pyclass(
    module = "noether.commalg",
    name = "RadicalIdeal",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRadicalIdeal {
    inner: RadicalIdealInner,
}

#[pymethods]
impl PyRadicalIdeal {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/RadicalIdeal@1"
    }

    /// Source ideal before nilpotents are removed.
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            RadicalIdealInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            RadicalIdealInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    /// Canonical radical ideal in the source context.
    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            RadicalIdealInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            RadicalIdealInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
        }
    }

    /// Multiplication characteristic polynomial for each source variable.
    #[getter]
    fn characteristic_polynomials(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .characteristic_polynomials()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            RadicalIdealInner::FiniteField(value) => value
                .characteristic_polynomials()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    /// Square-free coordinate polynomials adjoined to the source ideal.
    #[getter]
    fn coordinate_certificates(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .coordinate_certificates()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            RadicalIdealInner::FiniteField(value) => value
                .coordinate_certificates()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    /// Canonical name of the exact proof route.
    #[getter]
    fn proof_kind(&self) -> &'static str {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.proof_kind(),
            RadicalIdealInner::FiniteField(value) => value.proof_kind(),
        }
    }

    /// Irreducible factors and source multiplicities for a principal proof.
    #[getter]
    fn principal_factors(&self) -> Option<Vec<(PyPolynomial, u64)>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.principal_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (PyPolynomial::from_rational(factor.clone()), *multiplicity)
                    })
                    .collect()
            }),
            RadicalIdealInner::FiniteField(value) => value.principal_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (
                            PyPolynomial::from_finite_field(factor.clone()),
                            *multiplicity,
                        )
                    })
                    .collect()
            }),
        }
    }

    /// Canonical source and square-free monomial ideals for a monomial proof.
    #[getter]
    fn monomial_proof(&self) -> Option<(PyMonomialIdeal, PyMonomialIdeal)> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => {
                value.monomial_proof().map(|(source, radical)| {
                    (
                        PyMonomialIdeal::from_inner(MonomialIdealInner::Rational(source.clone())),
                        PyMonomialIdeal::from_inner(MonomialIdealInner::Rational(radical.clone())),
                    )
                })
            }
            RadicalIdealInner::FiniteField(value) => {
                value.monomial_proof().map(|(source, radical)| {
                    (
                        PyMonomialIdeal::from_inner(MonomialIdealInner::FiniteField(
                            source.clone(),
                        )),
                        PyMonomialIdeal::from_inner(MonomialIdealInner::FiniteField(
                            radical.clone(),
                        )),
                    )
                })
            }
        }
    }

    /// Sorted pivot variables eliminated by an affine-linear ideal.
    #[getter]
    fn affine_linear_pivots(&self) -> Option<Vec<usize>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => {
                value.affine_linear_pivots().map(<[usize]>::to_vec)
            }
            RadicalIdealInner::FiniteField(value) => {
                value.affine_linear_pivots().map(<[usize]>::to_vec)
            }
        }
    }

    /// Sorted pivot variables eliminated by a polynomial-graph ideal.
    #[getter]
    fn polynomial_graph_pivots(&self) -> Option<Vec<usize>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
            RadicalIdealInner::FiniteField(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
        }
    }

    /// Complete-intersection codimension retained by a Jacobian proof.
    #[getter]
    fn complete_intersection_codimension(&self) -> Option<usize> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.complete_intersection_codimension(),
            RadicalIdealInner::FiniteField(value) => value.complete_intersection_codimension(),
        }
    }

    /// Expected-codimension Jacobian minors retained by the radical proof.
    #[getter]
    fn jacobian_minors(&self) -> Option<Vec<PyPolynomial>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.jacobian_minors().map(|minors| {
                minors
                    .iter()
                    .cloned()
                    .map(PyPolynomial::from_rational)
                    .collect()
            }),
            RadicalIdealInner::FiniteField(value) => value.jacobian_minors().map(|minors| {
                minors
                    .iter()
                    .cloned()
                    .map(PyPolynomial::from_finite_field)
                    .collect()
            }),
        }
    }

    /// Singular-locus ideal retained by a complete-intersection proof.
    #[getter]
    fn singular_locus_ideal(&self) -> Option<PyIdeal> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .singular_locus_ideal()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            RadicalIdealInner::FiniteField(value) => value
                .singular_locus_ideal()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    /// Krull dimension of the retained singular locus.
    #[getter]
    fn singular_locus_dimension(&self) -> Option<i64> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.singular_locus_dimension(),
            RadicalIdealInner::FiniteField(value) => value.singular_locus_dimension(),
        }
    }

    /// Common irreducible factors and their multiplicities.
    #[getter]
    fn common_factors(&self) -> Option<Vec<(PyPolynomial, u64)>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.common_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (PyPolynomial::from_rational(factor.clone()), *multiplicity)
                    })
                    .collect()
            }),
            RadicalIdealInner::FiniteField(value) => value.common_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (
                            PyPolynomial::from_finite_field(factor.clone()),
                            *multiplicity,
                        )
                    })
                    .collect()
            }),
        }
    }

    /// Residual source and recursively certified radical for a common-factor split.
    #[getter]
    fn common_factor_residual(&self) -> Option<(PyIdeal, PyIdeal)> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .common_factor_residual_source()
                .zip(value.common_factor_residual_radical())
                .map(|(residual_source, residual_radical)| {
                    (
                        PyIdeal::from_rational_polynomial(residual_source.clone()),
                        PyIdeal::from_rational_polynomial(residual_radical.clone()),
                    )
                }),
            RadicalIdealInner::FiniteField(value) => value
                .common_factor_residual_source()
                .zip(value.common_factor_residual_radical())
                .map(|(residual_source, residual_radical)| {
                    (
                        PyIdeal::from_finite_field_polynomial(residual_source.clone()),
                        PyIdeal::from_finite_field_polynomial(residual_radical.clone()),
                    )
                }),
        }
    }

    /// Canonical generator whose irreducible factors define recursive branches.
    #[getter]
    fn factor_split_generator(&self) -> Option<PyPolynomial> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .factor_split_generator()
                .cloned()
                .map(PyPolynomial::from_rational),
            RadicalIdealInner::FiniteField(value) => value
                .factor_split_generator()
                .cloned()
                .map(PyPolynomial::from_finite_field),
        }
    }

    /// Irreducible factors and multiplicities of the splitting generator.
    #[getter]
    fn factor_split_factors(&self) -> Option<Vec<(PyPolynomial, u64)>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.factor_split_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (PyPolynomial::from_rational(factor.clone()), *multiplicity)
                    })
                    .collect()
            }),
            RadicalIdealInner::FiniteField(value) => value.factor_split_factors().map(|factors| {
                factors
                    .iter()
                    .map(|(factor, multiplicity)| {
                        (
                            PyPolynomial::from_finite_field(factor.clone()),
                            *multiplicity,
                        )
                    })
                    .collect()
            }),
        }
    }

    /// Strict branch ideals paired with their recursively certified radicals.
    #[getter]
    fn factor_split_branches(&self) -> Option<Vec<(PyIdeal, PyIdeal)>> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .factor_split_branch_sources()
                .zip(value.factor_split_branch_radicals())
                .map(|(sources, radicals)| {
                    sources
                        .iter()
                        .zip(radicals)
                        .map(|(source, radical)| {
                            (
                                PyIdeal::from_rational_polynomial(source.clone()),
                                PyIdeal::from_rational_polynomial(radical.clone()),
                            )
                        })
                        .collect()
                }),
            RadicalIdealInner::FiniteField(value) => value
                .factor_split_branch_sources()
                .zip(value.factor_split_branch_radicals())
                .map(|(sources, radicals)| {
                    sources
                        .iter()
                        .zip(radicals)
                        .map(|(source, radical)| {
                            (
                                PyIdeal::from_finite_field_polynomial(source.clone()),
                                PyIdeal::from_finite_field_polynomial(radical.clone()),
                            )
                        })
                        .collect()
                }),
        }
    }

    /// Polynomial defining the closed and principal-open saturation branches.
    #[getter]
    fn saturation_split_witness(&self) -> Option<PyPolynomial> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .saturation_split_witness()
                .cloned()
                .map(PyPolynomial::from_rational),
            RadicalIdealInner::FiniteField(value) => value
                .saturation_split_witness()
                .cloned()
                .map(PyPolynomial::from_finite_field),
        }
    }

    /// Closed and principal-open branch ideals paired with their radicals.
    #[getter]
    fn saturation_split_branches(&self) -> Option<((PyIdeal, PyIdeal), (PyIdeal, PyIdeal))> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value
                .saturation_split_closed_source()
                .zip(value.saturation_split_closed_radical())
                .zip(
                    value
                        .saturation_split_open_source()
                        .zip(value.saturation_split_open_radical()),
                )
                .map(
                    |((closed_source, closed_radical), (open_source, open_radical))| {
                        (
                            (
                                PyIdeal::from_rational_polynomial(closed_source.clone()),
                                PyIdeal::from_rational_polynomial(closed_radical.clone()),
                            ),
                            (
                                PyIdeal::from_rational_polynomial(open_source.clone()),
                                PyIdeal::from_rational_polynomial(open_radical.clone()),
                            ),
                        )
                    },
                ),
            RadicalIdealInner::FiniteField(value) => value
                .saturation_split_closed_source()
                .zip(value.saturation_split_closed_radical())
                .zip(
                    value
                        .saturation_split_open_source()
                        .zip(value.saturation_split_open_radical()),
                )
                .map(
                    |((closed_source, closed_radical), (open_source, open_radical))| {
                        (
                            (
                                PyIdeal::from_finite_field_polynomial(closed_source.clone()),
                                PyIdeal::from_finite_field_polynomial(closed_radical.clone()),
                            ),
                            (
                                PyIdeal::from_finite_field_polynomial(open_source.clone()),
                                PyIdeal::from_finite_field_polynomial(open_radical.clone()),
                            ),
                        )
                    },
                ),
        }
    }

    /// Exact theorem justifying the radical claim.
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.theorem(),
            RadicalIdealInner::FiniteField(value) => value.theorem(),
        }
    }

    /// Independently recompute every retained certificate and the radical.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.verify(),
            RadicalIdealInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Serialize the complete radical evidence.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            RadicalIdealInner::Rational(value) => value.to_json(),
            RadicalIdealInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Deserialize and independently recompute the radical evidence.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self {
                inner: RadicalIdealInner::Rational(
                    RadicalIdeal::<RationalField>::from_json(value).map_err(value_error)?,
                ),
            }),
            Some(FiniteField::TYPE_ID) => Ok(Self {
                inner: RadicalIdealInner::FiniteField(
                    RadicalIdeal::<FiniteField>::from_json(value).map_err(value_error)?,
                ),
            }),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported radical coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized radical has no polynomial coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            RadicalIdealInner::Rational(value) => canonical_hash(value),
            RadicalIdealInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            RadicalIdealInner::Rational(value) => format!(
                "RadicalIdeal(context=Q, proof={}, generators={})",
                value.proof_kind(),
                value.ideal().generators().len()
            ),
            RadicalIdealInner::FiniteField(value) => format!(
                "RadicalIdeal(context=FiniteField, proof={}, generators={})",
                value.proof_kind(),
                value.ideal().generators().len()
            ),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PolynomialPrimeCertificateInner {
    Rational(PolynomialPrimeCertificate<RationalField>),
    FiniteField(PolynomialPrimeCertificate<FiniteField>),
}

/// Exact evidence that a polynomial ideal is prime.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimeCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimeCertificate {
    inner: PolynomialPrimeCertificateInner,
}

impl PyPolynomialPrimeCertificate {
    pub(crate) fn from_rational(value: PolynomialPrimeCertificate<RationalField>) -> Self {
        Self {
            inner: PolynomialPrimeCertificateInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: PolynomialPrimeCertificate<FiniteField>) -> Self {
        Self {
            inner: PolynomialPrimeCertificateInner::FiniteField(value),
        }
    }

    pub(crate) fn as_rational(&self) -> Option<&PolynomialPrimeCertificate<RationalField>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => Some(value),
            PolynomialPrimeCertificateInner::FiniteField(_) => None,
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&PolynomialPrimeCertificate<FiniteField>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(_) => None,
            PolynomialPrimeCertificateInner::FiniteField(value) => Some(value),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PolynomialPrimaryCertificateInner {
    Rational(PolynomialPrimaryCertificate<RationalField>),
    FiniteField(PolynomialPrimaryCertificate<FiniteField>),
}

/// Exact evidence that a polynomial ideal is primary.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialPrimaryCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialPrimaryCertificate {
    inner: PolynomialPrimaryCertificateInner,
}

impl PyPolynomialPrimaryCertificate {
    fn from_rational(value: PolynomialPrimaryCertificate<RationalField>) -> Self {
        Self {
            inner: PolynomialPrimaryCertificateInner::Rational(value),
        }
    }

    fn from_finite_field(value: PolynomialPrimaryCertificate<FiniteField>) -> Self {
        Self {
            inner: PolynomialPrimaryCertificateInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyPolynomialPrimaryCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/PolynomialPrimaryCertificate@1"
    }

    /// Canonical polynomial ideal certified as primary.
    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            PolynomialPrimaryCertificateInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
        }
    }

    /// Exact prime equal to the radical of this primary ideal.
    #[getter]
    fn associated_prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.associated_prime().clone())
            }
            PolynomialPrimaryCertificateInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.associated_prime().clone())
            }
        }
    }

    /// Complete EHV associated-prime evidence used by the general primary route.
    #[getter]
    fn ehv_associated_primes(&self) -> Option<PyAssociatedPrimeSet> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value
                .ehv_associated_primes()
                .cloned()
                .map(PyAssociatedPrimeSet::from_rational),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value
                .ehv_associated_primes()
                .cloned()
                .map(PyAssociatedPrimeSet::from_finite_field),
        }
    }

    /// EHV equidimensional-hull evidence used by an isolated primary component.
    #[getter]
    fn ehv_equidimensional_hull(&self) -> Option<PyEquidimensionalHull> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value
                .ehv_equidimensional_hull()
                .cloned()
                .map(PyEquidimensionalHull::from_rational),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value
                .ehv_equidimensional_hull()
                .cloned()
                .map(PyEquidimensionalHull::from_finite_field),
        }
    }

    /// Canonical name of the exact primary proof route.
    #[getter]
    fn proof_kind(&self) -> &'static str {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value.proof_kind(),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value.proof_kind(),
        }
    }

    /// Nilpotence bound when the Artinian-local route applies.
    #[getter]
    fn nilpotence_bound(&self) -> Option<usize> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value.nilpotence_bound(),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value.nilpotence_bound(),
        }
    }

    /// Polynomial-graph base below a graph-relative primary component.
    #[getter]
    fn polynomial_graph_base(&self) -> Option<PyIdeal> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value
                .polynomial_graph_base()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value
                .polynomial_graph_base()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    /// Pivot variables eliminated by the polynomial-graph base.
    #[getter]
    fn polynomial_graph_pivots(&self) -> Option<Vec<usize>> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
            PolynomialPrimaryCertificateInner::FiniteField(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
        }
    }

    /// Monomial primary ideal pulled back through a polynomial-graph quotient.
    #[getter]
    fn polynomial_graph_monomial_source(&self) -> Option<PyMonomialIdeal> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value
                .polynomial_graph_monomial_source()
                .cloned()
                .map(MonomialIdealInner::Rational)
                .map(PyMonomialIdeal::from_inner),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value
                .polynomial_graph_monomial_source()
                .cloned()
                .map(MonomialIdealInner::FiniteField)
                .map(PyMonomialIdeal::from_inner),
        }
    }

    /// Irreducible free-variable factor defining the component radical.
    #[getter]
    fn polynomial_graph_factor(&self) -> Option<PyPolynomial> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value
                .polynomial_graph_factor()
                .cloned()
                .map(PyPolynomial::from_rational),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value
                .polynomial_graph_factor()
                .cloned()
                .map(PyPolynomial::from_finite_field),
        }
    }

    /// Positive multiplicity of the hypersurface factor over the graph base.
    #[getter]
    fn polynomial_graph_factor_exponent(&self) -> Option<u64> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => {
                value.polynomial_graph_factor_exponent()
            }
            PolynomialPrimaryCertificateInner::FiniteField(value) => {
                value.polynomial_graph_factor_exponent()
            }
        }
    }

    /// Irreducible regular-sequence factors and their positive multiplicities.
    #[getter]
    fn polynomial_graph_factors(&self) -> Option<Vec<(PyPolynomial, u64)>> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => {
                value.polynomial_graph_factors().map(|factors| {
                    factors
                        .iter()
                        .map(|(factor, exponent)| {
                            (PyPolynomial::from_rational(factor.clone()), *exponent)
                        })
                        .collect()
                })
            }
            PolynomialPrimaryCertificateInner::FiniteField(value) => {
                value.polynomial_graph_factors().map(|factors| {
                    factors
                        .iter()
                        .map(|(factor, exponent)| {
                            (PyPolynomial::from_finite_field(factor.clone()), *exponent)
                        })
                        .collect()
                })
            }
        }
    }

    /// Exact theorem certifying the recorded ideal as primary.
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value.theorem(),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value.theorem(),
        }
    }

    /// Independently recheck the complete primary evidence.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value.verify(),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Serialize the complete primary evidence.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => value.to_json(),
            PolynomialPrimaryCertificateInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Deserialize and exactly reverify polynomial primary evidence.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("ideal")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                PolynomialPrimaryCertificate::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                PolynomialPrimaryCertificate::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported polynomial-primary coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized polynomial-primary certificate has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PolynomialPrimaryCertificateInner::Rational(value) => canonical_hash(value),
            PolynomialPrimaryCertificateInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PolynomialPrimaryCertificate(proof={}, generators={})",
            self.proof_kind(),
            match &self.inner {
                PolynomialPrimaryCertificateInner::Rational(value) => {
                    value.ideal().generators().len()
                }
                PolynomialPrimaryCertificateInner::FiniteField(value) => {
                    value.ideal().generators().len()
                }
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum AssociatedPrimeExtStageInner {
    Rational(AssociatedPrimeExtStage<RationalField>),
    FiniteField(AssociatedPrimeExtStage<FiniteField>),
}

/// One verified Ext-annihilator stage in an associated-prime computation.
#[pyclass(
    module = "noether.commalg",
    name = "AssociatedPrimeExtStage",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAssociatedPrimeExtStage {
    inner: AssociatedPrimeExtStageInner,
}

impl PyAssociatedPrimeExtStage {
    pub(crate) fn from_rational(value: AssociatedPrimeExtStage<RationalField>) -> Self {
        Self {
            inner: AssociatedPrimeExtStageInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: AssociatedPrimeExtStage<FiniteField>) -> Self {
        Self {
            inner: AssociatedPrimeExtStageInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyAssociatedPrimeExtStage {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/AssociatedPrimeExtStage@1"
    }

    #[getter]
    fn codimension(&self) -> usize {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => value.codimension(),
            AssociatedPrimeExtStageInner::FiniteField(value) => value.codimension(),
        }
    }

    #[getter]
    fn ext_module(&self) -> PyPresentedModule {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.ext_module().clone())
            }
            AssociatedPrimeExtStageInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.ext_module().clone())
            }
        }
    }

    #[getter]
    fn annihilator(&self) -> PyIdeal {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.annihilator().clone())
            }
            AssociatedPrimeExtStageInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.annihilator().clone())
            }
        }
    }

    #[getter]
    fn minimal_primes(&self) -> Option<PyMinimalPrimeDecomposition> {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => {
                value
                    .minimal_primes()
                    .map(|inner| PyMinimalPrimeDecomposition {
                        inner: MinimalPrimeDecompositionInner::Rational(inner.clone()),
                    })
            }
            AssociatedPrimeExtStageInner::FiniteField(value) => {
                value
                    .minimal_primes()
                    .map(|inner| PyMinimalPrimeDecomposition {
                        inner: MinimalPrimeDecompositionInner::FiniteField(inner.clone()),
                    })
            }
        }
    }

    #[getter]
    fn associated_primes(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => value
                .associated_primes()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            AssociatedPrimeExtStageInner::FiniteField(value) => value
                .associated_primes()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => value.verify(),
            AssociatedPrimeExtStageInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => value.to_json(),
            AssociatedPrimeExtStageInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("annihilator")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                AssociatedPrimeExtStage::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                AssociatedPrimeExtStage::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported associated-prime Ext-stage coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized associated-prime Ext stage has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            AssociatedPrimeExtStageInner::Rational(value) => canonical_hash(value),
            AssociatedPrimeExtStageInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "AssociatedPrimeExtStage(codimension={}, associated_primes={})",
            self.codimension(),
            self.associated_primes().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum AssociatedPrimeSetInner {
    Rational(AssociatedPrimeSet<RationalField>),
    FiniteField(AssociatedPrimeSet<FiniteField>),
}

/// Complete verified associated primes of a polynomial ideal.
#[pyclass(
    module = "noether.commalg",
    name = "AssociatedPrimeSet",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAssociatedPrimeSet {
    inner: AssociatedPrimeSetInner,
}

impl PyAssociatedPrimeSet {
    fn from_rational(value: AssociatedPrimeSet<RationalField>) -> Self {
        Self {
            inner: AssociatedPrimeSetInner::Rational(value),
        }
    }

    fn from_finite_field(value: AssociatedPrimeSet<FiniteField>) -> Self {
        Self {
            inner: AssociatedPrimeSetInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyAssociatedPrimeSet {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/AssociatedPrimeSet@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            AssociatedPrimeSetInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn resolution(&self) -> PyFreeResolution {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => {
                PyFreeResolution::from_rational_polynomial(value.resolution().clone())
            }
            AssociatedPrimeSetInner::FiniteField(value) => {
                PyFreeResolution::from_finite_field_polynomial(value.resolution().clone())
            }
        }
    }

    #[getter]
    fn stages(&self) -> Vec<PyAssociatedPrimeExtStage> {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value
                .stages()
                .iter()
                .cloned()
                .map(PyAssociatedPrimeExtStage::from_rational)
                .collect(),
            AssociatedPrimeSetInner::FiniteField(value) => value
                .stages()
                .iter()
                .cloned()
                .map(PyAssociatedPrimeExtStage::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn prime_certificates(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            AssociatedPrimeSetInner::FiniteField(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn primes(&self) -> Vec<PyIdeal> {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value
                .primes()
                .into_iter()
                .cloned()
                .map(PyIdeal::from_rational_polynomial)
                .collect(),
            AssociatedPrimeSetInner::FiniteField(value) => value
                .primes()
                .into_iter()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial)
                .collect(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value.theorem(),
            AssociatedPrimeSetInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value.verify(),
            AssociatedPrimeSetInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => value.to_json(),
            AssociatedPrimeSetInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                AssociatedPrimeSet::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                AssociatedPrimeSet::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported associated-prime coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized associated-prime set has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            AssociatedPrimeSetInner::Rational(value) => canonical_hash(value),
            AssociatedPrimeSetInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "AssociatedPrimeSet(stages={}, associated_primes={})",
            self.stages().len(),
            self.prime_certificates().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PrimaryDecompositionInner {
    Rational(PrimaryDecomposition<RationalField>),
    FiniteField(PrimaryDecomposition<FiniteField>),
}

/// Complete verified primary decomposition of an admitted polynomial ideal.
#[pyclass(
    module = "noether.commalg",
    name = "PrimaryDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPrimaryDecomposition {
    inner: PrimaryDecompositionInner,
}

impl PyPrimaryDecomposition {
    fn from_rational(value: PrimaryDecomposition<RationalField>) -> Self {
        Self {
            inner: PrimaryDecompositionInner::Rational(value),
        }
    }

    fn from_finite_field(value: PrimaryDecomposition<FiniteField>) -> Self {
        Self {
            inner: PrimaryDecompositionInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyPrimaryDecomposition {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/PrimaryDecomposition@1"
    }

    /// Original ideal reconstructed by the primary components.
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            PrimaryDecompositionInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    /// Canonically ordered primary component ideals.
    #[getter]
    fn components(&self) -> Vec<PyIdeal> {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyIdeal::from_rational_polynomial)
                .collect(),
            PrimaryDecompositionInner::FiniteField(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial)
                .collect(),
        }
    }

    /// Exact primary certificate for each component.
    #[getter]
    fn certificates(&self) -> Vec<PyPolynomialPrimaryCertificate> {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value
                .certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimaryCertificate::from_rational)
                .collect(),
            PrimaryDecompositionInner::FiniteField(value) => value
                .certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimaryCertificate::from_finite_field)
                .collect(),
        }
    }

    /// Associated primes, including embedded primes.
    #[getter]
    fn associated_primes(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value
                .associated_primes()
                .into_iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            PrimaryDecompositionInner::FiniteField(value) => value
                .associated_primes()
                .into_iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    /// Exact theorem certifying completeness of the decomposition.
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value.theorem(),
            PrimaryDecompositionInner::FiniteField(value) => value.theorem(),
        }
    }

    /// Recheck every primary claim and the exact component intersection.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value.verify(),
            PrimaryDecompositionInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Serialize the complete primary decomposition evidence.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => value.to_json(),
            PrimaryDecompositionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Deserialize and exactly reverify a complete primary decomposition.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                PrimaryDecomposition::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                PrimaryDecomposition::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported primary-decomposition coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized primary decomposition has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PrimaryDecompositionInner::Rational(value) => canonical_hash(value),
            PrimaryDecompositionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PrimaryDecomposition(components={})",
            match &self.inner {
                PrimaryDecompositionInner::Rational(value) => value.components().len(),
                PrimaryDecompositionInner::FiniteField(value) => value.components().len(),
            }
        ))
    }
}

#[pymethods]
impl PyPolynomialPrimeCertificate {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/PolynomialPrimeCertificate@1"
    }

    /// Canonical polynomial ideal certified as prime.
    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
        }
    }

    /// Canonical name of the exact primality proof route.
    #[getter]
    fn proof_kind(&self) -> &'static str {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value.proof_kind(),
            PolynomialPrimeCertificateInner::FiniteField(value) => value.proof_kind(),
        }
    }

    /// Irreducible generator when the hypersurface route applies.
    #[getter]
    fn irreducible_factor(&self) -> Option<PyPolynomial> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .irreducible_factor()
                .cloned()
                .map(PyPolynomial::from_rational),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .irreducible_factor()
                .cloned()
                .map(PyPolynomial::from_finite_field),
        }
    }

    /// Sorted coordinate indices generating a coordinate prime.
    #[getter]
    fn coordinate_variables(&self) -> Option<Vec<usize>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                value.coordinate_variables().map(<[usize]>::to_vec)
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.coordinate_variables().map(<[usize]>::to_vec)
            }
        }
    }

    /// Sorted leading pivot variables for an affine-linear prime.
    #[getter]
    fn affine_linear_pivots(&self) -> Option<Vec<usize>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                value.affine_linear_pivots().map(<[usize]>::to_vec)
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.affine_linear_pivots().map(<[usize]>::to_vec)
            }
        }
    }

    /// Sorted pivot variables for a polynomial-graph prime.
    #[getter]
    fn polynomial_graph_pivots(&self) -> Option<Vec<usize>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.polynomial_graph_pivots().map(<[usize]>::to_vec)
            }
        }
    }

    /// Codimension in a smooth projective complete-intersection proof.
    #[getter]
    fn complete_intersection_codimension(&self) -> Option<usize> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                value.complete_intersection_codimension()
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.complete_intersection_codimension()
            }
        }
    }

    /// Expected-codimension Jacobian minors in the projective proof.
    #[getter]
    fn jacobian_minors(&self) -> Option<Vec<PyPolynomial>> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => {
                value.jacobian_minors().map(|minors| {
                    minors
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_rational)
                        .collect()
                })
            }
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.jacobian_minors().map(|minors| {
                    minors
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_finite_field)
                        .collect()
                })
            }
        }
    }

    /// Affine-cone singular-locus ideal in the projective proof.
    #[getter]
    fn singular_locus_ideal(&self) -> Option<PyIdeal> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .singular_locus_ideal()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .singular_locus_ideal()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    /// Saturation proving that the projective singular locus is empty.
    #[getter]
    fn projective_singular_saturation(&self) -> Option<PyIdeal> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .projective_singular_saturation()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .projective_singular_saturation()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    /// Homogeneous coordinate inverted by the affine-patch proof.
    #[getter]
    fn homogeneous_patch_variable(&self) -> Option<usize> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value.homogeneous_patch_variable(),
            PolynomialPrimeCertificateInner::FiniteField(value) => {
                value.homogeneous_patch_variable()
            }
        }
    }

    /// Saturation proving injection into the selected affine localization.
    #[getter]
    fn homogeneous_patch_saturation(&self) -> Option<PyIdeal> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .homogeneous_patch_saturation()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .homogeneous_patch_saturation()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    /// Prime certificate for the affine slice `I + (x_i - 1)`.
    #[getter]
    fn homogeneous_patch_prime(&self) -> Option<PyPolynomialPrimeCertificate> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .homogeneous_patch_prime()
                .map(PyPolynomialPrimeCertificate::from_rational),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .homogeneous_patch_prime()
                .map(PyPolynomialPrimeCertificate::from_finite_field),
        }
    }

    /// Irreducible RUR when the zero-dimensional field route applies.
    #[getter]
    fn rational_univariate_representation(&self) -> Option<PyRationalUnivariateRepresentation> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value
                .rational_univariate_representation()
                .cloned()
                .map(|inner| PyRationalUnivariateRepresentation {
                    inner: RationalUnivariateRepresentationInner::Rational(Box::new(inner)),
                }),
            PolynomialPrimeCertificateInner::FiniteField(value) => value
                .rational_univariate_representation()
                .cloned()
                .map(PyRationalUnivariateRepresentation::from_finite_field),
        }
    }

    /// Exact theorem certifying the recorded ideal as prime.
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value.theorem(),
            PolynomialPrimeCertificateInner::FiniteField(value) => value.theorem(),
        }
    }

    /// Independently recheck the complete primality evidence.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value.verify(),
            PolynomialPrimeCertificateInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Serialize the complete primality evidence.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => value.to_json(),
            PolynomialPrimeCertificateInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Deserialize and exactly reverify polynomial primality evidence.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("ideal")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                PolynomialPrimeCertificate::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                PolynomialPrimeCertificate::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported polynomial-prime coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized polynomial-prime certificate has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PolynomialPrimeCertificateInner::Rational(value) => canonical_hash(value),
            PolynomialPrimeCertificateInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PolynomialPrimeCertificate(proof={}, generators={})",
            self.proof_kind(),
            match &self.inner {
                PolynomialPrimeCertificateInner::Rational(value) => {
                    value.ideal().generators().len()
                }
                PolynomialPrimeCertificateInner::FiniteField(value) => {
                    value.ideal().generators().len()
                }
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum EquidimensionalHullInner {
    Rational(EquidimensionalHull<RationalField>),
    FiniteField(EquidimensionalHull<FiniteField>),
}

/// Verified scheme-theoretic top-dimensional hull of a polynomial ideal.
#[pyclass(
    module = "noether.commalg",
    name = "EquidimensionalHull",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEquidimensionalHull {
    inner: EquidimensionalHullInner,
}

impl PyEquidimensionalHull {
    fn from_rational(value: EquidimensionalHull<RationalField>) -> Self {
        Self {
            inner: EquidimensionalHullInner::Rational(value),
        }
    }

    fn from_finite_field(value: EquidimensionalHull<FiniteField>) -> Self {
        Self {
            inner: EquidimensionalHullInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyEquidimensionalHull {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/EquidimensionalHull@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            EquidimensionalHullInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn codimension(&self) -> usize {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => value.codimension(),
            EquidimensionalHullInner::FiniteField(value) => value.codimension(),
        }
    }

    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            EquidimensionalHullInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
        }
    }

    #[getter]
    fn associated_primes(&self) -> PyAssociatedPrimeSet {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => {
                PyAssociatedPrimeSet::from_rational(value.associated_primes().clone())
            }
            EquidimensionalHullInner::FiniteField(value) => {
                PyAssociatedPrimeSet::from_finite_field(value.associated_primes().clone())
            }
        }
    }

    #[getter]
    fn minimal_primes(&self) -> Option<PyMinimalPrimeDecomposition> {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => {
                value
                    .minimal_primes()
                    .cloned()
                    .map(|minimal_primes| PyMinimalPrimeDecomposition {
                        inner: MinimalPrimeDecompositionInner::Rational(minimal_primes),
                    })
            }
            EquidimensionalHullInner::FiniteField(value) => {
                value
                    .minimal_primes()
                    .cloned()
                    .map(|minimal_primes| PyMinimalPrimeDecomposition {
                        inner: MinimalPrimeDecompositionInner::FiniteField(minimal_primes),
                    })
            }
        }
    }

    #[getter]
    fn prime_certificates(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            EquidimensionalHullInner::FiniteField(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => value.theorem(),
            EquidimensionalHullInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => value.verify(),
            EquidimensionalHullInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => value.to_json(),
            EquidimensionalHullInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                EquidimensionalHull::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                EquidimensionalHull::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported equidimensional-hull coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized equidimensional hull has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            EquidimensionalHullInner::Rational(value) => canonical_hash(value),
            EquidimensionalHullInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EquidimensionalHull(codimension={}, generators={})",
            self.codimension(),
            match &self.inner {
                EquidimensionalHullInner::Rational(value) => value.ideal().generators().len(),
                EquidimensionalHullInner::FiniteField(value) => value.ideal().generators().len(),
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum GrauertRemmertStepInner {
    Rational(Box<GrauertRemmertStep<RationalField>>),
    FiniteField(Box<GrauertRemmertStep<FiniteField>>),
}

/// One verified finite birational Grauert--Remmert end-ring extension.
#[pyclass(
    module = "noether.commalg",
    name = "GrauertRemmertStep",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrauertRemmertStep {
    inner: GrauertRemmertStepInner,
}

impl PyGrauertRemmertStep {
    fn from_rational(value: GrauertRemmertStep<RationalField>) -> Self {
        Self {
            inner: GrauertRemmertStepInner::Rational(Box::new(value)),
        }
    }

    fn from_finite_field(value: GrauertRemmertStep<FiniteField>) -> Self {
        Self {
            inner: GrauertRemmertStepInner::FiniteField(Box::new(value)),
        }
    }
}

#[pymethods]
impl PyGrauertRemmertStep {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/GrauertRemmertStep@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn source_prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.source_prime().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.source_prime().clone())
            }
        }
    }

    #[getter]
    fn test_ideal(&self) -> PyIdeal {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.test_ideal().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.test_ideal().clone())
            }
        }
    }

    #[getter]
    fn numerator(&self) -> PyPolynomial {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyPolynomial::from_rational(value.numerator().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.numerator().clone())
            }
        }
    }

    #[getter]
    fn denominator(&self) -> PyPolynomial {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyPolynomial::from_rational(value.denominator().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.denominator().clone())
            }
        }
    }

    #[getter]
    fn source_algebra(&self) -> PyQuotientRing {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyQuotientRing::from_rational(value.source_algebra().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.source_algebra().clone())
            }
        }
    }

    #[getter]
    fn target_algebra(&self) -> PyQuotientRing {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyQuotientRing::from_rational(value.target_algebra().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.target_algebra().clone())
            }
        }
    }

    #[getter]
    fn target(&self) -> PyIdeal {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.target().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.target().clone())
            }
        }
    }

    #[getter]
    fn target_prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.target_prime().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.target_prime().clone())
            }
        }
    }

    #[getter]
    fn structure_map(&self) -> PyQuotientRingMap {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => {
                PyQuotientRingMap::from_rational_identity(value.structure_map().clone())
            }
            GrauertRemmertStepInner::FiniteField(value) => {
                PyQuotientRingMap::from_finite_identity(value.structure_map().clone())
            }
        }
    }

    #[getter]
    fn coordinate_images(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            GrauertRemmertStepInner::FiniteField(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn integral_equations(&self) -> Vec<Vec<PyPolynomial>> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value
                .integral_equations()
                .iter()
                .map(|equation| {
                    equation
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_rational)
                        .collect()
                })
                .collect(),
            GrauertRemmertStepInner::FiniteField(value) => value
                .integral_equations()
                .iter()
                .map(|equation| {
                    equation
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_finite_field)
                        .collect()
                })
                .collect(),
        }
    }

    #[getter]
    fn fraction_numerators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value
                .fraction_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            GrauertRemmertStepInner::FiniteField(value) => value
                .fraction_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn fraction_denominators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value
                .fraction_denominators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            GrauertRemmertStepInner::FiniteField(value) => value
                .fraction_denominators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn proof_kind(&self) -> &'static str {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value.proof_kind(),
            GrauertRemmertStepInner::FiniteField(value) => value.proof_kind(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value.theorem(),
            GrauertRemmertStepInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value.verify(),
            GrauertRemmertStepInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value.to_json(),
            GrauertRemmertStepInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                GrauertRemmertStep::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                GrauertRemmertStep::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported Grauert--Remmert coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized Grauert--Remmert step has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            GrauertRemmertStepInner::Rational(value) => canonical_hash(value.as_ref()),
            GrauertRemmertStepInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            GrauertRemmertStepInner::Rational(value) => value.to_string(),
            GrauertRemmertStepInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum AffineNormalizationInner {
    Rational(Box<AffineNormalization<RationalField>>),
    FiniteField(Box<AffineNormalization<FiniteField>>),
}

/// A verified finite birational normalization and its conductor.
#[pyclass(
    module = "noether.commalg",
    name = "AffineNormalization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAffineNormalization {
    inner: AffineNormalizationInner,
}

impl PyAffineNormalization {
    fn from_rational(value: AffineNormalization<RationalField>) -> Self {
        Self {
            inner: AffineNormalizationInner::Rational(Box::new(value)),
        }
    }

    fn from_finite_field(value: AffineNormalization<FiniteField>) -> Self {
        Self {
            inner: AffineNormalizationInner::FiniteField(Box::new(value)),
        }
    }
}

#[pymethods]
impl PyAffineNormalization {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/AffineNormalization@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn source_prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.source_prime().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.source_prime().clone())
            }
        }
    }

    #[getter]
    fn source_algebra(&self) -> PyQuotientRing {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyQuotientRing::from_rational(value.source_algebra().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.source_algebra().clone())
            }
        }
    }

    #[getter]
    fn normal_algebra(&self) -> PyQuotientRing {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyQuotientRing::from_rational(value.normal_algebra().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.normal_algebra().clone())
            }
        }
    }

    #[getter]
    fn normal_prime(&self) -> Option<PyPolynomialPrimeCertificate> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .normal_prime()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational),
            AffineNormalizationInner::FiniteField(value) => value
                .normal_prime()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field),
        }
    }

    #[getter]
    fn normal_jacobian_minors(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .normal_jacobian_minors()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            AffineNormalizationInner::FiniteField(value) => value
                .normal_jacobian_minors()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn normal_singular_locus(&self) -> Option<PyIdeal> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .normal_singular_locus()
                .cloned()
                .map(PyIdeal::from_rational_polynomial),
            AffineNormalizationInner::FiniteField(value) => value
                .normal_singular_locus()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial),
        }
    }

    #[getter]
    fn structure_map(&self) -> PyQuotientRingMap {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyQuotientRingMap::from_rational_identity(value.structure_map().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyQuotientRingMap::from_finite_identity(value.structure_map().clone())
            }
        }
    }

    #[getter]
    fn coordinate_images(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            AffineNormalizationInner::FiniteField(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn integral_equations(&self) -> Vec<Vec<PyPolynomial>> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .integral_equations()
                .iter()
                .map(|equation| {
                    equation
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_rational)
                        .collect()
                })
                .collect(),
            AffineNormalizationInner::FiniteField(value) => value
                .integral_equations()
                .iter()
                .map(|equation| {
                    equation
                        .iter()
                        .cloned()
                        .map(PyPolynomial::from_finite_field)
                        .collect()
                })
                .collect(),
        }
    }

    #[getter]
    fn fraction_numerators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .fraction_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            AffineNormalizationInner::FiniteField(value) => value
                .fraction_numerators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn fraction_denominators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value
                .fraction_denominators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            AffineNormalizationInner::FiniteField(value) => value
                .fraction_denominators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn source_conductor(&self) -> PyIdeal {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source_conductor().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source_conductor().clone())
            }
        }
    }

    #[getter]
    fn normal_conductor(&self) -> PyIdeal {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.normal_conductor().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.normal_conductor().clone())
            }
        }
    }

    #[getter]
    fn nonnormal_locus(&self) -> PyIdeal {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.nonnormal_locus().clone())
            }
            AffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.nonnormal_locus().clone())
            }
        }
    }

    #[getter]
    fn is_normal(&self) -> bool {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.is_normal(),
            AffineNormalizationInner::FiniteField(value) => value.is_normal(),
        }
    }

    #[getter]
    fn proof_kind(&self) -> &'static str {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.proof_kind(),
            AffineNormalizationInner::FiniteField(value) => value.proof_kind(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.theorem(),
            AffineNormalizationInner::FiniteField(value) => value.theorem(),
        }
    }

    #[getter]
    fn conductor_theorem(&self) -> &'static str {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.conductor_theorem(),
            AffineNormalizationInner::FiniteField(value) => value.conductor_theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.verify(),
            AffineNormalizationInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => value.to_json(),
            AffineNormalizationInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                AffineNormalization::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                AffineNormalization::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported affine-normalization coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized affine normalization has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            AffineNormalizationInner::Rational(value) => canonical_hash(value.as_ref()),
            AffineNormalizationInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "AffineNormalization(proof={}, is_normal={})",
            self.proof_kind(),
            self.is_normal()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ReducedAffineNormalizationInner {
    Rational(ReducedAffineNormalization<RationalField>),
    FiniteField(ReducedAffineNormalization<FiniteField>),
}

/// Componentwise normalization of a reduced affine algebra.
#[pyclass(
    module = "noether.commalg",
    name = "ReducedAffineNormalization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyReducedAffineNormalization {
    inner: ReducedAffineNormalizationInner,
}

impl PyReducedAffineNormalization {
    fn from_rational(value: ReducedAffineNormalization<RationalField>) -> Self {
        Self {
            inner: ReducedAffineNormalizationInner::Rational(value),
        }
    }

    fn from_finite_field(value: ReducedAffineNormalization<FiniteField>) -> Self {
        Self {
            inner: ReducedAffineNormalizationInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyReducedAffineNormalization {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/ReducedAffineNormalization@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            ReducedAffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn source_algebra(&self) -> PyQuotientRing {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => {
                PyQuotientRing::from_rational(value.source_algebra().clone())
            }
            ReducedAffineNormalizationInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.source_algebra().clone())
            }
        }
    }

    #[getter]
    fn minimal_primes(&self) -> PyMinimalPrimeDecomposition {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => PyMinimalPrimeDecomposition {
                inner: MinimalPrimeDecompositionInner::Rational(value.minimal_primes().clone()),
            },
            ReducedAffineNormalizationInner::FiniteField(value) => PyMinimalPrimeDecomposition {
                inner: MinimalPrimeDecompositionInner::FiniteField(value.minimal_primes().clone()),
            },
        }
    }

    #[getter]
    fn components(&self) -> Vec<PyAffineNormalization> {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyAffineNormalization::from_rational)
                .collect(),
            ReducedAffineNormalizationInner::FiniteField(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyAffineNormalization::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn component_maps(&self) -> Vec<PyQuotientRingMap> {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value
                .component_maps()
                .iter()
                .cloned()
                .map(PyQuotientRingMap::from_rational_identity)
                .collect(),
            ReducedAffineNormalizationInner::FiniteField(value) => value
                .component_maps()
                .iter()
                .cloned()
                .map(PyQuotientRingMap::from_finite_identity)
                .collect(),
        }
    }

    #[getter]
    fn component_count(&self) -> usize {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.component_count(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.component_count(),
        }
    }

    #[getter]
    fn source_conductor(&self) -> PyIdeal {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source_conductor().clone())
            }
            ReducedAffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source_conductor().clone())
            }
        }
    }

    #[getter]
    fn target_conductor_factors(&self) -> Vec<PyIdeal> {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value
                .target_conductor_factors()
                .iter()
                .cloned()
                .map(PyIdeal::from_rational_polynomial)
                .collect(),
            ReducedAffineNormalizationInner::FiniteField(value) => value
                .target_conductor_factors()
                .iter()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial)
                .collect(),
        }
    }

    #[getter]
    fn nonnormal_locus(&self) -> PyIdeal {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.nonnormal_locus().clone())
            }
            ReducedAffineNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.nonnormal_locus().clone())
            }
        }
    }

    #[getter]
    fn is_normal(&self) -> bool {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.is_normal(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.is_normal(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.theorem(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.theorem(),
        }
    }

    #[getter]
    fn conductor_theorem(&self) -> &'static str {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.conductor_theorem(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.conductor_theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.verify(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => value.to_json(),
            ReducedAffineNormalizationInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                ReducedAffineNormalization::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                ReducedAffineNormalization::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported reduced-normalization coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized reduced normalization has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            ReducedAffineNormalizationInner::Rational(value) => canonical_hash(value),
            ReducedAffineNormalizationInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ReducedAffineNormalization(components={}, is_normal={})",
            self.component_count(),
            self.is_normal()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum NoetherNormalizationInner {
    Rational(NoetherNormalization<RationalField>),
    FiniteField(NoetherNormalization<FiniteField>),
}

/// A verified finite polynomial-subring map into an affine algebra.
#[pyclass(
    module = "noether.commalg",
    name = "NoetherNormalization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNoetherNormalization {
    inner: NoetherNormalizationInner,
}

impl PyNoetherNormalization {
    fn from_rational(value: NoetherNormalization<RationalField>) -> Self {
        Self {
            inner: NoetherNormalizationInner::Rational(value),
        }
    }

    fn from_finite_field(value: NoetherNormalization<FiniteField>) -> Self {
        Self {
            inner: NoetherNormalizationInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyNoetherNormalization {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/NoetherNormalization@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            NoetherNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.dimension(),
            NoetherNormalizationInner::FiniteField(value) => value.dimension(),
        }
    }

    #[getter]
    fn parameter_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.parameter_ring().clone())
            }
            NoetherNormalizationInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.parameter_ring().clone())
            }
        }
    }

    #[getter]
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            NoetherNormalizationInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }

    #[getter]
    fn parameter_indices(&self) -> Vec<usize> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.parameter_indices().to_vec(),
            NoetherNormalizationInner::FiniteField(value) => value.parameter_indices().to_vec(),
        }
    }

    #[getter]
    fn dependent_indices(&self) -> Vec<usize> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.dependent_indices().to_vec(),
            NoetherNormalizationInner::FiniteField(value) => value.dependent_indices().to_vec(),
        }
    }

    #[getter]
    fn shear_coefficients(&self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value
                .shear_coefficients()
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|coefficient| rational_to_python(py, coefficient))
                        .collect()
                })
                .collect(),
            NoetherNormalizationInner::FiniteField(value) => value
                .shear_coefficients()
                .iter()
                .map(|row| {
                    row.iter()
                        .cloned()
                        .map(|inner| {
                            Py::new(py, PyFiniteFieldElement { inner })
                                .map(|value| value.into_any())
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[getter]
    fn coordinate_images(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            NoetherNormalizationInner::FiniteField(value) => value
                .coordinate_images()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn parameter_images(&self) -> Vec<PyQuotientElement> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value
                .parameter_images()
                .iter()
                .cloned()
                .map(PyQuotientElement::from_rational)
                .collect(),
            NoetherNormalizationInner::FiniteField(value) => value
                .parameter_images()
                .iter()
                .cloned()
                .map(PyQuotientElement::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn transformed_ideal(&self) -> PyIdeal {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.transformed_ideal().clone())
            }
            NoetherNormalizationInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.transformed_ideal().clone())
            }
        }
    }

    #[getter]
    fn pure_power_bounds(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.pure_power_bounds(),
            NoetherNormalizationInner::FiniteField(value) => value.pure_power_bounds(),
        }
        .iter()
        .map(|bound| integer_to_python(py, bound))
        .collect()
    }

    #[getter]
    fn module_generator_count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let count = match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.module_generator_count(),
            NoetherNormalizationInner::FiniteField(value) => value.module_generator_count(),
        };
        integer_to_python(py, &count)
    }

    fn module_generator_exponents(&self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let exponents = match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.module_generator_exponents(),
            NoetherNormalizationInner::FiniteField(value) => value.module_generator_exponents(),
        }
        .map_err(|error| match error {
            NoetherNormalizationError::ResourceLimit { .. } => {
                PyOverflowError::new_err(error.to_string())
            }
            _ => value_error(error),
        })?;
        exponents
            .iter()
            .map(|exponent| {
                exponent
                    .iter()
                    .map(|power| integer_to_python(py, power))
                    .collect()
            })
            .collect()
    }

    fn module_generators(&self) -> PyResult<Vec<PyQuotientElement>> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value
                .module_generators()
                .map(|values| {
                    values
                        .into_iter()
                        .map(PyQuotientElement::from_rational)
                        .collect()
                })
                .map_err(|error| normalization_error(RationalField::TYPE_ID, error)),
            NoetherNormalizationInner::FiniteField(value) => value
                .module_generators()
                .map(|values| {
                    values
                        .into_iter()
                        .map(PyQuotientElement::from_finite_field)
                        .collect()
                })
                .map_err(|error| normalization_error(FiniteField::TYPE_ID, error)),
        }
    }

    fn apply(&self, value: &PyPolynomial) -> PyResult<PyQuotientElement> {
        match &self.inner {
            NoetherNormalizationInner::Rational(normalization) => {
                let value = value.as_rational().ok_or_else(|| {
                    PyTypeError::new_err("normalization parameter must be a polynomial over Q")
                })?;
                normalization
                    .apply(value)
                    .map(PyQuotientElement::from_rational)
                    .map_err(|error| normalization_error(RationalField::TYPE_ID, error))
            }
            NoetherNormalizationInner::FiniteField(normalization) => {
                let value = value.as_finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "normalization parameter must be a polynomial over the normalization's finite field",
                    )
                })?;
                normalization
                    .apply(value)
                    .map(PyQuotientElement::from_finite_field)
                    .map_err(|error| normalization_error(FiniteField::TYPE_ID, error))
            }
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.theorem(),
            NoetherNormalizationInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.verify(),
            NoetherNormalizationInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => value.to_json(),
            NoetherNormalizationInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                NoetherNormalization::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                NoetherNormalization::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported Noether-normalization coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized Noether normalization has no coefficient context",
            )),
        }
    }

    fn __call__(&self, value: &PyPolynomial) -> PyResult<PyQuotientElement> {
        self.apply(value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            NoetherNormalizationInner::Rational(value) => canonical_hash(value),
            NoetherNormalizationInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "NoetherNormalization(dimension={}, module_generators={})",
            self.dimension(),
            match &self.inner {
                NoetherNormalizationInner::Rational(value) => value.module_generator_count(),
                NoetherNormalizationInner::FiniteField(value) => value.module_generator_count(),
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum EquidimensionalRadicalComponentInner {
    Rational(EquidimensionalRadicalComponent<RationalField>),
    FiniteField(EquidimensionalRadicalComponent<FiniteField>),
}

/// One verified equidimensional radical component.
#[pyclass(
    module = "noether.commalg",
    name = "EquidimensionalRadicalComponent",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEquidimensionalRadicalComponent {
    inner: EquidimensionalRadicalComponentInner,
}

impl PyEquidimensionalRadicalComponent {
    fn from_rational(value: EquidimensionalRadicalComponent<RationalField>) -> Self {
        Self {
            inner: EquidimensionalRadicalComponentInner::Rational(value),
        }
    }

    fn from_finite_field(value: EquidimensionalRadicalComponent<FiniteField>) -> Self {
        Self {
            inner: EquidimensionalRadicalComponentInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyEquidimensionalRadicalComponent {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/EquidimensionalRadicalComponent@1"
    }

    #[getter]
    fn dimension(&self) -> i64 {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => value.dimension(),
            EquidimensionalRadicalComponentInner::FiniteField(value) => value.dimension(),
        }
    }

    #[getter]
    fn ideal(&self) -> PyIdeal {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.ideal().clone())
            }
            EquidimensionalRadicalComponentInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.ideal().clone())
            }
        }
    }

    #[getter]
    fn prime_certificates(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            EquidimensionalRadicalComponentInner::FiniteField(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => value.theorem(),
            EquidimensionalRadicalComponentInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => value.verify(),
            EquidimensionalRadicalComponentInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => value.to_json(),
            EquidimensionalRadicalComponentInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("ideal")
            .and_then(|ideal| ideal.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                EquidimensionalRadicalComponent::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                EquidimensionalRadicalComponent::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported equidimensional-component coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized equidimensional component has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            EquidimensionalRadicalComponentInner::Rational(value) => canonical_hash(value),
            EquidimensionalRadicalComponentInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EquidimensionalRadicalComponent(dimension={}, minimal_primes={})",
            self.dimension(),
            self.prime_certificates().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum EquidimensionalRadicalDecompositionInner {
    Rational(EquidimensionalRadicalDecomposition<RationalField>),
    FiniteField(EquidimensionalRadicalDecomposition<FiniteField>),
}

/// Verified decomposition of a polynomial-ideal radical by dimension.
#[pyclass(
    module = "noether.commalg",
    name = "EquidimensionalRadicalDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEquidimensionalRadicalDecomposition {
    inner: EquidimensionalRadicalDecompositionInner,
}

impl PyEquidimensionalRadicalDecomposition {
    fn from_rational(value: EquidimensionalRadicalDecomposition<RationalField>) -> Self {
        Self {
            inner: EquidimensionalRadicalDecompositionInner::Rational(value),
        }
    }

    fn from_finite_field(value: EquidimensionalRadicalDecomposition<FiniteField>) -> Self {
        Self {
            inner: EquidimensionalRadicalDecompositionInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyEquidimensionalRadicalDecomposition {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/EquidimensionalRadicalDecomposition@1"
    }

    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn radical(&self) -> PyRadicalIdeal {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => PyRadicalIdeal {
                inner: RadicalIdealInner::Rational(value.radical().clone()),
            },
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => PyRadicalIdeal {
                inner: RadicalIdealInner::FiniteField(value.radical().clone()),
            },
        }
    }

    #[getter]
    fn minimal_primes(&self) -> PyMinimalPrimeDecomposition {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => {
                PyMinimalPrimeDecomposition {
                    inner: MinimalPrimeDecompositionInner::Rational(value.minimal_primes().clone()),
                }
            }
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => {
                PyMinimalPrimeDecomposition {
                    inner: MinimalPrimeDecompositionInner::FiniteField(
                        value.minimal_primes().clone(),
                    ),
                }
            }
        }
    }

    #[getter]
    fn components(&self) -> Vec<PyEquidimensionalRadicalComponent> {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyEquidimensionalRadicalComponent::from_rational)
                .collect(),
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyEquidimensionalRadicalComponent::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => value.theorem(),
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => value.verify(),
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => value.to_json(),
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("minimal_primes")
            .and_then(|minimal| minimal.get("source"))
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self::from_rational(
                EquidimensionalRadicalDecomposition::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(FiniteField::TYPE_ID) => Ok(Self::from_finite_field(
                EquidimensionalRadicalDecomposition::<FiniteField>::from_json(value)
                    .map_err(value_error)?,
            )),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported equidimensional-decomposition coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized equidimensional decomposition has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            EquidimensionalRadicalDecompositionInner::Rational(value) => canonical_hash(value),
            EquidimensionalRadicalDecompositionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EquidimensionalRadicalDecomposition(components={})",
            self.components().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum MinimalPrimeDecompositionInner {
    Rational(MinimalPrimeDecomposition<RationalField>),
    FiniteField(MinimalPrimeDecomposition<FiniteField>),
}

/// Verified minimal primes of an admitted polynomial ideal.
#[pyclass(
    module = "noether.commalg",
    name = "MinimalPrimeDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimalPrimeDecomposition {
    inner: MinimalPrimeDecompositionInner,
}

#[pymethods]
impl PyMinimalPrimeDecomposition {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        "noether/commalg/MinimalPrimeDecomposition@1"
    }

    /// Original ideal whose minimal primes are represented.
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            MinimalPrimeDecompositionInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    /// Verified radical of the source ideal.
    #[getter]
    fn radical(&self) -> PyRadicalIdeal {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => PyRadicalIdeal {
                inner: RadicalIdealInner::Rational(value.radical().clone()),
            },
            MinimalPrimeDecompositionInner::FiniteField(value) => PyRadicalIdeal {
                inner: RadicalIdealInner::FiniteField(value.radical().clone()),
            },
        }
    }

    /// Canonically ordered minimal prime ideals.
    #[getter]
    fn components(&self) -> Vec<PyIdeal> {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyIdeal::from_rational_polynomial)
                .collect(),
            MinimalPrimeDecompositionInner::FiniteField(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyIdeal::from_finite_field_polynomial)
                .collect(),
        }
    }

    /// Exact primality certificate for each component.
    #[getter]
    fn prime_certificates(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            MinimalPrimeDecompositionInner::FiniteField(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    /// Exact theorem certifying the recorded components as prime.
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => value.theorem(),
            MinimalPrimeDecompositionInner::FiniteField(value) => value.theorem(),
        }
    }

    /// Recheck radical equality, component intersection, and primality evidence.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => value.verify(),
            MinimalPrimeDecompositionInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    /// Serialize the complete minimal-prime evidence.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => value.to_json(),
            MinimalPrimeDecompositionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    /// Deserialize and exactly reverify minimal-prime evidence.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let coefficient_type = payload
            .get("source")
            .and_then(|source| source.get("ring"))
            .and_then(|ring| ring.get("coefficient_ring"))
            .and_then(|coefficient| coefficient.get("type"))
            .and_then(serde_json::Value::as_str);
        match coefficient_type {
            Some(RationalField::TYPE_ID) => Ok(Self {
                inner: MinimalPrimeDecompositionInner::Rational(
                    MinimalPrimeDecomposition::<RationalField>::from_json(value)
                        .map_err(value_error)?,
                ),
            }),
            Some(FiniteField::TYPE_ID) => Ok(Self {
                inner: MinimalPrimeDecompositionInner::FiniteField(
                    MinimalPrimeDecomposition::<FiniteField>::from_json(value)
                        .map_err(value_error)?,
                ),
            }),
            Some(type_id) => Err(PyTypeError::new_err(format!(
                "unsupported minimal-prime coefficient context {type_id:?}"
            ))),
            None => Err(PyTypeError::new_err(
                "serialized minimal-prime decomposition has no coefficient context",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            MinimalPrimeDecompositionInner::Rational(value) => canonical_hash(value),
            MinimalPrimeDecompositionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "MinimalPrimeDecomposition(components={})",
            match &self.inner {
                MinimalPrimeDecompositionInner::Rational(value) => value.components().len(),
                MinimalPrimeDecompositionInner::FiniteField(value) => value.components().len(),
            }
        ))
    }
}

#[pymethods]
impl PyIdealDecomposition {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        IdealDecomposition::<RationalFunctionField<FiniteField>>::TYPE_ID
    }

    /// The base-changed radical ideal reconstructed by the components.
    #[getter]
    fn source(&self) -> PyIdeal {
        PyIdeal::from_finite_rational_function_polynomial(self.inner.source().clone())
    }

    /// Canonically ordered top-dimensional minimal components.
    #[getter]
    fn components(&self) -> Vec<PyIdeal> {
        self.inner
            .components()
            .iter()
            .cloned()
            .map(PyIdeal::from_finite_rational_function_polynomial)
            .collect()
    }

    /// Absolute coefficient-Frobenius action on component indices.
    #[getter]
    fn frobenius_permutation(&self) -> Vec<usize> {
        self.inner
            .frobenius_permutation()
            .unwrap_or_default()
            .to_vec()
    }

    /// Standard Hilbert polynomial of every component.
    fn hilbert_polynomials(&self) -> PyResult<Vec<PyPolynomial>> {
        self.inner
            .components()
            .iter()
            .map(|component| {
                component
                    .hilbert_polynomial()
                    .map(PyPolynomial::from_rational)
                    .map_err(value_error)
            })
            .collect()
    }

    /// Recheck contexts, dimensions, and component intersection.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    /// Recompute and verify every coefficient-Frobenius image.
    fn verify_frobenius(&self) -> PyResult<()> {
        self.inner.verify_frobenius().map_err(value_error)
    }

    /// Serialize the source, components, and verified Frobenius action.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and reverify all decomposition evidence.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: IdealDecomposition::from_json(value).map_err(value_error)?,
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
                "IdealDecomposition(components={}, frobenius_permutation={:?})",
                self.inner.components().len(),
                self.inner.frobenius_permutation().unwrap_or_default()
            )
        })
    }
}

/// One exact element of a concrete number field.
#[pyclass(
    module = "noether.numthy",
    name = "NumberFieldElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyNumberFieldElement {
    pub(crate) inner: NumberFieldElement,
}

#[pymethods]
impl PyNumberFieldElement {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        NumberFieldElement::TYPE_ID
    }

    /// Concrete number-field context.
    #[getter]
    fn field(&self) -> PyNumberField {
        PyNumberField {
            inner: self.inner.field().clone(),
        }
    }

    /// Exact ascending power-basis coefficients as `Fraction` values.
    #[getter]
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .coefficients()
            .iter()
            .map(|coefficient| rational_to_python(py, coefficient))
            .collect()
    }

    /// Whether this element is zero.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Whether this element is one.
    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Multiplicative inverse, failing on zero.
    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .map_err(|error| PyZeroDivisionError::new_err(error.to_string()))
    }

    /// Minimal polynomial over `Q`.
    fn minimal_polynomial(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_rational(
            self.inner.minimal_polynomial().map_err(value_error)?,
        ))
    }

    /// Characteristic polynomial over `Q`.
    fn characteristic_polynomial(&self) -> PyResult<PyPolynomial> {
        Ok(PyPolynomial::from_rational(
            self.inner
                .characteristic_polynomial()
                .map_err(value_error)?,
        ))
    }

    /// Exact field trace to `Q`.
    fn trace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, &self.inner.trace())
    }

    /// Exact field norm to `Q`.
    fn norm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, &self.inner.norm())
    }

    /// Certified values under every complex embedding.
    #[pyo3(signature = (precision=53))]
    fn conjugates(&self, precision: u64) -> PyResult<Vec<PyComplexBall>> {
        let field = ComplexBallField::new(precision).map_err(value_error)?;
        Ok(self
            .inner
            .conjugates(field)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyComplexBall { inner })
            .collect())
    }

    /// Canonical JSON representation.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Restore and validate an exact element representation.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: NumberFieldElement::from_json(value).map_err(value_error)?,
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: self.inner.add(&other).map_err(value_error)?,
        })
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: self.inner.sub(&other).map_err(value_error)?,
        })
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: other.sub(&self.inner).map_err(value_error)?,
        })
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: self.inner.mul(&other).map_err(value_error)?,
        })
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: self.inner.checked_div(&other).map_err(division_error)?,
        })
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = extract_number_field_operand(py, self.inner.field(), other)?;
        Ok(Self {
            inner: other.checked_div(&self.inner).map_err(division_error)?,
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn __pow__(&self, exponent: u64, modulus: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulus.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "three-argument pow is not defined for number-field elements",
            ));
        }
        Ok(Self {
            inner: self.inner.pow(exponent).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("NumberFieldElement({})", self.inner))
    }
}
