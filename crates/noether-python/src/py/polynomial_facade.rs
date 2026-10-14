// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of generic sparse polynomial rings.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::cmp::Ordering;
use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    BaseChangeError, Degree, FreeAbelianGrading, MonomialMap, MonomialOrder, Polynomial,
    PolynomialConstantEmbedding, PolynomialError, PolynomialFactorization, PolynomialHomomorphism,
    PolynomialRing, RationalFunction, RationalFunctionField, RealRootIsolation, ResidueClass,
    ResidueRing, StrongGroebnerBasis,
};
use noether::numthy::{
    FiniteField, FiniteFieldElement, FiniteFieldEmbedding, NumberField, PadicElement, PadicField,
    PadicLocalField, PadicLocalFieldElement, PadicRing, QadicElement, QadicField, QadicRing,
};
use noether::{
    CompositeCoefficientMap, IdentityCoefficientMap, Integer, IntegerFiniteFieldMap,
    IntegerRationalEmbedding, IntegerReduction, IntegerRing, PrimeSubfieldEmbedding, RationalField,
    Ring, SemanticType, UnsupportedOperationError, Q, Z,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyList, PyString, PyTuple, PyType};
use serde::{Deserialize, Serialize};

use super::ball::PyRealBall;
use super::coefficient_maps::{
    PyIntegerFiniteFieldMap, PyIntegerRationalEmbedding, PyIntegerReduction,
    PyPrimeSubfieldEmbedding,
};
use super::commalg::{PyResidueClass, PyResidueRing};
use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::ideal_evidence::PyArithmeticNullstellensatzCertificate;
use super::monomial_ideal::{extract_exponent_vectors, MonomialIdealInner, PyMonomialIdeal};
use super::number_field::{
    extract_number_field_element, IdealInner, PyIdeal, PyNumberField, PyNumberFieldElement,
};
use super::numthy::{
    PyFiniteField, PyFiniteFieldElement, PyFiniteFieldEmbedding, PyPadicElement, PyPadicField,
    PyPadicFieldFrobenius, PyPadicLocalField, PyPadicLocalFieldCoefficientFrobenius,
    PyPadicLocalFieldElement, PyPadicRing, PyPadicRingFrobenius, PyQadicElement, PyQadicField,
    PyQadicFieldFrobenius, PyQadicRing, PyQadicRingFrobenius,
};
use super::quotient::{quotient_from_ideal, PyQuotientRing};
use super::rational_function::{
    PyRationalFunction, PyRationalFunctionField, PyRationalFunctionFieldMap,
};
use super::root::{PyIntegerRing, PyRationalField};
use super::series::{
    PyDegree, PyFreeAbelianGrading, PyHilbertSeries, PyPositiveGrading, PySeriesTruncation,
    PyWeightedGrading,
};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn base_change_error(error: BaseChangeError) -> PyErr {
    match error {
        BaseChangeError::ContextMismatch(error)
        | BaseChangeError::CoefficientMapping(noether::CoefficientMappingError::ContextMismatch(
            error,
        )) => super::errors::context_mismatch(error),
        error => value_error(error),
    }
}

const FIELD_POLYNOMIAL_CONTEXTS: &[&str] = &[
    RationalField::TYPE_ID,
    FiniteField::TYPE_ID,
    NumberField::TYPE_ID,
    RationalFunctionField::<FiniteField>::TYPE_ID,
];

const EXACT_GCD_POLYNOMIAL_CONTEXTS: &[&str] = &[
    IntegerRing::TYPE_ID,
    RationalField::TYPE_ID,
    FiniteField::TYPE_ID,
    NumberField::TYPE_ID,
    RationalFunctionField::<FiniteField>::TYPE_ID,
];

const FROBENIUS_POLYNOMIAL_CONTEXTS: &[&str] = &[
    PadicRing::TYPE_ID,
    PadicField::TYPE_ID,
    QadicRing::TYPE_ID,
    QadicField::TYPE_ID,
];

const SEMILINEAR_POLYNOMIAL_CONTEXTS: &[&str] = &[
    PadicRing::TYPE_ID,
    PadicField::TYPE_ID,
    QadicRing::TYPE_ID,
    QadicField::TYPE_ID,
    PadicLocalField::TYPE_ID,
];

fn unsupported_polynomial_operation(
    operation: &'static str,
    context: &'static str,
    supported_contexts: &[&'static str],
) -> PyErr {
    super::errors::unsupported_operation(
        UnsupportedOperationError::new(
            format!("noether/commalg/Polynomial/{operation}@1"),
            context,
        )
        .with_supported_contexts(supported_contexts.iter().copied()),
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

fn coefficient_type_from_json(value: &str, polynomial: bool) -> PyResult<String> {
    let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
    let ring = if polynomial {
        payload.get("ring")
    } else {
        Some(&payload)
    }
    .and_then(|ring| ring.get("coefficient_ring"))
    .ok_or_else(|| value_error("serialized polynomial context has no coefficient ring"))?;
    ring.get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| value_error("serialized coefficient ring has no string type identity"))
}

fn rational_function_base_type_from_json(value: &str, polynomial: bool) -> PyResult<String> {
    let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
    let field = if polynomial {
        payload.get("ring")
    } else {
        Some(&payload)
    }
    .and_then(|ring| ring.get("coefficient_ring"))
    .ok_or_else(|| value_error("serialized polynomial context has no coefficient ring"))?;
    field
        .get("coefficient_ring")
        .and_then(|context| context.get("type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| value_error("serialized rational-function field has no base-field type"))
}

fn extract_residue(ring: &ResidueRing, value: &Bound<'_, PyAny>) -> PyResult<ResidueClass> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyResidueClass>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error(
            "residue coefficient belongs to another context",
        ));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or ResidueClass",
    ))
}

pub(crate) fn extract_finite_field(
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
        return Err(value_error(
            "finite-field coefficient belongs to another context",
        ));
    }
    Err(PyTypeError::new_err(
        "coefficient must be a Python int or FiniteFieldElement",
    ))
}

pub(crate) fn extract_finite_rational_function(
    field: &RationalFunctionField<FiniteField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<RationalFunction<FiniteField>> {
    if let Ok(function) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        let function = function.as_finite_field().ok_or_else(|| {
            value_error("rational-function coefficient has a different base field")
        })?;
        if field.contains(function) {
            return Ok(function.clone());
        }
        return Err(value_error(
            "rational-function coefficient belongs to another field context",
        ));
    }
    field
        .constant(extract_finite_field(field.coefficient_ring(), value)?)
        .map_err(value_error)
}

pub(crate) fn extract_rational_function(
    py: Python<'_>,
    field: &RationalFunctionField<RationalField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<RationalFunction<RationalField>> {
    if let Ok(function) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        let function = function.as_rational().ok_or_else(|| {
            value_error("rational-function coefficient has a different base field")
        })?;
        if field.contains(function) {
            return Ok(function.clone());
        }
        return Err(value_error(
            "rational-function coefficient belongs to another field context",
        ));
    }
    field
        .constant(extract_rational(py, value)?)
        .map_err(value_error)
}

fn extract_padic(ring: &PadicRing, value: &Bound<'_, PyAny>) -> PyResult<PadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicElement>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(value_error("p-adic coefficient belongs to another context"));
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

fn extract_exponent(value: &Bound<'_, PyAny>) -> PyResult<Vec<Integer>> {
    value
        .try_iter()?
        .map(|power| extract_integer(&power?))
        .collect()
}

fn collect_terms<E>(
    terms: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<(Vec<Integer>, E)>> {
    terms
        .try_iter()?
        .map(|item| {
            let item = item?;
            let pair = item.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(
                    "each polynomial term must be an (exponent, coefficient) tuple",
                )
            })?;
            if pair.len() != 2 {
                return Err(PyTypeError::new_err(
                    "each polynomial term must contain exactly two items",
                ));
            }
            Ok((
                extract_exponent(&pair.get_item(0)?)?,
                extract(&pair.get_item(1)?)?,
            ))
        })
        .collect()
}

/// A checked monomial endomorphism on an ordered variable list.
#[pyclass(
    module = "noether.commalg",
    name = "MonomialMap",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMonomialMap {
    pub(crate) inner: MonomialMap,
}

#[pymethods]
impl PyMonomialMap {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        MonomialMap::TYPE_ID
    }

    #[new]
    fn new(images: &Bound<'_, PyAny>) -> PyResult<Self> {
        let images = images
            .try_iter()?
            .map(|row| extract_exponent(&row?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: MonomialMap::new(images).map_err(value_error)?,
        })
    }

    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, arity: usize) -> Self {
        Self {
            inner: MonomialMap::identity(arity),
        }
    }

    #[classmethod]
    fn diagonal_power(
        _cls: &Bound<'_, PyType>,
        arity: usize,
        power: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MonomialMap::diagonal_power(arity, extract_integer(power)?)
                .map_err(value_error)?,
        })
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    #[getter]
    fn images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyList::empty(py);
        for row in self.inner.images() {
            output.append(integers_to_python(py, row)?)?;
        }
        Ok(output.unbind().into_any())
    }

    fn apply(&self, py: Python<'_>, exponent: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        integers_to_python(
            py,
            &self
                .inner
                .apply(&extract_exponent(exponent)?)
                .map_err(value_error)?,
        )
    }

    fn compose(&self, inner: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.compose(&inner.inner).map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: MonomialMap::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("MonomialMap({:?})", self.inner.images()))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// One admissible monomial order.
#[pyclass(
    module = "noether.commalg",
    name = "MonomialOrder",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMonomialOrder {
    pub(super) inner: MonomialOrder,
}

#[pymethods]
impl PyMonomialOrder {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        MonomialOrder::TYPE_ID
    }

    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_order_name(name)?,
        })
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn LEX() -> Self {
        Self {
            inner: MonomialOrder::Lexicographic,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GRLEX() -> Self {
        Self {
            inner: MonomialOrder::GradedLexicographic,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GREVLEX() -> Self {
        Self {
            inner: MonomialOrder::GradedReverseLexicographic,
        }
    }

    /// Construct an order from `(variable_indices, component_order)` blocks.
    #[classmethod]
    fn block(_cls: &Bound<'_, PyType>, blocks: &Bound<'_, PyAny>) -> PyResult<Self> {
        let blocks = blocks
            .try_iter()?
            .map(|item| {
                let item = item?;
                let pair = item.cast::<PyTuple>().map_err(|_| {
                    PyTypeError::new_err(
                        "each monomial-order block must be an (indices, order) tuple",
                    )
                })?;
                if pair.len() != 2 {
                    return Err(PyTypeError::new_err(
                        "each monomial-order block must contain exactly two items",
                    ));
                }
                let variables = pair
                    .get_item(0)?
                    .try_iter()?
                    .map(|variable| variable?.extract::<usize>())
                    .collect::<PyResult<Vec<_>>>()?;
                let order_value = pair.get_item(1)?;
                let order = extract_order(Some(&order_value))?;
                Ok((variables, order))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: MonomialOrder::block(blocks).map_err(value_error)?,
        })
    }

    /// Put selected variables in a leading lexicographic elimination block.
    #[classmethod]
    fn elimination(
        _cls: &Bound<'_, PyType>,
        variable_count: usize,
        eliminated: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let eliminated = eliminated
            .try_iter()?
            .map(|variable| variable?.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: MonomialOrder::elimination(variable_count, eliminated).map_err(value_error)?,
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    fn blocks(&self) -> Option<Vec<(Vec<usize>, Self)>> {
        self.inner.blocks().map(|blocks| {
            blocks
                .into_iter()
                .map(|(variables, order)| (variables, Self { inner: order }))
                .collect()
        })
    }

    /// Serialize the stable order name as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and validate one admissible monomial order.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: MonomialOrder::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("MonomialOrder({})", self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

fn parse_order_name(name: &str) -> PyResult<MonomialOrder> {
    match name {
        "lex" | "lexicographic" => Ok(MonomialOrder::Lexicographic),
        "grlex" | "graded_lexicographic" => Ok(MonomialOrder::GradedLexicographic),
        "grevlex" | "graded_reverse_lexicographic" => Ok(MonomialOrder::GradedReverseLexicographic),
        _ => Err(PyValueError::new_err(format!(
            "unknown monomial order {name:?}; expected 'lex', 'grlex', or 'grevlex'"
        ))),
    }
}

pub(crate) fn extract_order(value: Option<&Bound<'_, PyAny>>) -> PyResult<MonomialOrder> {
    let Some(value) = value else {
        return Ok(MonomialOrder::default());
    };
    if value.is_instance_of::<PyString>() {
        return parse_order_name(&value.extract::<String>()?);
    }
    if let Ok(order) = value.extract::<PyRef<'_, PyMonomialOrder>>() {
        return Ok(order.inner.clone());
    }
    Err(PyTypeError::new_err(
        "order must be a MonomialOrder or one of 'lex', 'grlex', 'grevlex'",
    ))
}

#[derive(Clone)]
pub(super) enum PolynomialRingInner {
    Integer(PolynomialRing<IntegerRing>),
    Rational(PolynomialRing<RationalField>),
    Residue(PolynomialRing<ResidueRing>),
    FiniteField(PolynomialRing<FiniteField>),
    Padic(PolynomialRing<PadicRing>),
    PadicField(PolynomialRing<PadicField>),
    Qadic(PolynomialRing<QadicRing>),
    QadicField(PolynomialRing<QadicField>),
    PadicLocalField(PolynomialRing<PadicLocalField>),
    NumberField(PolynomialRing<NumberField>),
    RationalFunction(PolynomialRing<RationalFunctionField<RationalField>>),
    FiniteRationalFunction(PolynomialRing<RationalFunctionField<FiniteField>>),
}

impl PolynomialRingInner {
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
            Self::NumberField(ring) => ring.variables(),
            Self::RationalFunction(ring) => ring.variables(),
            Self::FiniteRationalFunction(ring) => ring.variables(),
        }
    }

    fn term_order(&self) -> MonomialOrder {
        match self {
            Self::Integer(ring) => ring.term_order(),
            Self::Rational(ring) => ring.term_order(),
            Self::Residue(ring) => ring.term_order(),
            Self::FiniteField(ring) => ring.term_order(),
            Self::Padic(ring) => ring.term_order(),
            Self::PadicField(ring) => ring.term_order(),
            Self::Qadic(ring) => ring.term_order(),
            Self::QadicField(ring) => ring.term_order(),
            Self::PadicLocalField(ring) => ring.term_order(),
            Self::NumberField(ring) => ring.term_order(),
            Self::RationalFunction(ring) => ring.term_order(),
            Self::FiniteRationalFunction(ring) => ring.term_order(),
        }
    }

    fn grading(&self) -> &FreeAbelianGrading {
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
            Self::NumberField(ring) => ring.grading(),
            Self::RationalFunction(ring) => ring.grading(),
            Self::FiniteRationalFunction(ring) => ring.grading(),
        }
    }

    fn weighted_grading(&self) -> Option<noether::commalg::WeightedGrading> {
        match self {
            Self::Integer(ring) => ring.weighted_grading(),
            Self::Rational(ring) => ring.weighted_grading(),
            Self::Residue(ring) => ring.weighted_grading(),
            Self::FiniteField(ring) => ring.weighted_grading(),
            Self::Padic(ring) => ring.weighted_grading(),
            Self::PadicField(ring) => ring.weighted_grading(),
            Self::Qadic(ring) => ring.weighted_grading(),
            Self::QadicField(ring) => ring.weighted_grading(),
            Self::PadicLocalField(ring) => ring.weighted_grading(),
            Self::NumberField(ring) => ring.weighted_grading(),
            Self::RationalFunction(ring) => ring.weighted_grading(),
            Self::FiniteRationalFunction(ring) => ring.weighted_grading(),
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
            Self::NumberField(ring) => canonical_hash(ring),
            Self::RationalFunction(ring) => canonical_hash(ring),
            Self::FiniteRationalFunction(ring) => canonical_hash(ring),
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
            Self::NumberField(ring) => ring.to_json(),
            Self::RationalFunction(ring) => ring.to_json(),
            Self::FiniteRationalFunction(ring) => ring.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        match coefficient_type_from_json(value, false)?.as_str() {
            IntegerRing::TYPE_ID => Ok(Self::Integer(
                PolynomialRing::<IntegerRing>::from_json(value).map_err(value_error)?,
            )),
            RationalField::TYPE_ID => Ok(Self::Rational(
                PolynomialRing::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            ResidueRing::TYPE_ID => Ok(Self::Residue(
                PolynomialRing::<ResidueRing>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::FiniteField(
                PolynomialRing::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            PadicRing::TYPE_ID => Ok(Self::Padic(
                PolynomialRing::<PadicRing>::from_json(value).map_err(value_error)?,
            )),
            PadicField::TYPE_ID => Ok(Self::PadicField(
                PolynomialRing::<PadicField>::from_json(value).map_err(value_error)?,
            )),
            QadicRing::TYPE_ID => Ok(Self::Qadic(
                PolynomialRing::<QadicRing>::from_json(value).map_err(value_error)?,
            )),
            QadicField::TYPE_ID => Ok(Self::QadicField(
                PolynomialRing::<QadicField>::from_json(value).map_err(value_error)?,
            )),
            PadicLocalField::TYPE_ID => Ok(Self::PadicLocalField(
                PolynomialRing::<PadicLocalField>::from_json(value).map_err(value_error)?,
            )),
            NumberField::TYPE_ID => Ok(Self::NumberField(
                PolynomialRing::<NumberField>::from_json(value).map_err(value_error)?,
            )),
            RationalFunctionField::<FiniteField>::TYPE_ID
                if rational_function_base_type_from_json(value, false)?
                    == RationalField::TYPE_ID =>
            {
                Ok(Self::RationalFunction(
                    PolynomialRing::<RationalFunctionField<RationalField>>::from_json(value)
                        .map_err(value_error)?,
                ))
            }
            RationalFunctionField::<FiniteField>::TYPE_ID
                if rational_function_base_type_from_json(value, false)? == FiniteField::TYPE_ID =>
            {
                Ok(Self::FiniteRationalFunction(
                    PolynomialRing::<RationalFunctionField<FiniteField>>::from_json(value)
                        .map_err(value_error)?,
                ))
            }
            type_id => Err(value_error(format!(
                "unsupported polynomial coefficient context {type_id:?}"
            ))),
        }
    }

    fn homogeneous_monomials(
        &self,
        degree: u64,
        max_count: usize,
    ) -> Result<Vec<PolynomialInner>, PolynomialError> {
        match self {
            Self::Integer(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::Integer,
            ),
            Self::Rational(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::Rational,
            ),
            Self::Residue(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::Residue,
            ),
            Self::FiniteField(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::FiniteField,
            ),
            Self::Padic(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::Padic,
            ),
            Self::PadicField(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::PadicField,
            ),
            Self::Qadic(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::Qadic,
            ),
            Self::QadicField(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::QadicField,
            ),
            Self::PadicLocalField(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::PadicLocalField,
            ),
            Self::NumberField(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::NumberField,
            ),
            Self::RationalFunction(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::RationalFunction,
            ),
            Self::FiniteRationalFunction(ring) => map_polynomial_values(
                ring.homogeneous_monomials(degree, max_count),
                PolynomialInner::FiniteRationalFunction,
            ),
        }
    }

    fn homogeneous_monomials_of_degree(
        &self,
        degree: &Degree,
        positivity: &noether::commalg::PositiveGrading,
        max_count: usize,
    ) -> Result<Vec<PolynomialInner>, PolynomialError> {
        match self {
            Self::Integer(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::Integer,
            ),
            Self::Rational(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::Rational,
            ),
            Self::Residue(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::Residue,
            ),
            Self::FiniteField(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::FiniteField,
            ),
            Self::Padic(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::Padic,
            ),
            Self::PadicField(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::PadicField,
            ),
            Self::Qadic(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::Qadic,
            ),
            Self::QadicField(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::QadicField,
            ),
            Self::PadicLocalField(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::PadicLocalField,
            ),
            Self::NumberField(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::NumberField,
            ),
            Self::RationalFunction(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::RationalFunction,
            ),
            Self::FiniteRationalFunction(ring) => map_polynomial_values(
                ring.homogeneous_monomials_of_degree(degree, positivity, max_count),
                PolynomialInner::FiniteRationalFunction,
            ),
        }
    }

    fn monomials_in_region(
        &self,
        region: &noether::commalg::SeriesTruncation,
        max_count: usize,
    ) -> Result<Vec<PolynomialInner>, PolynomialError> {
        match self {
            Self::Integer(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::Integer,
            ),
            Self::Rational(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::Rational,
            ),
            Self::Residue(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::Residue,
            ),
            Self::FiniteField(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::FiniteField,
            ),
            Self::Padic(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::Padic,
            ),
            Self::PadicField(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::PadicField,
            ),
            Self::Qadic(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::Qadic,
            ),
            Self::QadicField(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::QadicField,
            ),
            Self::PadicLocalField(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::PadicLocalField,
            ),
            Self::NumberField(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::NumberField,
            ),
            Self::RationalFunction(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::RationalFunction,
            ),
            Self::FiniteRationalFunction(ring) => map_polynomial_values(
                ring.monomials_in_region(region, max_count),
                PolynomialInner::FiniteRationalFunction,
            ),
        }
    }
}

impl PartialEq for PolynomialRingInner {
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

impl Eq for PolynomialRingInner {}

/// A generic sparse polynomial ring over one accepted coefficient context.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialRing {
    pub(super) inner: PolynomialRingInner,
}

impl PyPolynomialRing {
    pub(crate) fn from_integer(ring: PolynomialRing<IntegerRing>) -> Self {
        Self {
            inner: PolynomialRingInner::Integer(ring),
        }
    }

    pub(crate) fn from_rational(ring: PolynomialRing<RationalField>) -> Self {
        Self {
            inner: PolynomialRingInner::Rational(ring),
        }
    }

    pub(crate) fn from_residue(ring: PolynomialRing<ResidueRing>) -> Self {
        Self {
            inner: PolynomialRingInner::Residue(ring),
        }
    }

    pub(crate) fn from_finite_field(ring: PolynomialRing<FiniteField>) -> Self {
        Self {
            inner: PolynomialRingInner::FiniteField(ring),
        }
    }

    pub(crate) fn from_finite_rational_function(
        ring: PolynomialRing<RationalFunctionField<FiniteField>>,
    ) -> Self {
        Self {
            inner: PolynomialRingInner::FiniteRationalFunction(ring),
        }
    }

    pub(crate) fn from_rational_function(
        ring: PolynomialRing<RationalFunctionField<RationalField>>,
    ) -> Self {
        Self {
            inner: PolynomialRingInner::RationalFunction(ring),
        }
    }

    fn coefficient_ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PolynomialRingInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            PolynomialRingInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            PolynomialRingInner::Residue(ring) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::FiniteField(ring) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::Padic(ring) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::PadicField(ring) => Ok(Py::new(
                py,
                PyPadicField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::Qadic(ring) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::QadicField(ring) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::PadicLocalField(ring) => Ok(Py::new(
                py,
                PyPadicLocalField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::NumberField(ring) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: ring.coefficient_ring().clone(),
                },
            )?
            .into_any()),
            PolynomialRingInner::RationalFunction(ring) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(ring.coefficient_ring().clone()),
            )?
            .into_any()),
            PolynomialRingInner::FiniteRationalFunction(ring) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_finite_field(ring.coefficient_ring().clone()),
            )?
            .into_any()),
        }
    }

    fn resolve_variable(&self, variable: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(index) = variable.extract::<usize>() {
            if index < self.inner.variables().len() {
                return Ok(index);
            }
            return Err(value_error(format!(
                "polynomial variable index {index} is outside arity {}",
                self.inner.variables().len()
            )));
        }
        if value_is_string(variable) {
            let name = variable.extract::<String>()?;
            return self
                .inner
                .variables()
                .iter()
                .position(|candidate| candidate == &name)
                .ok_or_else(|| value_error(format!("unknown polynomial variable {name:?}")));
        }
        Err(PyTypeError::new_err(
            "polynomial variable must be a nonnegative index or declared name",
        ))
    }

    fn build_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyPolynomial> {
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring) => PolynomialInner::Integer(
                ring.from_terms(collect_terms(terms, extract_integer)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Rational(ring) => PolynomialInner::Rational(
                ring.from_terms(collect_terms(terms, |value| extract_rational(py, value))?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Residue(ring) => PolynomialInner::Residue(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_residue(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::FiniteField(ring) => PolynomialInner::FiniteField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_finite_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::Padic(ring) => PolynomialInner::Padic(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::PadicField(ring) => PolynomialInner::PadicField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::Qadic(ring) => PolynomialInner::Qadic(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_qadic(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::QadicField(ring) => PolynomialInner::QadicField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_qadic_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::PadicLocalField(ring) => PolynomialInner::PadicLocalField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_padic_local_field(ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::NumberField(ring) => PolynomialInner::NumberField(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_number_field_element(py, ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::RationalFunction(ring) => PolynomialInner::RationalFunction(
                ring.from_terms(collect_terms(terms, |value| {
                    extract_rational_function(py, ring.coefficient_ring(), value)
                })?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                PolynomialInner::FiniteRationalFunction(
                    ring.from_terms(collect_terms(terms, |value| {
                        extract_finite_rational_function(ring.coefficient_ring(), value)
                    })?)
                    .map_err(value_error)?,
                )
            }
        };
        Ok(PyPolynomial { inner })
    }

    fn constant_from_python(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyPolynomial> {
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring) => PolynomialInner::Integer(
                ring.constant(extract_integer(value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Rational(ring) => PolynomialInner::Rational(
                ring.constant(extract_rational(py, value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Residue(ring) => PolynomialInner::Residue(
                ring.constant(extract_residue(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::FiniteField(ring) => PolynomialInner::FiniteField(
                ring.constant(extract_finite_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Padic(ring) => PolynomialInner::Padic(
                ring.constant(extract_padic(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::PadicField(ring) => PolynomialInner::PadicField(
                ring.constant(extract_padic_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::Qadic(ring) => PolynomialInner::Qadic(
                ring.constant(extract_qadic(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::QadicField(ring) => PolynomialInner::QadicField(
                ring.constant(extract_qadic_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::PadicLocalField(ring) => PolynomialInner::PadicLocalField(
                ring.constant(extract_padic_local_field(ring.coefficient_ring(), value)?)
                    .map_err(value_error)?,
            ),
            PolynomialRingInner::NumberField(ring) => PolynomialInner::NumberField(
                ring.constant(extract_number_field_element(
                    py,
                    ring.coefficient_ring(),
                    value,
                )?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::RationalFunction(ring) => PolynomialInner::RationalFunction(
                ring.constant(extract_rational_function(
                    py,
                    ring.coefficient_ring(),
                    value,
                )?)
                .map_err(value_error)?,
            ),
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                PolynomialInner::FiniteRationalFunction(
                    ring.constant(extract_finite_rational_function(
                        ring.coefficient_ring(),
                        value,
                    )?)
                    .map_err(value_error)?,
                )
            }
        };
        Ok(PyPolynomial { inner })
    }
}

#[pymethods]
impl PyPolynomialRing {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialRing::<IntegerRing>::TYPE_ID
    }

    #[new]
    #[pyo3(signature = (coefficient_ring, variables, *, order=None, grading=None))]
    fn new(
        coefficient_ring: &Bound<'_, PyAny>,
        variables: Vec<String>,
        order: Option<&Bound<'_, PyAny>>,
        grading: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let order = extract_order(order)?;
        let grading = extract_polynomial_grading(grading)?;
        let inner = if coefficient_ring.is_instance_of::<PyIntegerRing>() {
            PolynomialRingInner::Integer(build_ring(Z, variables, order, grading)?)
        } else if coefficient_ring.is_instance_of::<PyRationalField>() {
            PolynomialRingInner::Rational(build_ring(Q, variables, order, grading)?)
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyResidueRing>>() {
            PolynomialRingInner::Residue(build_ring(ring.inner.clone(), variables, order, grading)?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyFiniteField>>() {
            PolynomialRingInner::FiniteField(build_ring(
                field.inner.clone(),
                variables,
                order,
                grading,
            )?)
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyPadicRing>>() {
            PolynomialRingInner::Padic(build_ring(ring.inner.clone(), variables, order, grading)?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyPadicField>>() {
            PolynomialRingInner::PadicField(build_ring(
                field.inner.clone(),
                variables,
                order,
                grading,
            )?)
        } else if let Ok(ring) = coefficient_ring.extract::<PyRef<'_, PyQadicRing>>() {
            PolynomialRingInner::Qadic(build_ring(ring.inner.clone(), variables, order, grading)?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyQadicField>>() {
            PolynomialRingInner::QadicField(build_ring(
                field.inner.clone(),
                variables,
                order,
                grading,
            )?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            PolynomialRingInner::PadicLocalField(build_ring(
                field.inner.clone(),
                variables,
                order,
                grading,
            )?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyNumberField>>() {
            PolynomialRingInner::NumberField(build_ring(
                field.inner.clone(),
                variables,
                order,
                grading,
            )?)
        } else if let Ok(field) = coefficient_ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            if let Some(field) = field.as_rational() {
                PolynomialRingInner::RationalFunction(build_ring(
                    field.clone(),
                    variables,
                    order,
                    grading,
                )?)
            } else if let Some(field) = field.as_finite_field() {
                PolynomialRingInner::FiniteRationalFunction(build_ring(
                    field.clone(),
                    variables,
                    order,
                    grading,
                )?)
            } else {
                return Err(PyTypeError::new_err(
                    "unsupported rational-function coefficient context",
                ));
            }
        } else {
            return Err(PyTypeError::new_err(
                "PolynomialRing supports Z, Q, ResidueRing, FiniteField, NumberField, RationalFunctionField over Q or a finite field, PadicRing/PadicField, QadicRing/QadicField, and PadicLocalField coefficient contexts",
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
    fn term_order(&self) -> PyMonomialOrder {
        PyMonomialOrder {
            inner: self.inner.term_order(),
        }
    }

    #[getter]
    fn grading(&self) -> PyFreeAbelianGrading {
        PyFreeAbelianGrading {
            inner: self.inner.grading().clone(),
        }
    }

    /// Positive rank-one view of the grading, when one exists.
    #[getter]
    fn weighted_grading(&self) -> Option<PyWeightedGrading> {
        self.inner
            .weighted_grading()
            .map(|inner| PyWeightedGrading { inner })
    }

    fn zero(&self) -> PyPolynomial {
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring) => PolynomialInner::Integer(ring.zero()),
            PolynomialRingInner::Rational(ring) => PolynomialInner::Rational(ring.zero()),
            PolynomialRingInner::Residue(ring) => PolynomialInner::Residue(ring.zero()),
            PolynomialRingInner::FiniteField(ring) => PolynomialInner::FiniteField(ring.zero()),
            PolynomialRingInner::Padic(ring) => PolynomialInner::Padic(ring.zero()),
            PolynomialRingInner::PadicField(ring) => PolynomialInner::PadicField(ring.zero()),
            PolynomialRingInner::Qadic(ring) => PolynomialInner::Qadic(ring.zero()),
            PolynomialRingInner::QadicField(ring) => PolynomialInner::QadicField(ring.zero()),
            PolynomialRingInner::PadicLocalField(ring) => {
                PolynomialInner::PadicLocalField(ring.zero())
            }
            PolynomialRingInner::NumberField(ring) => PolynomialInner::NumberField(ring.zero()),
            PolynomialRingInner::RationalFunction(ring) => {
                PolynomialInner::RationalFunction(ring.zero())
            }
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                PolynomialInner::FiniteRationalFunction(ring.zero())
            }
        };
        PyPolynomial { inner }
    }

    fn one(&self) -> PyPolynomial {
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring) => PolynomialInner::Integer(ring.one()),
            PolynomialRingInner::Rational(ring) => PolynomialInner::Rational(ring.one()),
            PolynomialRingInner::Residue(ring) => PolynomialInner::Residue(ring.one()),
            PolynomialRingInner::FiniteField(ring) => PolynomialInner::FiniteField(ring.one()),
            PolynomialRingInner::Padic(ring) => PolynomialInner::Padic(ring.one()),
            PolynomialRingInner::PadicField(ring) => PolynomialInner::PadicField(ring.one()),
            PolynomialRingInner::Qadic(ring) => PolynomialInner::Qadic(ring.one()),
            PolynomialRingInner::QadicField(ring) => PolynomialInner::QadicField(ring.one()),
            PolynomialRingInner::PadicLocalField(ring) => {
                PolynomialInner::PadicLocalField(ring.one())
            }
            PolynomialRingInner::NumberField(ring) => PolynomialInner::NumberField(ring.one()),
            PolynomialRingInner::RationalFunction(ring) => {
                PolynomialInner::RationalFunction(ring.one())
            }
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                PolynomialInner::FiniteRationalFunction(ring.one())
            }
        };
        PyPolynomial { inner }
    }

    fn constant(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyPolynomial> {
        self.constant_from_python(py, value)
    }

    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyPolynomial> {
        self.constant_from_python(py, value)
    }

    fn from_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyPolynomial> {
        self.build_terms(py, terms)
    }

    fn generator(&self, variable: &Bound<'_, PyAny>) -> PyResult<PyPolynomial> {
        let variable = self.resolve_variable(variable)?;
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring) => {
                PolynomialInner::Integer(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::Rational(ring) => {
                PolynomialInner::Rational(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::Residue(ring) => {
                PolynomialInner::Residue(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::FiniteField(ring) => {
                PolynomialInner::FiniteField(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::Padic(ring) => {
                PolynomialInner::Padic(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::PadicField(ring) => {
                PolynomialInner::PadicField(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::Qadic(ring) => {
                PolynomialInner::Qadic(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::QadicField(ring) => {
                PolynomialInner::QadicField(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::PadicLocalField(ring) => {
                PolynomialInner::PadicLocalField(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::NumberField(ring) => {
                PolynomialInner::NumberField(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::RationalFunction(ring) => {
                PolynomialInner::RationalFunction(ring.generator(variable).map_err(value_error)?)
            }
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                PolynomialInner::FiniteRationalFunction(
                    ring.generator(variable).map_err(value_error)?,
                )
            }
        };
        Ok(PyPolynomial { inner })
    }

    fn generators(&self) -> PyResult<Vec<PyPolynomial>> {
        (0..self.inner.variables().len())
            .map(|variable| {
                let inner = match &self.inner {
                    PolynomialRingInner::Integer(ring) => {
                        PolynomialInner::Integer(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::Rational(ring) => {
                        PolynomialInner::Rational(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::Residue(ring) => {
                        PolynomialInner::Residue(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::FiniteField(ring) => {
                        PolynomialInner::FiniteField(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::Padic(ring) => {
                        PolynomialInner::Padic(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::PadicField(ring) => {
                        PolynomialInner::PadicField(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::Qadic(ring) => {
                        PolynomialInner::Qadic(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::QadicField(ring) => {
                        PolynomialInner::QadicField(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::PadicLocalField(ring) => PolynomialInner::PadicLocalField(
                        ring.generator(variable).map_err(value_error)?,
                    ),
                    PolynomialRingInner::NumberField(ring) => {
                        PolynomialInner::NumberField(ring.generator(variable).map_err(value_error)?)
                    }
                    PolynomialRingInner::RationalFunction(ring) => {
                        PolynomialInner::RationalFunction(
                            ring.generator(variable).map_err(value_error)?,
                        )
                    }
                    PolynomialRingInner::FiniteRationalFunction(ring) => {
                        PolynomialInner::FiniteRationalFunction(
                            ring.generator(variable).map_err(value_error)?,
                        )
                    }
                };
                Ok(PyPolynomial { inner })
            })
            .collect()
    }

    /// Monomials of one exact ordinary or weighted degree, in term order.
    #[pyo3(signature = (degree, *, max_count))]
    fn homogeneous_monomials(&self, degree: u64, max_count: usize) -> PyResult<Vec<PyPolynomial>> {
        Ok(self
            .inner
            .homogeneous_monomials(degree, max_count)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyPolynomial { inner })
            .collect())
    }

    /// Monomials in one exact free-abelian degree, with finite-piece evidence.
    #[pyo3(signature = (degree, positivity, *, max_count))]
    fn homogeneous_monomials_of_degree(
        &self,
        degree: &PyDegree,
        positivity: &PyPositiveGrading,
        max_count: usize,
    ) -> PyResult<Vec<PyPolynomial>> {
        Ok(self
            .inner
            .homogeneous_monomials_of_degree(&degree.inner, &positivity.inner, max_count)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyPolynomial { inner })
            .collect())
    }

    /// Monomials in one explicit finite truncation region, in term order.
    #[pyo3(signature = (region, *, max_count))]
    fn monomials_in_region(
        &self,
        region: &PySeriesTruncation,
        max_count: usize,
    ) -> PyResult<Vec<PyPolynomial>> {
        Ok(self
            .inner
            .monomials_in_region(&region.inner, max_count)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyPolynomial { inner })
            .collect())
    }

    /// Construct a canonical monomial ideal from exponent vectors.
    fn monomial_ideal(&self, generators: &Bound<'_, PyAny>) -> PyResult<PyMonomialIdeal> {
        Ok(PyMonomialIdeal::from_inner(MonomialIdealInner::from_ring(
            &self.inner,
            extract_exponent_vectors(generators)?,
        )?))
    }

    /// Construct a canonical polynomial ideal over `Z` or an accepted field.
    fn ideal(&self, py: Python<'_>, generators: Vec<Py<PyPolynomial>>) -> PyResult<PyIdeal> {
        match &self.inner {
            PolynomialRingInner::Integer(ring) => {
                let mut values = Vec::with_capacity(generators.len());
                for generator in generators {
                    let generator = generator.borrow(py);
                    let value = generator.as_integer().ok_or_else(|| {
                        value_error("ideal generator must be a polynomial over Z")
                    })?;
                    values.push(value.clone());
                }
                Ok(PyIdeal::from_integer_polynomial(
                    ring.ideal(values).map_err(value_error)?,
                ))
            }
            PolynomialRingInner::Rational(ring) => {
                let mut values = Vec::with_capacity(generators.len());
                for generator in generators {
                    let generator = generator.borrow(py);
                    let value = generator.as_rational().ok_or_else(|| {
                        value_error("ideal generator must be a polynomial over Q")
                    })?;
                    values.push(value.clone());
                }
                Ok(PyIdeal::from_rational_polynomial(
                    ring.ideal(values).map_err(value_error)?,
                ))
            }
            PolynomialRingInner::FiniteField(ring) => {
                let mut values = Vec::with_capacity(generators.len());
                for generator in generators {
                    let generator = generator.borrow(py);
                    let value = generator.as_finite_field().ok_or_else(|| {
                        value_error(
                            "ideal generator must be a polynomial over the ring's finite field",
                        )
                    })?;
                    values.push(value.clone());
                }
                Ok(PyIdeal::from_finite_field_polynomial(
                    ring.ideal(values).map_err(value_error)?,
                ))
            }
            PolynomialRingInner::RationalFunction(ring) => {
                let mut values = Vec::with_capacity(generators.len());
                for generator in generators {
                    let generator = generator.borrow(py);
                    let value = generator.as_rational_function().ok_or_else(|| {
                        value_error(
                            "ideal generator must be a polynomial over the ring's rational-function field",
                        )
                    })?;
                    values.push(value.clone());
                }
                Ok(PyIdeal::from_rational_function_polynomial(
                    ring.ideal(values).map_err(value_error)?,
                ))
            }
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                let mut values = Vec::with_capacity(generators.len());
                for generator in generators {
                    let generator = generator.borrow(py);
                    let value = generator.as_finite_rational_function().ok_or_else(|| {
                        value_error(
                            "ideal generator must be a polynomial over the ring's rational-function field",
                        )
                    })?;
                    values.push(value.clone());
                }
                Ok(PyIdeal::from_finite_rational_function_polynomial(
                    ring.ideal(values).map_err(value_error)?,
                ))
            }
            PolynomialRingInner::Residue(_)
            | PolynomialRingInner::Padic(_)
            | PolynomialRingInner::PadicField(_)
            | PolynomialRingInner::Qadic(_)
            | PolynomialRingInner::QadicField(_)
            | PolynomialRingInner::PadicLocalField(_)
            | PolynomialRingInner::NumberField(_) => Err(PyTypeError::new_err(
                "the polynomial ideal projection currently supports Z, Q, finite fields, and rational-function fields over Q or a finite field",
            )),
        }
    }

    /// Form the exact quotient by an ideal in this polynomial ring.
    fn quotient(&self, ideal: &PyIdeal) -> PyResult<PyQuotientRing> {
        quotient_from_ideal(self, ideal)
    }

    /// Exact Kähler differentials over the coefficient field.
    fn kahler_differentials(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::differentials::PyKahlerDifferentials> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact module construction.
        super::differentials::kahler_differentials_from_polynomial_ring(py, self)
    }

    /// Cotangent fiber at an explicit coefficient-field point.
    fn cotangent_space_at(
        &self,
        py: Python<'_>,
        point: Vec<Py<PyAny>>,
    ) -> PyResult<super::differentials::PyCotangentSpace> {
        // NOETHER_RELEASES_GIL: delegated helper detaches exact fiber construction.
        super::differentials::cotangent_space_from_polynomial_ring(py, self, point)
    }

    /// Construct a field-polynomial ideal with an exact Hilbert-series target.
    fn ideal_with_hilbert_series(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPolynomial>>,
        target: &PyHilbertSeries,
    ) -> PyResult<PyIdeal> {
        match &self.inner {
            PolynomialRingInner::Rational(ring) => {
                let values = generators
                    .into_iter()
                    .map(|generator| {
                        generator.borrow(py).as_rational().cloned().ok_or_else(|| {
                            value_error("ideal generator must be a polynomial over Q")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyIdeal::from_rational_polynomial(
                    ring.ideal_with_hilbert_series(values, &target.inner)
                        .map_err(value_error)?,
                ))
            }
            PolynomialRingInner::FiniteField(ring) => {
                let values = generators
                    .into_iter()
                    .map(|generator| {
                        generator
                            .borrow(py)
                            .as_finite_field()
                            .cloned()
                            .ok_or_else(|| {
                                value_error(
                                    "ideal generator must be a polynomial over the ring's finite field",
                                )
                            })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyIdeal::from_finite_field_polynomial(
                    ring.ideal_with_hilbert_series(values, &target.inner)
                        .map_err(value_error)?,
                ))
            }
            PolynomialRingInner::RationalFunction(ring) => {
                let values = generators
                    .into_iter()
                    .map(|generator| {
                        generator
                            .borrow(py)
                            .as_rational_function()
                            .cloned()
                            .ok_or_else(|| {
                                value_error(
                                    "ideal generator must be a polynomial over the ring's rational-function field",
                                )
                            })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyIdeal::from_rational_function_polynomial(
                    ring.ideal_with_hilbert_series(values, &target.inner)
                        .map_err(value_error)?,
                ))
            }
            PolynomialRingInner::FiniteRationalFunction(ring) => {
                let values = generators
                    .into_iter()
                    .map(|generator| {
                        generator
                            .borrow(py)
                            .as_finite_rational_function()
                            .cloned()
                            .ok_or_else(|| {
                                value_error(
                                    "ideal generator must be a polynomial over the ring's rational-function field",
                                )
                            })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyIdeal::from_finite_rational_function_polynomial(
                    ring.ideal_with_hilbert_series(values, &target.inner)
                        .map_err(value_error)?,
                ))
            }
            PolynomialRingInner::Integer(_)
            | PolynomialRingInner::Residue(_)
            | PolynomialRingInner::Padic(_)
            | PolynomialRingInner::PadicField(_)
            | PolynomialRingInner::Qadic(_)
            | PolynomialRingInner::QadicField(_)
            | PolynomialRingInner::PadicLocalField(_)
            | PolynomialRingInner::NumberField(_) => Err(PyTypeError::new_err(
                "ideal_with_hilbert_series currently supports the projected Groebner fields",
            )),
        }
    }

    /// Compute rational Nullstellensatz multipliers for integral generators
    /// without first constructing a strong basis over `Z`.
    fn nullstellensatz_certificate(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPolynomial>>,
    ) -> PyResult<PyArithmeticNullstellensatzCertificate> {
        let PolynomialRingInner::Integer(ring) = &self.inner else {
            return Err(PyTypeError::new_err(
                "nullstellensatz_certificate requires a polynomial ring over Z",
            ));
        };
        let values = generators
            .into_iter()
            .map(|generator| {
                generator.borrow(py).as_integer().cloned().ok_or_else(|| {
                    value_error("Nullstellensatz generator must be a polynomial over Z")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyArithmeticNullstellensatzCertificate {
            inner: py
                .detach(|| ring.nullstellensatz_certificate(values))
                .map_err(value_error)?,
        })
    }

    /// Compute a normalized strong Groebner basis over `Z`.
    fn strong_groebner_basis(
        &self,
        py: Python<'_>,
        generators: Vec<Py<PyPolynomial>>,
    ) -> PyResult<PyStrongGroebnerBasis> {
        let PolynomialRingInner::Integer(ring) = &self.inner else {
            return Err(PyTypeError::new_err(
                "strong_groebner_basis requires a polynomial ring over Z",
            ));
        };
        let mut values = Vec::with_capacity(generators.len());
        for generator in generators {
            let generator = generator.borrow(py);
            let value = generator
                .as_integer()
                .ok_or_else(|| value_error("strong basis generator must be a polynomial over Z"))?;
            values.push(value.clone());
        }
        Ok(PyStrongGroebnerBasis {
            inner: py
                .detach(|| ring.strong_groebner_basis(values))
                .map_err(value_error)?,
        })
    }

    /// Extend coefficients along one checked standard map while preserving the
    /// complete polynomial presentation.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialRingInner::Integer(ring)
                if map.is_instance_of::<PyIntegerRationalEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerRationalEmbedding>>()?;
                PolynomialRingInner::Rational(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialRingInner::Integer(ring) if map.is_instance_of::<PyIntegerReduction>() => {
                let map = map.extract::<PyRef<'_, PyIntegerReduction>>()?;
                PolynomialRingInner::Residue(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialRingInner::Integer(ring)
                if map.is_instance_of::<PyIntegerFiniteFieldMap>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerFiniteFieldMap>>()?;
                PolynomialRingInner::FiniteField(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialRingInner::FiniteField(ring)
                if map.is_instance_of::<PyPrimeSubfieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>()?;
                PolynomialRingInner::FiniteField(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialRingInner::FiniteField(ring)
                if map.is_instance_of::<PyFiniteFieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>()?;
                PolynomialRingInner::FiniteField(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialRingInner::FiniteRationalFunction(ring)
                if map.is_instance_of::<PyRationalFunctionFieldMap>() =>
            {
                let map = map.extract::<PyRef<'_, PyRationalFunctionFieldMap>>()?;
                PolynomialRingInner::FiniteRationalFunction(
                    ring.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "unsupported polynomial-ring base-change context or coefficient map",
                ));
            }
        };
        Ok(Self { inner })
    }

    /// Construct a checked homomorphism to a polynomial or coefficient ring.
    #[pyo3(signature = (target, images, *, coefficient_map=None))]
    fn homomorphism(
        &self,
        py: Python<'_>,
        target: &Bound<'_, PyAny>,
        images: &Bound<'_, PyAny>,
        coefficient_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyPolynomialHomomorphism> {
        build_polynomial_homomorphism(py, self, target, images, coefficient_map)
    }

    /// Serialize the complete coefficient, variable, order, and grading context.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a supported polynomial-ring context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PolynomialRingInner::from_json(value)?,
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
            if let Some(grading) = self.inner.weighted_grading() {
                format!(
                    "PolynomialRing(variables={:?}, order={:?}, weights={:?})",
                    self.inner.variables(),
                    self.inner.term_order().name(),
                    grading.weights()
                )
            } else {
                format!(
                    "PolynomialRing(variables={:?}, order={:?}, grading={})",
                    self.inner.variables(),
                    self.inner.term_order().name(),
                    self.inner.grading()
                )
            }
        })
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A verified normalized strong Groebner basis over `Z`.
#[pyclass(
    module = "noether.commalg",
    name = "StrongGroebnerBasis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStrongGroebnerBasis {
    inner: StrongGroebnerBasis,
}

#[pymethods]
impl PyStrongGroebnerBasis {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        StrongGroebnerBasis::TYPE_ID
    }

    /// Integral polynomial-ring context.
    #[getter]
    fn ring(&self) -> PyPolynomialRing {
        PyPolynomialRing::from_integer(self.inner.ring().clone())
    }

    /// Deterministically normalized strong basis.
    #[getter]
    fn basis(&self) -> Vec<PyPolynomial> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(PyPolynomial::from_integer)
            .collect()
    }

    /// Euclidean strong normal form.
    fn normal_form(&self, value: &PyPolynomial) -> PyResult<PyPolynomial> {
        let value = value
            .as_integer()
            .ok_or_else(|| PyTypeError::new_err("normal_form requires a polynomial over Z"))?;
        self.inner
            .normal_form(value)
            .map(PyPolynomial::from_integer)
            .map_err(value_error)
    }

    /// Verify exact membership through strong reduction.
    fn __contains__(&self, value: &PyPolynomial) -> PyResult<bool> {
        let value = value
            .as_integer()
            .ok_or_else(|| PyTypeError::new_err("membership requires a polynomial over Z"))?;
        self.inner.contains(value).map_err(value_error)
    }

    /// Recompute and verify the normalized strong basis.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    /// Serialize the context and strong basis as typed JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a typed strong basis.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StrongGroebnerBasis::from_json(value).map_err(value_error)?,
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
            "StrongGroebnerBasis(variables={}, basis={})",
            self.inner.ring().variables().len(),
            self.inner.basis().len()
        ))
    }
}

#[derive(Clone)]
enum PolynomialFactorizationInner {
    Integer(PolynomialFactorization<IntegerRing>),
    Rational(PolynomialFactorization<RationalField>),
    FiniteField(PolynomialFactorization<FiniteField>),
}

impl PartialEq for PolynomialFactorizationInner {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for PolynomialFactorizationInner {}

/// Canonical complete factorization of one supported polynomial.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialFactorization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomialFactorization {
    inner: PolynomialFactorizationInner,
}

#[pymethods]
impl PyPolynomialFactorization {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialFactorization::<IntegerRing>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPolynomial {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => {
                PyPolynomial::from_integer(value.source().clone())
            }
            PolynomialFactorizationInner::Rational(value) => {
                PyPolynomial::from_rational(value.source().clone())
            }
            PolynomialFactorizationInner::FiniteField(value) => {
                PyPolynomial::from_finite_field(value.source().clone())
            }
        }
    }

    #[getter]
    fn coefficient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => {
                integer_to_python(py, value.coefficient())
            }
            PolynomialFactorizationInner::Rational(value) => {
                rational_to_python(py, value.coefficient())
            }
            PolynomialFactorizationInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.coefficient().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn factors(&self) -> Vec<(PyPolynomial, u64)> {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => value
                .factors()
                .iter()
                .map(|(factor, multiplicity)| {
                    (PyPolynomial::from_integer(factor.clone()), *multiplicity)
                })
                .collect(),
            PolynomialFactorizationInner::Rational(value) => value
                .factors()
                .iter()
                .map(|(factor, multiplicity)| {
                    (PyPolynomial::from_rational(factor.clone()), *multiplicity)
                })
                .collect(),
            PolynomialFactorizationInner::FiniteField(value) => value
                .factors()
                .iter()
                .map(|(factor, multiplicity)| {
                    (
                        PyPolynomial::from_finite_field(factor.clone()),
                        *multiplicity,
                    )
                })
                .collect(),
        }
    }

    #[getter]
    fn value(&self) -> PyResult<PyPolynomial> {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => value
                .value()
                .map(PyPolynomial::from_integer)
                .map_err(value_error),
            PolynomialFactorizationInner::Rational(value) => value
                .value()
                .map(PyPolynomial::from_rational)
                .map_err(value_error),
            PolynomialFactorizationInner::FiniteField(value) => value
                .value()
                .map(PyPolynomial::from_finite_field)
                .map_err(value_error),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => value.verify(),
            PolynomialFactorizationInner::Rational(value) => value.verify(),
            PolynomialFactorizationInner::FiniteField(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => value.to_json(),
            PolynomialFactorizationInner::Rational(value) => value.to_json(),
            PolynomialFactorizationInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let source = payload
            .get("source")
            .ok_or_else(|| value_error("factorization JSON is missing its source polynomial"))?;
        let source = serde_json::to_string(source).map_err(value_error)?;
        let inner = match coefficient_type_from_json(&source, true)?.as_str() {
            IntegerRing::TYPE_ID => PolynomialFactorizationInner::Integer(
                PolynomialFactorization::<IntegerRing>::from_json(value).map_err(value_error)?,
            ),
            RationalField::TYPE_ID => PolynomialFactorizationInner::Rational(
                PolynomialFactorization::<RationalField>::from_json(value).map_err(value_error)?,
            ),
            FiniteField::TYPE_ID => PolynomialFactorizationInner::FiniteField(
                PolynomialFactorization::<FiniteField>::from_json(value).map_err(value_error)?,
            ),
            context => {
                return Err(value_error(format!(
                    "unsupported polynomial factorization coefficient context {context:?}"
                )))
            }
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            PolynomialFactorizationInner::Integer(value) => canonical_hash(value),
            PolynomialFactorizationInner::Rational(value) => canonical_hash(value),
            PolynomialFactorizationInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PolynomialFactorization(factors={})",
            self.factors().len()
        ))
    }
}

/// One exact real root isolated between rational endpoints.
#[pyclass(
    module = "noether.commalg",
    name = "RealRootIsolation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRealRootIsolation {
    pub(crate) inner: RealRootIsolation,
}

#[pymethods]
impl PyRealRootIsolation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RealRootIsolation::TYPE_ID
    }

    /// Exact rational polynomial whose root is isolated.
    #[getter]
    fn polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_rational(self.inner.polynomial().clone())
    }

    /// Monic square-free defining factor carrying this root.
    #[getter]
    fn defining_factor(&self) -> PyPolynomial {
        PyPolynomial::from_rational(self.inner.defining_factor().clone())
    }

    /// Strict lower rational endpoint.
    #[getter]
    fn lower(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, self.inner.lower())
    }

    /// Strict upper rational endpoint.
    #[getter]
    fn upper(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, self.inner.upper())
    }

    /// Exact root multiplicity.
    #[getter]
    fn multiplicity(&self) -> usize {
        self.inner.multiplicity()
    }

    /// Refine to an interval of width at most `2^-bits`.
    fn refine(&self, bits: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.refine(bits).map_err(value_error)?,
        })
    }

    /// Convert an exact refinement into a rigorous Arb enclosure.
    fn ball(&self, precision: u64) -> PyResult<PyRealBall> {
        Ok(PyRealBall {
            inner: self.inner.ball(precision).map_err(value_error)?,
        })
    }

    /// Exact comparison with a rational, returned as `-1`, `0`, or `1`.
    fn compare_rational(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<i8> {
        self.inner
            .compare_rational(&extract_rational(py, value)?)
            .map(ordering_sign)
            .map_err(value_error)
    }

    /// Exact comparison with another isolated root.
    fn compare(&self, other: &Self) -> PyResult<i8> {
        self.inner
            .compare(&other.inner)
            .map(ordering_sign)
            .map_err(value_error)
    }

    /// Exact sign, `-1`, `0`, or `1`.
    fn sign(&self) -> PyResult<i8> {
        self.inner.sign().map_err(value_error)
    }

    /// Greatest integer not exceeding this root.
    fn floor(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.floor().map_err(value_error)?)
    }

    /// Least integer not smaller than this root.
    fn ceiling(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.ceiling().map_err(value_error)?)
    }

    /// Verify the exact Sturm and multiplicity certificate.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RealRootIsolation::from_json(value).map_err(value_error)?,
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

fn ordering_sign(ordering: Ordering) -> i8 {
    match ordering {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

fn build_ring<R: noether::CommutativeRing>(
    coefficient_ring: R,
    variables: Vec<String>,
    order: MonomialOrder,
    grading: Option<FreeAbelianGrading>,
) -> PyResult<PolynomialRing<R>> {
    match grading {
        Some(grading) => PolynomialRing::with_grading(coefficient_ring, variables, order, grading)
            .map_err(value_error),
        None => PolynomialRing::new(coefficient_ring, variables, order).map_err(value_error),
    }
}

fn extract_polynomial_grading(
    grading: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<FreeAbelianGrading>> {
    let Some(grading) = grading else {
        return Ok(None);
    };
    if let Ok(grading) = grading.extract::<PyRef<'_, PyFreeAbelianGrading>>() {
        return Ok(Some(grading.inner.clone()));
    }
    if let Ok(grading) = grading.extract::<PyRef<'_, PyWeightedGrading>>() {
        return Ok(Some(grading.inner.as_free_abelian()));
    }
    Err(PyTypeError::new_err(
        "grading must be a FreeAbelianGrading or WeightedGrading",
    ))
}

#[derive(Clone)]
pub(super) enum PolynomialInner {
    Integer(Polynomial<IntegerRing>),
    Rational(Polynomial<RationalField>),
    Residue(Polynomial<ResidueRing>),
    FiniteField(Polynomial<FiniteField>),
    Padic(Polynomial<PadicRing>),
    PadicField(Polynomial<PadicField>),
    Qadic(Polynomial<QadicRing>),
    QadicField(Polynomial<QadicField>),
    PadicLocalField(Polynomial<PadicLocalField>),
    NumberField(Polynomial<NumberField>),
    RationalFunction(Polynomial<RationalFunctionField<RationalField>>),
    FiniteRationalFunction(Polynomial<RationalFunctionField<FiniteField>>),
}

fn map_polynomial_values<R: noether::CommutativeRing>(
    values: Result<Vec<Polynomial<R>>, PolynomialError>,
    wrap: fn(Polynomial<R>) -> PolynomialInner,
) -> Result<Vec<PolynomialInner>, PolynomialError> {
    values.map(|values| values.into_iter().map(wrap).collect())
}

impl PolynomialInner {
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
            Self::NumberField(_) => NumberField::TYPE_ID,
            Self::RationalFunction(_) => RationalFunctionField::<RationalField>::TYPE_ID,
            Self::FiniteRationalFunction(_) => RationalFunctionField::<FiniteField>::TYPE_ID,
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
            Self::NumberField(value) => value.nonzero_term_count(),
            Self::RationalFunction(value) => value.nonzero_term_count(),
            Self::FiniteRationalFunction(value) => value.nonzero_term_count(),
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
            Self::NumberField(value) => value.is_zero(),
            Self::RationalFunction(value) => value.is_zero(),
            Self::FiniteRationalFunction(value) => value.is_zero(),
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
            Self::NumberField(value) => value.is_one(),
            Self::RationalFunction(value) => value.is_one(),
            Self::FiniteRationalFunction(value) => value.is_one(),
        }
    }

    fn multihomogeneous_degree(&self) -> Result<Option<Degree>, PolynomialError> {
        match self {
            Self::Integer(value) => value.multihomogeneous_degree(),
            Self::Rational(value) => value.multihomogeneous_degree(),
            Self::Residue(value) => value.multihomogeneous_degree(),
            Self::FiniteField(value) => value.multihomogeneous_degree(),
            Self::Padic(value) => value.multihomogeneous_degree(),
            Self::PadicField(value) => value.multihomogeneous_degree(),
            Self::Qadic(value) => value.multihomogeneous_degree(),
            Self::QadicField(value) => value.multihomogeneous_degree(),
            Self::PadicLocalField(value) => value.multihomogeneous_degree(),
            Self::NumberField(value) => value.multihomogeneous_degree(),
            Self::RationalFunction(value) => value.multihomogeneous_degree(),
            Self::FiniteRationalFunction(value) => value.multihomogeneous_degree(),
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
            Self::NumberField(value) => canonical_hash(value),
            Self::RationalFunction(value) => canonical_hash(value),
            Self::FiniteRationalFunction(value) => canonical_hash(value),
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
            Self::NumberField(value) => value.to_json(),
            Self::RationalFunction(value) => value.to_json(),
            Self::FiniteRationalFunction(value) => value.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        match coefficient_type_from_json(value, true)?.as_str() {
            IntegerRing::TYPE_ID => Ok(Self::Integer(
                Polynomial::<IntegerRing>::from_json(value).map_err(value_error)?,
            )),
            RationalField::TYPE_ID => Ok(Self::Rational(
                Polynomial::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            ResidueRing::TYPE_ID => Ok(Self::Residue(
                Polynomial::<ResidueRing>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::FiniteField(
                Polynomial::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            PadicRing::TYPE_ID => Ok(Self::Padic(
                Polynomial::<PadicRing>::from_json(value).map_err(value_error)?,
            )),
            PadicField::TYPE_ID => Ok(Self::PadicField(
                Polynomial::<PadicField>::from_json(value).map_err(value_error)?,
            )),
            QadicRing::TYPE_ID => Ok(Self::Qadic(
                Polynomial::<QadicRing>::from_json(value).map_err(value_error)?,
            )),
            QadicField::TYPE_ID => Ok(Self::QadicField(
                Polynomial::<QadicField>::from_json(value).map_err(value_error)?,
            )),
            PadicLocalField::TYPE_ID => Ok(Self::PadicLocalField(
                Polynomial::<PadicLocalField>::from_json(value).map_err(value_error)?,
            )),
            NumberField::TYPE_ID => Ok(Self::NumberField(
                Polynomial::<NumberField>::from_json(value).map_err(value_error)?,
            )),
            RationalFunctionField::<FiniteField>::TYPE_ID
                if rational_function_base_type_from_json(value, true)?
                    == RationalField::TYPE_ID =>
            {
                Ok(Self::RationalFunction(
                    Polynomial::<RationalFunctionField<RationalField>>::from_json(value)
                        .map_err(value_error)?,
                ))
            }
            RationalFunctionField::<FiniteField>::TYPE_ID
                if rational_function_base_type_from_json(value, true)? == FiniteField::TYPE_ID =>
            {
                Ok(Self::FiniteRationalFunction(
                    Polynomial::<RationalFunctionField<FiniteField>>::from_json(value)
                        .map_err(value_error)?,
                ))
            }
            type_id => Err(value_error(format!(
                "unsupported polynomial coefficient context {type_id:?}"
            ))),
        }
    }

    fn coefficient_polynomial(
        &self,
        variable: usize,
        exponent: &Integer,
    ) -> Result<Self, PolynomialError> {
        match self {
            Self::Integer(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::Integer),
            Self::Rational(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::Rational),
            Self::Residue(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::Residue),
            Self::FiniteField(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::FiniteField),
            Self::Padic(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::Padic),
            Self::PadicField(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::PadicField),
            Self::Qadic(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::Qadic),
            Self::QadicField(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::QadicField),
            Self::PadicLocalField(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::PadicLocalField),
            Self::NumberField(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::NumberField),
            Self::RationalFunction(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::RationalFunction),
            Self::FiniteRationalFunction(value) => value
                .coefficient_polynomial(variable, exponent)
                .map(Self::FiniteRationalFunction),
        }
    }

    fn coefficient_polynomials(
        &self,
        variable: usize,
        max_count: usize,
    ) -> Result<Vec<Self>, PolynomialError> {
        match self {
            Self::Integer(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::Integer,
            ),
            Self::Rational(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::Rational,
            ),
            Self::Residue(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::Residue,
            ),
            Self::FiniteField(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::FiniteField,
            ),
            Self::Padic(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::Padic,
            ),
            Self::PadicField(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::PadicField,
            ),
            Self::Qadic(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::Qadic,
            ),
            Self::QadicField(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::QadicField,
            ),
            Self::PadicLocalField(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::PadicLocalField,
            ),
            Self::NumberField(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::NumberField,
            ),
            Self::RationalFunction(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::RationalFunction,
            ),
            Self::FiniteRationalFunction(value) => map_polynomial_values(
                value.coefficient_polynomials(variable, max_count),
                Self::FiniteRationalFunction,
            ),
        }
    }

    fn univariate_view(&self, variable: usize) -> Result<Self, PolynomialError> {
        match self {
            Self::Integer(value) => value.univariate_view(variable).map(Self::Integer),
            Self::Rational(value) => value.univariate_view(variable).map(Self::Rational),
            Self::Residue(value) => value.univariate_view(variable).map(Self::Residue),
            Self::FiniteField(value) => value.univariate_view(variable).map(Self::FiniteField),
            Self::Padic(value) => value.univariate_view(variable).map(Self::Padic),
            Self::PadicField(value) => value.univariate_view(variable).map(Self::PadicField),
            Self::Qadic(value) => value.univariate_view(variable).map(Self::Qadic),
            Self::QadicField(value) => value.univariate_view(variable).map(Self::QadicField),
            Self::PadicLocalField(value) => {
                value.univariate_view(variable).map(Self::PadicLocalField)
            }
            Self::NumberField(value) => value.univariate_view(variable).map(Self::NumberField),
            Self::RationalFunction(value) => {
                value.univariate_view(variable).map(Self::RationalFunction)
            }
            Self::FiniteRationalFunction(value) => value
                .univariate_view(variable)
                .map(Self::FiniteRationalFunction),
        }
    }
}

impl PartialEq for PolynomialInner {
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

impl Eq for PolynomialInner {}

pub(super) type IdentityPolynomialMap<R> =
    CompositeCoefficientMap<IdentityCoefficientMap<R>, PolynomialConstantEmbedding<R>>;
pub(super) type IdentityPolynomialHomomorphism<R> =
    PolynomialHomomorphism<R, PolynomialRing<R>, IdentityPolynomialMap<R>>;
type IntegerRationalPolynomialMap =
    CompositeCoefficientMap<IntegerRationalEmbedding, PolynomialConstantEmbedding<RationalField>>;
type IntegerRationalPolynomialHomomorphism = PolynomialHomomorphism<
    IntegerRing,
    PolynomialRing<RationalField>,
    IntegerRationalPolynomialMap,
>;
type IntegerResiduePolynomialMap =
    CompositeCoefficientMap<IntegerReduction, PolynomialConstantEmbedding<ResidueRing>>;
type IntegerResiduePolynomialHomomorphism =
    PolynomialHomomorphism<IntegerRing, PolynomialRing<ResidueRing>, IntegerResiduePolynomialMap>;
type IntegerFinitePolynomialMap =
    CompositeCoefficientMap<IntegerFiniteFieldMap, PolynomialConstantEmbedding<FiniteField>>;
type IntegerFinitePolynomialHomomorphism =
    PolynomialHomomorphism<IntegerRing, PolynomialRing<FiniteField>, IntegerFinitePolynomialMap>;
type PrimeFinitePolynomialMap =
    CompositeCoefficientMap<PrimeSubfieldEmbedding, PolynomialConstantEmbedding<FiniteField>>;
pub(super) type PrimeFinitePolynomialHomomorphism =
    PolynomialHomomorphism<FiniteField, PolynomialRing<FiniteField>, PrimeFinitePolynomialMap>;
type FiniteEmbeddingPolynomialMap =
    CompositeCoefficientMap<FiniteFieldEmbedding, PolynomialConstantEmbedding<FiniteField>>;
pub(super) type FiniteEmbeddingPolynomialHomomorphism =
    PolynomialHomomorphism<FiniteField, PolynomialRing<FiniteField>, FiniteEmbeddingPolynomialMap>;

#[derive(Clone, PartialEq, Eq)]
pub(super) enum PolynomialHomomorphismInner {
    IntegerIdentity(IdentityPolynomialHomomorphism<IntegerRing>),
    RationalIdentity(IdentityPolynomialHomomorphism<RationalField>),
    ResidueIdentity(IdentityPolynomialHomomorphism<ResidueRing>),
    FiniteIdentity(IdentityPolynomialHomomorphism<FiniteField>),
    IntegerRationalPolynomial(IntegerRationalPolynomialHomomorphism),
    IntegerResiduePolynomial(IntegerResiduePolynomialHomomorphism),
    IntegerFinitePolynomial(IntegerFinitePolynomialHomomorphism),
    PrimeFinitePolynomial(PrimeFinitePolynomialHomomorphism),
    FiniteEmbeddingPolynomial(FiniteEmbeddingPolynomialHomomorphism),
    IntegerRationalScalar(
        PolynomialHomomorphism<IntegerRing, RationalField, IntegerRationalEmbedding>,
    ),
    IntegerResidueScalar(PolynomialHomomorphism<IntegerRing, ResidueRing, IntegerReduction>),
    IntegerFiniteScalar(PolynomialHomomorphism<IntegerRing, FiniteField, IntegerFiniteFieldMap>),
    PrimeFiniteScalar(PolynomialHomomorphism<FiniteField, FiniteField, PrimeSubfieldEmbedding>),
    FiniteEmbeddingScalar(PolynomialHomomorphism<FiniteField, FiniteField, FiniteFieldEmbedding>),
}

impl PolynomialHomomorphismInner {
    fn from_json(value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value).map_err(value_error)?;
        let source_type = payload
            .get("source")
            .and_then(|source| source.get("coefficient_ring"))
            .and_then(|source| source.get("type"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| value_error("polynomial homomorphism source has no coefficient type"))?;
        let target = payload
            .get("target")
            .ok_or_else(|| value_error("polynomial homomorphism has no target context"))?;
        let target_type = target
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| value_error("polynomial homomorphism target has no type identity"))?;
        let coefficient_map = payload
            .get("coefficient_map")
            .ok_or_else(|| value_error("polynomial homomorphism has no coefficient map"))?;

        if target_type == PolynomialRing::<IntegerRing>::TYPE_ID {
            let target_coefficient_type = target
                .get("coefficient_ring")
                .and_then(|context| context.get("type"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    value_error("polynomial homomorphism target has no coefficient type")
                })?;
            let inner_map_type = coefficient_map
                .get("inner")
                .and_then(|map| map.get("type"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    value_error("polynomial-target homomorphism has no inner coefficient map")
                })?;
            let identity_type = IdentityCoefficientMap::<IntegerRing>::TYPE_ID;
            return match (source_type, target_coefficient_type, inner_map_type) {
                (IntegerRing::TYPE_ID, IntegerRing::TYPE_ID, kind) if kind == identity_type => {
                    Ok(Self::IntegerIdentity(
                        IdentityPolynomialHomomorphism::<IntegerRing>::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (RationalField::TYPE_ID, RationalField::TYPE_ID, kind)
                    if kind == identity_type =>
                {
                    Ok(Self::RationalIdentity(
                        IdentityPolynomialHomomorphism::<RationalField>::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (ResidueRing::TYPE_ID, ResidueRing::TYPE_ID, kind) if kind == identity_type => {
                    Ok(Self::ResidueIdentity(
                        IdentityPolynomialHomomorphism::<ResidueRing>::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (FiniteField::TYPE_ID, FiniteField::TYPE_ID, kind) if kind == identity_type => {
                    Ok(Self::FiniteIdentity(
                        IdentityPolynomialHomomorphism::<FiniteField>::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (IntegerRing::TYPE_ID, RationalField::TYPE_ID, IntegerRationalEmbedding::TYPE_ID) => {
                    Ok(Self::IntegerRationalPolynomial(
                        IntegerRationalPolynomialHomomorphism::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (IntegerRing::TYPE_ID, ResidueRing::TYPE_ID, IntegerReduction::TYPE_ID) => {
                    Ok(Self::IntegerResiduePolynomial(
                        IntegerResiduePolynomialHomomorphism::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (IntegerRing::TYPE_ID, FiniteField::TYPE_ID, IntegerFiniteFieldMap::TYPE_ID) => {
                    Ok(Self::IntegerFinitePolynomial(
                        IntegerFinitePolynomialHomomorphism::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (FiniteField::TYPE_ID, FiniteField::TYPE_ID, PrimeSubfieldEmbedding::TYPE_ID) => {
                    Ok(Self::PrimeFinitePolynomial(
                        PrimeFinitePolynomialHomomorphism::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                (FiniteField::TYPE_ID, FiniteField::TYPE_ID, FiniteFieldEmbedding::TYPE_ID) => {
                    Ok(Self::FiniteEmbeddingPolynomial(
                        FiniteEmbeddingPolynomialHomomorphism::from_json(value)
                            .map_err(value_error)?,
                    ))
                }
                _ => Err(value_error(format!(
                    "unsupported polynomial homomorphism context triple ({source_type}, {target_coefficient_type}, {inner_map_type})"
                ))),
            };
        }

        let map_type = coefficient_map
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| value_error("polynomial homomorphism coefficient map has no type"))?;
        match (source_type, target_type, map_type) {
            (IntegerRing::TYPE_ID, RationalField::TYPE_ID, IntegerRationalEmbedding::TYPE_ID) => {
                Ok(Self::IntegerRationalScalar(
                    PolynomialHomomorphism::from_json(value).map_err(value_error)?,
                ))
            }
            (IntegerRing::TYPE_ID, ResidueRing::TYPE_ID, IntegerReduction::TYPE_ID) => {
                Ok(Self::IntegerResidueScalar(
                    PolynomialHomomorphism::from_json(value).map_err(value_error)?,
                ))
            }
            (IntegerRing::TYPE_ID, FiniteField::TYPE_ID, IntegerFiniteFieldMap::TYPE_ID) => {
                Ok(Self::IntegerFiniteScalar(
                    PolynomialHomomorphism::from_json(value).map_err(value_error)?,
                ))
            }
            (FiniteField::TYPE_ID, FiniteField::TYPE_ID, PrimeSubfieldEmbedding::TYPE_ID) => {
                Ok(Self::PrimeFiniteScalar(
                    PolynomialHomomorphism::from_json(value).map_err(value_error)?,
                ))
            }
            (FiniteField::TYPE_ID, FiniteField::TYPE_ID, FiniteFieldEmbedding::TYPE_ID) => {
                Ok(Self::FiniteEmbeddingScalar(
                    PolynomialHomomorphism::from_json(value).map_err(value_error)?,
                ))
            }
            _ => Err(value_error(format!(
                "unsupported scalar-target polynomial homomorphism context triple ({source_type}, {target_type}, {map_type})"
            ))),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::IntegerIdentity(value) => value.to_json(),
            Self::RationalIdentity(value) => value.to_json(),
            Self::ResidueIdentity(value) => value.to_json(),
            Self::FiniteIdentity(value) => value.to_json(),
            Self::IntegerRationalPolynomial(value) => value.to_json(),
            Self::IntegerResiduePolynomial(value) => value.to_json(),
            Self::IntegerFinitePolynomial(value) => value.to_json(),
            Self::PrimeFinitePolynomial(value) => value.to_json(),
            Self::FiniteEmbeddingPolynomial(value) => value.to_json(),
            Self::IntegerRationalScalar(value) => value.to_json(),
            Self::IntegerResidueScalar(value) => value.to_json(),
            Self::IntegerFiniteScalar(value) => value.to_json(),
            Self::PrimeFiniteScalar(value) => value.to_json(),
            Self::FiniteEmbeddingScalar(value) => value.to_json(),
        }
    }

    fn canonical_hash(&self) -> isize {
        match self {
            Self::IntegerIdentity(value) => canonical_hash(value),
            Self::RationalIdentity(value) => canonical_hash(value),
            Self::ResidueIdentity(value) => canonical_hash(value),
            Self::FiniteIdentity(value) => canonical_hash(value),
            Self::IntegerRationalPolynomial(value) => canonical_hash(value),
            Self::IntegerResiduePolynomial(value) => canonical_hash(value),
            Self::IntegerFinitePolynomial(value) => canonical_hash(value),
            Self::PrimeFinitePolynomial(value) => canonical_hash(value),
            Self::FiniteEmbeddingPolynomial(value) => canonical_hash(value),
            Self::IntegerRationalScalar(value) => canonical_hash(value),
            Self::IntegerResidueScalar(value) => canonical_hash(value),
            Self::IntegerFiniteScalar(value) => canonical_hash(value),
            Self::PrimeFiniteScalar(value) => canonical_hash(value),
            Self::FiniteEmbeddingScalar(value) => canonical_hash(value),
        }
    }
}

impl Serialize for PolynomialHomomorphismInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::IntegerIdentity(value) => value.serialize(serializer),
            Self::RationalIdentity(value) => value.serialize(serializer),
            Self::ResidueIdentity(value) => value.serialize(serializer),
            Self::FiniteIdentity(value) => value.serialize(serializer),
            Self::IntegerRationalPolynomial(value) => value.serialize(serializer),
            Self::IntegerResiduePolynomial(value) => value.serialize(serializer),
            Self::IntegerFinitePolynomial(value) => value.serialize(serializer),
            Self::PrimeFinitePolynomial(value) => value.serialize(serializer),
            Self::FiniteEmbeddingPolynomial(value) => value.serialize(serializer),
            Self::IntegerRationalScalar(value) => value.serialize(serializer),
            Self::IntegerResidueScalar(value) => value.serialize(serializer),
            Self::IntegerFiniteScalar(value) => value.serialize(serializer),
            Self::PrimeFiniteScalar(value) => value.serialize(serializer),
            Self::FiniteEmbeddingScalar(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PolynomialHomomorphismInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let payload = serde_json::Value::deserialize(deserializer)?;
        let json = serde_json::to_string(&payload).map_err(serde::de::Error::custom)?;
        Self::from_json(&json).map_err(serde::de::Error::custom)
    }
}

/// A checked polynomial-ring homomorphism with explicit endpoints and images.
#[pyclass(
    module = "noether.commalg",
    name = "PolynomialHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyPolynomialHomomorphism {
    pub(super) inner: PolynomialHomomorphismInner,
}

#[pymethods]
impl PyPolynomialHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        PolynomialHomomorphism::<IntegerRing, RationalField, IntegerRationalEmbedding>::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyPolynomialRing {
        let inner = match &self.inner {
            PolynomialHomomorphismInner::IntegerIdentity(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::RationalIdentity(value) => {
                PolynomialRingInner::Rational(value.source().clone())
            }
            PolynomialHomomorphismInner::ResidueIdentity(value) => {
                PolynomialRingInner::Residue(value.source().clone())
            }
            PolynomialHomomorphismInner::FiniteIdentity(value) => {
                PolynomialRingInner::FiniteField(value.source().clone())
            }
            PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => {
                PolynomialRingInner::FiniteField(value.source().clone())
            }
            PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => {
                PolynomialRingInner::FiniteField(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerRationalScalar(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerResidueScalar(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::IntegerFiniteScalar(value) => {
                PolynomialRingInner::Integer(value.source().clone())
            }
            PolynomialHomomorphismInner::PrimeFiniteScalar(value) => {
                PolynomialRingInner::FiniteField(value.source().clone())
            }
            PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => {
                PolynomialRingInner::FiniteField(value.source().clone())
            }
        };
        PyPolynomialRing { inner }
    }

    #[getter]
    fn target(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PolynomialHomomorphismInner::IntegerIdentity(value) => {
                Ok(Py::new(py, PyPolynomialRing::from_integer(value.target().clone()))?.into_any())
            }
            PolynomialHomomorphismInner::RationalIdentity(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_rational(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_rational(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::ResidueIdentity(value) => Ok(Py::new(
                py,
                PyPolynomialRing {
                    inner: PolynomialRingInner::Residue(value.target().clone()),
                },
            )?
            .into_any()),
            PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => Ok(Py::new(
                py,
                PyPolynomialRing {
                    inner: PolynomialRingInner::Residue(value.target().clone()),
                },
            )?
            .into_any()),
            PolynomialHomomorphismInner::FiniteIdentity(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(value.target().clone()),
            )?
            .into_any()),
            PolynomialHomomorphismInner::IntegerRationalScalar(_value) => {
                Ok(Py::new(py, PyRationalField)?.into_any())
            }
            PolynomialHomomorphismInner::IntegerResidueScalar(value) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: value.target().clone(),
                },
            )?
            .into_any()),
            PolynomialHomomorphismInner::IntegerFiniteScalar(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.target().clone(),
                },
            )?
            .into_any()),
            PolynomialHomomorphismInner::PrimeFiniteScalar(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.target().clone(),
                },
            )?
            .into_any()),
            PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.target().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn coefficient_map(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            PolynomialHomomorphismInner::IntegerIdentity(_)
            | PolynomialHomomorphismInner::RationalIdentity(_)
            | PolynomialHomomorphismInner::ResidueIdentity(_)
            | PolynomialHomomorphismInner::FiniteIdentity(_) => Ok(None),
            PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerRationalEmbedding {
                        inner: *value.coefficient_map().inner(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerReduction {
                        inner: value.coefficient_map().inner().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerFiniteFieldMap {
                        inner: value.coefficient_map().inner().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => Ok(Some(
                Py::new(
                    py,
                    PyPrimeSubfieldEmbedding {
                        inner: value.coefficient_map().inner().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => Ok(Some(
                Py::new(
                    py,
                    PyFiniteFieldEmbedding {
                        inner: value.coefficient_map().inner().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::IntegerRationalScalar(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerRationalEmbedding {
                        inner: *value.coefficient_map(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::IntegerResidueScalar(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerReduction {
                        inner: value.coefficient_map().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::IntegerFiniteScalar(value) => Ok(Some(
                Py::new(
                    py,
                    PyIntegerFiniteFieldMap {
                        inner: value.coefficient_map().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::PrimeFiniteScalar(value) => Ok(Some(
                Py::new(
                    py,
                    PyPrimeSubfieldEmbedding {
                        inner: value.coefficient_map().clone(),
                    },
                )?
                .into_any(),
            )),
            PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => Ok(Some(
                Py::new(
                    py,
                    PyFiniteFieldEmbedding {
                        inner: value.coefficient_map().clone(),
                    },
                )?
                .into_any(),
            )),
        }
    }

    #[getter]
    fn images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyList::empty(py);
        match &self.inner {
            PolynomialHomomorphismInner::IntegerIdentity(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_integer(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::RationalIdentity(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_rational(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_rational(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::ResidueIdentity(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_residue(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_residue(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::FiniteIdentity(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_finite_field(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_finite_field(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_finite_field(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => {
                for image in value.images() {
                    output.append(PyPolynomial::from_finite_field(image.clone()))?;
                }
            }
            PolynomialHomomorphismInner::IntegerRationalScalar(value) => {
                for image in value.images() {
                    output.append(rational_to_python(py, image)?)?;
                }
            }
            PolynomialHomomorphismInner::IntegerResidueScalar(value) => {
                for image in value.images() {
                    output.append(PyResidueClass {
                        inner: image.clone(),
                    })?;
                }
            }
            PolynomialHomomorphismInner::IntegerFiniteScalar(value) => {
                for image in value.images() {
                    output.append(PyFiniteFieldElement {
                        inner: image.clone(),
                    })?;
                }
            }
            PolynomialHomomorphismInner::PrimeFiniteScalar(value) => {
                for image in value.images() {
                    output.append(PyFiniteFieldElement {
                        inner: image.clone(),
                    })?;
                }
            }
            PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => {
                for image in value.images() {
                    output.append(PyFiniteFieldElement {
                        inner: image.clone(),
                    })?;
                }
            }
        }
        Ok(output.unbind().into_any())
    }

    fn apply(&self, py: Python<'_>, value: &PyPolynomial) -> PyResult<Py<PyAny>> {
        match (&self.inner, &value.inner) {
            (
                PolynomialHomomorphismInner::IntegerIdentity(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_integer(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::RationalIdentity(map),
                PolynomialInner::Rational(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_rational(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::IntegerRationalPolynomial(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_rational(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::ResidueIdentity(map),
                PolynomialInner::Residue(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_residue(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::IntegerResiduePolynomial(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_residue(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::FiniteIdentity(map),
                PolynomialInner::FiniteField(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_finite_field(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::IntegerFinitePolynomial(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_finite_field(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::PrimeFinitePolynomial(map),
                PolynomialInner::FiniteField(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_finite_field(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(map),
                PolynomialInner::FiniteField(value),
            ) => Ok(Py::new(
                py,
                PyPolynomial::from_finite_field(map.apply(value).map_err(value_error)?),
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::IntegerRationalScalar(map),
                PolynomialInner::Integer(value),
            ) => rational_to_python(py, &map.apply(value).map_err(value_error)?),
            (
                PolynomialHomomorphismInner::IntegerResidueScalar(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: map.apply(value).map_err(value_error)?,
                },
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::IntegerFiniteScalar(map),
                PolynomialInner::Integer(value),
            ) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: map.apply(value).map_err(value_error)?,
                },
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::PrimeFiniteScalar(map),
                PolynomialInner::FiniteField(value),
            ) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: map.apply(value).map_err(value_error)?,
                },
            )?
            .into_any()),
            (
                PolynomialHomomorphismInner::FiniteEmbeddingScalar(map),
                PolynomialInner::FiniteField(value),
            ) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: map.apply(value).map_err(value_error)?,
                },
            )?
            .into_any()),
            _ => Err(value_error(
                "polynomial does not belong to this homomorphism's source context",
            )),
        }
    }

    /// Extend an ideal along this map and recanonicalize it in the target
    /// polynomial ring.
    fn extend_ideal(&self, ideal: &PyIdeal) -> PyResult<PyIdeal> {
        match (&self.inner, &ideal.inner) {
            (
                PolynomialHomomorphismInner::RationalIdentity(map),
                IdealInner::RationalPolynomial(ideal),
            ) => Ok(PyIdeal::from_rational_polynomial(
                map.extend_ideal(ideal).map_err(base_change_error)?,
            )),
            (
                PolynomialHomomorphismInner::FiniteIdentity(map),
                IdealInner::FiniteFieldPolynomial(ideal),
            ) => Ok(PyIdeal::from_finite_field_polynomial(
                map.extend_ideal(ideal).map_err(base_change_error)?,
            )),
            (
                PolynomialHomomorphismInner::PrimeFinitePolynomial(map),
                IdealInner::FiniteFieldPolynomial(ideal),
            ) => Ok(PyIdeal::from_finite_field_polynomial(
                map.extend_ideal(ideal).map_err(base_change_error)?,
            )),
            (
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(map),
                IdealInner::FiniteFieldPolynomial(ideal),
            ) => Ok(PyIdeal::from_finite_field_polynomial(
                map.extend_ideal(ideal).map_err(base_change_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "extend_ideal requires a polynomial-target map and an ideal in its exact source ring",
            )),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PolynomialHomomorphismInner::IntegerIdentity(value) => value.verify(),
            PolynomialHomomorphismInner::RationalIdentity(value) => value.verify(),
            PolynomialHomomorphismInner::ResidueIdentity(value) => value.verify(),
            PolynomialHomomorphismInner::FiniteIdentity(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => value.verify(),
            PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => value.verify(),
            PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerRationalScalar(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerResidueScalar(value) => value.verify(),
            PolynomialHomomorphismInner::IntegerFiniteScalar(value) => value.verify(),
            PolynomialHomomorphismInner::PrimeFiniteScalar(value) => value.verify(),
            PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => value.verify(),
        }
        .map_err(value_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PolynomialHomomorphismInner::from_json(value)?,
        })
    }

    fn __call__(&self, py: Python<'_>, value: &PyPolynomial) -> PyResult<Py<PyAny>> {
        self.apply(py, value)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        self.inner.canonical_hash()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PolynomialHomomorphism(source={:?}, image_count={})",
            self.source().inner.variables(),
            match &self.inner {
                PolynomialHomomorphismInner::IntegerIdentity(value) => value.images().len(),
                PolynomialHomomorphismInner::RationalIdentity(value) => value.images().len(),
                PolynomialHomomorphismInner::ResidueIdentity(value) => value.images().len(),
                PolynomialHomomorphismInner::FiniteIdentity(value) => value.images().len(),
                PolynomialHomomorphismInner::IntegerRationalPolynomial(value) => {
                    value.images().len()
                }
                PolynomialHomomorphismInner::IntegerResiduePolynomial(value) => {
                    value.images().len()
                }
                PolynomialHomomorphismInner::IntegerFinitePolynomial(value) => {
                    value.images().len()
                }
                PolynomialHomomorphismInner::PrimeFinitePolynomial(value) => value.images().len(),
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(value) => {
                    value.images().len()
                }
                PolynomialHomomorphismInner::IntegerRationalScalar(value) => value.images().len(),
                PolynomialHomomorphismInner::IntegerResidueScalar(value) => value.images().len(),
                PolynomialHomomorphismInner::IntegerFiniteScalar(value) => value.images().len(),
                PolynomialHomomorphismInner::PrimeFiniteScalar(value) => value.images().len(),
                PolynomialHomomorphismInner::FiniteEmbeddingScalar(value) => value.images().len(),
            }
        ))
    }
}

fn collect_polynomial_homomorphism_images<R: noether::CommutativeRing>(
    images: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&PolynomialInner) -> Option<Polynomial<R>>,
) -> PyResult<Vec<Polynomial<R>>> {
    images
        .try_iter()?
        .enumerate()
        .map(|(index, image)| {
            let image = image?;
            let image = image.extract::<PyRef<'_, PyPolynomial>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "image {index} must be a polynomial in the declared target ring"
                ))
            })?;
            extract(&image.inner).ok_or_else(|| {
                value_error(format!(
                    "image {index} has the wrong polynomial coefficient context"
                ))
            })
        })
        .collect()
}

fn collect_scalar_homomorphism_images<E>(
    images: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<E>> {
    images.try_iter()?.map(|image| extract(&image?)).collect()
}

fn build_polynomial_homomorphism(
    py: Python<'_>,
    source: &PyPolynomialRing,
    target: &Bound<'_, PyAny>,
    images: &Bound<'_, PyAny>,
    coefficient_map: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyPolynomialHomomorphism> {
    if let Ok(target) = target.extract::<PyRef<'_, PyPolynomialRing>>() {
        let inner = match (&source.inner, &target.inner, coefficient_map) {
            (PolynomialRingInner::Integer(source), PolynomialRingInner::Integer(target), None) => {
                PolynomialHomomorphismInner::IntegerIdentity(
                    source
                        .homomorphism(
                            target,
                            IdentityCoefficientMap::new(Z),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::Integer(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::Rational(source),
                PolynomialRingInner::Rational(target),
                None,
            ) => PolynomialHomomorphismInner::RationalIdentity(
                source
                    .homomorphism(
                        target,
                        IdentityCoefficientMap::new(Q),
                        collect_polynomial_homomorphism_images(images, |image| match image {
                            PolynomialInner::Rational(value) => Some(value.clone()),
                            _ => None,
                        })?,
                    )
                    .map_err(value_error)?,
            ),
            (PolynomialRingInner::Residue(source), PolynomialRingInner::Residue(target), None) => {
                PolynomialHomomorphismInner::ResidueIdentity(
                    source
                        .homomorphism(
                            target,
                            IdentityCoefficientMap::new(source.coefficient_ring().clone()),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::Residue(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::FiniteField(source),
                PolynomialRingInner::FiniteField(target),
                None,
            ) => PolynomialHomomorphismInner::FiniteIdentity(
                source
                    .homomorphism(
                        target,
                        IdentityCoefficientMap::new(source.coefficient_ring().clone()),
                        collect_polynomial_homomorphism_images(images, |image| match image {
                            PolynomialInner::FiniteField(value) => Some(value.clone()),
                            _ => None,
                        })?,
                    )
                    .map_err(value_error)?,
            ),
            (
                PolynomialRingInner::Integer(source),
                PolynomialRingInner::Rational(target),
                Some(map),
            ) if map.is_instance_of::<PyIntegerRationalEmbedding>() => {
                let map = map.extract::<PyRef<'_, PyIntegerRationalEmbedding>>()?;
                PolynomialHomomorphismInner::IntegerRationalPolynomial(
                    source
                        .homomorphism(
                            target,
                            map.inner,
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::Rational(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::Integer(source),
                PolynomialRingInner::Residue(target),
                Some(map),
            ) if map.is_instance_of::<PyIntegerReduction>() => {
                let map = map.extract::<PyRef<'_, PyIntegerReduction>>()?;
                PolynomialHomomorphismInner::IntegerResiduePolynomial(
                    source
                        .homomorphism(
                            target,
                            map.inner.clone(),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::Residue(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::Integer(source),
                PolynomialRingInner::FiniteField(target),
                Some(map),
            ) if map.is_instance_of::<PyIntegerFiniteFieldMap>() => {
                let map = map.extract::<PyRef<'_, PyIntegerFiniteFieldMap>>()?;
                PolynomialHomomorphismInner::IntegerFinitePolynomial(
                    source
                        .homomorphism(
                            target,
                            map.inner.clone(),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::FiniteField(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::FiniteField(source),
                PolynomialRingInner::FiniteField(target),
                Some(map),
            ) if map.is_instance_of::<PyPrimeSubfieldEmbedding>() => {
                let map = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>()?;
                PolynomialHomomorphismInner::PrimeFinitePolynomial(
                    source
                        .homomorphism(
                            target,
                            map.inner.clone(),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::FiniteField(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            (
                PolynomialRingInner::FiniteField(source),
                PolynomialRingInner::FiniteField(target),
                Some(map),
            ) if map.is_instance_of::<PyFiniteFieldEmbedding>() => {
                let map = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>()?;
                PolynomialHomomorphismInner::FiniteEmbeddingPolynomial(
                    source
                        .homomorphism(
                            target,
                            map.inner.clone(),
                            collect_polynomial_homomorphism_images(images, |image| match image {
                                PolynomialInner::FiniteField(value) => Some(value.clone()),
                                _ => None,
                            })?,
                        )
                        .map_err(value_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "unsupported polynomial homomorphism endpoints or coefficient map",
                ));
            }
        };
        return Ok(PyPolynomialHomomorphism { inner });
    }

    let inner = if target.is_instance_of::<PyRationalField>() {
        let PolynomialRingInner::Integer(source) = &source.inner else {
            return Err(PyTypeError::new_err(
                "a Q-valued projected homomorphism currently requires a source over Z",
            ));
        };
        let map = coefficient_map
            .ok_or_else(|| PyTypeError::new_err("coefficient_map is required"))?
            .extract::<PyRef<'_, PyIntegerRationalEmbedding>>()?;
        PolynomialHomomorphismInner::IntegerRationalScalar(
            PolynomialHomomorphism::new(
                source.clone(),
                Q,
                map.inner,
                collect_scalar_homomorphism_images(images, |image| extract_rational(py, image))?,
            )
            .map_err(value_error)?,
        )
    } else if let Ok(target) = target.extract::<PyRef<'_, PyResidueRing>>() {
        let PolynomialRingInner::Integer(source) = &source.inner else {
            return Err(PyTypeError::new_err(
                "a residue-valued projected homomorphism currently requires a source over Z",
            ));
        };
        let map = coefficient_map
            .ok_or_else(|| PyTypeError::new_err("coefficient_map is required"))?
            .extract::<PyRef<'_, PyIntegerReduction>>()?;
        PolynomialHomomorphismInner::IntegerResidueScalar(
            PolynomialHomomorphism::new(
                source.clone(),
                target.inner.clone(),
                map.inner.clone(),
                collect_scalar_homomorphism_images(images, |image| {
                    extract_residue(&target.inner, image)
                })?,
            )
            .map_err(value_error)?,
        )
    } else if let Ok(target) = target.extract::<PyRef<'_, PyFiniteField>>() {
        let map =
            coefficient_map.ok_or_else(|| PyTypeError::new_err("coefficient_map is required"))?;
        match &source.inner {
            PolynomialRingInner::Integer(source)
                if map.is_instance_of::<PyIntegerFiniteFieldMap>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerFiniteFieldMap>>()?;
                PolynomialHomomorphismInner::IntegerFiniteScalar(
                    PolynomialHomomorphism::new(
                        source.clone(),
                        target.inner.clone(),
                        map.inner.clone(),
                        collect_scalar_homomorphism_images(images, |image| {
                            extract_finite_field(&target.inner, image)
                        })?,
                    )
                    .map_err(value_error)?,
                )
            }
            PolynomialRingInner::FiniteField(source)
                if map.is_instance_of::<PyPrimeSubfieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>()?;
                PolynomialHomomorphismInner::PrimeFiniteScalar(
                    PolynomialHomomorphism::new(
                        source.clone(),
                        target.inner.clone(),
                        map.inner.clone(),
                        collect_scalar_homomorphism_images(images, |image| {
                            extract_finite_field(&target.inner, image)
                        })?,
                    )
                    .map_err(value_error)?,
                )
            }
            PolynomialRingInner::FiniteField(source)
                if map.is_instance_of::<PyFiniteFieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>()?;
                PolynomialHomomorphismInner::FiniteEmbeddingScalar(
                    PolynomialHomomorphism::new(
                        source.clone(),
                        target.inner.clone(),
                        map.inner.clone(),
                        collect_scalar_homomorphism_images(images, |image| {
                            extract_finite_field(&target.inner, image)
                        })?,
                    )
                    .map_err(value_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "unsupported finite-field homomorphism source or coefficient map",
                ));
            }
        }
    } else {
        return Err(PyTypeError::new_err(
            "homomorphism target must be a supported polynomial or coefficient ring",
        ));
    };
    Ok(PyPolynomialHomomorphism { inner })
}

/// A sparse polynomial carrying its complete coefficient and variable context.
#[pyclass(
    module = "noether.commalg",
    name = "Polynomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPolynomial {
    pub(super) inner: PolynomialInner,
}

impl PyPolynomial {
    pub(crate) fn from_integer(value: Polynomial<IntegerRing>) -> Self {
        Self {
            inner: PolynomialInner::Integer(value),
        }
    }

    pub(crate) fn as_integer(&self) -> Option<&Polynomial<IntegerRing>> {
        match &self.inner {
            PolynomialInner::Integer(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_rational(value: Polynomial<RationalField>) -> Self {
        Self {
            inner: PolynomialInner::Rational(value),
        }
    }

    pub(crate) fn as_rational(&self) -> Option<&Polynomial<RationalField>> {
        match &self.inner {
            PolynomialInner::Rational(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_residue(value: Polynomial<ResidueRing>) -> Self {
        Self {
            inner: PolynomialInner::Residue(value),
        }
    }

    pub(crate) fn as_residue(&self) -> Option<&Polynomial<ResidueRing>> {
        match &self.inner {
            PolynomialInner::Residue(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_padic(&self) -> Option<&Polynomial<PadicRing>> {
        match &self.inner {
            PolynomialInner::Padic(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_padic_field(&self) -> Option<&Polynomial<PadicField>> {
        match &self.inner {
            PolynomialInner::PadicField(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_qadic(&self) -> Option<&Polynomial<QadicRing>> {
        match &self.inner {
            PolynomialInner::Qadic(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_qadic_field(&self) -> Option<&Polynomial<QadicField>> {
        match &self.inner {
            PolynomialInner::QadicField(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_padic_local_field(&self) -> Option<&Polynomial<PadicLocalField>> {
        match &self.inner {
            PolynomialInner::PadicLocalField(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_finite_field(value: Polynomial<FiniteField>) -> Self {
        Self {
            inner: PolynomialInner::FiniteField(value),
        }
    }

    pub(crate) fn as_finite_field(&self) -> Option<&Polynomial<FiniteField>> {
        match &self.inner {
            PolynomialInner::FiniteField(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_padic(value: Polynomial<PadicRing>) -> Self {
        Self {
            inner: PolynomialInner::Padic(value),
        }
    }

    pub(crate) fn from_padic_field(value: Polynomial<PadicField>) -> Self {
        Self {
            inner: PolynomialInner::PadicField(value),
        }
    }

    pub(crate) fn from_qadic(value: Polynomial<QadicRing>) -> Self {
        Self {
            inner: PolynomialInner::Qadic(value),
        }
    }

    pub(crate) fn from_qadic_field(value: Polynomial<QadicField>) -> Self {
        Self {
            inner: PolynomialInner::QadicField(value),
        }
    }

    pub(crate) fn from_padic_local_field(value: Polynomial<PadicLocalField>) -> Self {
        Self {
            inner: PolynomialInner::PadicLocalField(value),
        }
    }

    pub(crate) fn from_number_field(value: Polynomial<NumberField>) -> Self {
        Self {
            inner: PolynomialInner::NumberField(value),
        }
    }

    pub(crate) fn as_number_field(&self) -> Option<&Polynomial<NumberField>> {
        match &self.inner {
            PolynomialInner::NumberField(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_rational_function(
        value: Polynomial<RationalFunctionField<RationalField>>,
    ) -> Self {
        Self {
            inner: PolynomialInner::RationalFunction(value),
        }
    }

    pub(crate) fn as_rational_function(
        &self,
    ) -> Option<&Polynomial<RationalFunctionField<RationalField>>> {
        match &self.inner {
            PolynomialInner::RationalFunction(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn from_finite_rational_function(
        value: Polynomial<RationalFunctionField<FiniteField>>,
    ) -> Self {
        Self {
            inner: PolynomialInner::FiniteRationalFunction(value),
        }
    }

    pub(crate) fn as_finite_rational_function(
        &self,
    ) -> Option<&Polynomial<RationalFunctionField<FiniteField>>> {
        match &self.inner {
            PolynomialInner::FiniteRationalFunction(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn term_count(&self) -> usize {
        self.inner.nonzero_term_count()
    }

    fn ring_value(&self) -> PyPolynomialRing {
        PyPolynomialRing {
            inner: self.inner.ring(),
        }
    }

    fn coerce_operand(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(polynomial) = other.extract::<PyRef<'_, Self>>() {
            return Ok(polynomial.clone());
        }
        self.ring_value().constant_from_python(py, other)
    }

    fn binary(&self, other: &Self, operation: &str) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (PolynomialInner::Integer(left), PolynomialInner::Integer(right)) => {
                PolynomialInner::Integer(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::Rational(left), PolynomialInner::Rational(right)) => {
                PolynomialInner::Rational(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::Residue(left), PolynomialInner::Residue(right)) => {
                PolynomialInner::Residue(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::FiniteField(left), PolynomialInner::FiniteField(right)) => {
                PolynomialInner::FiniteField(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::Padic(left), PolynomialInner::Padic(right)) => {
                PolynomialInner::Padic(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::PadicField(left), PolynomialInner::PadicField(right)) => {
                PolynomialInner::PadicField(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::Qadic(left), PolynomialInner::Qadic(right)) => {
                PolynomialInner::Qadic(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::QadicField(left), PolynomialInner::QadicField(right)) => {
                PolynomialInner::QadicField(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::PadicLocalField(left), PolynomialInner::PadicLocalField(right)) => {
                PolynomialInner::PadicLocalField(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::NumberField(left), PolynomialInner::NumberField(right)) => {
                PolynomialInner::NumberField(binary_polynomial(left, right, operation)?)
            }
            (PolynomialInner::RationalFunction(left), PolynomialInner::RationalFunction(right)) => {
                PolynomialInner::RationalFunction(binary_polynomial(left, right, operation)?)
            }
            (
                PolynomialInner::FiniteRationalFunction(left),
                PolynomialInner::FiniteRationalFunction(right),
            ) => {
                PolynomialInner::FiniteRationalFunction(binary_polynomial(left, right, operation)?)
            }
            _ => {
                return Err(value_error(
                    "polynomials have different coefficient contexts",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn field_div_rem(&self, other: &Self) -> PyResult<(Self, Self)> {
        let (quotient, remainder) = match (&self.inner, &other.inner) {
            (PolynomialInner::Rational(left), PolynomialInner::Rational(right)) => {
                let (quotient, remainder) = left.div_rem(right).map_err(value_error)?;
                (
                    PolynomialInner::Rational(quotient),
                    PolynomialInner::Rational(remainder),
                )
            }
            (PolynomialInner::FiniteField(left), PolynomialInner::FiniteField(right)) => {
                let (quotient, remainder) = left.div_rem(right).map_err(value_error)?;
                (
                    PolynomialInner::FiniteField(quotient),
                    PolynomialInner::FiniteField(remainder),
                )
            }
            (PolynomialInner::NumberField(left), PolynomialInner::NumberField(right)) => {
                let (quotient, remainder) = left.div_rem(right).map_err(value_error)?;
                (
                    PolynomialInner::NumberField(quotient),
                    PolynomialInner::NumberField(remainder),
                )
            }
            (PolynomialInner::RationalFunction(left), PolynomialInner::RationalFunction(right)) => {
                let (quotient, remainder) = left.div_rem(right).map_err(value_error)?;
                (
                    PolynomialInner::RationalFunction(quotient),
                    PolynomialInner::RationalFunction(remainder),
                )
            }
            (
                PolynomialInner::FiniteRationalFunction(left),
                PolynomialInner::FiniteRationalFunction(right),
            ) => {
                let (quotient, remainder) = left.div_rem(right).map_err(value_error)?;
                (
                    PolynomialInner::FiniteRationalFunction(quotient),
                    PolynomialInner::FiniteRationalFunction(remainder),
                )
            }
            (PolynomialInner::Integer(_), PolynomialInner::Integer(_))
            | (PolynomialInner::Residue(_), PolynomialInner::Residue(_))
            | (PolynomialInner::Padic(_), PolynomialInner::Padic(_))
            | (PolynomialInner::PadicField(_), PolynomialInner::PadicField(_))
            | (PolynomialInner::Qadic(_), PolynomialInner::Qadic(_))
            | (PolynomialInner::QadicField(_), PolynomialInner::QadicField(_)) => {
                return Err(unsupported_polynomial_operation(
                    "div_rem",
                    self.inner.context_type_id(),
                    FIELD_POLYNOMIAL_CONTEXTS,
                ));
            }
            _ => {
                return Err(value_error(
                    "polynomials have different coefficient contexts",
                ));
            }
        };
        Ok((Self { inner: quotient }, Self { inner: remainder }))
    }

    fn exact_division(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (PolynomialInner::Integer(left), PolynomialInner::Integer(right)) => {
                PolynomialInner::Integer(left.exact_div(right).map_err(value_error)?)
            }
            (PolynomialInner::Rational(left), PolynomialInner::Rational(right)) => {
                PolynomialInner::Rational(left.exact_div(right).map_err(value_error)?)
            }
            (PolynomialInner::FiniteField(left), PolynomialInner::FiniteField(right)) => {
                PolynomialInner::FiniteField(left.exact_div(right).map_err(value_error)?)
            }
            (PolynomialInner::NumberField(left), PolynomialInner::NumberField(right)) => {
                PolynomialInner::NumberField(left.exact_div(right).map_err(value_error)?)
            }
            (PolynomialInner::RationalFunction(left), PolynomialInner::RationalFunction(right)) => {
                PolynomialInner::RationalFunction(left.exact_div(right).map_err(value_error)?)
            }
            (
                PolynomialInner::FiniteRationalFunction(left),
                PolynomialInner::FiniteRationalFunction(right),
            ) => {
                PolynomialInner::FiniteRationalFunction(left.exact_div(right).map_err(value_error)?)
            }
            (PolynomialInner::Residue(_), PolynomialInner::Residue(_))
            | (PolynomialInner::Padic(_), PolynomialInner::Padic(_))
            | (PolynomialInner::PadicField(_), PolynomialInner::PadicField(_))
            | (PolynomialInner::Qadic(_), PolynomialInner::Qadic(_))
            | (PolynomialInner::QadicField(_), PolynomialInner::QadicField(_)) => {
                return Err(unsupported_polynomial_operation(
                    "exact_div",
                    self.inner.context_type_id(),
                    EXACT_GCD_POLYNOMIAL_CONTEXTS,
                ));
            }
            _ => {
                return Err(value_error(
                    "polynomials have different coefficient contexts",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn polynomial_gcd(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (PolynomialInner::Integer(left), PolynomialInner::Integer(right)) => {
                PolynomialInner::Integer(left.gcd(right).map_err(value_error)?)
            }
            (PolynomialInner::Rational(left), PolynomialInner::Rational(right)) => {
                PolynomialInner::Rational(left.gcd(right).map_err(value_error)?)
            }
            (PolynomialInner::FiniteField(left), PolynomialInner::FiniteField(right)) => {
                PolynomialInner::FiniteField(left.gcd(right).map_err(value_error)?)
            }
            (PolynomialInner::NumberField(left), PolynomialInner::NumberField(right)) => {
                PolynomialInner::NumberField(left.gcd(right).map_err(value_error)?)
            }
            (PolynomialInner::RationalFunction(left), PolynomialInner::RationalFunction(right)) => {
                PolynomialInner::RationalFunction(left.gcd(right).map_err(value_error)?)
            }
            (
                PolynomialInner::FiniteRationalFunction(left),
                PolynomialInner::FiniteRationalFunction(right),
            ) => PolynomialInner::FiniteRationalFunction(left.gcd(right).map_err(value_error)?),
            (PolynomialInner::Residue(_), PolynomialInner::Residue(_))
            | (PolynomialInner::Padic(_), PolynomialInner::Padic(_))
            | (PolynomialInner::PadicField(_), PolynomialInner::PadicField(_))
            | (PolynomialInner::Qadic(_), PolynomialInner::Qadic(_))
            | (PolynomialInner::QadicField(_), PolynomialInner::QadicField(_)) => {
                return Err(unsupported_polynomial_operation(
                    "gcd",
                    self.inner.context_type_id(),
                    EXACT_GCD_POLYNOMIAL_CONTEXTS,
                ));
            }
            _ => {
                return Err(value_error(
                    "polynomials have different coefficient contexts",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn images_from_python(&self, images: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let variables = self.inner.ring().variables().to_vec();
        if let Ok(mapping) = images.cast::<PyDict>() {
            return variables
                .iter()
                .map(|variable| -> PyResult<Self> {
                    Ok(mapping
                        .get_item(variable)?
                        .ok_or_else(|| value_error(format!("missing image for {variable:?}")))?
                        .extract::<PyRef<'_, Self>>()?
                        .clone())
                })
                .collect();
        }
        images
            .try_iter()?
            .map(|image| -> PyResult<Self> { Ok(image?.extract::<PyRef<'_, Self>>()?.clone()) })
            .collect()
    }
}

fn binary_polynomial<R: noether::CommutativeRing>(
    left: &Polynomial<R>,
    right: &Polynomial<R>,
    operation: &str,
) -> PyResult<Polynomial<R>> {
    match operation {
        "add" => (left + right).map_err(value_error),
        "sub" => (left - right).map_err(value_error),
        "mul" => (left * right).map_err(value_error),
        _ => Err(PyValueError::new_err("unknown polynomial operation")),
    }
}

#[pymethods]
impl PyPolynomial {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Polynomial::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn ring(&self) -> PyPolynomialRing {
        self.ring_value()
    }

    /// Extend coefficients along one checked standard map while preserving the
    /// complete polynomial presentation.
    fn base_change(&self, map: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Integer(value)
                if map.is_instance_of::<PyIntegerRationalEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyIntegerRationalEmbedding>>()?;
                PolynomialInner::Rational(value.base_change(&map.inner).map_err(base_change_error)?)
            }
            PolynomialInner::Integer(value) if map.is_instance_of::<PyIntegerReduction>() => {
                let map = map.extract::<PyRef<'_, PyIntegerReduction>>()?;
                PolynomialInner::Residue(value.base_change(&map.inner).map_err(base_change_error)?)
            }
            PolynomialInner::Integer(value) if map.is_instance_of::<PyIntegerFiniteFieldMap>() => {
                let map = map.extract::<PyRef<'_, PyIntegerFiniteFieldMap>>()?;
                PolynomialInner::FiniteField(
                    value.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialInner::FiniteField(value)
                if map.is_instance_of::<PyPrimeSubfieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyPrimeSubfieldEmbedding>>()?;
                PolynomialInner::FiniteField(
                    value.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialInner::FiniteField(value)
                if map.is_instance_of::<PyFiniteFieldEmbedding>() =>
            {
                let map = map.extract::<PyRef<'_, PyFiniteFieldEmbedding>>()?;
                PolynomialInner::FiniteField(
                    value.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            PolynomialInner::FiniteRationalFunction(value)
                if map.is_instance_of::<PyRationalFunctionFieldMap>() =>
            {
                let map = map.extract::<PyRef<'_, PyRationalFunctionFieldMap>>()?;
                PolynomialInner::FiniteRationalFunction(
                    value.base_change(&map.inner).map_err(base_change_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "unsupported polynomial base-change context or coefficient map",
                ));
            }
        };
        Ok(Self { inner })
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

    /// Isolate every distinct real root of an exact integer/rational polynomial.
    fn real_roots(&self, py: Python<'_>) -> PyResult<Vec<PyRealRootIsolation>> {
        py.detach(|| {
            let roots = match &self.inner {
                PolynomialInner::Integer(value) => value.real_roots(),
                PolynomialInner::Rational(value) => value.real_roots(),
                _ => {
                    return Err(unsupported_polynomial_operation(
                        "real_roots",
                        self.inner.context_type_id(),
                        &[IntegerRing::TYPE_ID, RationalField::TYPE_ID],
                    ));
                }
            }
            .map_err(value_error)?;
            Ok(roots
                .into_iter()
                .map(|inner| PyRealRootIsolation { inner })
                .collect())
        })
    }

    /// Canonical complete factorization over `Z`, `Q`, or a finite field.
    fn factor(&self, py: Python<'_>) -> PyResult<PyPolynomialFactorization> {
        py.detach(|| {
            let inner = match &self.inner {
                PolynomialInner::Integer(value) => {
                    PolynomialFactorizationInner::Integer(value.factor().map_err(value_error)?)
                }
                PolynomialInner::Rational(value) => {
                    PolynomialFactorizationInner::Rational(value.factor().map_err(value_error)?)
                }
                PolynomialInner::FiniteField(value) => {
                    PolynomialFactorizationInner::FiniteField(value.factor().map_err(value_error)?)
                }
                _ => {
                    return Err(unsupported_polynomial_operation(
                        "factor",
                        self.inner.context_type_id(),
                        &[
                            IntegerRing::TYPE_ID,
                            RationalField::TYPE_ID,
                            FiniteField::TYPE_ID,
                        ],
                    ))
                }
            };
            Ok(PyPolynomialFactorization { inner })
        })
    }

    /// Whether this supported univariate polynomial is an exact power.
    fn is_nth_power(&self, exponent: u64) -> PyResult<bool> {
        match &self.inner {
            PolynomialInner::Integer(value) => value.is_nth_power(exponent),
            PolynomialInner::Rational(value) => value.is_nth_power(exponent),
            PolynomialInner::FiniteField(value) => value.is_nth_power(exponent),
            _ => {
                return Err(unsupported_polynomial_operation(
                    "is_nth_power",
                    self.inner.context_type_id(),
                    &[
                        IntegerRing::TYPE_ID,
                        RationalField::TYPE_ID,
                        FiniteField::TYPE_ID,
                    ],
                ))
            }
        }
        .map_err(value_error)
    }

    /// Extract a deterministic exact root, or return `None` when none exists.
    fn nth_root(&self, exponent: u64) -> PyResult<Option<Self>> {
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => value
                .nth_root(exponent)
                .map(|root| root.map(PolynomialInner::Integer)),
            PolynomialInner::Rational(value) => value
                .nth_root(exponent)
                .map(|root| root.map(PolynomialInner::Rational)),
            PolynomialInner::FiniteField(value) => value
                .nth_root(exponent)
                .map(|root| root.map(PolynomialInner::FiniteField)),
            _ => {
                return Err(unsupported_polynomial_operation(
                    "nth_root",
                    self.inner.context_type_id(),
                    &[
                        IntegerRing::TYPE_ID,
                        RationalField::TYPE_ID,
                        FiniteField::TYPE_ID,
                    ],
                ))
            }
        }
        .map_err(value_error)?;
        Ok(inner.map(|inner| Self { inner }))
    }

    fn terms(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyList::empty(py);
        match &self.inner {
            PolynomialInner::Integer(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        integer_to_python(py, coefficient)?,
                    ))?;
                }
            }
            PolynomialInner::Rational(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        rational_to_python(py, coefficient)?,
                    ))?;
                }
            }
            PolynomialInner::Residue(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyResidueClass {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::FiniteField(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyFiniteFieldElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::Padic(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyPadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::PadicField(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyPadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::Qadic(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyQadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::QadicField(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyQadicElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::PadicLocalField(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyPadicLocalFieldElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::NumberField(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyNumberFieldElement {
                                inner: coefficient.clone(),
                            },
                        )?,
                    ))?;
                }
            }
            PolynomialInner::RationalFunction(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(py, PyRationalFunction::from_rational(coefficient.clone()))?,
                    ))?;
                }
            }
            PolynomialInner::FiniteRationalFunction(value) => {
                for (exponent, coefficient) in value.terms() {
                    output.append((
                        integers_to_python(py, exponent)?,
                        Py::new(
                            py,
                            PyRationalFunction::from_finite_field(coefficient.clone()),
                        )?,
                    ))?;
                }
            }
        }
        Ok(output.unbind().into_any())
    }

    /// Serialize this polynomial with its complete ring and ordered terms.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    /// Deserialize and revalidate a polynomial over one supported coefficient ring.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PolynomialInner::from_json(value)?,
        })
    }

    fn coefficient(&self, py: Python<'_>, exponent: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let exponent = extract_exponent(exponent)?;
        match &self.inner {
            PolynomialInner::Integer(value) => {
                integer_to_python(py, &value.coefficient(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Rational(value) => {
                rational_to_python(py, &value.coefficient(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Residue(value) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::Padic(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::PadicField(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::Qadic(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::QadicField(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::PadicLocalField(value) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::NumberField(value) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: value.coefficient(&exponent).map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::RationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_rational(
                    value.coefficient(&exponent).map_err(value_error)?,
                ),
            )?
            .into_any()),
            PolynomialInner::FiniteRationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_finite_field(
                    value.coefficient(&exponent).map_err(value_error)?,
                ),
            )?
            .into_any()),
        }
    }

    /// Coefficient of one selected-variable power in the remaining variables.
    fn coefficient_polynomial(
        &self,
        variable: &Bound<'_, PyAny>,
        exponent: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let variable = self.ring_value().resolve_variable(variable)?;
        Ok(Self {
            inner: self
                .inner
                .coefficient_polynomial(variable, &extract_integer(exponent)?)
                .map_err(value_error)?,
        })
    }

    /// Dense selected-variable coefficients through its degree.
    #[pyo3(signature = (variable, *, max_count))]
    fn coefficient_polynomials(
        &self,
        variable: &Bound<'_, PyAny>,
        max_count: usize,
    ) -> PyResult<Vec<Self>> {
        let variable = self.ring_value().resolve_variable(variable)?;
        Ok(self
            .inner
            .coefficient_polynomials(variable, max_count)
            .map_err(value_error)?
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// Checked one-variable view retaining only the selected indeterminate.
    fn univariate_view(&self, variable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let variable = self.ring_value().resolve_variable(variable)?;
        Ok(Self {
            inner: self.inner.univariate_view(variable).map_err(value_error)?,
        })
    }

    fn constant_coefficient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PolynomialInner::Integer(value) => integer_to_python(py, &value.constant_coefficient()),
            PolynomialInner::Rational(value) => {
                rational_to_python(py, &value.constant_coefficient())
            }
            PolynomialInner::Residue(value) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::Padic(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::PadicField(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::Qadic(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::QadicField(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::PadicLocalField(value) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::NumberField(value) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: value.constant_coefficient(),
                },
            )?
            .into_any()),
            PolynomialInner::RationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_rational(value.constant_coefficient()),
            )?
            .into_any()),
            PolynomialInner::FiniteRationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_finite_field(value.constant_coefficient()),
            )?
            .into_any()),
        }
    }

    #[pyo3(signature = (variable=None))]
    fn degree(
        &self,
        py: Python<'_>,
        variable: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let variable = variable
            .map(|value| self.ring_value().resolve_variable(value))
            .transpose()?;
        let degree = match &self.inner {
            PolynomialInner::Integer(value) => value.degree(variable),
            PolynomialInner::Rational(value) => value.degree(variable),
            PolynomialInner::Residue(value) => value.degree(variable),
            PolynomialInner::FiniteField(value) => value.degree(variable),
            PolynomialInner::Padic(value) => value.degree(variable),
            PolynomialInner::PadicField(value) => value.degree(variable),
            PolynomialInner::Qadic(value) => value.degree(variable),
            PolynomialInner::QadicField(value) => value.degree(variable),
            PolynomialInner::PadicLocalField(value) => value.degree(variable),
            PolynomialInner::NumberField(value) => value.degree(variable),
            PolynomialInner::RationalFunction(value) => value.degree(variable),
            PolynomialInner::FiniteRationalFunction(value) => value.degree(variable),
        }
        .map_err(value_error)?;
        degree
            .map(|value| integer_to_python(py, &value))
            .transpose()
    }

    fn homogeneous_degree(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let degree = match &self.inner {
            PolynomialInner::Integer(value) => value.homogeneous_degree(),
            PolynomialInner::Rational(value) => value.homogeneous_degree(),
            PolynomialInner::Residue(value) => value.homogeneous_degree(),
            PolynomialInner::FiniteField(value) => value.homogeneous_degree(),
            PolynomialInner::Padic(value) => value.homogeneous_degree(),
            PolynomialInner::PadicField(value) => value.homogeneous_degree(),
            PolynomialInner::Qadic(value) => value.homogeneous_degree(),
            PolynomialInner::QadicField(value) => value.homogeneous_degree(),
            PolynomialInner::PadicLocalField(value) => value.homogeneous_degree(),
            PolynomialInner::NumberField(value) => value.homogeneous_degree(),
            PolynomialInner::RationalFunction(value) => value.homogeneous_degree(),
            PolynomialInner::FiniteRationalFunction(value) => value.homogeneous_degree(),
        }
        .map_err(value_error)?;
        degree
            .map(|value| integer_to_python(py, &value))
            .transpose()
    }

    fn homogeneous_components(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyList::empty(py);
        match &self.inner {
            PolynomialInner::Integer(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::Integer,
            )?,
            PolynomialInner::Rational(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::Rational,
            )?,
            PolynomialInner::Residue(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::Residue,
            )?,
            PolynomialInner::FiniteField(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::FiniteField,
            )?,
            PolynomialInner::Padic(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::Padic,
            )?,
            PolynomialInner::PadicField(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::PadicField,
            )?,
            PolynomialInner::Qadic(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::Qadic,
            )?,
            PolynomialInner::QadicField(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::QadicField,
            )?,
            PolynomialInner::PadicLocalField(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::PadicLocalField,
            )?,
            PolynomialInner::NumberField(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::NumberField,
            )?,
            PolynomialInner::RationalFunction(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::RationalFunction,
            )?,
            PolynomialInner::FiniteRationalFunction(value) => append_components(
                py,
                &output,
                value.homogeneous_components().map_err(value_error)?,
                PolynomialInner::FiniteRationalFunction,
            )?,
        }
        Ok(output.unbind().into_any())
    }

    /// Degree shared by every term in the full free-abelian grading.
    fn multihomogeneous_degree(&self) -> PyResult<Option<PyDegree>> {
        Ok(self
            .inner
            .multihomogeneous_degree()
            .map_err(value_error)?
            .map(|inner| PyDegree { inner }))
    }

    /// Split into components in the full free-abelian grading.
    fn multihomogeneous_components(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyList::empty(py);
        match &self.inner {
            PolynomialInner::Integer(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::Integer,
            )?,
            PolynomialInner::Rational(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::Rational,
            )?,
            PolynomialInner::Residue(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::Residue,
            )?,
            PolynomialInner::FiniteField(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::FiniteField,
            )?,
            PolynomialInner::Padic(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::Padic,
            )?,
            PolynomialInner::PadicField(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::PadicField,
            )?,
            PolynomialInner::Qadic(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::Qadic,
            )?,
            PolynomialInner::QadicField(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::QadicField,
            )?,
            PolynomialInner::PadicLocalField(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::PadicLocalField,
            )?,
            PolynomialInner::NumberField(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::NumberField,
            )?,
            PolynomialInner::RationalFunction(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::RationalFunction,
            )?,
            PolynomialInner::FiniteRationalFunction(value) => append_multicomponents(
                py,
                &output,
                value.multihomogeneous_components().map_err(value_error)?,
                PolynomialInner::FiniteRationalFunction,
            )?,
        }
        Ok(output.unbind().into_any())
    }

    fn derivative(&self, variable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let variable = self.ring_value().resolve_variable(variable)?;
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => {
                PolynomialInner::Integer(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::Rational(value) => {
                PolynomialInner::Rational(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::Residue(value) => {
                PolynomialInner::Residue(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::FiniteField(value) => {
                PolynomialInner::FiniteField(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::Padic(value) => {
                PolynomialInner::Padic(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::PadicField(value) => {
                PolynomialInner::PadicField(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::Qadic(value) => {
                PolynomialInner::Qadic(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::QadicField(value) => {
                PolynomialInner::QadicField(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::PadicLocalField(value) => {
                PolynomialInner::PadicLocalField(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::NumberField(value) => {
                PolynomialInner::NumberField(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::RationalFunction(value) => {
                PolynomialInner::RationalFunction(value.derivative(variable).map_err(value_error)?)
            }
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction(
                    value.derivative(variable).map_err(value_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn evaluate(&self, py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let ordered = ordered_values(values, self.inner.ring().variables())?;
        match &self.inner {
            PolynomialInner::Integer(value) => integer_to_python(
                py,
                &value
                    .evaluate(&extract_many(&ordered, extract_integer)?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Rational(value) => rational_to_python(
                py,
                &value
                    .evaluate(&extract_many(&ordered, |item| extract_rational(py, item))?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Residue(value) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_residue(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_finite_field(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::Padic(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_padic(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::PadicField(value) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_padic_field(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::Qadic(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_qadic(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::QadicField(value) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_qadic_field(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::PadicLocalField(value) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_padic_local_field(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::NumberField(value) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_number_field_element(py, value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                },
            )?
            .into_any()),
            PolynomialInner::RationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_rational(
                    value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_rational_function(py, value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                ),
            )?
            .into_any()),
            PolynomialInner::FiniteRationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunction::from_finite_field(
                    value
                        .evaluate(&extract_many(&ordered, |item| {
                            extract_finite_rational_function(value.ring().coefficient_ring(), item)
                        })?)
                        .map_err(value_error)?,
                ),
            )?
            .into_any()),
        }
    }

    fn substitute(&self, images: &Bound<'_, PyAny>) -> PyResult<Self> {
        let images = self.images_from_python(images)?;
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => PolynomialInner::Integer(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::Integer(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Rational(value) => PolynomialInner::Rational(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::Rational(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Residue(value) => PolynomialInner::Residue(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::Residue(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::FiniteField(value) => PolynomialInner::FiniteField(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::FiniteField(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Padic(value) => PolynomialInner::Padic(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::Padic(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::PadicField(value) => PolynomialInner::PadicField(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::PadicField(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Qadic(value) => PolynomialInner::Qadic(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::Qadic(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::QadicField(value) => PolynomialInner::QadicField(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::QadicField(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::PadicLocalField(value) => PolynomialInner::PadicLocalField(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::PadicLocalField(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::NumberField(value) => PolynomialInner::NumberField(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::NumberField(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::RationalFunction(value) => PolynomialInner::RationalFunction(
                value
                    .substitute(&typed_images(&images, |inner| match inner {
                        PolynomialInner::RationalFunction(value) => Some(value.clone()),
                        _ => None,
                    })?)
                    .map_err(value_error)?,
            ),
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction(
                    value
                        .substitute(&typed_images(&images, |inner| match inner {
                            PolynomialInner::FiniteRationalFunction(value) => Some(value.clone()),
                            _ => None,
                        })?)
                        .map_err(value_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn monomial_substitution(&self, monomial_map: &PyMonomialMap) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => PolynomialInner::Integer(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Rational(value) => PolynomialInner::Rational(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Residue(value) => PolynomialInner::Residue(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::FiniteField(value) => PolynomialInner::FiniteField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Padic(value) => PolynomialInner::Padic(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::PadicField(value) => PolynomialInner::PadicField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::Qadic(value) => PolynomialInner::Qadic(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::QadicField(value) => PolynomialInner::QadicField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::PadicLocalField(value) => PolynomialInner::PadicLocalField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::NumberField(value) => PolynomialInner::NumberField(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::RationalFunction(value) => PolynomialInner::RationalFunction(
                value
                    .monomial_substitution(&monomial_map.inner)
                    .map_err(value_error)?,
            ),
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction(
                    value
                        .monomial_substitution(&monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn semilinear_substitution(
        &self,
        coefficient_map: &Bound<'_, PyAny>,
        monomial_map: &PyMonomialMap,
    ) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Padic(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicRingFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a polynomial over PadicRing requires PadicRingFrobenius",
                        )
                    })?;
                PolynomialInner::Padic(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            PolynomialInner::PadicField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicFieldFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a polynomial over PadicField requires PadicFieldFrobenius",
                        )
                    })?;
                PolynomialInner::PadicField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            PolynomialInner::Qadic(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyQadicRingFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a polynomial over QadicRing requires QadicRingFrobenius",
                        )
                    })?;
                PolynomialInner::Qadic(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            PolynomialInner::QadicField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyQadicFieldFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a polynomial over QadicField requires QadicFieldFrobenius",
                        )
                    })?;
                PolynomialInner::QadicField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            PolynomialInner::PadicLocalField(value) => {
                let coefficient_map = coefficient_map
                    .extract::<PyRef<'_, PyPadicLocalFieldCoefficientFrobenius>>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "a polynomial over PadicLocalField requires PadicLocalFieldCoefficientFrobenius",
                        )
                    })?;
                PolynomialInner::PadicLocalField(
                    value
                        .semilinear_substitution(&coefficient_map.inner, &monomial_map.inner)
                        .map_err(value_error)?,
                )
            }
            _ => {
                return Err(unsupported_polynomial_operation(
                    "semilinear_substitution",
                    self.inner.context_type_id(),
                    SEMILINEAR_POLYNOMIAL_CONTEXTS,
                ));
            }
        };
        Ok(Self { inner })
    }

    fn frobenius_lift(&self, exponent: u64) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Padic(value) => {
                PolynomialInner::Padic(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            PolynomialInner::PadicField(value) => {
                PolynomialInner::PadicField(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            PolynomialInner::Qadic(value) => {
                PolynomialInner::Qadic(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            PolynomialInner::QadicField(value) => {
                PolynomialInner::QadicField(value.frobenius_lift(exponent).map_err(value_error)?)
            }
            _ => {
                return Err(unsupported_polynomial_operation(
                    "frobenius_lift",
                    self.inner.context_type_id(),
                    FROBENIUS_POLYNOMIAL_CONTEXTS,
                ));
            }
        };
        Ok(Self { inner })
    }

    fn pow(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exponent = extract_integer(exponent)?;
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => {
                PolynomialInner::Integer(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Rational(value) => {
                PolynomialInner::Rational(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Residue(value) => {
                PolynomialInner::Residue(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::FiniteField(value) => {
                PolynomialInner::FiniteField(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Padic(value) => {
                PolynomialInner::Padic(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::PadicField(value) => {
                PolynomialInner::PadicField(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::Qadic(value) => {
                PolynomialInner::Qadic(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::QadicField(value) => {
                PolynomialInner::QadicField(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::PadicLocalField(value) => {
                PolynomialInner::PadicLocalField(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::NumberField(value) => {
                PolynomialInner::NumberField(value.pow_integer(&exponent).map_err(value_error)?)
            }
            PolynomialInner::RationalFunction(value) => PolynomialInner::RationalFunction(
                value.pow_integer(&exponent).map_err(value_error)?,
            ),
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction(
                    value.pow_integer(&exponent).map_err(value_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn leading_coefficient(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            PolynomialInner::Integer(value) => value
                .leading_coefficient()
                .map(|coefficient| integer_to_python(py, &coefficient))
                .transpose(),
            PolynomialInner::Rational(value) => value
                .leading_coefficient()
                .map(|coefficient| rational_to_python(py, &coefficient))
                .transpose(),
            PolynomialInner::Residue(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyResidueClass { inner: coefficient }).map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::FiniteField(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyFiniteFieldElement { inner: coefficient })
                        .map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::Padic(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyPadicElement { inner: coefficient }).map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::PadicField(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyPadicElement { inner: coefficient }).map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::Qadic(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyQadicElement { inner: coefficient }).map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::QadicField(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyQadicElement { inner: coefficient }).map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::PadicLocalField(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyPadicLocalFieldElement { inner: coefficient })
                        .map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::NumberField(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyNumberFieldElement { inner: coefficient })
                        .map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::RationalFunction(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyRationalFunction::from_rational(coefficient))
                        .map(|value| value.into_any())
                })
                .transpose(),
            PolynomialInner::FiniteRationalFunction(value) => value
                .leading_coefficient()
                .map(|coefficient| {
                    Py::new(py, PyRationalFunction::from_finite_field(coefficient))
                        .map(|value| value.into_any())
                })
                .transpose(),
        }
    }

    fn monic(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Rational(value) => {
                PolynomialInner::Rational(value.monic().map_err(value_error)?)
            }
            PolynomialInner::FiniteField(value) => {
                PolynomialInner::FiniteField(value.monic().map_err(value_error)?)
            }
            PolynomialInner::NumberField(value) => {
                PolynomialInner::NumberField(value.monic().map_err(value_error)?)
            }
            PolynomialInner::RationalFunction(value) => {
                PolynomialInner::RationalFunction(value.monic().map_err(value_error)?)
            }
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction(value.monic().map_err(value_error)?)
            }
            _ => {
                return Err(unsupported_polynomial_operation(
                    "monic",
                    self.inner.context_type_id(),
                    FIELD_POLYNOMIAL_CONTEXTS,
                ));
            }
        };
        Ok(Self { inner })
    }

    fn div_rem(&self, py: Python<'_>, divisor: &Bound<'_, PyAny>) -> PyResult<(Self, Self)> {
        let divisor = self.coerce_operand(py, divisor)?;
        py.detach(|| self.field_div_rem(&divisor))
    }

    fn exact_div(&self, py: Python<'_>, divisor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let divisor = self.coerce_operand(py, divisor)?;
        py.detach(|| self.exact_division(&divisor))
    }

    fn gcd(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let right = self.coerce_operand(py, other)?;
        py.detach(|| self.polynomial_gcd(&right))
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

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let right = self.coerce_operand(py, other)?;
        py.detach(|| self.binary(&right, "mul"))
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, other)
    }

    fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div_rem(py, other).map(|(quotient, _)| quotient)
    }

    fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dividend = self.coerce_operand(py, other)?;
        py.detach(|| dividend.field_div_rem(self).map(|(quotient, _)| quotient))
    }

    fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div_rem(py, other).map(|(_, remainder)| remainder)
    }

    fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dividend = self.coerce_operand(py, other)?;
        py.detach(|| dividend.field_div_rem(self).map(|(_, remainder)| remainder))
    }

    fn __divmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<(Self, Self)> {
        self.div_rem(py, other)
    }

    fn __rdivmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<(Self, Self)> {
        let dividend = self.coerce_operand(py, other)?;
        py.detach(|| dividend.field_div_rem(self))
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            PolynomialInner::Integer(value) => {
                PolynomialInner::Integer((-value).map_err(value_error)?)
            }
            PolynomialInner::Rational(value) => {
                PolynomialInner::Rational((-value).map_err(value_error)?)
            }
            PolynomialInner::Residue(value) => {
                PolynomialInner::Residue((-value).map_err(value_error)?)
            }
            PolynomialInner::FiniteField(value) => {
                PolynomialInner::FiniteField((-value).map_err(value_error)?)
            }
            PolynomialInner::Padic(value) => PolynomialInner::Padic((-value).map_err(value_error)?),
            PolynomialInner::PadicField(value) => {
                PolynomialInner::PadicField((-value).map_err(value_error)?)
            }
            PolynomialInner::Qadic(value) => PolynomialInner::Qadic((-value).map_err(value_error)?),
            PolynomialInner::QadicField(value) => {
                PolynomialInner::QadicField((-value).map_err(value_error)?)
            }
            PolynomialInner::PadicLocalField(value) => {
                PolynomialInner::PadicLocalField((-value).map_err(value_error)?)
            }
            PolynomialInner::NumberField(value) => {
                PolynomialInner::NumberField((-value).map_err(value_error)?)
            }
            PolynomialInner::RationalFunction(value) => {
                PolynomialInner::RationalFunction((-value).map_err(value_error)?)
            }
            PolynomialInner::FiniteRationalFunction(value) => {
                PolynomialInner::FiniteRationalFunction((-value).map_err(value_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulus.is_some() {
            return Err(PyTypeError::new_err(
                "Polynomial power does not accept Python's third modulus argument",
            ));
        }
        self.pow(exponent)
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
                "Polynomial(nonzero_terms={}, variables={:?})",
                self.inner.nonzero_term_count(),
                self.inner.ring().variables()
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
        let weights = ring.weighted_grading();
        summary.set_item("type", "noether/commalg/Polynomial@1")?;
        summary.set_item(
            "variables",
            variables
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .collect::<Vec<_>>(),
        )?;
        summary.set_item("term_order", self.inner.ring().term_order().name())?;
        summary.set_item("grading_rank", ring.grading().rank())?;
        summary.set_item(
            "weights",
            weights.as_ref().map(|grading| {
                grading
                    .weights()
                    .iter()
                    .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                    .copied()
                    .collect::<Vec<_>>()
            }),
        )?;
        if variables.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("variables_truncated", true)?;
        }
        summary.set_item("nonzero_term_count", self.inner.nonzero_term_count())?;
        Ok(summary.unbind())
    }
}

fn value_is_string(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyString>()
}

fn integers_to_python(py: Python<'_>, values: &[Integer]) -> PyResult<Py<PyAny>> {
    let output = PyList::empty(py);
    for value in values {
        output.append(integer_to_python(py, value)?)?;
    }
    Ok(output.unbind().into_any())
}

fn append_components<R: noether::CommutativeRing>(
    py: Python<'_>,
    output: &Bound<'_, PyList>,
    components: Vec<(Integer, Polynomial<R>)>,
    wrap: impl Fn(Polynomial<R>) -> PolynomialInner,
) -> PyResult<()> {
    for (degree, polynomial) in components {
        output.append((
            integer_to_python(py, &degree)?,
            Py::new(
                py,
                PyPolynomial {
                    inner: wrap(polynomial),
                },
            )?,
        ))?;
    }
    Ok(())
}

fn append_multicomponents<R: noether::CommutativeRing>(
    py: Python<'_>,
    output: &Bound<'_, PyList>,
    components: Vec<(Degree, Polynomial<R>)>,
    wrap: impl Fn(Polynomial<R>) -> PolynomialInner,
) -> PyResult<()> {
    for (degree, polynomial) in components {
        output.append((
            Py::new(py, PyDegree { inner: degree })?,
            Py::new(
                py,
                PyPolynomial {
                    inner: wrap(polynomial),
                },
            )?,
        ))?;
    }
    Ok(())
}

fn ordered_values<'py>(
    values: &Bound<'py, PyAny>,
    variables: &[String],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(mapping) = values.cast::<PyDict>() {
        return variables
            .iter()
            .map(|variable| {
                mapping
                    .get_item(variable)?
                    .ok_or_else(|| value_error(format!("missing value for {variable:?}")))
            })
            .collect();
    }
    values.try_iter()?.collect()
}

fn extract_many<E>(
    values: &[Bound<'_, PyAny>],
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<E>> {
    values.iter().map(&mut extract).collect()
}

fn typed_images<E>(
    images: &[PyPolynomial],
    mut extract: impl FnMut(&PolynomialInner) -> Option<E>,
) -> PyResult<Vec<E>> {
    images
        .iter()
        .map(|image| {
            extract(&image.inner).ok_or_else(|| {
                value_error("substitution image has a different coefficient context")
            })
        })
        .collect()
}
