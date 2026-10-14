// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of Mora standard bases for local modules.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::LocalPolynomialRing;
use noether::homalg::{
    FreeModule, LocalModuleError, LocalModuleOrder, LocalModuleReduction, LocalModuleStandardBasis,
    ModuleElement,
};
use noether::linalg::Vector;
use noether::numthy::FiniteField;
use noether::{RationalField, SemanticType};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::local::{
    LocalPolynomialInner, LocalRingInner, PyLocalPolynomial, PyLocalPolynomialRing,
};

fn local_module_error(error: LocalModuleError) -> PyErr {
    match error {
        LocalModuleError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
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

/// A module-monomial order extending a local polynomial order.
#[pyclass(
    module = "noether.homalg",
    name = "LocalModuleOrder",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyLocalModuleOrder {
    inner: LocalModuleOrder,
}

#[pymethods]
impl PyLocalModuleOrder {
    #[classattr]
    fn type_id() -> &'static str {
        LocalModuleOrder::TYPE_ID
    }

    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "term_over_position" => LocalModuleOrder::TermOverPosition,
            "position_over_term" => LocalModuleOrder::PositionOverTerm,
            _ => return Err(PyValueError::new_err(
                "unknown local module order; expected 'term_over_position' or 'position_over_term'",
            )),
        };
        Ok(Self { inner })
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn TERM_OVER_POSITION() -> Self {
        Self {
            inner: LocalModuleOrder::TermOverPosition,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn POSITION_OVER_TERM() -> Self {
        Self {
            inner: LocalModuleOrder::PositionOverTerm,
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: LocalModuleOrder::from_json(value).map_err(value_error)?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("LocalModuleOrder({})", self.inner.name()))
    }

    fn __str__(&self) -> &'static str {
        self.inner.name()
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalModuleStandardBasisInner {
    Rational(LocalModuleStandardBasis<RationalField>),
    FiniteField(LocalModuleStandardBasis<FiniteField>),
}

/// A verified Mora standard basis of a submodule over a polynomial local ring.
#[pyclass(
    module = "noether.homalg",
    name = "LocalModuleStandardBasis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalModuleStandardBasis {
    inner: LocalModuleStandardBasisInner,
}

impl PyLocalModuleStandardBasis {
    fn from_rational(inner: LocalModuleStandardBasis<RationalField>) -> Self {
        Self {
            inner: LocalModuleStandardBasisInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalModuleStandardBasis<FiniteField>) -> Self {
        Self {
            inner: LocalModuleStandardBasisInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalModuleStandardBasis {
    #[classattr]
    fn type_id() -> &'static str {
        LocalModuleStandardBasis::<RationalField>::TYPE_ID
    }

    #[classmethod]
    #[pyo3(signature = (ring, rank, generators, *, order=None))]
    fn from_generators(
        _cls: &Bound<'_, PyType>,
        ring: &PyLocalPolynomialRing,
        rank: usize,
        generators: &Bound<'_, PyAny>,
        order: Option<&PyLocalModuleOrder>,
    ) -> PyResult<Self> {
        let order = order.map_or_else(LocalModuleOrder::default, |value| value.inner);
        match &ring.inner {
            LocalRingInner::Rational(ring) => {
                let ambient = FreeModule::new(ring.clone(), rank);
                let generators = collect_rational_vectors(&ambient, generators)?;
                Ok(Self::from_rational(
                    LocalModuleStandardBasis::from_generators(ambient, &generators, order)
                        .map_err(local_module_error)?,
                ))
            }
            LocalRingInner::FiniteField(ring) => {
                let ambient = FreeModule::new(ring.clone(), rank);
                let generators = collect_finite_field_vectors(&ambient, generators)?;
                Ok(Self::from_finite_field(
                    LocalModuleStandardBasis::from_generators(ambient, &generators, order)
                        .map_err(local_module_error)?,
                ))
            }
        }
    }

    #[getter]
    fn ring(&self) -> PyLocalPolynomialRing {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => {
                PyLocalPolynomialRing::from_rational(value.ambient().ring().clone())
            }
            LocalModuleStandardBasisInner::FiniteField(value) => {
                PyLocalPolynomialRing::from_finite_field(value.ambient().ring().clone())
            }
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value.ambient().rank(),
            LocalModuleStandardBasisInner::FiniteField(value) => value.ambient().rank(),
        }
    }

    #[getter]
    fn order(&self) -> PyLocalModuleOrder {
        let inner = match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value.order(),
            LocalModuleStandardBasisInner::FiniteField(value) => value.order(),
        };
        PyLocalModuleOrder { inner }
    }

    #[getter]
    fn generators(&self) -> PyResult<Vec<Vec<PyLocalPolynomial>>> {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value
                .generators()
                .map_err(local_module_error)?
                .iter()
                .map(rational_coordinates)
                .collect(),
            LocalModuleStandardBasisInner::FiniteField(value) => value
                .generators()
                .map_err(local_module_error)?
                .iter()
                .map(finite_field_coordinates)
                .collect(),
        }
    }

    #[getter]
    fn basis(&self) -> PyResult<Vec<Vec<PyLocalPolynomial>>> {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value
                .basis()
                .map_err(local_module_error)?
                .iter()
                .map(rational_coordinates)
                .collect(),
            LocalModuleStandardBasisInner::FiniteField(value) => value
                .basis()
                .map_err(local_module_error)?
                .iter()
                .map(finite_field_coordinates)
                .collect(),
        }
    }

    fn reduce(&self, value: &Bound<'_, PyAny>) -> PyResult<PyLocalModuleReduction> {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(basis) => {
                let value = collect_rational_vector(basis.ambient(), value)?;
                Ok(PyLocalModuleReduction::from_rational(
                    basis.reduce(&value).map_err(local_module_error)?,
                ))
            }
            LocalModuleStandardBasisInner::FiniteField(basis) => {
                let value = collect_finite_field_vector(basis.ambient(), value)?;
                Ok(PyLocalModuleReduction::from_finite_field(
                    basis.reduce(&value).map_err(local_module_error)?,
                ))
            }
        }
    }

    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.reduce(value)?.is_member())
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => {
                value.verify().map_err(local_module_error)
            }
            LocalModuleStandardBasisInner::FiniteField(value) => {
                value.verify().map_err(local_module_error)
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value.to_json(),
            LocalModuleStandardBasisInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/ambient/ring/ambient_ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalModuleStandardBasis::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalModuleStandardBasis::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local module coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => canonical_hash(value),
            LocalModuleStandardBasisInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalModuleStandardBasisInner::Rational(value) => value.to_string(),
            LocalModuleStandardBasisInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum LocalModuleReductionInner {
    Rational(LocalModuleReduction<RationalField>),
    FiniteField(LocalModuleReduction<FiniteField>),
}

/// Replayable weak-normal-form evidence for one local module reduction.
#[pyclass(
    module = "noether.homalg",
    name = "LocalModuleReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLocalModuleReduction {
    inner: LocalModuleReductionInner,
}

impl PyLocalModuleReduction {
    fn from_rational(inner: LocalModuleReduction<RationalField>) -> Self {
        Self {
            inner: LocalModuleReductionInner::Rational(inner),
        }
    }

    fn from_finite_field(inner: LocalModuleReduction<FiniteField>) -> Self {
        Self {
            inner: LocalModuleReductionInner::FiniteField(inner),
        }
    }
}

#[pymethods]
impl PyLocalModuleReduction {
    #[classattr]
    fn type_id() -> &'static str {
        LocalModuleReduction::<RationalField>::TYPE_ID
    }

    #[getter]
    fn standard_basis(&self) -> PyLocalModuleStandardBasis {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => {
                PyLocalModuleStandardBasis::from_rational(value.standard_basis().clone())
            }
            LocalModuleReductionInner::FiniteField(value) => {
                PyLocalModuleStandardBasis::from_finite_field(value.standard_basis().clone())
            }
        }
    }

    #[getter]
    fn value(&self) -> PyResult<Vec<PyLocalPolynomial>> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => rational_coordinates(value.value()),
            LocalModuleReductionInner::FiniteField(value) => {
                finite_field_coordinates(value.value())
            }
        }
    }

    #[getter]
    fn multiplier(&self) -> PyResult<PyLocalPolynomial> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => Ok(PyLocalPolynomial::from_rational(
                value.multiplier().map_err(local_module_error)?,
            )),
            LocalModuleReductionInner::FiniteField(value) => {
                Ok(PyLocalPolynomial::from_finite_field(
                    value.multiplier().map_err(local_module_error)?,
                ))
            }
        }
    }

    #[getter]
    fn quotients(&self) -> PyResult<Vec<PyLocalPolynomial>> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => Ok(value
                .quotients()
                .map_err(local_module_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_rational)
                .collect()),
            LocalModuleReductionInner::FiniteField(value) => Ok(value
                .quotients()
                .map_err(local_module_error)?
                .into_iter()
                .map(PyLocalPolynomial::from_finite_field)
                .collect()),
        }
    }

    #[getter]
    fn remainder(&self) -> PyResult<Vec<PyLocalPolynomial>> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => {
                rational_coordinates(&value.remainder().map_err(local_module_error)?)
            }
            LocalModuleReductionInner::FiniteField(value) => {
                finite_field_coordinates(&value.remainder().map_err(local_module_error)?)
            }
        }
    }

    #[getter]
    fn is_member(&self) -> bool {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => value.is_member(),
            LocalModuleReductionInner::FiniteField(value) => value.is_member(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => {
                value.verify().map_err(local_module_error)
            }
            LocalModuleReductionInner::FiniteField(value) => {
                value.verify().map_err(local_module_error)
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => value.to_json(),
            LocalModuleReductionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(
            value,
            "/standard_basis/ambient/ring/ambient_ring/coefficient_ring/type",
        )?
        .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                LocalModuleReduction::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                LocalModuleReduction::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported local module coefficient context {coefficient}"
            ))),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match &self.inner {
            LocalModuleReductionInner::Rational(value) => canonical_hash(value),
            LocalModuleReductionInner::FiniteField(value) => canonical_hash(value),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            LocalModuleReductionInner::Rational(value) => value.to_string(),
            LocalModuleReductionInner::FiniteField(value) => value.to_string(),
        })
    }
}

fn collect_rational_vectors(
    ambient: &FreeModule<LocalPolynomialRing<RationalField>>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<ModuleElement<LocalPolynomialRing<RationalField>>>> {
    values
        .try_iter()?
        .map(|value| collect_rational_vector(ambient, &value?))
        .collect()
}

fn collect_rational_vector(
    ambient: &FreeModule<LocalPolynomialRing<RationalField>>,
    value: &Bound<'_, PyAny>,
) -> PyResult<ModuleElement<LocalPolynomialRing<RationalField>>> {
    let coordinates = value
        .try_iter()?
        .map(|coordinate| {
            let coordinate = coordinate?.extract::<PyRef<'_, PyLocalPolynomial>>()?;
            match &coordinate.inner {
                LocalPolynomialInner::Rational(value) => Ok(value.clone()),
                LocalPolynomialInner::FiniteField(_) => Err(PyTypeError::new_err(
                    "local module coordinate must belong to the rational local ring",
                )),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    build_element(ambient, coordinates)
}

fn collect_finite_field_vectors(
    ambient: &FreeModule<LocalPolynomialRing<FiniteField>>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<ModuleElement<LocalPolynomialRing<FiniteField>>>> {
    values
        .try_iter()?
        .map(|value| collect_finite_field_vector(ambient, &value?))
        .collect()
}

fn collect_finite_field_vector(
    ambient: &FreeModule<LocalPolynomialRing<FiniteField>>,
    value: &Bound<'_, PyAny>,
) -> PyResult<ModuleElement<LocalPolynomialRing<FiniteField>>> {
    let coordinates = value
        .try_iter()?
        .map(|coordinate| {
            let coordinate = coordinate?.extract::<PyRef<'_, PyLocalPolynomial>>()?;
            match &coordinate.inner {
                LocalPolynomialInner::Rational(_) => Err(PyTypeError::new_err(
                    "local module coordinate must belong to the finite-field local ring",
                )),
                LocalPolynomialInner::FiniteField(value) => Ok(value.clone()),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    build_element(ambient, coordinates)
}

fn build_element<R: noether::Field>(
    ambient: &FreeModule<LocalPolynomialRing<R>>,
    coordinates: Vec<noether::commalg::LocalPolynomial<R>>,
) -> PyResult<ModuleElement<LocalPolynomialRing<R>>> {
    let vector = Vector::from_values(ambient.ring().clone(), coordinates).map_err(value_error)?;
    ModuleElement::from_coordinates(ambient.clone(), vector).map_err(value_error)
}

fn rational_coordinates(
    value: &ModuleElement<LocalPolynomialRing<RationalField>>,
) -> PyResult<Vec<PyLocalPolynomial>> {
    Ok(value
        .coordinates()
        .values()
        .map(PyLocalPolynomial::from_rational)
        .collect())
}

fn finite_field_coordinates(
    value: &ModuleElement<LocalPolynomialRing<FiniteField>>,
) -> PyResult<Vec<PyLocalPolynomial>> {
    Ok(value
        .coordinates()
        .values()
        .map(PyLocalPolynomial::from_finite_field)
        .collect())
}

fn serialized_coefficient(value: &str, pointer: &str) -> PyResult<String> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    document
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| PyValueError::new_err("serialized local module has no coefficient identity"))
}
