// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of verified construction algebras.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{
    AssociatedGradedAlgebra, ConstructionAlgebraError, ReesAlgebra, SymmetricAlgebra,
};
use noether::numthy::FiniteField;
use noether::{RationalField, SemanticType};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use super::module_facade::PyPresentedModule;
use super::number_field::PyIdeal;
use super::polynomial_facade::PyPolynomialRing;
use super::quotient::{PyQuotientElement, PyQuotientRing};

pub(crate) fn construction_error(error: ConstructionAlgebraError) -> PyErr {
    match error {
        ConstructionAlgebraError::ResourceLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        ConstructionAlgebraError::ContextMismatch(error) => super::errors::context_mismatch(error),
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

fn serialized_coefficient(value: &str, pointer: &str) -> PyResult<String> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    document
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PyValueError::new_err("serialized construction algebra has no coefficient identity")
        })
}

#[derive(Clone, PartialEq, Eq)]
enum SymmetricAlgebraInner {
    Rational(Box<SymmetricAlgebra<RationalField>>),
    FiniteField(Box<SymmetricAlgebra<FiniteField>>),
}

/// A verified symmetric-algebra quotient presentation.
#[pyclass(
    module = "noether.commalg",
    name = "SymmetricAlgebra",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySymmetricAlgebra {
    inner: SymmetricAlgebraInner,
}

impl PySymmetricAlgebra {
    pub(crate) fn from_rational(inner: SymmetricAlgebra<RationalField>) -> Self {
        Self {
            inner: SymmetricAlgebraInner::Rational(Box::new(inner)),
        }
    }
    pub(crate) fn from_finite_field(inner: SymmetricAlgebra<FiniteField>) -> Self {
        Self {
            inner: SymmetricAlgebraInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PySymmetricAlgebra {
    #[classattr]
    fn type_id() -> &'static str {
        SymmetricAlgebra::<RationalField>::TYPE_ID
    }
    #[getter]
    fn module(&self) -> PyPresentedModule {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.module().clone())
            }
            SymmetricAlgebraInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.module().clone())
            }
        }
    }
    #[getter]
    fn ambient_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.ambient_ring().clone())
            }
            SymmetricAlgebraInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.ambient_ring().clone())
            }
        }
    }
    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.defining_ideal().clone())
            }
            SymmetricAlgebraInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.defining_ideal().clone())
            }
        }
    }
    #[getter]
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            SymmetricAlgebraInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }
    #[getter]
    fn base_images(&self) -> PyResult<Vec<PyQuotientElement>> {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => Ok(value
                .base_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_rational)
                .collect()),
            SymmetricAlgebraInner::FiniteField(value) => Ok(value
                .base_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_finite_field)
                .collect()),
        }
    }
    #[getter]
    fn generator_images(&self) -> PyResult<Vec<PyQuotientElement>> {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => Ok(value
                .generator_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_rational)
                .collect()),
            SymmetricAlgebraInner::FiniteField(value) => Ok(value
                .generator_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_finite_field)
                .collect()),
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => value.verify(),
            SymmetricAlgebraInner::FiniteField(value) => value.verify(),
        }
        .map_err(construction_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => value.to_json(),
            SymmetricAlgebraInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(
            value,
            "/module/presentation/source/ring/coefficient_ring/type",
        )?
        .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                SymmetricAlgebra::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                SymmetricAlgebra::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported symmetric-algebra coefficient context {coefficient}"
            ))),
        }
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            SymmetricAlgebraInner::Rational(value) => canonical_hash(value.as_ref()),
            SymmetricAlgebraInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            SymmetricAlgebraInner::Rational(value) => value.to_string(),
            SymmetricAlgebraInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ReesAlgebraInner {
    Rational(Box<ReesAlgebra<RationalField>>),
    FiniteField(Box<ReesAlgebra<FiniteField>>),
}

/// A verified elimination presentation of an ideal's Rees algebra.
#[pyclass(
    module = "noether.commalg",
    name = "ReesAlgebra",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyReesAlgebra {
    inner: ReesAlgebraInner,
}

impl PyReesAlgebra {
    pub(crate) fn from_rational(inner: ReesAlgebra<RationalField>) -> Self {
        Self {
            inner: ReesAlgebraInner::Rational(Box::new(inner)),
        }
    }
    pub(crate) fn from_finite_field(inner: ReesAlgebra<FiniteField>) -> Self {
        Self {
            inner: ReesAlgebraInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyReesAlgebra {
    #[classattr]
    fn type_id() -> &'static str {
        ReesAlgebra::<RationalField>::TYPE_ID
    }
    #[getter]
    fn source(&self) -> PyIdeal {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.source().clone())
            }
            ReesAlgebraInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.source().clone())
            }
        }
    }
    #[getter]
    fn ambient_ring(&self) -> PyPolynomialRing {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => {
                PyPolynomialRing::from_rational(value.ambient_ring().clone())
            }
            ReesAlgebraInner::FiniteField(value) => {
                PyPolynomialRing::from_finite_field(value.ambient_ring().clone())
            }
        }
    }
    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.defining_ideal().clone())
            }
            ReesAlgebraInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.defining_ideal().clone())
            }
        }
    }
    #[getter]
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            ReesAlgebraInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }
    #[getter]
    fn base_images(&self) -> PyResult<Vec<PyQuotientElement>> {
        rees_images(&self.inner, false)
    }
    #[getter]
    fn generator_images(&self) -> PyResult<Vec<PyQuotientElement>> {
        rees_images(&self.inner, true)
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => value.verify(),
            ReesAlgebraInner::FiniteField(value) => value.verify(),
        }
        .map_err(construction_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => value.to_json(),
            ReesAlgebraInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/source/ring/coefficient_ring/type")?.as_str() {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                ReesAlgebra::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                ReesAlgebra::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported Rees-algebra coefficient context {coefficient}"
            ))),
        }
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            ReesAlgebraInner::Rational(value) => canonical_hash(value.as_ref()),
            ReesAlgebraInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ReesAlgebraInner::Rational(value) => value.to_string(),
            ReesAlgebraInner::FiniteField(value) => value.to_string(),
        })
    }
}

fn rees_images(inner: &ReesAlgebraInner, generators: bool) -> PyResult<Vec<PyQuotientElement>> {
    match inner {
        ReesAlgebraInner::Rational(value) => {
            let values = if generators {
                value.generator_images()
            } else {
                value.base_images()
            }
            .map_err(construction_error)?;
            Ok(values
                .into_iter()
                .map(PyQuotientElement::from_rational)
                .collect())
        }
        ReesAlgebraInner::FiniteField(value) => {
            let values = if generators {
                value.generator_images()
            } else {
                value.base_images()
            }
            .map_err(construction_error)?;
            Ok(values
                .into_iter()
                .map(PyQuotientElement::from_finite_field)
                .collect())
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum AssociatedGradedAlgebraInner {
    Rational(Box<AssociatedGradedAlgebra<RationalField>>),
    FiniteField(Box<AssociatedGradedAlgebra<FiniteField>>),
}

/// A verified presentation of the associated graded algebra of an ideal.
#[pyclass(
    module = "noether.commalg",
    name = "AssociatedGradedAlgebra",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAssociatedGradedAlgebra {
    inner: AssociatedGradedAlgebraInner,
}

impl PyAssociatedGradedAlgebra {
    pub(crate) fn from_rational(inner: AssociatedGradedAlgebra<RationalField>) -> Self {
        Self {
            inner: AssociatedGradedAlgebraInner::Rational(Box::new(inner)),
        }
    }
    pub(crate) fn from_finite_field(inner: AssociatedGradedAlgebra<FiniteField>) -> Self {
        Self {
            inner: AssociatedGradedAlgebraInner::FiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyAssociatedGradedAlgebra {
    #[classattr]
    fn type_id() -> &'static str {
        AssociatedGradedAlgebra::<RationalField>::TYPE_ID
    }
    #[getter]
    fn rees_algebra(&self) -> PyReesAlgebra {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => {
                PyReesAlgebra::from_rational(value.rees_algebra().clone())
            }
            AssociatedGradedAlgebraInner::FiniteField(value) => {
                PyReesAlgebra::from_finite_field(value.rees_algebra().clone())
            }
        }
    }
    #[getter]
    fn defining_ideal(&self) -> PyIdeal {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.defining_ideal().clone())
            }
            AssociatedGradedAlgebraInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.defining_ideal().clone())
            }
        }
    }
    #[getter]
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            AssociatedGradedAlgebraInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }
    #[getter]
    fn generator_images(&self) -> PyResult<Vec<PyQuotientElement>> {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => Ok(value
                .generator_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_rational)
                .collect()),
            AssociatedGradedAlgebraInner::FiniteField(value) => Ok(value
                .generator_images()
                .map_err(construction_error)?
                .into_iter()
                .map(PyQuotientElement::from_finite_field)
                .collect()),
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => value.verify(),
            AssociatedGradedAlgebraInner::FiniteField(value) => value.verify(),
        }
        .map_err(construction_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => value.to_json(),
            AssociatedGradedAlgebraInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_coefficient(value, "/rees_algebra/source/ring/coefficient_ring/type")?
            .as_str()
        {
            RationalField::TYPE_ID => Ok(Self::from_rational(
                AssociatedGradedAlgebra::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            FiniteField::TYPE_ID => Ok(Self::from_finite_field(
                AssociatedGradedAlgebra::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            coefficient => Err(PyValueError::new_err(format!(
                "unsupported associated-graded coefficient context {coefficient}"
            ))),
        }
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => canonical_hash(value.as_ref()),
            AssociatedGradedAlgebraInner::FiniteField(value) => canonical_hash(value.as_ref()),
        }
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            AssociatedGradedAlgebraInner::Rational(value) => value.to_string(),
            AssociatedGradedAlgebraInner::FiniteField(value) => value.to_string(),
        })
    }
}
