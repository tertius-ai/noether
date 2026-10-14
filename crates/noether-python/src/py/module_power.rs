// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of symmetric and exterior module powers.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::commalg::{PolynomialRing, QuotientRing, ResidueClass, ResidueRing};
use noether::homalg::{
    ExteriorAlgebra, ExteriorAlgebraError, ExteriorElement, FreeModulePower, ModulePowerError,
    ModulePowerHomomorphism, ModulePowerKind, PresentedModulePower,
    PresentedModulePowerHomomorphism,
};
use noether::numthy::FiniteField;
use noether::{IntegerRing, RationalField, Ring, SemanticType};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyType};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::commalg::PyResidueClass;
use super::facade_scalar::{extract_integer, integer_to_python};
use super::linalg::{
    extract_finite_field_polynomial, extract_integer_polynomial, extract_rational_polynomial,
    extract_residue_polynomial,
};
use super::module_facade::{
    serialized_module_ring, serialized_scalar_module_ring, FreeModuleInner, ModuleElementInner,
    ModuleHomomorphismInner, PresentedModuleHomomorphismInner, PresentedModuleInner, PyFreeModule,
    PyModuleElement, PyModuleHomomorphism, PyPresentedModule, PyPresentedModuleHomomorphism,
    SerializedModuleRing, SerializedScalarModuleRing,
};
use super::polynomial_facade::PyPolynomial;
use super::quotient::{PyQuotientElement, QuotientElementInner};

pub(crate) fn module_power_error(error: ModulePowerError) -> PyErr {
    match error {
        ModulePowerError::RankLimit { .. }
        | ModulePowerError::BasisTableLimit { .. }
        | ModulePowerError::MatrixCellLimit { .. }
        | ModulePowerError::ExpansionWorkLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        ModulePowerError::ContextMismatch(error) => super::errors::context_mismatch(error),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn exterior_error(error: ExteriorAlgebraError) -> PyErr {
    match error {
        ExteriorAlgebraError::TermPairLimit { .. } | ExteriorAlgebraError::DegreeOverflow => {
            PyOverflowError::new_err(error.to_string())
        }
        ExteriorAlgebraError::ContextMismatch(error) => super::errors::context_mismatch(error),
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

macro_rules! match_exterior_algebra {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            ExteriorAlgebraInner::ScalarInteger($value) => $body,
            ExteriorAlgebraInner::ScalarResidue($value) => $body,
            ExteriorAlgebraInner::Integer($value) => $body,
            ExteriorAlgebraInner::Rational($value) => $body,
            ExteriorAlgebraInner::Residue($value) => $body,
            ExteriorAlgebraInner::FiniteField($value) => $body,
            ExteriorAlgebraInner::QuotientRational($value) => $body,
            ExteriorAlgebraInner::QuotientFiniteField($value) => $body,
        }
    };
}

macro_rules! match_exterior_element {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            ExteriorElementInner::ScalarInteger($value) => $body,
            ExteriorElementInner::ScalarResidue($value) => $body,
            ExteriorElementInner::Integer($value) => $body,
            ExteriorElementInner::Rational($value) => $body,
            ExteriorElementInner::Residue($value) => $body,
            ExteriorElementInner::FiniteField($value) => $body,
            ExteriorElementInner::QuotientRational($value) => $body,
            ExteriorElementInner::QuotientFiniteField($value) => $body,
        }
    };
}

#[derive(Clone, PartialEq, Eq)]
enum ExteriorAlgebraInner {
    ScalarInteger(Box<ExteriorAlgebra<IntegerRing>>),
    ScalarResidue(Box<ExteriorAlgebra<ResidueRing>>),
    Integer(Box<ExteriorAlgebra<PolynomialRing<IntegerRing>>>),
    Rational(Box<ExteriorAlgebra<PolynomialRing<RationalField>>>),
    Residue(Box<ExteriorAlgebra<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<ExteriorAlgebra<PolynomialRing<FiniteField>>>),
    QuotientRational(Box<ExteriorAlgebra<QuotientRing<PolynomialRing<RationalField>>>>),
    QuotientFiniteField(Box<ExteriorAlgebra<QuotientRing<PolynomialRing<FiniteField>>>>),
}

/// The exact exterior-algebra parent of a finite based free module.
#[pyclass(
    module = "noether.homalg",
    name = "ExteriorAlgebra",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyExteriorAlgebra {
    inner: ExteriorAlgebraInner,
}

impl PyExteriorAlgebra {
    pub(super) fn from_module(source: &FreeModuleInner) -> Self {
        let inner = match source {
            FreeModuleInner::ScalarInteger(value) => {
                ExteriorAlgebraInner::ScalarInteger(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::ScalarResidue(value) => {
                ExteriorAlgebraInner::ScalarResidue(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::Integer(value) => {
                ExteriorAlgebraInner::Integer(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::Rational(value) => {
                ExteriorAlgebraInner::Rational(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::Residue(value) => {
                ExteriorAlgebraInner::Residue(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::FiniteField(value) => {
                ExteriorAlgebraInner::FiniteField(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::QuotientRational(value) => {
                ExteriorAlgebraInner::QuotientRational(Box::new(value.exterior_algebra()))
            }
            FreeModuleInner::QuotientFiniteField(value) => {
                ExteriorAlgebraInner::QuotientFiniteField(Box::new(value.exterior_algebra()))
            }
        };
        Self { inner }
    }

    fn wrap_element(&self, value: ExteriorElementInner) -> PyExteriorElement {
        PyExteriorElement { inner: value }
    }
}

impl Serialize for PyExteriorAlgebra {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match_exterior_algebra!(&self.inner, |value| value.serialize(serializer))
    }
}

impl<'de> Deserialize<'de> for PyExteriorAlgebra {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = serde_json::Value::deserialize(deserializer)?;
        macro_rules! attempt {
            ($type:ty, $variant:ident) => {
                if let Ok(value) = serde_json::from_value::<$type>(payload.clone()) {
                    return Ok(Self {
                        inner: ExteriorAlgebraInner::$variant(Box::new(value)),
                    });
                }
            };
        }
        attempt!(ExteriorAlgebra<IntegerRing>, ScalarInteger);
        attempt!(ExteriorAlgebra<ResidueRing>, ScalarResidue);
        attempt!(ExteriorAlgebra<PolynomialRing<IntegerRing>>, Integer);
        attempt!(ExteriorAlgebra<PolynomialRing<RationalField>>, Rational);
        attempt!(ExteriorAlgebra<PolynomialRing<ResidueRing>>, Residue);
        attempt!(ExteriorAlgebra<PolynomialRing<FiniteField>>, FiniteField);
        attempt!(
            ExteriorAlgebra<QuotientRing<PolynomialRing<RationalField>>>,
            QuotientRational
        );
        attempt!(
            ExteriorAlgebra<QuotientRing<PolynomialRing<FiniteField>>>,
            QuotientFiniteField
        );
        Err(serde::de::Error::custom(
            "invalid exterior algebra or unsupported coefficient context",
        ))
    }
}

fn parse_exterior_terms<E>(
    terms: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<(Vec<usize>, E)>> {
    terms
        .try_iter()?
        .map(|item| {
            let item = item?;
            let pair = item.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err("each exterior term must be a (basis, coefficient) tuple")
            })?;
            if pair.len() != 2 {
                return Err(PyTypeError::new_err(
                    "each exterior term must contain exactly two entries",
                ));
            }
            let basis = pair.get_item(0)?.extract::<Vec<usize>>()?;
            let coefficient = extract(&pair.get_item(1)?)?;
            Ok((basis, coefficient))
        })
        .collect()
}

fn extract_residue_scalar(ring: &ResidueRing, value: &Bound<'_, PyAny>) -> PyResult<ResidueClass> {
    if let Ok(integer) = extract_integer(value) {
        return Ok(ring.element(integer));
    }
    if let Ok(residue) = value.extract::<PyRef<'_, PyResidueClass>>() {
        if ring.contains(&residue.inner) {
            return Ok(residue.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            residue.inner.ring(),
            ring
        )));
    }
    Err(PyTypeError::new_err(
        "residue coefficient must be a Python int or ResidueClass",
    ))
}

#[pymethods]
impl PyExteriorAlgebra {
    /// Construct from one finite based free module and retain its exact context.
    #[new]
    fn new(source: &PyFreeModule) -> Self {
        Self::from_module(&source.inner)
    }

    #[classattr]
    fn type_id() -> &'static str {
        ExteriorAlgebra::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// The ordered degree-one source module.
    #[getter]
    fn source(&self) -> PyFreeModule {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => {
                FreeModuleInner::ScalarInteger(value.source().clone())
            }
            ExteriorAlgebraInner::ScalarResidue(value) => {
                FreeModuleInner::ScalarResidue(value.source().clone())
            }
            ExteriorAlgebraInner::Integer(value) => {
                FreeModuleInner::Integer(value.source().clone())
            }
            ExteriorAlgebraInner::Rational(value) => {
                FreeModuleInner::Rational(value.source().clone())
            }
            ExteriorAlgebraInner::Residue(value) => {
                FreeModuleInner::Residue(value.source().clone())
            }
            ExteriorAlgebraInner::FiniteField(value) => {
                FreeModuleInner::FiniteField(value.source().clone())
            }
            ExteriorAlgebraInner::QuotientRational(value) => {
                FreeModuleInner::QuotientRational(value.source().clone())
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                FreeModuleInner::QuotientFiniteField(value.source().clone())
            }
        };
        PyFreeModule { inner }
    }

    /// The exact coefficient context.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.source().inner.ring_to_python(py)
    }

    /// Rank of the degree-one source module.
    #[getter]
    fn rank(&self) -> usize {
        match_exterior_algebra!(&self.inner, |value| value.rank())
    }

    /// Canonical coordinate module for one homogeneous exterior degree.
    fn power(&self, py: Python<'_>, degree: usize) -> PyResult<PyFreeModulePower> {
        match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => {
                Ok(PyFreeModulePower::from_scalar_integer(
                    py.detach(|| value.power(degree)).map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::ScalarResidue(value) => {
                Ok(PyFreeModulePower::from_scalar_residue(
                    py.detach(|| value.power(degree)).map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::Integer(value) => Ok(PyFreeModulePower::from_integer(
                py.detach(|| value.power(degree)).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Rational(value) => Ok(PyFreeModulePower::from_rational(
                py.detach(|| value.power(degree)).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Residue(value) => Ok(PyFreeModulePower::from_residue(
                py.detach(|| value.power(degree)).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::FiniteField(value) => Ok(PyFreeModulePower::from_finite_field(
                py.detach(|| value.power(degree)).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::QuotientRational(value) => {
                Ok(PyFreeModulePower::from_quotient_rational(
                    py.detach(|| value.power(degree)).map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                Ok(PyFreeModulePower::from_quotient_finite_field(
                    py.detach(|| value.power(degree)).map_err(exterior_error)?,
                ))
            }
        }
    }

    /// Increasing basis-position sequences in canonical coordinate order.
    fn basis(&self, py: Python<'_>, degree: usize) -> PyResult<Vec<Vec<usize>>> {
        py.detach(|| match_exterior_algebra!(&self.inner, |value| value.basis(degree)))
            .map_err(exterior_error)
    }

    /// Additive identity, represented by an empty canonical sparse table.
    fn zero(&self) -> PyExteriorElement {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => {
                ExteriorElementInner::ScalarInteger(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::ScalarResidue(value) => {
                ExteriorElementInner::ScalarResidue(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::Integer(value) => {
                ExteriorElementInner::Integer(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::Rational(value) => {
                ExteriorElementInner::Rational(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::Residue(value) => {
                ExteriorElementInner::Residue(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::FiniteField(value) => {
                ExteriorElementInner::FiniteField(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::QuotientRational(value) => {
                ExteriorElementInner::QuotientRational(Box::new(value.zero()))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(value.zero()))
            }
        };
        self.wrap_element(inner)
    }

    /// Multiplicative identity in exterior degree zero.
    fn one(&self) -> PyExteriorElement {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => {
                ExteriorElementInner::ScalarInteger(Box::new(value.one()))
            }
            ExteriorAlgebraInner::ScalarResidue(value) => {
                ExteriorElementInner::ScalarResidue(Box::new(value.one()))
            }
            ExteriorAlgebraInner::Integer(value) => {
                ExteriorElementInner::Integer(Box::new(value.one()))
            }
            ExteriorAlgebraInner::Rational(value) => {
                ExteriorElementInner::Rational(Box::new(value.one()))
            }
            ExteriorAlgebraInner::Residue(value) => {
                ExteriorElementInner::Residue(Box::new(value.one()))
            }
            ExteriorAlgebraInner::FiniteField(value) => {
                ExteriorElementInner::FiniteField(Box::new(value.one()))
            }
            ExteriorAlgebraInner::QuotientRational(value) => {
                ExteriorElementInner::QuotientRational(Box::new(value.one()))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(value.one()))
            }
        };
        self.wrap_element(inner)
    }

    /// Degree-one element at a zero-based source-basis position.
    fn generator(&self, position: usize) -> PyResult<PyExteriorElement> {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => ExteriorElementInner::ScalarInteger(
                Box::new(value.generator(position).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::ScalarResidue(value) => ExteriorElementInner::ScalarResidue(
                Box::new(value.generator(position).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::Integer(value) => ExteriorElementInner::Integer(Box::new(
                value.generator(position).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Rational(value) => ExteriorElementInner::Rational(Box::new(
                value.generator(position).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Residue(value) => ExteriorElementInner::Residue(Box::new(
                value.generator(position).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::FiniteField(value) => ExteriorElementInner::FiniteField(
                Box::new(value.generator(position).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::QuotientRational(value) => {
                ExteriorElementInner::QuotientRational(Box::new(
                    value.generator(position).map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(
                    value.generator(position).map_err(exterior_error)?,
                ))
            }
        };
        Ok(self.wrap_element(inner))
    }

    /// Exterior basis element for a strictly increasing position sequence.
    fn basis_element(&self, basis: Vec<usize>) -> PyResult<PyExteriorElement> {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => ExteriorElementInner::ScalarInteger(
                Box::new(value.basis_element(basis).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::ScalarResidue(value) => ExteriorElementInner::ScalarResidue(
                Box::new(value.basis_element(basis).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::Integer(value) => ExteriorElementInner::Integer(Box::new(
                value.basis_element(basis).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Rational(value) => ExteriorElementInner::Rational(Box::new(
                value.basis_element(basis).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::Residue(value) => ExteriorElementInner::Residue(Box::new(
                value.basis_element(basis).map_err(exterior_error)?,
            )),
            ExteriorAlgebraInner::FiniteField(value) => ExteriorElementInner::FiniteField(
                Box::new(value.basis_element(basis).map_err(exterior_error)?),
            ),
            ExteriorAlgebraInner::QuotientRational(value) => {
                ExteriorElementInner::QuotientRational(Box::new(
                    value.basis_element(basis).map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(
                    value.basis_element(basis).map_err(exterior_error)?,
                ))
            }
        };
        Ok(self.wrap_element(inner))
    }

    /// Construct and canonicalize sparse `(basis, coefficient)` terms.
    fn from_terms(&self, py: Python<'_>, terms: &Bound<'_, PyAny>) -> PyResult<PyExteriorElement> {
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => {
                let terms = parse_exterior_terms(terms, extract_integer)?;
                ExteriorElementInner::ScalarInteger(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::ScalarResidue(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    extract_residue_scalar(value.ring(), entry)
                })?;
                ExteriorElementInner::ScalarResidue(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::Integer(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    extract_integer_polynomial(value.ring(), entry)
                })?;
                ExteriorElementInner::Integer(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::Rational(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    extract_rational_polynomial(py, value.ring(), entry)
                })?;
                ExteriorElementInner::Rational(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::Residue(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    extract_residue_polynomial(value.ring(), entry)
                })?;
                ExteriorElementInner::Residue(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::FiniteField(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    extract_finite_field_polynomial(value.ring(), entry)
                })?;
                ExteriorElementInner::FiniteField(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::QuotientRational(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    let scalar = entry.extract::<PyRef<'_, PyQuotientElement>>()?;
                    let QuotientElementInner::Rational(scalar) = &scalar.inner else {
                        return Err(PyTypeError::new_err(
                            "quotient coefficient context mismatch",
                        ));
                    };
                    Ok(scalar.clone())
                })?;
                ExteriorElementInner::QuotientRational(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                let terms = parse_exterior_terms(terms, |entry| {
                    let scalar = entry.extract::<PyRef<'_, PyQuotientElement>>()?;
                    let QuotientElementInner::FiniteField(scalar) = &scalar.inner else {
                        return Err(PyTypeError::new_err(
                            "quotient coefficient context mismatch",
                        ));
                    };
                    Ok(scalar.clone())
                })?;
                ExteriorElementInner::QuotientFiniteField(Box::new(
                    py.detach(|| value.from_terms(terms))
                        .map_err(exterior_error)?,
                ))
            }
        };
        Ok(self.wrap_element(inner))
    }

    /// Universal alternating product of degree-one module vectors.
    fn alternating_product(
        &self,
        py: Python<'_>,
        vectors: Vec<PyModuleElement>,
    ) -> PyResult<PyExteriorElement> {
        macro_rules! apply {
            ($algebra:expr, $variant:ident) => {{
                let values = vectors
                    .iter()
                    .map(|value| match &value.inner {
                        ModuleElementInner::$variant(value) => Ok(value),
                        _ => Err(PyTypeError::new_err("module vector context mismatch")),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                ExteriorElementInner::$variant(Box::new(
                    py.detach(|| $algebra.alternating_product(values))
                        .map_err(exterior_error)?,
                ))
            }};
        }
        let inner = match &self.inner {
            ExteriorAlgebraInner::ScalarInteger(value) => apply!(value, ScalarInteger),
            ExteriorAlgebraInner::ScalarResidue(value) => apply!(value, ScalarResidue),
            ExteriorAlgebraInner::Integer(value) => apply!(value, Integer),
            ExteriorAlgebraInner::Rational(value) => apply!(value, Rational),
            ExteriorAlgebraInner::Residue(value) => apply!(value, Residue),
            ExteriorAlgebraInner::FiniteField(value) => apply!(value, FiniteField),
            ExteriorAlgebraInner::QuotientRational(value) => apply!(value, QuotientRational),
            ExteriorAlgebraInner::QuotientFiniteField(value) => {
                apply!(value, QuotientFiniteField)
            }
        };
        Ok(self.wrap_element(inner))
    }

    /// Embed a coordinate vector from one canonical homogeneous exterior power.
    fn from_power_element(
        &self,
        py: Python<'_>,
        degree: usize,
        element: &PyModuleElement,
    ) -> PyResult<PyExteriorElement> {
        macro_rules! apply {
            ($algebra:expr, $element:expr, $variant:ident) => {
                ExteriorElementInner::$variant(Box::new(
                    py.detach(|| $algebra.from_power_element(degree, $element))
                        .map_err(exterior_error)?,
                ))
            };
        }
        let inner = match (&self.inner, &element.inner) {
            (ExteriorAlgebraInner::ScalarInteger(a), ModuleElementInner::ScalarInteger(e)) => {
                apply!(a, e, ScalarInteger)
            }
            (ExteriorAlgebraInner::ScalarResidue(a), ModuleElementInner::ScalarResidue(e)) => {
                apply!(a, e, ScalarResidue)
            }
            (ExteriorAlgebraInner::Integer(a), ModuleElementInner::Integer(e)) => {
                apply!(a, e, Integer)
            }
            (ExteriorAlgebraInner::Rational(a), ModuleElementInner::Rational(e)) => {
                apply!(a, e, Rational)
            }
            (ExteriorAlgebraInner::Residue(a), ModuleElementInner::Residue(e)) => {
                apply!(a, e, Residue)
            }
            (ExteriorAlgebraInner::FiniteField(a), ModuleElementInner::FiniteField(e)) => {
                apply!(a, e, FiniteField)
            }
            (
                ExteriorAlgebraInner::QuotientRational(a),
                ModuleElementInner::QuotientRational(e),
            ) => apply!(a, e, QuotientRational),
            (
                ExteriorAlgebraInner::QuotientFiniteField(a),
                ModuleElementInner::QuotientFiniteField(e),
            ) => apply!(a, e, QuotientFiniteField),
            _ => return Err(PyTypeError::new_err("power element context mismatch")),
        };
        Ok(self.wrap_element(inner))
    }

    /// Serialize the parent context as faithful typed JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| match_exterior_algebra!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }

    /// Restore and validate an exterior-algebra parent from faithful typed JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| serde_json::from_str(value))
            .map_err(value_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match_exterior_algebra!(&self.inner, |value| canonical_hash(value.as_ref()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(
            match_exterior_algebra!(&self.inner, |value| value.to_string())
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ExteriorElementInner {
    ScalarInteger(Box<ExteriorElement<IntegerRing>>),
    ScalarResidue(Box<ExteriorElement<ResidueRing>>),
    Integer(Box<ExteriorElement<PolynomialRing<IntegerRing>>>),
    Rational(Box<ExteriorElement<PolynomialRing<RationalField>>>),
    Residue(Box<ExteriorElement<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<ExteriorElement<PolynomialRing<FiniteField>>>),
    QuotientRational(Box<ExteriorElement<QuotientRing<PolynomialRing<RationalField>>>>),
    QuotientFiniteField(Box<ExteriorElement<QuotientRing<PolynomialRing<FiniteField>>>>),
}

/// A canonical sparse, possibly inhomogeneous, exact exterior-algebra element.
#[pyclass(
    module = "noether.homalg",
    name = "ExteriorElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyExteriorElement {
    inner: ExteriorElementInner,
}

impl Serialize for PyExteriorElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match_exterior_element!(&self.inner, |value| value.serialize(serializer))
    }
}

impl<'de> Deserialize<'de> for PyExteriorElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = serde_json::Value::deserialize(deserializer)?;
        macro_rules! attempt {
            ($type:ty, $variant:ident) => {
                if let Ok(value) = serde_json::from_value::<$type>(payload.clone()) {
                    return Ok(Self {
                        inner: ExteriorElementInner::$variant(Box::new(value)),
                    });
                }
            };
        }
        attempt!(ExteriorElement<IntegerRing>, ScalarInteger);
        attempt!(ExteriorElement<ResidueRing>, ScalarResidue);
        attempt!(ExteriorElement<PolynomialRing<IntegerRing>>, Integer);
        attempt!(ExteriorElement<PolynomialRing<RationalField>>, Rational);
        attempt!(ExteriorElement<PolynomialRing<ResidueRing>>, Residue);
        attempt!(ExteriorElement<PolynomialRing<FiniteField>>, FiniteField);
        attempt!(
            ExteriorElement<QuotientRing<PolynomialRing<RationalField>>>,
            QuotientRational
        );
        attempt!(
            ExteriorElement<QuotientRing<PolynomialRing<FiniteField>>>,
            QuotientFiniteField
        );
        Err(serde::de::Error::custom(
            "invalid exterior element or unsupported coefficient context",
        ))
    }
}

impl PyExteriorElement {
    fn algebra_inner(&self) -> ExteriorAlgebraInner {
        match &self.inner {
            ExteriorElementInner::ScalarInteger(value) => {
                ExteriorAlgebraInner::ScalarInteger(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::ScalarResidue(value) => {
                ExteriorAlgebraInner::ScalarResidue(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::Integer(value) => {
                ExteriorAlgebraInner::Integer(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::Rational(value) => {
                ExteriorAlgebraInner::Rational(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::Residue(value) => {
                ExteriorAlgebraInner::Residue(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::FiniteField(value) => {
                ExteriorAlgebraInner::FiniteField(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::QuotientRational(value) => {
                ExteriorAlgebraInner::QuotientRational(Box::new(value.algebra().clone()))
            }
            ExteriorElementInner::QuotientFiniteField(value) => {
                ExteriorAlgebraInner::QuotientFiniteField(Box::new(value.algebra().clone()))
            }
        }
    }

    fn coefficient_objects(&self, py: Python<'_>) -> PyResult<Vec<(Vec<usize>, Py<PyAny>)>> {
        macro_rules! polynomial_terms {
            ($value:expr, $ctor:expr) => {{
                $value
                    .terms()
                    .map(|(basis, coefficient)| {
                        Ok((
                            basis.to_vec(),
                            Py::new(py, $ctor(coefficient.clone()))?.into_any(),
                        ))
                    })
                    .collect()
            }};
        }
        match &self.inner {
            ExteriorElementInner::ScalarInteger(value) => value
                .terms()
                .map(|(basis, coefficient)| {
                    Ok((basis.to_vec(), integer_to_python(py, coefficient)?))
                })
                .collect(),
            ExteriorElementInner::ScalarResidue(value) => value
                .terms()
                .map(|(basis, coefficient)| {
                    Ok((
                        basis.to_vec(),
                        Py::new(
                            py,
                            PyResidueClass {
                                inner: coefficient.clone(),
                            },
                        )?
                        .into_any(),
                    ))
                })
                .collect(),
            ExteriorElementInner::Integer(value) => {
                polynomial_terms!(value, PyPolynomial::from_integer)
            }
            ExteriorElementInner::Rational(value) => {
                polynomial_terms!(value, PyPolynomial::from_rational)
            }
            ExteriorElementInner::Residue(value) => {
                polynomial_terms!(value, PyPolynomial::from_residue)
            }
            ExteriorElementInner::FiniteField(value) => {
                polynomial_terms!(value, PyPolynomial::from_finite_field)
            }
            ExteriorElementInner::QuotientRational(value) => value
                .terms()
                .map(|(basis, coefficient)| {
                    Ok((
                        basis.to_vec(),
                        Py::new(py, PyQuotientElement::from_rational(coefficient.clone()))?
                            .into_any(),
                    ))
                })
                .collect(),
            ExteriorElementInner::QuotientFiniteField(value) => value
                .terms()
                .map(|(basis, coefficient)| {
                    Ok((
                        basis.to_vec(),
                        Py::new(
                            py,
                            PyQuotientElement::from_finite_field(coefficient.clone()),
                        )?
                        .into_any(),
                    ))
                })
                .collect(),
        }
    }

    fn binary(&self, py: Python<'_>, rhs: &Self, operation: &str) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {{
                let value = py
                    .detach(|| match operation {
                        "add" => $left.add($right),
                        "sub" => $left.sub($right),
                        "wedge" => $left.wedge($right),
                        _ => unreachable!("private operation selector"),
                    })
                    .map_err(exterior_error)?;
                ExteriorElementInner::$variant(Box::new(value))
            }};
        }
        let inner = match (&self.inner, &rhs.inner) {
            (ExteriorElementInner::ScalarInteger(a), ExteriorElementInner::ScalarInteger(b)) => {
                apply!(a, b, ScalarInteger)
            }
            (ExteriorElementInner::ScalarResidue(a), ExteriorElementInner::ScalarResidue(b)) => {
                apply!(a, b, ScalarResidue)
            }
            (ExteriorElementInner::Integer(a), ExteriorElementInner::Integer(b)) => {
                apply!(a, b, Integer)
            }
            (ExteriorElementInner::Rational(a), ExteriorElementInner::Rational(b)) => {
                apply!(a, b, Rational)
            }
            (ExteriorElementInner::Residue(a), ExteriorElementInner::Residue(b)) => {
                apply!(a, b, Residue)
            }
            (ExteriorElementInner::FiniteField(a), ExteriorElementInner::FiniteField(b)) => {
                apply!(a, b, FiniteField)
            }
            (
                ExteriorElementInner::QuotientRational(a),
                ExteriorElementInner::QuotientRational(b),
            ) => apply!(a, b, QuotientRational),
            (
                ExteriorElementInner::QuotientFiniteField(a),
                ExteriorElementInner::QuotientFiniteField(b),
            ) => apply!(a, b, QuotientFiniteField),
            _ => {
                return Err(PyValueError::new_err(
                    "exterior algebra contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyExteriorElement {
    #[classattr]
    fn type_id() -> &'static str {
        ExteriorElement::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Exact exterior-algebra parent carried by this element.
    #[getter]
    fn algebra(&self) -> PyExteriorAlgebra {
        PyExteriorAlgebra {
            inner: self.algebra_inner(),
        }
    }

    /// Number of nonzero canonical sparse terms.
    #[getter]
    fn term_count(&self) -> usize {
        match_exterior_element!(&self.inner, |value| value.term_count())
    }

    /// Whether the canonical sparse term table is empty.
    fn is_zero(&self) -> bool {
        match_exterior_element!(&self.inner, |value| value.is_zero())
    }

    /// Canonical `(increasing basis positions, coefficient)` terms.
    #[getter]
    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(Vec<usize>, Py<PyAny>)>> {
        self.coefficient_objects(py)
    }

    /// Unique exterior degree, or `None` for a nonzero inhomogeneous value.
    #[getter]
    fn homogeneous_degree(&self) -> Option<usize> {
        match_exterior_element!(&self.inner, |value| value.homogeneous_degree())
    }

    /// Sorted exterior degrees carrying nonzero homogeneous components.
    #[getter]
    fn degrees(&self) -> Vec<usize> {
        match_exterior_element!(&self.inner, |value| value.degrees())
    }

    /// Extract one homogeneous component without changing its parent.
    fn homogeneous_component(&self, degree: usize) -> Self {
        let inner = match &self.inner {
            ExteriorElementInner::ScalarInteger(v) => {
                ExteriorElementInner::ScalarInteger(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::ScalarResidue(v) => {
                ExteriorElementInner::ScalarResidue(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::Integer(v) => {
                ExteriorElementInner::Integer(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::Rational(v) => {
                ExteriorElementInner::Rational(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::Residue(v) => {
                ExteriorElementInner::Residue(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::FiniteField(v) => {
                ExteriorElementInner::FiniteField(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::QuotientRational(v) => {
                ExteriorElementInner::QuotientRational(Box::new(v.homogeneous_component(degree)))
            }
            ExteriorElementInner::QuotientFiniteField(v) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(v.homogeneous_component(degree)))
            }
        };
        Self { inner }
    }

    /// Coefficient of one strictly increasing exterior basis sequence.
    fn coefficient(&self, py: Python<'_>, basis: Vec<usize>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ExteriorElementInner::ScalarInteger(v) => {
                integer_to_python(py, &v.coefficient(basis).map_err(exterior_error)?)
            }
            ExteriorElementInner::ScalarResidue(v) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: v.coefficient(basis).map_err(exterior_error)?,
                },
            )?
            .into_any()),
            ExteriorElementInner::Integer(v) => Ok(Py::new(
                py,
                PyPolynomial::from_integer(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
            ExteriorElementInner::Rational(v) => Ok(Py::new(
                py,
                PyPolynomial::from_rational(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
            ExteriorElementInner::Residue(v) => Ok(Py::new(
                py,
                PyPolynomial::from_residue(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
            ExteriorElementInner::FiniteField(v) => Ok(Py::new(
                py,
                PyPolynomial::from_finite_field(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
            ExteriorElementInner::QuotientRational(v) => Ok(Py::new(
                py,
                PyQuotientElement::from_rational(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
            ExteriorElementInner::QuotientFiniteField(v) => Ok(Py::new(
                py,
                PyQuotientElement::from_finite_field(v.coefficient(basis).map_err(exterior_error)?),
            )?
            .into_any()),
        }
    }

    /// Exact wedge product; material sparse work runs without the Python GIL.
    fn wedge(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: binary detaches the Rust operation.
        self.binary(py, rhs, "wedge")
    }

    /// Multiply every term by a scalar in the exact coefficient context.
    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            ExteriorElementInner::ScalarInteger(v) => {
                let scalar = extract_integer(scalar)?;
                ExteriorElementInner::ScalarInteger(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::ScalarResidue(v) => {
                let scalar = extract_residue_scalar(v.algebra().ring(), scalar)?;
                ExteriorElementInner::ScalarResidue(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::Integer(v) => {
                let scalar = extract_integer_polynomial(v.algebra().ring(), scalar)?;
                ExteriorElementInner::Integer(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::Rational(v) => {
                let scalar = extract_rational_polynomial(py, v.algebra().ring(), scalar)?;
                ExteriorElementInner::Rational(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::Residue(v) => {
                let scalar = extract_residue_polynomial(v.algebra().ring(), scalar)?;
                ExteriorElementInner::Residue(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::FiniteField(v) => {
                let scalar = extract_finite_field_polynomial(v.algebra().ring(), scalar)?;
                ExteriorElementInner::FiniteField(Box::new(
                    py.detach(|| v.scalar_mul(&scalar))
                        .map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::QuotientRational(v) => {
                let scalar = scalar.extract::<PyRef<'_, PyQuotientElement>>()?;
                let QuotientElementInner::Rational(scalar) = &scalar.inner else {
                    return Err(PyTypeError::new_err(
                        "quotient coefficient context mismatch",
                    ));
                };
                ExteriorElementInner::QuotientRational(Box::new(
                    py.detach(|| v.scalar_mul(scalar)).map_err(exterior_error)?,
                ))
            }
            ExteriorElementInner::QuotientFiniteField(v) => {
                let scalar = scalar.extract::<PyRef<'_, PyQuotientElement>>()?;
                let QuotientElementInner::FiniteField(scalar) = &scalar.inner else {
                    return Err(PyTypeError::new_err(
                        "quotient coefficient context mismatch",
                    ));
                };
                ExteriorElementInner::QuotientFiniteField(Box::new(
                    py.detach(|| v.scalar_mul(scalar)).map_err(exterior_error)?,
                ))
            }
        };
        Ok(Self { inner })
    }

    /// Contract with a covector from the exact dual based module.
    fn contract(&self, py: Python<'_>, covector: &PyModuleElement) -> PyResult<Self> {
        macro_rules! apply {
            ($value:expr, $covector:expr, $variant:ident) => {
                ExteriorElementInner::$variant(Box::new(
                    py.detach(|| $value.contract($covector))
                        .map_err(exterior_error)?,
                ))
            };
        }
        let inner = match (&self.inner, &covector.inner) {
            (ExteriorElementInner::ScalarInteger(v), ModuleElementInner::ScalarInteger(c)) => {
                apply!(v, c, ScalarInteger)
            }
            (ExteriorElementInner::ScalarResidue(v), ModuleElementInner::ScalarResidue(c)) => {
                apply!(v, c, ScalarResidue)
            }
            (ExteriorElementInner::Integer(v), ModuleElementInner::Integer(c)) => {
                apply!(v, c, Integer)
            }
            (ExteriorElementInner::Rational(v), ModuleElementInner::Rational(c)) => {
                apply!(v, c, Rational)
            }
            (ExteriorElementInner::Residue(v), ModuleElementInner::Residue(c)) => {
                apply!(v, c, Residue)
            }
            (ExteriorElementInner::FiniteField(v), ModuleElementInner::FiniteField(c)) => {
                apply!(v, c, FiniteField)
            }
            (
                ExteriorElementInner::QuotientRational(v),
                ModuleElementInner::QuotientRational(c),
            ) => apply!(v, c, QuotientRational),
            (
                ExteriorElementInner::QuotientFiniteField(v),
                ModuleElementInner::QuotientFiniteField(c),
            ) => apply!(v, c, QuotientFiniteField),
            _ => {
                return Err(PyTypeError::new_err(
                    "covector coefficient context mismatch",
                ))
            }
        };
        Ok(Self { inner })
    }

    /// Convert a homogeneous value to canonical exterior-power coordinates.
    fn to_power_element(&self, py: Python<'_>, degree: usize) -> PyResult<PyModuleElement> {
        let inner = match &self.inner {
            ExteriorElementInner::ScalarInteger(v) => ModuleElementInner::ScalarInteger(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::ScalarResidue(v) => ModuleElementInner::ScalarResidue(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::Integer(v) => ModuleElementInner::Integer(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::Rational(v) => ModuleElementInner::Rational(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::Residue(v) => ModuleElementInner::Residue(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::FiniteField(v) => ModuleElementInner::FiniteField(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::QuotientRational(v) => ModuleElementInner::QuotientRational(
                py.detach(|| v.to_power_element(degree))
                    .map_err(exterior_error)?,
            ),
            ExteriorElementInner::QuotientFiniteField(v) => {
                ModuleElementInner::QuotientFiniteField(
                    py.detach(|| v.to_power_element(degree))
                        .map_err(exterior_error)?,
                )
            }
        };
        Ok(PyModuleElement { inner })
    }

    /// Replay canonicality and coefficient-context checks.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| match_exterior_element!(&self.inner, |value| value.verify()))
            .map_err(exterior_error)
    }

    /// Serialize the value as faithful typed JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| match_exterior_element!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }

    /// Restore and revalidate a canonical exterior element from typed JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| serde_json::from_str(value))
            .map_err(value_error)
    }

    fn __add__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: binary detaches the Rust operation.
        self.binary(py, rhs, "add")
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: binary detaches the Rust operation.
        self.binary(py, rhs, "sub")
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = match &self.inner {
            ExteriorElementInner::ScalarInteger(v) => ExteriorElementInner::ScalarInteger(
                Box::new(py.detach(|| v.neg()).map_err(exterior_error)?),
            ),
            ExteriorElementInner::ScalarResidue(v) => ExteriorElementInner::ScalarResidue(
                Box::new(py.detach(|| v.neg()).map_err(exterior_error)?),
            ),
            ExteriorElementInner::Integer(v) => ExteriorElementInner::Integer(Box::new(
                py.detach(|| v.neg()).map_err(exterior_error)?,
            )),
            ExteriorElementInner::Rational(v) => ExteriorElementInner::Rational(Box::new(
                py.detach(|| v.neg()).map_err(exterior_error)?,
            )),
            ExteriorElementInner::Residue(v) => ExteriorElementInner::Residue(Box::new(
                py.detach(|| v.neg()).map_err(exterior_error)?,
            )),
            ExteriorElementInner::FiniteField(v) => ExteriorElementInner::FiniteField(Box::new(
                py.detach(|| v.neg()).map_err(exterior_error)?,
            )),
            ExteriorElementInner::QuotientRational(v) => ExteriorElementInner::QuotientRational(
                Box::new(py.detach(|| v.neg()).map_err(exterior_error)?),
            ),
            ExteriorElementInner::QuotientFiniteField(v) => {
                ExteriorElementInner::QuotientFiniteField(Box::new(
                    py.detach(|| v.neg()).map_err(exterior_error)?,
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: wedge and scalar_mul detach the Rust operation.
        if let Ok(rhs) = rhs.extract::<PyRef<'_, Self>>() {
            self.wedge(py, &rhs)
        } else {
            self.scalar_mul(py, rhs)
        }
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: scalar_mul detaches the Rust operation.
        self.scalar_mul(py, scalar)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match_exterior_element!(&self.inner, |value| canonical_hash(value.as_ref()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(
            match_exterior_element!(&self.inner, |value| value.to_string())
        )
    }
}

/// Symmetric or exterior module-power kind.
#[pyclass(
    module = "noether.homalg",
    name = "ModulePowerKind",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyModulePowerKind {
    inner: ModulePowerKind,
}

#[pymethods]
impl PyModulePowerKind {
    #[classattr]
    fn type_id() -> &'static str {
        ModulePowerKind::TYPE_ID
    }
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "symmetric" => ModulePowerKind::Symmetric,
            "exterior" => ModulePowerKind::Exterior,
            _ => return Err(PyValueError::new_err("expected 'symmetric' or 'exterior'")),
        };
        Ok(Self { inner })
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn SYMMETRIC() -> Self {
        Self {
            inner: ModulePowerKind::Symmetric,
        }
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn EXTERIOR() -> Self {
        Self {
            inner: ModulePowerKind::Exterior,
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
            inner: ModulePowerKind::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("ModulePowerKind({})", self.inner.name()))
    }
}

macro_rules! match_free_power {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            FreeModulePowerInner::ScalarInteger($value) => $body,
            FreeModulePowerInner::ScalarResidue($value) => $body,
            FreeModulePowerInner::Integer($value) => $body,
            FreeModulePowerInner::Rational($value) => $body,
            FreeModulePowerInner::Residue($value) => $body,
            FreeModulePowerInner::FiniteField($value) => $body,
            FreeModulePowerInner::QuotientRational($value) => $body,
            FreeModulePowerInner::QuotientFiniteField($value) => $body,
        }
    };
}

macro_rules! match_presented_power {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            PresentedModulePowerInner::ScalarInteger($value) => $body,
            PresentedModulePowerInner::ScalarResidue($value) => $body,
            PresentedModulePowerInner::Integer($value) => $body,
            PresentedModulePowerInner::Rational($value) => $body,
            PresentedModulePowerInner::Residue($value) => $body,
            PresentedModulePowerInner::FiniteField($value) => $body,
            PresentedModulePowerInner::QuotientRational($value) => $body,
            PresentedModulePowerInner::QuotientFiniteField($value) => $body,
        }
    };
}

#[derive(Clone, PartialEq, Eq)]
enum FreeModulePowerInner {
    ScalarInteger(Box<FreeModulePower<IntegerRing>>),
    ScalarResidue(Box<FreeModulePower<ResidueRing>>),
    Integer(Box<FreeModulePower<PolynomialRing<IntegerRing>>>),
    Rational(Box<FreeModulePower<PolynomialRing<RationalField>>>),
    Residue(Box<FreeModulePower<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<FreeModulePower<PolynomialRing<FiniteField>>>),
    QuotientRational(Box<FreeModulePower<QuotientRing<PolynomialRing<RationalField>>>>),
    QuotientFiniteField(Box<FreeModulePower<QuotientRing<PolynomialRing<FiniteField>>>>),
}

/// A canonical symmetric or exterior power of a based free module.
#[pyclass(
    module = "noether.homalg",
    name = "FreeModulePower",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFreeModulePower {
    inner: FreeModulePowerInner,
}

impl PyFreeModulePower {
    pub(crate) fn from_scalar_integer(inner: FreeModulePower<IntegerRing>) -> Self {
        Self {
            inner: FreeModulePowerInner::ScalarInteger(Box::new(inner)),
        }
    }
    pub(crate) fn from_scalar_residue(inner: FreeModulePower<ResidueRing>) -> Self {
        Self {
            inner: FreeModulePowerInner::ScalarResidue(Box::new(inner)),
        }
    }
    pub(crate) fn from_integer(inner: FreeModulePower<PolynomialRing<IntegerRing>>) -> Self {
        Self {
            inner: FreeModulePowerInner::Integer(Box::new(inner)),
        }
    }
    pub(crate) fn from_rational(inner: FreeModulePower<PolynomialRing<RationalField>>) -> Self {
        Self {
            inner: FreeModulePowerInner::Rational(Box::new(inner)),
        }
    }
    pub(crate) fn from_finite_field(inner: FreeModulePower<PolynomialRing<FiniteField>>) -> Self {
        Self {
            inner: FreeModulePowerInner::FiniteField(Box::new(inner)),
        }
    }
    pub(crate) fn from_residue(inner: FreeModulePower<PolynomialRing<ResidueRing>>) -> Self {
        Self {
            inner: FreeModulePowerInner::Residue(Box::new(inner)),
        }
    }
    pub(crate) fn from_quotient_rational(
        inner: FreeModulePower<QuotientRing<PolynomialRing<RationalField>>>,
    ) -> Self {
        Self {
            inner: FreeModulePowerInner::QuotientRational(Box::new(inner)),
        }
    }
    pub(crate) fn from_quotient_finite_field(
        inner: FreeModulePower<QuotientRing<PolynomialRing<FiniteField>>>,
    ) -> Self {
        Self {
            inner: FreeModulePowerInner::QuotientFiniteField(Box::new(inner)),
        }
    }
}

#[pymethods]
impl PyFreeModulePower {
    #[classattr]
    fn type_id() -> &'static str {
        FreeModulePower::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[getter]
    fn source(&self, py: Python<'_>) -> PyFreeModule {
        // NOETHER_RELEASES_GIL: basis metadata can be material.
        py.detach(|| match &self.inner {
            FreeModulePowerInner::ScalarInteger(value) => PyFreeModule {
                inner: FreeModuleInner::ScalarInteger(value.source().clone()),
            },
            FreeModulePowerInner::ScalarResidue(value) => PyFreeModule {
                inner: FreeModuleInner::ScalarResidue(value.source().clone()),
            },
            FreeModulePowerInner::Integer(value) => PyFreeModule {
                inner: FreeModuleInner::Integer(value.source().clone()),
            },
            FreeModulePowerInner::Rational(value) => PyFreeModule {
                inner: FreeModuleInner::Rational(value.source().clone()),
            },
            FreeModulePowerInner::Residue(value) => PyFreeModule {
                inner: FreeModuleInner::Residue(value.source().clone()),
            },
            FreeModulePowerInner::FiniteField(value) => PyFreeModule {
                inner: FreeModuleInner::FiniteField(value.source().clone()),
            },
            FreeModulePowerInner::QuotientRational(value) => PyFreeModule {
                inner: FreeModuleInner::QuotientRational(value.source().clone()),
            },
            FreeModulePowerInner::QuotientFiniteField(value) => PyFreeModule {
                inner: FreeModuleInner::QuotientFiniteField(value.source().clone()),
            },
        })
    }
    #[getter]
    fn kind(&self) -> PyModulePowerKind {
        PyModulePowerKind {
            inner: match_free_power!(&self.inner, |value| value.kind()),
        }
    }
    #[getter]
    fn degree(&self) -> usize {
        match_free_power!(&self.inner, |value| value.degree())
    }
    #[getter]
    fn basis_exponents(&self, py: Python<'_>) -> Vec<Vec<usize>> {
        // NOETHER_RELEASES_GIL: this copies the complete canonical basis table.
        py.detach(|| match_free_power!(&self.inner, |value| value.basis_exponents().to_vec()))
    }
    #[getter]
    fn module(&self, py: Python<'_>) -> PyFreeModule {
        // NOETHER_RELEASES_GIL: basis metadata can be material.
        py.detach(|| match &self.inner {
            FreeModulePowerInner::ScalarInteger(value) => PyFreeModule {
                inner: FreeModuleInner::ScalarInteger(value.module().clone()),
            },
            FreeModulePowerInner::ScalarResidue(value) => PyFreeModule {
                inner: FreeModuleInner::ScalarResidue(value.module().clone()),
            },
            FreeModulePowerInner::Integer(value) => PyFreeModule {
                inner: FreeModuleInner::Integer(value.module().clone()),
            },
            FreeModulePowerInner::Rational(value) => PyFreeModule {
                inner: FreeModuleInner::Rational(value.module().clone()),
            },
            FreeModulePowerInner::Residue(value) => PyFreeModule {
                inner: FreeModuleInner::Residue(value.module().clone()),
            },
            FreeModulePowerInner::FiniteField(value) => PyFreeModule {
                inner: FreeModuleInner::FiniteField(value.module().clone()),
            },
            FreeModulePowerInner::QuotientRational(value) => PyFreeModule {
                inner: FreeModuleInner::QuotientRational(value.module().clone()),
            },
            FreeModulePowerInner::QuotientFiniteField(value) => PyFreeModule {
                inner: FreeModuleInner::QuotientFiniteField(value.module().clone()),
            },
        })
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // NOETHER_RELEASES_GIL: verification reconstructs the canonical power.
        py.detach(|| match_free_power!(&self.inner, |value| value.verify()))
            .map_err(module_power_error)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: serialization enumerates the canonical basis table.
        py.detach(|| match_free_power!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        // NOETHER_RELEASES_GIL: deserialization replays the complete construction.
        py.detach(move || {
            if let Some(ring) = serialized_scalar_module_ring(&value, "/source/ring")? {
                return match ring {
                    SerializedScalarModuleRing::Integer => Ok(Self::from_scalar_integer(
                        FreeModulePower::<IntegerRing>::from_json(&value).map_err(value_error)?,
                    )),
                    SerializedScalarModuleRing::Residue => Ok(Self::from_scalar_residue(
                        FreeModulePower::<ResidueRing>::from_json(&value).map_err(value_error)?,
                    )),
                };
            }
            match serialized_module_ring(&value, "/source/ring")? {
                SerializedModuleRing::PolynomialInteger => Ok(Self::from_integer(
                    FreeModulePower::<PolynomialRing<IntegerRing>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialRational => Ok(Self::from_rational(
                    FreeModulePower::<PolynomialRing<RationalField>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialResidue => Ok(Self::from_residue(
                    FreeModulePower::<PolynomialRing<ResidueRing>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialFiniteField => Ok(Self::from_finite_field(
                    FreeModulePower::<PolynomialRing<FiniteField>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::QuotientRational => Ok(Self::from_quotient_rational(
                    FreeModulePower::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                        &value,
                    )
                    .map_err(value_error)?,
                )),
                SerializedModuleRing::QuotientFiniteField => Ok(Self::from_quotient_finite_field(
                    FreeModulePower::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(&value)
                        .map_err(value_error)?,
                )),
            }
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match_free_power!(&self.inner, |value| canonical_hash(value.as_ref()))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match_free_power!(&self.inner, |value| value.to_string()))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PresentedModulePowerInner {
    ScalarInteger(Box<PresentedModulePower<IntegerRing>>),
    ScalarResidue(Box<PresentedModulePower<ResidueRing>>),
    Integer(Box<PresentedModulePower<PolynomialRing<IntegerRing>>>),
    Rational(Box<PresentedModulePower<PolynomialRing<RationalField>>>),
    Residue(Box<PresentedModulePower<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<PresentedModulePower<PolynomialRing<FiniteField>>>),
    QuotientRational(Box<PresentedModulePower<QuotientRing<PolynomialRing<RationalField>>>>),
    QuotientFiniteField(Box<PresentedModulePower<QuotientRing<PolynomialRing<FiniteField>>>>),
}

/// A canonical symmetric or exterior power presentation.
#[pyclass(
    module = "noether.homalg",
    name = "PresentedModulePower",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPresentedModulePower {
    inner: PresentedModulePowerInner,
}

impl PyPresentedModulePower {
    pub(crate) fn from_scalar_integer(inner: PresentedModulePower<IntegerRing>) -> Self {
        Self {
            inner: PresentedModulePowerInner::ScalarInteger(Box::new(inner)),
        }
    }
    pub(crate) fn from_scalar_residue(inner: PresentedModulePower<ResidueRing>) -> Self {
        Self {
            inner: PresentedModulePowerInner::ScalarResidue(Box::new(inner)),
        }
    }
    pub(crate) fn from_integer(inner: PresentedModulePower<PolynomialRing<IntegerRing>>) -> Self {
        Self {
            inner: PresentedModulePowerInner::Integer(Box::new(inner)),
        }
    }
    pub(crate) fn from_rational(
        inner: PresentedModulePower<PolynomialRing<RationalField>>,
    ) -> Self {
        Self {
            inner: PresentedModulePowerInner::Rational(Box::new(inner)),
        }
    }
    pub(crate) fn from_finite_field(
        inner: PresentedModulePower<PolynomialRing<FiniteField>>,
    ) -> Self {
        Self {
            inner: PresentedModulePowerInner::FiniteField(Box::new(inner)),
        }
    }
    pub(crate) fn from_residue(inner: PresentedModulePower<PolynomialRing<ResidueRing>>) -> Self {
        Self {
            inner: PresentedModulePowerInner::Residue(Box::new(inner)),
        }
    }
    pub(crate) fn from_quotient_rational(
        inner: PresentedModulePower<QuotientRing<PolynomialRing<RationalField>>>,
    ) -> Self {
        Self {
            inner: PresentedModulePowerInner::QuotientRational(Box::new(inner)),
        }
    }
    pub(crate) fn from_quotient_finite_field(
        inner: PresentedModulePower<QuotientRing<PolynomialRing<FiniteField>>>,
    ) -> Self {
        Self {
            inner: PresentedModulePowerInner::QuotientFiniteField(Box::new(inner)),
        }
    }
}

macro_rules! match_power_map {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            ModulePowerHomomorphismInner::ScalarInteger($value) => $body,
            ModulePowerHomomorphismInner::ScalarResidue($value) => $body,
            ModulePowerHomomorphismInner::Integer($value) => $body,
            ModulePowerHomomorphismInner::Rational($value) => $body,
            ModulePowerHomomorphismInner::Residue($value) => $body,
            ModulePowerHomomorphismInner::FiniteField($value) => $body,
            ModulePowerHomomorphismInner::QuotientRational($value) => $body,
            ModulePowerHomomorphismInner::QuotientFiniteField($value) => $body,
        }
    };
}

#[derive(Clone, PartialEq, Eq)]
enum ModulePowerHomomorphismInner {
    ScalarInteger(Box<ModulePowerHomomorphism<IntegerRing>>),
    ScalarResidue(Box<ModulePowerHomomorphism<ResidueRing>>),
    Integer(Box<ModulePowerHomomorphism<PolynomialRing<IntegerRing>>>),
    Rational(Box<ModulePowerHomomorphism<PolynomialRing<RationalField>>>),
    Residue(Box<ModulePowerHomomorphism<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<ModulePowerHomomorphism<PolynomialRing<FiniteField>>>),
    QuotientRational(Box<ModulePowerHomomorphism<QuotientRing<PolynomialRing<RationalField>>>>),
    QuotientFiniteField(Box<ModulePowerHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>>),
}

/// A replay-verifiable map induced on a symmetric or exterior power.
#[pyclass(
    module = "noether.homalg",
    name = "ModulePowerHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModulePowerHomomorphism {
    inner: ModulePowerHomomorphismInner,
}

impl PyModulePowerHomomorphism {
    pub(super) fn from_map(
        map: &ModuleHomomorphismInner,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let inner =
            match map {
                ModuleHomomorphismInner::ScalarInteger(value) => {
                    ModulePowerHomomorphismInner::ScalarInteger(Box::new(
                        ModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                    ))
                }
                ModuleHomomorphismInner::ScalarResidue(value) => {
                    ModulePowerHomomorphismInner::ScalarResidue(Box::new(
                        ModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                    ))
                }
                ModuleHomomorphismInner::Integer(value) => ModulePowerHomomorphismInner::Integer(
                    Box::new(ModulePowerHomomorphism::new(value.clone(), kind, degree)?),
                ),
                ModuleHomomorphismInner::Rational(value) => ModulePowerHomomorphismInner::Rational(
                    Box::new(ModulePowerHomomorphism::new(value.clone(), kind, degree)?),
                ),
                ModuleHomomorphismInner::Residue(value) => ModulePowerHomomorphismInner::Residue(
                    Box::new(ModulePowerHomomorphism::new(value.clone(), kind, degree)?),
                ),
                ModuleHomomorphismInner::FiniteField(value) => {
                    ModulePowerHomomorphismInner::FiniteField(Box::new(
                        ModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                    ))
                }
                ModuleHomomorphismInner::QuotientRational(value) => {
                    ModulePowerHomomorphismInner::QuotientRational(Box::new(
                        ModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                    ))
                }
                ModuleHomomorphismInner::QuotientFiniteField(value) => {
                    ModulePowerHomomorphismInner::QuotientFiniteField(Box::new(
                        ModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                    ))
                }
            };
        Ok(Self { inner })
    }
}

impl Serialize for PyModulePowerHomomorphism {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match_power_map!(&self.inner, |value| value.serialize(serializer))
    }
}

impl<'de> Deserialize<'de> for PyModulePowerHomomorphism {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = serde_json::Value::deserialize(deserializer)?;
        macro_rules! attempt {
            ($type:ty, $variant:ident) => {
                if let Ok(value) = serde_json::from_value::<$type>(payload.clone()) {
                    return Ok(Self {
                        inner: ModulePowerHomomorphismInner::$variant(Box::new(value)),
                    });
                }
            };
        }
        attempt!(ModulePowerHomomorphism<IntegerRing>, ScalarInteger);
        attempt!(ModulePowerHomomorphism<ResidueRing>, ScalarResidue);
        attempt!(
            ModulePowerHomomorphism<PolynomialRing<IntegerRing>>,
            Integer
        );
        attempt!(
            ModulePowerHomomorphism<PolynomialRing<RationalField>>,
            Rational
        );
        attempt!(
            ModulePowerHomomorphism<PolynomialRing<ResidueRing>>,
            Residue
        );
        attempt!(
            ModulePowerHomomorphism<PolynomialRing<FiniteField>>,
            FiniteField
        );
        attempt!(
            ModulePowerHomomorphism<QuotientRing<PolynomialRing<RationalField>>>,
            QuotientRational
        );
        attempt!(
            ModulePowerHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>,
            QuotientFiniteField
        );
        Err(serde::de::Error::custom(
            "invalid induced module-power map or unsupported coefficient context",
        ))
    }
}

#[pymethods]
impl PyModulePowerHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        ModulePowerHomomorphism::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Degree-one linear map from which this evidence was computed.
    #[getter]
    fn source_map(&self) -> PyModuleHomomorphism {
        let inner = match &self.inner {
            ModulePowerHomomorphismInner::ScalarInteger(v) => {
                ModuleHomomorphismInner::ScalarInteger(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::ScalarResidue(v) => {
                ModuleHomomorphismInner::ScalarResidue(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::Integer(v) => {
                ModuleHomomorphismInner::Integer(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::Rational(v) => {
                ModuleHomomorphismInner::Rational(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::Residue(v) => {
                ModuleHomomorphismInner::Residue(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::FiniteField(v) => {
                ModuleHomomorphismInner::FiniteField(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::QuotientRational(v) => {
                ModuleHomomorphismInner::QuotientRational(v.source_map().clone())
            }
            ModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                ModuleHomomorphismInner::QuotientFiniteField(v.source_map().clone())
            }
        };
        PyModuleHomomorphism { inner }
    }

    /// Whether this is symmetric- or exterior-power evidence.
    #[getter]
    fn kind(&self) -> PyModulePowerKind {
        PyModulePowerKind {
            inner: match_power_map!(&self.inner, |value| value.kind()),
        }
    }

    /// Homogeneous power degree.
    #[getter]
    fn degree(&self) -> usize {
        match_power_map!(&self.inner, |value| value.degree())
    }

    /// Canonical source power and its ordered coordinate basis.
    #[getter]
    fn source_power(&self, py: Python<'_>) -> PyFreeModulePower {
        // NOETHER_RELEASES_GIL: cloning currently copies the canonical basis table.
        py.detach(|| match &self.inner {
            ModulePowerHomomorphismInner::ScalarInteger(v) => {
                PyFreeModulePower::from_scalar_integer(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::ScalarResidue(v) => {
                PyFreeModulePower::from_scalar_residue(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::Integer(v) => {
                PyFreeModulePower::from_integer(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::Rational(v) => {
                PyFreeModulePower::from_rational(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::Residue(v) => {
                PyFreeModulePower::from_residue(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::FiniteField(v) => {
                PyFreeModulePower::from_finite_field(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::QuotientRational(v) => {
                PyFreeModulePower::from_quotient_rational(v.source_power().clone())
            }
            ModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PyFreeModulePower::from_quotient_finite_field(v.source_power().clone())
            }
        })
    }

    /// Canonical target power and its ordered coordinate basis.
    #[getter]
    fn target_power(&self, py: Python<'_>) -> PyFreeModulePower {
        // NOETHER_RELEASES_GIL: cloning currently copies the canonical basis table.
        py.detach(|| match &self.inner {
            ModulePowerHomomorphismInner::ScalarInteger(v) => {
                PyFreeModulePower::from_scalar_integer(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::ScalarResidue(v) => {
                PyFreeModulePower::from_scalar_residue(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::Integer(v) => {
                PyFreeModulePower::from_integer(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::Rational(v) => {
                PyFreeModulePower::from_rational(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::Residue(v) => {
                PyFreeModulePower::from_residue(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::FiniteField(v) => {
                PyFreeModulePower::from_finite_field(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::QuotientRational(v) => {
                PyFreeModulePower::from_quotient_rational(v.target_power().clone())
            }
            ModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PyFreeModulePower::from_quotient_finite_field(v.target_power().clone())
            }
        })
    }

    /// Fully materialized induced coordinate map.
    #[getter]
    fn induced(&self) -> PyModuleHomomorphism {
        let inner = match &self.inner {
            ModulePowerHomomorphismInner::ScalarInteger(v) => {
                ModuleHomomorphismInner::ScalarInteger(v.induced().clone())
            }
            ModulePowerHomomorphismInner::ScalarResidue(v) => {
                ModuleHomomorphismInner::ScalarResidue(v.induced().clone())
            }
            ModulePowerHomomorphismInner::Integer(v) => {
                ModuleHomomorphismInner::Integer(v.induced().clone())
            }
            ModulePowerHomomorphismInner::Rational(v) => {
                ModuleHomomorphismInner::Rational(v.induced().clone())
            }
            ModulePowerHomomorphismInner::Residue(v) => {
                ModuleHomomorphismInner::Residue(v.induced().clone())
            }
            ModulePowerHomomorphismInner::FiniteField(v) => {
                ModuleHomomorphismInner::FiniteField(v.induced().clone())
            }
            ModulePowerHomomorphismInner::QuotientRational(v) => {
                ModuleHomomorphismInner::QuotientRational(v.induced().clone())
            }
            ModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                ModuleHomomorphismInner::QuotientFiniteField(v.induced().clone())
            }
        };
        PyModuleHomomorphism { inner }
    }

    /// Compose compatible induced maps and replay functoriality evidence.
    fn compose(&self, py: Python<'_>, inner: &Self) -> PyResult<Self> {
        macro_rules! apply {
            ($outer:expr, $inner:expr, $variant:ident) => {
                ModulePowerHomomorphismInner::$variant(Box::new(
                    py.detach(|| $outer.compose($inner))
                        .map_err(module_power_error)?,
                ))
            };
        }
        let value = match (&self.inner, &inner.inner) {
            (
                ModulePowerHomomorphismInner::ScalarInteger(a),
                ModulePowerHomomorphismInner::ScalarInteger(b),
            ) => apply!(a, b, ScalarInteger),
            (
                ModulePowerHomomorphismInner::ScalarResidue(a),
                ModulePowerHomomorphismInner::ScalarResidue(b),
            ) => apply!(a, b, ScalarResidue),
            (
                ModulePowerHomomorphismInner::Integer(a),
                ModulePowerHomomorphismInner::Integer(b),
            ) => apply!(a, b, Integer),
            (
                ModulePowerHomomorphismInner::Rational(a),
                ModulePowerHomomorphismInner::Rational(b),
            ) => apply!(a, b, Rational),
            (
                ModulePowerHomomorphismInner::Residue(a),
                ModulePowerHomomorphismInner::Residue(b),
            ) => apply!(a, b, Residue),
            (
                ModulePowerHomomorphismInner::FiniteField(a),
                ModulePowerHomomorphismInner::FiniteField(b),
            ) => apply!(a, b, FiniteField),
            (
                ModulePowerHomomorphismInner::QuotientRational(a),
                ModulePowerHomomorphismInner::QuotientRational(b),
            ) => apply!(a, b, QuotientRational),
            (
                ModulePowerHomomorphismInner::QuotientFiniteField(a),
                ModulePowerHomomorphismInner::QuotientFiniteField(b),
            ) => apply!(a, b, QuotientFiniteField),
            _ => return Err(PyValueError::new_err("induced-map contexts do not match")),
        };
        Ok(Self { inner: value })
    }

    /// Apply exterior-power evidence to the matching homogeneous component.
    fn apply_exterior(
        &self,
        py: Python<'_>,
        element: &PyExteriorElement,
    ) -> PyResult<PyExteriorElement> {
        macro_rules! apply {
            ($map:expr, $element:expr, $variant:ident) => {
                ExteriorElementInner::$variant(Box::new(
                    py.detach(|| $map.apply_exterior($element))
                        .map_err(module_power_error)?,
                ))
            };
        }
        let inner = match (&self.inner, &element.inner) {
            (
                ModulePowerHomomorphismInner::ScalarInteger(m),
                ExteriorElementInner::ScalarInteger(e),
            ) => apply!(m, e, ScalarInteger),
            (
                ModulePowerHomomorphismInner::ScalarResidue(m),
                ExteriorElementInner::ScalarResidue(e),
            ) => apply!(m, e, ScalarResidue),
            (ModulePowerHomomorphismInner::Integer(m), ExteriorElementInner::Integer(e)) => {
                apply!(m, e, Integer)
            }
            (ModulePowerHomomorphismInner::Rational(m), ExteriorElementInner::Rational(e)) => {
                apply!(m, e, Rational)
            }
            (ModulePowerHomomorphismInner::Residue(m), ExteriorElementInner::Residue(e)) => {
                apply!(m, e, Residue)
            }
            (
                ModulePowerHomomorphismInner::FiniteField(m),
                ExteriorElementInner::FiniteField(e),
            ) => apply!(m, e, FiniteField),
            (
                ModulePowerHomomorphismInner::QuotientRational(m),
                ExteriorElementInner::QuotientRational(e),
            ) => apply!(m, e, QuotientRational),
            (
                ModulePowerHomomorphismInner::QuotientFiniteField(m),
                ExteriorElementInner::QuotientFiniteField(e),
            ) => apply!(m, e, QuotientFiniteField),
            _ => {
                return Err(PyTypeError::new_err(
                    "induced map and exterior element contexts do not match",
                ))
            }
        };
        Ok(PyExteriorElement { inner })
    }

    /// Recompute the induced matrix and reject stale or tampered evidence.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| match_power_map!(&self.inner, |value| value.verify()))
            .map_err(module_power_error)
    }

    /// Serialize replay-verifiable evidence as faithful typed JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| match_power_map!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }

    /// Restore and replay-verify induced-map evidence from faithful typed JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| serde_json::from_str(value))
            .map_err(value_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        match_power_map!(&self.inner, |value| canonical_hash(value.as_ref()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match_power_map!(&self.inner, |value| value.to_string()))
    }
}

macro_rules! match_presented_power_map {
    ($inner:expr, |$value:ident| $body:expr) => {
        match $inner {
            PresentedModulePowerHomomorphismInner::ScalarInteger($value) => $body,
            PresentedModulePowerHomomorphismInner::ScalarResidue($value) => $body,
            PresentedModulePowerHomomorphismInner::Integer($value) => $body,
            PresentedModulePowerHomomorphismInner::Rational($value) => $body,
            PresentedModulePowerHomomorphismInner::Residue($value) => $body,
            PresentedModulePowerHomomorphismInner::FiniteField($value) => $body,
            PresentedModulePowerHomomorphismInner::QuotientRational($value) => $body,
            PresentedModulePowerHomomorphismInner::QuotientFiniteField($value) => $body,
        }
    };
}

#[derive(Clone, PartialEq, Eq)]
enum PresentedModulePowerHomomorphismInner {
    ScalarInteger(Box<PresentedModulePowerHomomorphism<IntegerRing>>),
    ScalarResidue(Box<PresentedModulePowerHomomorphism<ResidueRing>>),
    Integer(Box<PresentedModulePowerHomomorphism<PolynomialRing<IntegerRing>>>),
    Rational(Box<PresentedModulePowerHomomorphism<PolynomialRing<RationalField>>>),
    Residue(Box<PresentedModulePowerHomomorphism<PolynomialRing<ResidueRing>>>),
    FiniteField(Box<PresentedModulePowerHomomorphism<PolynomialRing<FiniteField>>>),
    QuotientRational(
        Box<PresentedModulePowerHomomorphism<QuotientRing<PolynomialRing<RationalField>>>>,
    ),
    QuotientFiniteField(
        Box<PresentedModulePowerHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>>,
    ),
}

/// A replay-verifiable induced map between symmetric or exterior presentations.
#[pyclass(
    module = "noether.homalg",
    name = "PresentedModulePowerHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPresentedModulePowerHomomorphism {
    inner: PresentedModulePowerHomomorphismInner,
}

impl PyPresentedModulePowerHomomorphism {
    pub(super) fn from_map(
        map: &PresentedModuleHomomorphismInner,
        kind: ModulePowerKind,
        degree: usize,
    ) -> Result<Self, ModulePowerError> {
        let inner = match map {
            PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                PresentedModulePowerHomomorphismInner::ScalarInteger(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                PresentedModulePowerHomomorphismInner::ScalarResidue(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::Integer(value) => {
                PresentedModulePowerHomomorphismInner::Integer(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::Rational(value) => {
                PresentedModulePowerHomomorphismInner::Rational(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::Residue(value) => {
                PresentedModulePowerHomomorphismInner::Residue(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::FiniteField(value) => {
                PresentedModulePowerHomomorphismInner::FiniteField(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::QuotientRational(value) => {
                PresentedModulePowerHomomorphismInner::QuotientRational(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
            PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                PresentedModulePowerHomomorphismInner::QuotientFiniteField(Box::new(
                    PresentedModulePowerHomomorphism::new(value.clone(), kind, degree)?,
                ))
            }
        };
        Ok(Self { inner })
    }
}

impl Serialize for PyPresentedModulePowerHomomorphism {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match_presented_power_map!(&self.inner, |value| value.serialize(serializer))
    }
}

impl<'de> Deserialize<'de> for PyPresentedModulePowerHomomorphism {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = serde_json::Value::deserialize(deserializer)?;
        macro_rules! attempt {
            ($type:ty, $variant:ident) => {
                if let Ok(value) = serde_json::from_value::<$type>(payload.clone()) {
                    return Ok(Self {
                        inner: PresentedModulePowerHomomorphismInner::$variant(Box::new(value)),
                    });
                }
            };
        }
        attempt!(PresentedModulePowerHomomorphism<IntegerRing>, ScalarInteger);
        attempt!(PresentedModulePowerHomomorphism<ResidueRing>, ScalarResidue);
        attempt!(
            PresentedModulePowerHomomorphism<PolynomialRing<IntegerRing>>,
            Integer
        );
        attempt!(
            PresentedModulePowerHomomorphism<PolynomialRing<RationalField>>,
            Rational
        );
        attempt!(
            PresentedModulePowerHomomorphism<PolynomialRing<ResidueRing>>,
            Residue
        );
        attempt!(
            PresentedModulePowerHomomorphism<PolynomialRing<FiniteField>>,
            FiniteField
        );
        attempt!(
            PresentedModulePowerHomomorphism<QuotientRing<PolynomialRing<RationalField>>>,
            QuotientRational
        );
        attempt!(
            PresentedModulePowerHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>,
            QuotientFiniteField
        );
        Err(serde::de::Error::custom(
            "invalid presented induced-power map or unsupported coefficient context",
        ))
    }
}

#[pymethods]
impl PyPresentedModulePowerHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        PresentedModulePowerHomomorphism::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    /// Presented map from which this induced evidence was computed.
    #[getter]
    fn source_map(&self) -> PyPresentedModuleHomomorphism {
        let inner = match &self.inner {
            PresentedModulePowerHomomorphismInner::ScalarInteger(v) => {
                PresentedModuleHomomorphismInner::ScalarInteger(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::ScalarResidue(v) => {
                PresentedModuleHomomorphismInner::ScalarResidue(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::Integer(v) => {
                PresentedModuleHomomorphismInner::Integer(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::Rational(v) => {
                PresentedModuleHomomorphismInner::Rational(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::Residue(v) => {
                PresentedModuleHomomorphismInner::Residue(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::FiniteField(v) => {
                PresentedModuleHomomorphismInner::FiniteField(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientRational(v) => {
                PresentedModuleHomomorphismInner::QuotientRational(v.source_map().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PresentedModuleHomomorphismInner::QuotientFiniteField(v.source_map().clone())
            }
        };
        PyPresentedModuleHomomorphism { inner }
    }
    /// Whether this is symmetric- or exterior-power evidence.
    #[getter]
    fn kind(&self) -> PyModulePowerKind {
        PyModulePowerKind {
            inner: match_presented_power_map!(&self.inner, |value| value.kind()),
        }
    }
    /// Homogeneous power degree.
    #[getter]
    fn degree(&self) -> usize {
        match_presented_power_map!(&self.inner, |value| value.degree())
    }
    /// Verified source power presentation.
    #[getter]
    fn source_power(&self) -> PyPresentedModulePower {
        match &self.inner {
            PresentedModulePowerHomomorphismInner::ScalarInteger(v) => {
                PyPresentedModulePower::from_scalar_integer(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::ScalarResidue(v) => {
                PyPresentedModulePower::from_scalar_residue(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Integer(v) => {
                PyPresentedModulePower::from_integer(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Rational(v) => {
                PyPresentedModulePower::from_rational(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Residue(v) => {
                PyPresentedModulePower::from_residue(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::FiniteField(v) => {
                PyPresentedModulePower::from_finite_field(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientRational(v) => {
                PyPresentedModulePower::from_quotient_rational(v.source_power().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PyPresentedModulePower::from_quotient_finite_field(v.source_power().clone())
            }
        }
    }
    /// Verified target power presentation.
    #[getter]
    fn target_power(&self) -> PyPresentedModulePower {
        match &self.inner {
            PresentedModulePowerHomomorphismInner::ScalarInteger(v) => {
                PyPresentedModulePower::from_scalar_integer(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::ScalarResidue(v) => {
                PyPresentedModulePower::from_scalar_residue(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Integer(v) => {
                PyPresentedModulePower::from_integer(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Rational(v) => {
                PyPresentedModulePower::from_rational(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::Residue(v) => {
                PyPresentedModulePower::from_residue(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::FiniteField(v) => {
                PyPresentedModulePower::from_finite_field(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientRational(v) => {
                PyPresentedModulePower::from_quotient_rational(v.target_power().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PyPresentedModulePower::from_quotient_finite_field(v.target_power().clone())
            }
        }
    }
    /// Induced map with generator and relation lifts.
    #[getter]
    fn induced(&self) -> PyPresentedModuleHomomorphism {
        let inner = match &self.inner {
            PresentedModulePowerHomomorphismInner::ScalarInteger(v) => {
                PresentedModuleHomomorphismInner::ScalarInteger(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::ScalarResidue(v) => {
                PresentedModuleHomomorphismInner::ScalarResidue(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::Integer(v) => {
                PresentedModuleHomomorphismInner::Integer(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::Rational(v) => {
                PresentedModuleHomomorphismInner::Rational(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::Residue(v) => {
                PresentedModuleHomomorphismInner::Residue(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::FiniteField(v) => {
                PresentedModuleHomomorphismInner::FiniteField(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientRational(v) => {
                PresentedModuleHomomorphismInner::QuotientRational(v.induced().clone())
            }
            PresentedModulePowerHomomorphismInner::QuotientFiniteField(v) => {
                PresentedModuleHomomorphismInner::QuotientFiniteField(v.induced().clone())
            }
        };
        PyPresentedModuleHomomorphism { inner }
    }
    /// Compose compatible presented induced maps with verified relation lifts.
    fn compose(&self, py: Python<'_>, inner: &Self) -> PyResult<Self> {
        macro_rules! apply {
            ($a:expr, $b:expr, $variant:ident) => {
                PresentedModulePowerHomomorphismInner::$variant(Box::new(
                    py.detach(|| $a.compose($b)).map_err(module_power_error)?,
                ))
            };
        }
        let value = match (&self.inner, &inner.inner) {
            (
                PresentedModulePowerHomomorphismInner::ScalarInteger(a),
                PresentedModulePowerHomomorphismInner::ScalarInteger(b),
            ) => apply!(a, b, ScalarInteger),
            (
                PresentedModulePowerHomomorphismInner::ScalarResidue(a),
                PresentedModulePowerHomomorphismInner::ScalarResidue(b),
            ) => apply!(a, b, ScalarResidue),
            (
                PresentedModulePowerHomomorphismInner::Integer(a),
                PresentedModulePowerHomomorphismInner::Integer(b),
            ) => apply!(a, b, Integer),
            (
                PresentedModulePowerHomomorphismInner::Rational(a),
                PresentedModulePowerHomomorphismInner::Rational(b),
            ) => apply!(a, b, Rational),
            (
                PresentedModulePowerHomomorphismInner::Residue(a),
                PresentedModulePowerHomomorphismInner::Residue(b),
            ) => apply!(a, b, Residue),
            (
                PresentedModulePowerHomomorphismInner::FiniteField(a),
                PresentedModulePowerHomomorphismInner::FiniteField(b),
            ) => apply!(a, b, FiniteField),
            (
                PresentedModulePowerHomomorphismInner::QuotientRational(a),
                PresentedModulePowerHomomorphismInner::QuotientRational(b),
            ) => apply!(a, b, QuotientRational),
            (
                PresentedModulePowerHomomorphismInner::QuotientFiniteField(a),
                PresentedModulePowerHomomorphismInner::QuotientFiniteField(b),
            ) => apply!(a, b, QuotientFiniteField),
            _ => {
                return Err(PyValueError::new_err(
                    "presented induced-map contexts do not match",
                ))
            }
        };
        Ok(Self { inner: value })
    }
    /// Replay power presentations, relation lifts, and the commuting square.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| match_presented_power_map!(&self.inner, |value| value.verify()))
            .map_err(module_power_error)
    }
    /// Serialize replay-verifiable evidence as faithful typed JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| match_presented_power_map!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }
    /// Restore and replay-verify presented-map evidence from typed JSON.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| serde_json::from_str(value))
            .map_err(value_error)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match_presented_power_map!(&self.inner, |value| canonical_hash(value.as_ref()))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(
            match_presented_power_map!(&self.inner, |value| value.to_string())
        )
    }
}

#[pymethods]
impl PyPresentedModulePower {
    #[classattr]
    fn type_id() -> &'static str {
        PresentedModulePower::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[getter]
    fn source(&self, py: Python<'_>) -> PyPresentedModule {
        // NOETHER_RELEASES_GIL: presentation metadata can be material.
        py.detach(|| match &self.inner {
            PresentedModulePowerInner::ScalarInteger(value) => PyPresentedModule {
                inner: PresentedModuleInner::ScalarInteger(value.source().clone()),
            },
            PresentedModulePowerInner::ScalarResidue(value) => PyPresentedModule {
                inner: PresentedModuleInner::ScalarResidue(value.source().clone()),
            },
            PresentedModulePowerInner::Integer(value) => PyPresentedModule {
                inner: PresentedModuleInner::Integer(value.source().clone()),
            },
            PresentedModulePowerInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.source().clone())
            }
            PresentedModulePowerInner::Residue(value) => PyPresentedModule {
                inner: PresentedModuleInner::Residue(value.source().clone()),
            },
            PresentedModulePowerInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.source().clone())
            }
            PresentedModulePowerInner::QuotientRational(value) => PyPresentedModule {
                inner: PresentedModuleInner::QuotientRational(value.source().clone()),
            },
            PresentedModulePowerInner::QuotientFiniteField(value) => PyPresentedModule {
                inner: PresentedModuleInner::QuotientFiniteField(value.source().clone()),
            },
        })
    }
    #[getter]
    fn kind(&self) -> PyModulePowerKind {
        PyModulePowerKind {
            inner: match_presented_power!(&self.inner, |value| value.kind()),
        }
    }
    #[getter]
    fn degree(&self) -> usize {
        match_presented_power!(&self.inner, |value| value.degree())
    }
    #[getter]
    fn module(&self, py: Python<'_>) -> PyPresentedModule {
        // NOETHER_RELEASES_GIL: presentation matrices and basis metadata can be material.
        py.detach(|| match &self.inner {
            PresentedModulePowerInner::ScalarInteger(value) => PyPresentedModule {
                inner: PresentedModuleInner::ScalarInteger(value.module().clone()),
            },
            PresentedModulePowerInner::ScalarResidue(value) => PyPresentedModule {
                inner: PresentedModuleInner::ScalarResidue(value.module().clone()),
            },
            PresentedModulePowerInner::Integer(value) => PyPresentedModule {
                inner: PresentedModuleInner::Integer(value.module().clone()),
            },
            PresentedModulePowerInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.module().clone())
            }
            PresentedModulePowerInner::Residue(value) => PyPresentedModule {
                inner: PresentedModuleInner::Residue(value.module().clone()),
            },
            PresentedModulePowerInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.module().clone())
            }
            PresentedModulePowerInner::QuotientRational(value) => PyPresentedModule {
                inner: PresentedModuleInner::QuotientRational(value.module().clone()),
            },
            PresentedModulePowerInner::QuotientFiniteField(value) => PyPresentedModule {
                inner: PresentedModuleInner::QuotientFiniteField(value.module().clone()),
            },
        })
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // NOETHER_RELEASES_GIL: verification reconstructs the presentation.
        py.detach(|| match_presented_power!(&self.inner, |value| value.verify()))
            .map_err(module_power_error)
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: serialization traverses the presentation.
        py.detach(|| match_presented_power!(&self.inner, |value| value.to_json()))
            .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        // NOETHER_RELEASES_GIL: deserialization reconstructs and verifies the presentation.
        py.detach(move || {
            if let Some(ring) =
                serialized_scalar_module_ring(&value, "/source/presentation/source/ring")?
            {
                return match ring {
                    SerializedScalarModuleRing::Integer => Ok(Self::from_scalar_integer(
                        PresentedModulePower::<IntegerRing>::from_json(&value)
                            .map_err(value_error)?,
                    )),
                    SerializedScalarModuleRing::Residue => Ok(Self::from_scalar_residue(
                        PresentedModulePower::<ResidueRing>::from_json(&value)
                            .map_err(value_error)?,
                    )),
                };
            }
            match serialized_module_ring(&value, "/source/presentation/source/ring")? {
                SerializedModuleRing::PolynomialInteger => Ok(Self::from_integer(
                    PresentedModulePower::<PolynomialRing<IntegerRing>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialRational => Ok(Self::from_rational(
                    PresentedModulePower::<PolynomialRing<RationalField>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialResidue => Ok(Self::from_residue(
                    PresentedModulePower::<PolynomialRing<ResidueRing>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::PolynomialFiniteField => Ok(Self::from_finite_field(
                    PresentedModulePower::<PolynomialRing<FiniteField>>::from_json(&value)
                        .map_err(value_error)?,
                )),
                SerializedModuleRing::QuotientRational => Ok(Self::from_quotient_rational(
                    PresentedModulePower::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                        &value,
                    )
                    .map_err(value_error)?,
                )),
                SerializedModuleRing::QuotientFiniteField => Ok(Self::from_quotient_finite_field(
                    PresentedModulePower::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                        &value,
                    )
                    .map_err(value_error)?,
                )),
            }
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __hash__(&self) -> isize {
        match_presented_power!(&self.inner, |value| canonical_hash(value.as_ref()))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match_presented_power!(&self.inner, |value| value.to_string()))
    }
}
