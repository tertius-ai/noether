// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of polynomial free modules and module maps.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::commalg::{
    AssociatedPrimeError, DecompositionError, PolynomialRing, QuotientRing, ResidueRing,
};
use noether::homalg::{
    DerivedError, DerivedModule, DerivedModuleHomomorphism, FreeModule, FreeResolution,
    GradedBettiTable, GradedBettiTableError, GradedModuleInvariants, ImageSubmodule,
    KernelSubmodule, ModuleAssociatedPrimeSet, ModuleElement, ModuleError, ModuleGroebnerBasis,
    ModuleGroebnerError, ModuleHomomorphism, ModuleInvariantError, ModuleMonomial,
    ModuleMonomialOrder, ModulePrimaryComponent, ModulePrimaryDecomposition,
    ModulePrimaryDecompositionError, ModuleReduction, PresentedModule, PresentedModuleHomology,
    PresentedModuleHomomorphism, ResolutionCompleteness, ResolutionError, ResolutionMinimality,
    StrongModuleGroebnerBasis, StrongModuleReduction, StrongModuleTerm, Submodule,
    SubmoduleMembership,
};
use noether::numthy::FiniteField;
use noether::{IntegerRing, RationalField, SemanticType, Z};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::commalg::{PyResidueClass, PyResidueRing};
use super::construction::{construction_error, PySymmetricAlgebra};
use super::facade_scalar::{extract_integer, integer_to_python};
use super::homalg::PyFinitelyGeneratedAbelianGroup;
use super::linalg::{
    extract_finite_field_polynomial, extract_integer_polynomial, extract_rational_polynomial,
    extract_residue_polynomial, MatrixInner, PolynomialMatrixInner, PolynomialSparseMatrixInner,
    PolynomialVectorInner, PyMatrix, PySparseMatrix, PyVector, QuotientMatrixInner,
    QuotientSparseMatrixInner, QuotientVectorInner, SparseMatrixInner, VectorInner,
};
use super::module_power::{module_power_error, PyFreeModulePower, PyPresentedModulePower};
use super::number_field::{PyAssociatedPrimeExtStage, PyIdeal, PyPolynomialPrimeCertificate};
use super::polynomial_facade::{
    PolynomialInner, PolynomialRingInner, PyMonomialOrder, PyPolynomial, PyPolynomialRing,
};
use super::quotient::{PyQuotientElement, PyQuotientRing, QuotientElementInner, QuotientRingInner};
use super::root::PyIntegerRing;
use super::series::{PyDegree, PyFreeAbelianGrading, PyHilbertSeries};

fn module_error(error: ModuleError) -> PyErr {
    match error {
        ModuleError::ContextMismatch(error) => super::errors::context_mismatch(error),
        ModuleError::RankOverflow { .. } => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn module_groebner_error(error: ModuleGroebnerError) -> PyErr {
    match error {
        ModuleGroebnerError::Module(error) => module_error(error),
        ModuleGroebnerError::ContextMismatch(error) => super::errors::context_mismatch(error),
        ModuleGroebnerError::ResourceLimit { .. } => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn resolution_error(error: ResolutionError) -> PyErr {
    match error {
        ResolutionError::Module(error) => module_error(error),
        ResolutionError::Groebner(error) => module_groebner_error(error),
        ResolutionError::HomologicalDegreeLimit { .. }
        | ResolutionError::ProjectiveDimensionBound { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        ResolutionError::Homology(_) | ResolutionError::InvalidEvidence(_) => {
            PyValueError::new_err(error.to_string())
        }
    }
}

fn betti_error(error: GradedBettiTableError) -> PyErr {
    match error {
        GradedBettiTableError::Resolution(error) => resolution_error(error),
        GradedBettiTableError::TooManyEntries { .. } => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn derived_error(error: DerivedError) -> PyErr {
    match error {
        DerivedError::Resolution(error) => resolution_error(error),
        DerivedError::Module(error) => module_error(error),
        DerivedError::Groebner(error) => module_groebner_error(error),
        DerivedError::DegreeNotJustified { .. } | DerivedError::InvalidEvidence(_) => {
            PyValueError::new_err(error.to_string())
        }
    }
}

fn module_invariant_error(error: ModuleInvariantError) -> PyErr {
    match error {
        ModuleInvariantError::Module(error) => module_error(error),
        ModuleInvariantError::Groebner(error) => module_groebner_error(error),
        ModuleInvariantError::MinorCountLimit { .. } => PyOverflowError::new_err(error.to_string()),
        ModuleInvariantError::Ideal(_)
        | ModuleInvariantError::LinearAlgebra(_)
        | ModuleInvariantError::Resolution(_)
        | ModuleInvariantError::Betti(_)
        | ModuleInvariantError::Hilbert(_)
        | ModuleInvariantError::InvalidEvidence(_) => PyValueError::new_err(error.to_string()),
    }
}

fn associated_prime_error(error: AssociatedPrimeError) -> PyErr {
    match error {
        AssociatedPrimeError::Module(error) => module_error(error),
        AssociatedPrimeError::Resolution(error) => resolution_error(error),
        AssociatedPrimeError::Derived(error) => derived_error(error),
        AssociatedPrimeError::Invariant(error) => module_invariant_error(error),
        AssociatedPrimeError::Decomposition(DecompositionError::ResourceLimit { .. }) => {
            PyOverflowError::new_err(error.to_string())
        }
        other => value_error(other),
    }
}

fn module_primary_error(error: ModulePrimaryDecompositionError) -> PyErr {
    match error {
        ModulePrimaryDecompositionError::AssociatedPrimes(error) => associated_prime_error(error),
        ModulePrimaryDecompositionError::Groebner(error) => module_groebner_error(error),
        ModulePrimaryDecompositionError::Module(error) => module_error(error),
        ModulePrimaryDecompositionError::ContextMismatch(error) => {
            super::errors::context_mismatch(error)
        }
        ModulePrimaryDecompositionError::ResourceLimit { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        ModulePrimaryDecompositionError::Ideal(_)
        | ModulePrimaryDecompositionError::Polynomial(_)
        | ModulePrimaryDecompositionError::Unsupported(_)
        | ModulePrimaryDecompositionError::Invalid(_) => PyValueError::new_err(error.to_string()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SerializedModuleRing {
    PolynomialRational,
    PolynomialFiniteField,
    PolynomialInteger,
    PolynomialResidue,
    QuotientRational,
    QuotientFiniteField,
}

pub(crate) fn serialized_module_ring(
    value: &str,
    ring_pointer: &str,
) -> PyResult<SerializedModuleRing> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    let ring = document
        .pointer(ring_pointer)
        .ok_or_else(|| PyValueError::new_err("serialized module has no ring identity"))?;
    let ring_type = ring
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("serialized module ring has no type identity"))?;
    let coefficient = if ring_type == PolynomialRing::<RationalField>::TYPE_ID {
        ring.pointer("/coefficient_ring/type")
    } else if ring_type == QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID {
        ring.pointer("/defining_ideal/ring/coefficient_ring/type")
    } else {
        None
    }
    .and_then(serde_json::Value::as_str)
    .ok_or_else(|| PyValueError::new_err("serialized module has no coefficient-ring identity"))?;
    match (ring_type, coefficient) {
        (kind, RationalField::TYPE_ID) if kind == PolynomialRing::<RationalField>::TYPE_ID => {
            Ok(SerializedModuleRing::PolynomialRational)
        }
        (kind, FiniteField::TYPE_ID) if kind == PolynomialRing::<RationalField>::TYPE_ID => {
            Ok(SerializedModuleRing::PolynomialFiniteField)
        }
        (kind, IntegerRing::TYPE_ID) if kind == PolynomialRing::<RationalField>::TYPE_ID => {
            Ok(SerializedModuleRing::PolynomialInteger)
        }
        (kind, ResidueRing::TYPE_ID) if kind == PolynomialRing::<RationalField>::TYPE_ID => {
            Ok(SerializedModuleRing::PolynomialResidue)
        }
        (kind, RationalField::TYPE_ID)
            if kind == QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID =>
        {
            Ok(SerializedModuleRing::QuotientRational)
        }
        (kind, FiniteField::TYPE_ID)
            if kind == QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID =>
        {
            Ok(SerializedModuleRing::QuotientFiniteField)
        }
        _ => Err(PyTypeError::new_err(
            "unsupported module coefficient context",
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SerializedScalarModuleRing {
    Integer,
    Residue,
}

pub(crate) fn serialized_scalar_module_ring(
    value: &str,
    ring_pointer: &str,
) -> PyResult<Option<SerializedScalarModuleRing>> {
    let document = serde_json::from_str::<serde_json::Value>(value).map_err(value_error)?;
    let ring_type = document
        .pointer(ring_pointer)
        .and_then(|ring| ring.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("serialized module has no ring identity"))?;
    Ok(if ring_type == IntegerRing::TYPE_ID {
        Some(SerializedScalarModuleRing::Integer)
    } else if ring_type == ResidueRing::TYPE_ID {
        Some(SerializedScalarModuleRing::Residue)
    } else {
        None
    })
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum FreeModuleInner {
    ScalarInteger(FreeModule<IntegerRing>),
    ScalarResidue(FreeModule<ResidueRing>),
    Integer(FreeModule<PolynomialRing<IntegerRing>>),
    Rational(FreeModule<PolynomialRing<RationalField>>),
    Residue(FreeModule<PolynomialRing<ResidueRing>>),
    FiniteField(FreeModule<PolynomialRing<FiniteField>>),
    QuotientRational(FreeModule<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(FreeModule<QuotientRing<PolynomialRing<FiniteField>>>),
}

impl FreeModuleInner {
    fn rank(&self) -> usize {
        match self {
            Self::ScalarInteger(value) => value.rank(),
            Self::ScalarResidue(value) => value.rank(),
            Self::Integer(value) => value.rank(),
            Self::Rational(value) => value.rank(),
            Self::Residue(value) => value.rank(),
            Self::FiniteField(value) => value.rank(),
            Self::QuotientRational(value) => value.rank(),
            Self::QuotientFiniteField(value) => value.rank(),
        }
    }

    fn basis_labels(&self) -> Option<Vec<String>> {
        match self {
            Self::ScalarInteger(value) => value.basis_labels(),
            Self::ScalarResidue(value) => value.basis_labels(),
            Self::Integer(value) => value.basis_labels(),
            Self::Rational(value) => value.basis_labels(),
            Self::Residue(value) => value.basis_labels(),
            Self::FiniteField(value) => value.basis_labels(),
            Self::QuotientRational(value) => value.basis_labels(),
            Self::QuotientFiniteField(value) => value.basis_labels(),
        }
        .map(<[String]>::to_vec)
    }

    fn grading_rank(&self) -> Option<usize> {
        match self {
            Self::ScalarInteger(value) => value.grading_rank(),
            Self::ScalarResidue(value) => value.grading_rank(),
            Self::Integer(value) => value.grading_rank(),
            Self::Rational(value) => value.grading_rank(),
            Self::Residue(value) => value.grading_rank(),
            Self::FiniteField(value) => value.grading_rank(),
            Self::QuotientRational(value) => value.grading_rank(),
            Self::QuotientFiniteField(value) => value.grading_rank(),
        }
    }

    fn degree_shifts(&self) -> Option<Vec<PyDegree>> {
        match self {
            Self::ScalarInteger(value) => value.degree_shifts(),
            Self::ScalarResidue(value) => value.degree_shifts(),
            Self::Integer(value) => value.degree_shifts(),
            Self::Rational(value) => value.degree_shifts(),
            Self::Residue(value) => value.degree_shifts(),
            Self::FiniteField(value) => value.degree_shifts(),
            Self::QuotientRational(value) => value.degree_shifts(),
            Self::QuotientFiniteField(value) => value.degree_shifts(),
        }
        .map(|values| {
            values
                .iter()
                .cloned()
                .map(|inner| PyDegree { inner })
                .collect()
        })
    }

    pub(super) fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::ScalarInteger(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            Self::ScalarResidue(value) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: value.ring().clone(),
                },
            )?
            .into_any()),
            Self::Integer(value) => {
                Ok(Py::new(py, PyPolynomialRing::from_integer(value.ring().clone()))?.into_any())
            }
            Self::Rational(value) => {
                Ok(Py::new(py, PyPolynomialRing::from_rational(value.ring().clone()))?.into_any())
            }
            Self::Residue(value) => {
                Ok(Py::new(py, PyPolynomialRing::from_residue(value.ring().clone()))?.into_any())
            }
            Self::FiniteField(value) => Ok(Py::new(
                py,
                PyPolynomialRing::from_finite_field(value.ring().clone()),
            )?
            .into_any()),
            Self::QuotientRational(value) => {
                Ok(Py::new(py, PyQuotientRing::from_rational(value.ring().clone()))?.into_any())
            }
            Self::QuotientFiniteField(value) => Ok(Py::new(
                py,
                PyQuotientRing::from_finite_field(value.ring().clone()),
            )?
            .into_any()),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::ScalarInteger(value) => value.to_json(),
            Self::ScalarResidue(value) => value.to_json(),
            Self::Integer(value) => value.to_json(),
            Self::Rational(value) => value.to_json(),
            Self::Residue(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
            Self::QuotientRational(value) => value.to_json(),
            Self::QuotientFiniteField(value) => value.to_json(),
        }
    }
}

/// A finite based free module over a supported polynomial or quotient ring.
#[pyclass(module = "noether.homalg", name = "FreeModule", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFreeModule {
    pub(crate) inner: FreeModuleInner,
}

#[pymethods]
impl PyFreeModule {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FreeModule::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[new]
    #[pyo3(signature = (ring, rank, *, basis_labels=None, grading_rank=None, degree_shifts=None))]
    fn new(
        ring: &Bound<'_, PyAny>,
        rank: usize,
        basis_labels: Option<Vec<String>>,
        grading_rank: Option<usize>,
        degree_shifts: Option<Vec<PyDegree>>,
    ) -> PyResult<Self> {
        let shifts = degree_shifts.map(|values| {
            values
                .into_iter()
                .map(|value| value.inner)
                .collect::<Vec<_>>()
        });
        let inner = if ring.extract::<PyRef<'_, PyIntegerRing>>().is_ok() {
            FreeModuleInner::ScalarInteger(
                FreeModule::with_basis_data(Z, rank, basis_labels, grading_rank, shifts)
                    .map_err(module_error)?,
            )
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            FreeModuleInner::ScalarResidue(
                FreeModule::with_basis_data(
                    ring.inner.clone(),
                    rank,
                    basis_labels,
                    grading_rank,
                    shifts,
                )
                .map_err(module_error)?,
            )
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            match &ring.inner {
                PolynomialRingInner::Integer(ring) => FreeModuleInner::Integer(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
                PolynomialRingInner::Rational(ring) => FreeModuleInner::Rational(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
                PolynomialRingInner::Residue(ring) => FreeModuleInner::Residue(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
                PolynomialRingInner::FiniteField(ring) => FreeModuleInner::FiniteField(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
                _ => {
                    return Err(PyTypeError::new_err(
                        "unsupported polynomial module coefficient context",
                    ));
                }
            }
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            match &ring.inner {
                QuotientRingInner::Rational(ring) => FreeModuleInner::QuotientRational(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
                QuotientRingInner::FiniteField(ring) => FreeModuleInner::QuotientFiniteField(
                    FreeModule::with_basis_data(
                        ring.clone(),
                        rank,
                        basis_labels,
                        grading_rank,
                        shifts,
                    )
                    .map_err(module_error)?,
                ),
            }
        } else {
            return Err(PyTypeError::new_err(
                "FreeModule requires IntegerRing, ResidueRing, or a supported PolynomialRing or QuotientRing",
            ));
        };
        Ok(Self { inner })
    }

    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.ring_to_python(py)
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn basis_labels(&self) -> Option<Vec<String>> {
        self.inner.basis_labels()
    }

    #[getter]
    fn grading_rank(&self) -> Option<usize> {
        self.inner.grading_rank()
    }

    #[getter]
    fn degree_shifts(&self) -> Option<Vec<PyDegree>> {
        self.inner.degree_shifts()
    }

    fn is_graded(&self) -> bool {
        self.inner.grading_rank().is_some()
    }

    fn basis_element(&self, position: usize) -> PyResult<PyModuleElement> {
        let inner = match &self.inner {
            FreeModuleInner::ScalarInteger(value) => ModuleElementInner::ScalarInteger(
                value.basis_element(position).map_err(module_error)?,
            ),
            FreeModuleInner::ScalarResidue(value) => ModuleElementInner::ScalarResidue(
                value.basis_element(position).map_err(module_error)?,
            ),
            FreeModuleInner::Integer(value) => {
                ModuleElementInner::Integer(value.basis_element(position).map_err(module_error)?)
            }
            FreeModuleInner::Rational(value) => {
                ModuleElementInner::Rational(value.basis_element(position).map_err(module_error)?)
            }
            FreeModuleInner::Residue(value) => {
                ModuleElementInner::Residue(value.basis_element(position).map_err(module_error)?)
            }
            FreeModuleInner::FiniteField(value) => ModuleElementInner::FiniteField(
                value.basis_element(position).map_err(module_error)?,
            ),
            FreeModuleInner::QuotientRational(value) => ModuleElementInner::QuotientRational(
                value.basis_element(position).map_err(module_error)?,
            ),
            FreeModuleInner::QuotientFiniteField(value) => ModuleElementInner::QuotientFiniteField(
                value.basis_element(position).map_err(module_error)?,
            ),
        };
        Ok(PyModuleElement { inner })
    }

    fn direct_sum(&self, rhs: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &rhs.inner) {
            (FreeModuleInner::ScalarInteger(left), FreeModuleInner::ScalarInteger(right)) => {
                FreeModuleInner::ScalarInteger(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::ScalarResidue(left), FreeModuleInner::ScalarResidue(right)) => {
                FreeModuleInner::ScalarResidue(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::Integer(left), FreeModuleInner::Integer(right)) => {
                FreeModuleInner::Integer(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::Rational(left), FreeModuleInner::Rational(right)) => {
                FreeModuleInner::Rational(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::Residue(left), FreeModuleInner::Residue(right)) => {
                FreeModuleInner::Residue(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::FiniteField(left), FreeModuleInner::FiniteField(right)) => {
                FreeModuleInner::FiniteField(left.direct_sum(right).map_err(module_error)?)
            }
            (FreeModuleInner::QuotientRational(left), FreeModuleInner::QuotientRational(right)) => {
                FreeModuleInner::QuotientRational(left.direct_sum(right).map_err(module_error)?)
            }
            (
                FreeModuleInner::QuotientFiniteField(left),
                FreeModuleInner::QuotientFiniteField(right),
            ) => {
                FreeModuleInner::QuotientFiniteField(left.direct_sum(right).map_err(module_error)?)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "module coefficient contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn tensor(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: tensor bases can be material.
        py.detach(|| {
            let inner = match (&self.inner, &rhs.inner) {
                (FreeModuleInner::ScalarInteger(left), FreeModuleInner::ScalarInteger(right)) => {
                    FreeModuleInner::ScalarInteger(left.tensor(right).map_err(module_error)?)
                }
                (FreeModuleInner::ScalarResidue(left), FreeModuleInner::ScalarResidue(right)) => {
                    FreeModuleInner::ScalarResidue(left.tensor(right).map_err(module_error)?)
                }
                (FreeModuleInner::Integer(left), FreeModuleInner::Integer(right)) => {
                    FreeModuleInner::Integer(left.tensor(right).map_err(module_error)?)
                }
                (FreeModuleInner::Rational(left), FreeModuleInner::Rational(right)) => {
                    FreeModuleInner::Rational(left.tensor(right).map_err(module_error)?)
                }
                (FreeModuleInner::Residue(left), FreeModuleInner::Residue(right)) => {
                    FreeModuleInner::Residue(left.tensor(right).map_err(module_error)?)
                }
                (FreeModuleInner::FiniteField(left), FreeModuleInner::FiniteField(right)) => {
                    FreeModuleInner::FiniteField(left.tensor(right).map_err(module_error)?)
                }
                (
                    FreeModuleInner::QuotientRational(left),
                    FreeModuleInner::QuotientRational(right),
                ) => FreeModuleInner::QuotientRational(left.tensor(right).map_err(module_error)?),
                (
                    FreeModuleInner::QuotientFiniteField(left),
                    FreeModuleInner::QuotientFiniteField(right),
                ) => {
                    FreeModuleInner::QuotientFiniteField(left.tensor(right).map_err(module_error)?)
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "module coefficient contexts do not match",
                    ));
                }
            };
            Ok(Self { inner })
        })
    }

    fn dual(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            FreeModuleInner::ScalarInteger(value) => {
                FreeModuleInner::ScalarInteger(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::ScalarResidue(value) => {
                FreeModuleInner::ScalarResidue(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::Integer(value) => {
                FreeModuleInner::Integer(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::Rational(value) => {
                FreeModuleInner::Rational(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::Residue(value) => {
                FreeModuleInner::Residue(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::FiniteField(value) => {
                FreeModuleInner::FiniteField(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::QuotientRational(value) => {
                FreeModuleInner::QuotientRational(value.dual().map_err(module_error)?)
            }
            FreeModuleInner::QuotientFiniteField(value) => {
                FreeModuleInner::QuotientFiniteField(value.dual().map_err(module_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn symmetric_power(&self, py: Python<'_>, degree: usize) -> PyResult<PyFreeModulePower> {
        // NOETHER_RELEASES_GIL: basis enumeration can be material.
        py.detach(|| match &self.inner {
            FreeModuleInner::ScalarInteger(value) => Ok(PyFreeModulePower::from_scalar_integer(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::ScalarResidue(value) => Ok(PyFreeModulePower::from_scalar_residue(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Integer(value) => Ok(PyFreeModulePower::from_integer(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Rational(value) => Ok(PyFreeModulePower::from_rational(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Residue(value) => Ok(PyFreeModulePower::from_residue(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::FiniteField(value) => Ok(PyFreeModulePower::from_finite_field(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::QuotientRational(value) => {
                Ok(PyFreeModulePower::from_quotient_rational(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
            FreeModuleInner::QuotientFiniteField(value) => {
                Ok(PyFreeModulePower::from_quotient_finite_field(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
        })
    }

    fn exterior_power(&self, py: Python<'_>, degree: usize) -> PyResult<PyFreeModulePower> {
        // NOETHER_RELEASES_GIL: basis enumeration can be material.
        py.detach(|| match &self.inner {
            FreeModuleInner::ScalarInteger(value) => Ok(PyFreeModulePower::from_scalar_integer(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::ScalarResidue(value) => Ok(PyFreeModulePower::from_scalar_residue(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Integer(value) => Ok(PyFreeModulePower::from_integer(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Rational(value) => Ok(PyFreeModulePower::from_rational(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::Residue(value) => Ok(PyFreeModulePower::from_residue(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::FiniteField(value) => Ok(PyFreeModulePower::from_finite_field(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            FreeModuleInner::QuotientRational(value) => {
                Ok(PyFreeModulePower::from_quotient_rational(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
            FreeModuleInner::QuotientFiniteField(value) => {
                Ok(PyFreeModulePower::from_quotient_finite_field(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
        })
    }

    /// Construct the exact exterior algebra on this based module.
    fn exterior_algebra(&self) -> super::module_power::PyExteriorAlgebra {
        super::module_power::PyExteriorAlgebra::from_module(&self.inner)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        if let Some(kind) = serialized_scalar_module_ring(value, "/ring")? {
            let inner = match kind {
                SerializedScalarModuleRing::Integer => FreeModuleInner::ScalarInteger(
                    FreeModule::<IntegerRing>::from_json(value).map_err(value_error)?,
                ),
                SerializedScalarModuleRing::Residue => FreeModuleInner::ScalarResidue(
                    FreeModule::<ResidueRing>::from_json(value).map_err(value_error)?,
                ),
            };
            return Ok(Self { inner });
        }
        let inner = match serialized_module_ring(value, "/ring")? {
            SerializedModuleRing::PolynomialInteger => FreeModuleInner::Integer(
                FreeModule::<PolynomialRing<IntegerRing>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialRational => FreeModuleInner::Rational(
                FreeModule::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialResidue => FreeModuleInner::Residue(
                FreeModule::<PolynomialRing<ResidueRing>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => FreeModuleInner::FiniteField(
                FreeModule::<PolynomialRing<FiniteField>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => FreeModuleInner::QuotientRational(
                FreeModule::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => FreeModuleInner::QuotientFiniteField(
                FreeModule::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("FreeModule(rank={})", self.rank()))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum ModuleElementInner {
    ScalarInteger(ModuleElement<IntegerRing>),
    ScalarResidue(ModuleElement<ResidueRing>),
    Integer(ModuleElement<PolynomialRing<IntegerRing>>),
    Rational(ModuleElement<PolynomialRing<RationalField>>),
    Residue(ModuleElement<PolynomialRing<ResidueRing>>),
    FiniteField(ModuleElement<PolynomialRing<FiniteField>>),
    QuotientRational(ModuleElement<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(ModuleElement<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// An exact coordinate vector in one based free module.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleElement",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleElement {
    pub(super) inner: ModuleElementInner,
}

impl PyModuleElement {
    pub(crate) fn from_rational_quotient(
        value: ModuleElement<QuotientRing<PolynomialRing<RationalField>>>,
    ) -> Self {
        Self {
            inner: ModuleElementInner::QuotientRational(value),
        }
    }

    pub(crate) fn from_finite_field_quotient(
        value: ModuleElement<QuotientRing<PolynomialRing<FiniteField>>>,
    ) -> Self {
        Self {
            inner: ModuleElementInner::QuotientFiniteField(value),
        }
    }
}

#[pymethods]
impl PyModuleElement {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleElement::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[new]
    fn new(module: &PyFreeModule, coordinates: &PyVector) -> PyResult<Self> {
        let inner = match (&module.inner, &coordinates.inner) {
            (FreeModuleInner::ScalarInteger(module), VectorInner::Integer(values)) => {
                ModuleElementInner::ScalarInteger(
                    ModuleElement::from_coordinates(module.clone(), values.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::ScalarResidue(module), VectorInner::Residue(values)) => {
                ModuleElementInner::ScalarResidue(
                    ModuleElement::from_coordinates(module.clone(), values.clone())
                        .map_err(module_error)?,
                )
            }
            (
                FreeModuleInner::Integer(module),
                VectorInner::Polynomial(PolynomialVectorInner::Integer(values)),
            ) => ModuleElementInner::Integer(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            (
                FreeModuleInner::Rational(module),
                VectorInner::Polynomial(PolynomialVectorInner::Rational(values)),
            ) => ModuleElementInner::Rational(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            (
                FreeModuleInner::Residue(module),
                VectorInner::Polynomial(PolynomialVectorInner::Residue(values)),
            ) => ModuleElementInner::Residue(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            (
                FreeModuleInner::FiniteField(module),
                VectorInner::Polynomial(PolynomialVectorInner::FiniteField(values)),
            ) => ModuleElementInner::FiniteField(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            (
                FreeModuleInner::QuotientRational(module),
                VectorInner::Quotient(QuotientVectorInner::Rational(values)),
            ) => ModuleElementInner::QuotientRational(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            (
                FreeModuleInner::QuotientFiniteField(module),
                VectorInner::Quotient(QuotientVectorInner::FiniteField(values)),
            ) => ModuleElementInner::QuotientFiniteField(
                ModuleElement::from_coordinates(module.clone(), values.clone())
                    .map_err(module_error)?,
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "module and coordinate vector contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>, module: &PyFreeModule) -> Self {
        let inner = match &module.inner {
            FreeModuleInner::ScalarInteger(value) => {
                ModuleElementInner::ScalarInteger(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::ScalarResidue(value) => {
                ModuleElementInner::ScalarResidue(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::Integer(value) => {
                ModuleElementInner::Integer(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::Rational(value) => {
                ModuleElementInner::Rational(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::Residue(value) => {
                ModuleElementInner::Residue(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::FiniteField(value) => {
                ModuleElementInner::FiniteField(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::QuotientRational(value) => {
                ModuleElementInner::QuotientRational(ModuleElement::zero(value.clone()))
            }
            FreeModuleInner::QuotientFiniteField(value) => {
                ModuleElementInner::QuotientFiniteField(ModuleElement::zero(value.clone()))
            }
        };
        Self { inner }
    }

    #[getter]
    fn module(&self) -> PyFreeModule {
        let inner = match &self.inner {
            ModuleElementInner::ScalarInteger(value) => {
                FreeModuleInner::ScalarInteger(value.module().clone())
            }
            ModuleElementInner::ScalarResidue(value) => {
                FreeModuleInner::ScalarResidue(value.module().clone())
            }
            ModuleElementInner::Integer(value) => FreeModuleInner::Integer(value.module().clone()),
            ModuleElementInner::Rational(value) => {
                FreeModuleInner::Rational(value.module().clone())
            }
            ModuleElementInner::Residue(value) => FreeModuleInner::Residue(value.module().clone()),
            ModuleElementInner::FiniteField(value) => {
                FreeModuleInner::FiniteField(value.module().clone())
            }
            ModuleElementInner::QuotientRational(value) => {
                FreeModuleInner::QuotientRational(value.module().clone())
            }
            ModuleElementInner::QuotientFiniteField(value) => {
                FreeModuleInner::QuotientFiniteField(value.module().clone())
            }
        };
        PyFreeModule { inner }
    }

    #[getter]
    fn coordinates(&self) -> PyVector {
        let inner = match &self.inner {
            ModuleElementInner::ScalarInteger(value) => {
                VectorInner::Integer(value.coordinates().clone())
            }
            ModuleElementInner::ScalarResidue(value) => {
                VectorInner::Residue(value.coordinates().clone())
            }
            ModuleElementInner::Integer(value) => {
                VectorInner::Polynomial(PolynomialVectorInner::Integer(value.coordinates().clone()))
            }
            ModuleElementInner::Rational(value) => VectorInner::Polynomial(
                PolynomialVectorInner::Rational(value.coordinates().clone()),
            ),
            ModuleElementInner::Residue(value) => {
                VectorInner::Polynomial(PolynomialVectorInner::Residue(value.coordinates().clone()))
            }
            ModuleElementInner::FiniteField(value) => VectorInner::Polynomial(
                PolynomialVectorInner::FiniteField(value.coordinates().clone()),
            ),
            ModuleElementInner::QuotientRational(value) => {
                VectorInner::Quotient(QuotientVectorInner::Rational(value.coordinates().clone()))
            }
            ModuleElementInner::QuotientFiniteField(value) => VectorInner::Quotient(
                QuotientVectorInner::FiniteField(value.coordinates().clone()),
            ),
        };
        PyVector { inner }
    }

    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &rhs.inner) {
            (ModuleElementInner::ScalarInteger(left), ModuleElementInner::ScalarInteger(right)) => {
                ModuleElementInner::ScalarInteger(left.add(right).map_err(module_error)?)
            }
            (ModuleElementInner::ScalarResidue(left), ModuleElementInner::ScalarResidue(right)) => {
                ModuleElementInner::ScalarResidue(left.add(right).map_err(module_error)?)
            }
            (ModuleElementInner::Integer(left), ModuleElementInner::Integer(right)) => {
                ModuleElementInner::Integer(left.add(right).map_err(module_error)?)
            }
            (ModuleElementInner::Rational(left), ModuleElementInner::Rational(right)) => {
                ModuleElementInner::Rational(left.add(right).map_err(module_error)?)
            }
            (ModuleElementInner::Residue(left), ModuleElementInner::Residue(right)) => {
                ModuleElementInner::Residue(left.add(right).map_err(module_error)?)
            }
            (ModuleElementInner::FiniteField(left), ModuleElementInner::FiniteField(right)) => {
                ModuleElementInner::FiniteField(left.add(right).map_err(module_error)?)
            }
            (
                ModuleElementInner::QuotientRational(left),
                ModuleElementInner::QuotientRational(right),
            ) => ModuleElementInner::QuotientRational(left.add(right).map_err(module_error)?),
            (
                ModuleElementInner::QuotientFiniteField(left),
                ModuleElementInner::QuotientFiniteField(right),
            ) => ModuleElementInner::QuotientFiniteField(left.add(right).map_err(module_error)?),
            _ => {
                return Err(PyValueError::new_err(
                    "module element contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &rhs.inner) {
            (ModuleElementInner::ScalarInteger(left), ModuleElementInner::ScalarInteger(right)) => {
                ModuleElementInner::ScalarInteger(left.sub(right).map_err(module_error)?)
            }
            (ModuleElementInner::ScalarResidue(left), ModuleElementInner::ScalarResidue(right)) => {
                ModuleElementInner::ScalarResidue(left.sub(right).map_err(module_error)?)
            }
            (ModuleElementInner::Integer(left), ModuleElementInner::Integer(right)) => {
                ModuleElementInner::Integer(left.sub(right).map_err(module_error)?)
            }
            (ModuleElementInner::Rational(left), ModuleElementInner::Rational(right)) => {
                ModuleElementInner::Rational(left.sub(right).map_err(module_error)?)
            }
            (ModuleElementInner::Residue(left), ModuleElementInner::Residue(right)) => {
                ModuleElementInner::Residue(left.sub(right).map_err(module_error)?)
            }
            (ModuleElementInner::FiniteField(left), ModuleElementInner::FiniteField(right)) => {
                ModuleElementInner::FiniteField(left.sub(right).map_err(module_error)?)
            }
            (
                ModuleElementInner::QuotientRational(left),
                ModuleElementInner::QuotientRational(right),
            ) => ModuleElementInner::QuotientRational(left.sub(right).map_err(module_error)?),
            (
                ModuleElementInner::QuotientFiniteField(left),
                ModuleElementInner::QuotientFiniteField(right),
            ) => ModuleElementInner::QuotientFiniteField(left.sub(right).map_err(module_error)?),
            _ => {
                return Err(PyValueError::new_err(
                    "module element contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            ModuleElementInner::ScalarInteger(value) => {
                ModuleElementInner::ScalarInteger(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::ScalarResidue(value) => {
                ModuleElementInner::ScalarResidue(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::Integer(value) => {
                ModuleElementInner::Integer(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::Rational(value) => {
                ModuleElementInner::Rational(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::Residue(value) => {
                ModuleElementInner::Residue(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::FiniteField(value) => {
                ModuleElementInner::FiniteField(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::QuotientRational(value) => {
                ModuleElementInner::QuotientRational(value.neg().map_err(module_error)?)
            }
            ModuleElementInner::QuotientFiniteField(value) => {
                ModuleElementInner::QuotientFiniteField(value.neg().map_err(module_error)?)
            }
        };
        Ok(Self { inner })
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            ModuleElementInner::ScalarInteger(value) => ModuleElementInner::ScalarInteger(
                value
                    .scalar_mul(&extract_integer(scalar)?)
                    .map_err(module_error)?,
            ),
            ModuleElementInner::ScalarResidue(value) => {
                let scalar = scalar.extract::<PyRef<'_, PyResidueClass>>()?;
                ModuleElementInner::ScalarResidue(
                    value.scalar_mul(&scalar.inner).map_err(module_error)?,
                )
            }
            ModuleElementInner::Integer(value) => ModuleElementInner::Integer(
                value
                    .scalar_mul(&extract_integer_polynomial(value.module().ring(), scalar)?)
                    .map_err(module_error)?,
            ),
            ModuleElementInner::Rational(value) => ModuleElementInner::Rational(
                value
                    .scalar_mul(&extract_rational_polynomial(
                        py,
                        value.module().ring(),
                        scalar,
                    )?)
                    .map_err(module_error)?,
            ),
            ModuleElementInner::Residue(value) => ModuleElementInner::Residue(
                value
                    .scalar_mul(&extract_residue_polynomial(value.module().ring(), scalar)?)
                    .map_err(module_error)?,
            ),
            ModuleElementInner::FiniteField(value) => ModuleElementInner::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_polynomial(
                        value.module().ring(),
                        scalar,
                    )?)
                    .map_err(module_error)?,
            ),
            ModuleElementInner::QuotientRational(value) => {
                let scalar = scalar.extract::<PyRef<'_, PyQuotientElement>>()?;
                let QuotientElementInner::Rational(scalar) = &scalar.inner else {
                    return Err(PyTypeError::new_err(
                        "module and quotient scalar contexts do not match",
                    ));
                };
                ModuleElementInner::QuotientRational(
                    value.scalar_mul(scalar).map_err(module_error)?,
                )
            }
            ModuleElementInner::QuotientFiniteField(value) => {
                let scalar = scalar.extract::<PyRef<'_, PyQuotientElement>>()?;
                let QuotientElementInner::FiniteField(scalar) = &scalar.inner else {
                    return Err(PyTypeError::new_err(
                        "module and quotient scalar contexts do not match",
                    ));
                };
                ModuleElementInner::QuotientFiniteField(
                    value.scalar_mul(scalar).map_err(module_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scalar_mul(py, scalar)
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.scalar_mul(py, scalar)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModuleElementInner::ScalarInteger(value) => value.to_json(),
            ModuleElementInner::ScalarResidue(value) => value.to_json(),
            ModuleElementInner::Integer(value) => value.to_json(),
            ModuleElementInner::Rational(value) => value.to_json(),
            ModuleElementInner::Residue(value) => value.to_json(),
            ModuleElementInner::FiniteField(value) => value.to_json(),
            ModuleElementInner::QuotientRational(value) => value.to_json(),
            ModuleElementInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        if let Some(kind) = serialized_scalar_module_ring(value, "/module/ring")? {
            let inner = match kind {
                SerializedScalarModuleRing::Integer => ModuleElementInner::ScalarInteger(
                    ModuleElement::<IntegerRing>::from_json(value).map_err(value_error)?,
                ),
                SerializedScalarModuleRing::Residue => ModuleElementInner::ScalarResidue(
                    ModuleElement::<ResidueRing>::from_json(value).map_err(value_error)?,
                ),
            };
            return Ok(Self { inner });
        }
        let inner = match serialized_module_ring(value, "/module/ring")? {
            SerializedModuleRing::PolynomialInteger => ModuleElementInner::Integer(
                ModuleElement::<PolynomialRing<IntegerRing>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialRational => ModuleElementInner::Rational(
                ModuleElement::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialResidue => ModuleElementInner::Residue(
                ModuleElement::<PolynomialRing<ResidueRing>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => ModuleElementInner::FiniteField(
                ModuleElement::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => ModuleElementInner::QuotientRational(
                ModuleElement::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => ModuleElementInner::QuotientFiniteField(
                ModuleElement::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ModuleElement(module_rank={})",
            self.module().rank()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum ModuleHomomorphismInner {
    ScalarInteger(ModuleHomomorphism<IntegerRing>),
    ScalarResidue(ModuleHomomorphism<ResidueRing>),
    Integer(ModuleHomomorphism<PolynomialRing<IntegerRing>>),
    Rational(ModuleHomomorphism<PolynomialRing<RationalField>>),
    Residue(ModuleHomomorphism<PolynomialRing<ResidueRing>>),
    FiniteField(ModuleHomomorphism<PolynomialRing<FiniteField>>),
    QuotientRational(ModuleHomomorphism<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(ModuleHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// An exact homomorphism between two based free modules.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleHomomorphism {
    pub(super) inner: ModuleHomomorphismInner,
}

#[pymethods]
impl PyModuleHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleHomomorphism::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[new]
    fn new(
        source: &PyFreeModule,
        target: &PyFreeModule,
        matrix: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let inner = if let Ok(matrix) = matrix.extract::<PyRef<'_, PyMatrix>>() {
            match (&source.inner, &target.inner, &matrix.inner) {
                (
                    FreeModuleInner::ScalarInteger(source),
                    FreeModuleInner::ScalarInteger(target),
                    MatrixInner::Integer(matrix),
                ) => ModuleHomomorphismInner::ScalarInteger(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::ScalarResidue(source),
                    FreeModuleInner::ScalarResidue(target),
                    MatrixInner::Residue(matrix),
                ) => ModuleHomomorphismInner::ScalarResidue(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Integer(source),
                    FreeModuleInner::Integer(target),
                    MatrixInner::Polynomial(PolynomialMatrixInner::Integer(matrix)),
                ) => ModuleHomomorphismInner::Integer(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Rational(source),
                    FreeModuleInner::Rational(target),
                    MatrixInner::Polynomial(PolynomialMatrixInner::Rational(matrix)),
                ) => ModuleHomomorphismInner::Rational(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Residue(source),
                    FreeModuleInner::Residue(target),
                    MatrixInner::Polynomial(PolynomialMatrixInner::Residue(matrix)),
                ) => ModuleHomomorphismInner::Residue(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::FiniteField(source),
                    FreeModuleInner::FiniteField(target),
                    MatrixInner::Polynomial(PolynomialMatrixInner::FiniteField(matrix)),
                ) => ModuleHomomorphismInner::FiniteField(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientRational(source),
                    FreeModuleInner::QuotientRational(target),
                    MatrixInner::Quotient(QuotientMatrixInner::Rational(matrix)),
                ) => ModuleHomomorphismInner::QuotientRational(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientFiniteField(source),
                    FreeModuleInner::QuotientFiniteField(target),
                    MatrixInner::Quotient(QuotientMatrixInner::FiniteField(matrix)),
                ) => ModuleHomomorphismInner::QuotientFiniteField(
                    ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix.clone())
                        .map_err(module_error)?,
                ),
                _ => {
                    return Err(PyTypeError::new_err(
                        "module endpoints and dense matrix contexts do not match",
                    ))
                }
            }
        } else if let Ok(matrix) = matrix.extract::<PyRef<'_, PySparseMatrix>>() {
            match (&source.inner, &target.inner, &matrix.inner) {
                (
                    FreeModuleInner::ScalarInteger(source),
                    FreeModuleInner::ScalarInteger(target),
                    SparseMatrixInner::Integer(matrix),
                ) => ModuleHomomorphismInner::ScalarInteger(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::ScalarResidue(source),
                    FreeModuleInner::ScalarResidue(target),
                    SparseMatrixInner::Residue(matrix),
                ) => ModuleHomomorphismInner::ScalarResidue(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Integer(source),
                    FreeModuleInner::Integer(target),
                    SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Integer(matrix)),
                ) => ModuleHomomorphismInner::Integer(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Rational(source),
                    FreeModuleInner::Rational(target),
                    SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Rational(matrix)),
                ) => ModuleHomomorphismInner::Rational(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::Residue(source),
                    FreeModuleInner::Residue(target),
                    SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Residue(matrix)),
                ) => ModuleHomomorphismInner::Residue(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::FiniteField(source),
                    FreeModuleInner::FiniteField(target),
                    SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::FiniteField(matrix)),
                ) => ModuleHomomorphismInner::FiniteField(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientRational(source),
                    FreeModuleInner::QuotientRational(target),
                    SparseMatrixInner::Quotient(QuotientSparseMatrixInner::Rational(matrix)),
                ) => ModuleHomomorphismInner::QuotientRational(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientFiniteField(source),
                    FreeModuleInner::QuotientFiniteField(target),
                    SparseMatrixInner::Quotient(QuotientSparseMatrixInner::FiniteField(matrix)),
                ) => ModuleHomomorphismInner::QuotientFiniteField(
                    ModuleHomomorphism::from_sparse_matrix(
                        source.clone(),
                        target.clone(),
                        matrix.clone(),
                    )
                    .map_err(module_error)?,
                ),
                _ => {
                    return Err(PyTypeError::new_err(
                        "module endpoints and sparse matrix contexts do not match",
                    ))
                }
            }
        } else {
            return Err(PyTypeError::new_err(
                "ModuleHomomorphism requires Matrix or SparseMatrix",
            ));
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn zero(
        _cls: &Bound<'_, PyType>,
        source: &PyFreeModule,
        target: &PyFreeModule,
    ) -> PyResult<Self> {
        let inner = match (&source.inner, &target.inner) {
            (FreeModuleInner::ScalarInteger(source), FreeModuleInner::ScalarInteger(target)) => {
                ModuleHomomorphismInner::ScalarInteger(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::ScalarResidue(source), FreeModuleInner::ScalarResidue(target)) => {
                ModuleHomomorphismInner::ScalarResidue(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::Integer(source), FreeModuleInner::Integer(target)) => {
                ModuleHomomorphismInner::Integer(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::Rational(source), FreeModuleInner::Rational(target)) => {
                ModuleHomomorphismInner::Rational(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::Residue(source), FreeModuleInner::Residue(target)) => {
                ModuleHomomorphismInner::Residue(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (FreeModuleInner::FiniteField(source), FreeModuleInner::FiniteField(target)) => {
                ModuleHomomorphismInner::FiniteField(
                    ModuleHomomorphism::zero(source.clone(), target.clone())
                        .map_err(module_error)?,
                )
            }
            (
                FreeModuleInner::QuotientRational(source),
                FreeModuleInner::QuotientRational(target),
            ) => ModuleHomomorphismInner::QuotientRational(
                ModuleHomomorphism::zero(source.clone(), target.clone()).map_err(module_error)?,
            ),
            (
                FreeModuleInner::QuotientFiniteField(source),
                FreeModuleInner::QuotientFiniteField(target),
            ) => ModuleHomomorphismInner::QuotientFiniteField(
                ModuleHomomorphism::zero(source.clone(), target.clone()).map_err(module_error)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "module coefficient contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, module: &PyFreeModule) -> PyResult<Self> {
        let inner = match &module.inner {
            FreeModuleInner::ScalarInteger(value) => ModuleHomomorphismInner::ScalarInteger(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::ScalarResidue(value) => ModuleHomomorphismInner::ScalarResidue(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::Integer(value) => ModuleHomomorphismInner::Integer(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::Rational(value) => ModuleHomomorphismInner::Rational(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::Residue(value) => ModuleHomomorphismInner::Residue(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::FiniteField(value) => ModuleHomomorphismInner::FiniteField(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::QuotientRational(value) => ModuleHomomorphismInner::QuotientRational(
                ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            FreeModuleInner::QuotientFiniteField(value) => {
                ModuleHomomorphismInner::QuotientFiniteField(
                    ModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn source(&self) -> PyFreeModule {
        PyFreeModule {
            inner: match &self.inner {
                ModuleHomomorphismInner::ScalarInteger(value) => {
                    FreeModuleInner::ScalarInteger(value.source().clone())
                }
                ModuleHomomorphismInner::ScalarResidue(value) => {
                    FreeModuleInner::ScalarResidue(value.source().clone())
                }
                ModuleHomomorphismInner::Integer(value) => {
                    FreeModuleInner::Integer(value.source().clone())
                }
                ModuleHomomorphismInner::Rational(value) => {
                    FreeModuleInner::Rational(value.source().clone())
                }
                ModuleHomomorphismInner::Residue(value) => {
                    FreeModuleInner::Residue(value.source().clone())
                }
                ModuleHomomorphismInner::FiniteField(value) => {
                    FreeModuleInner::FiniteField(value.source().clone())
                }
                ModuleHomomorphismInner::QuotientRational(value) => {
                    FreeModuleInner::QuotientRational(value.source().clone())
                }
                ModuleHomomorphismInner::QuotientFiniteField(value) => {
                    FreeModuleInner::QuotientFiniteField(value.source().clone())
                }
            },
        }
    }

    #[getter]
    fn target(&self) -> PyFreeModule {
        PyFreeModule {
            inner: match &self.inner {
                ModuleHomomorphismInner::ScalarInteger(value) => {
                    FreeModuleInner::ScalarInteger(value.target().clone())
                }
                ModuleHomomorphismInner::ScalarResidue(value) => {
                    FreeModuleInner::ScalarResidue(value.target().clone())
                }
                ModuleHomomorphismInner::Integer(value) => {
                    FreeModuleInner::Integer(value.target().clone())
                }
                ModuleHomomorphismInner::Rational(value) => {
                    FreeModuleInner::Rational(value.target().clone())
                }
                ModuleHomomorphismInner::Residue(value) => {
                    FreeModuleInner::Residue(value.target().clone())
                }
                ModuleHomomorphismInner::FiniteField(value) => {
                    FreeModuleInner::FiniteField(value.target().clone())
                }
                ModuleHomomorphismInner::QuotientRational(value) => {
                    FreeModuleInner::QuotientRational(value.target().clone())
                }
                ModuleHomomorphismInner::QuotientFiniteField(value) => {
                    FreeModuleInner::QuotientFiniteField(value.target().clone())
                }
            },
        }
    }

    #[getter]
    fn matrix(&self) -> PyMatrix {
        let inner = match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => MatrixInner::Integer(value.matrix()),
            ModuleHomomorphismInner::ScalarResidue(value) => MatrixInner::Residue(value.matrix()),
            ModuleHomomorphismInner::Integer(value) => {
                MatrixInner::Polynomial(PolynomialMatrixInner::Integer(value.matrix()))
            }
            ModuleHomomorphismInner::Rational(value) => {
                MatrixInner::Polynomial(PolynomialMatrixInner::Rational(value.matrix()))
            }
            ModuleHomomorphismInner::Residue(value) => {
                MatrixInner::Polynomial(PolynomialMatrixInner::Residue(value.matrix()))
            }
            ModuleHomomorphismInner::FiniteField(value) => {
                MatrixInner::Polynomial(PolynomialMatrixInner::FiniteField(value.matrix()))
            }
            ModuleHomomorphismInner::QuotientRational(value) => {
                MatrixInner::Quotient(QuotientMatrixInner::Rational(value.matrix()))
            }
            ModuleHomomorphismInner::QuotientFiniteField(value) => {
                MatrixInner::Quotient(QuotientMatrixInner::FiniteField(value.matrix()))
            }
        };
        PyMatrix { inner }
    }

    #[getter]
    fn sparse_matrix(&self) -> Option<PySparseMatrix> {
        match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Integer(value),
                })
            }
            ModuleHomomorphismInner::ScalarResidue(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Residue(value),
                })
            }
            ModuleHomomorphismInner::Integer(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Integer(
                        value,
                    )),
                })
            }
            ModuleHomomorphismInner::Rational(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Rational(
                        value,
                    )),
                })
            }
            ModuleHomomorphismInner::Residue(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::Residue(
                        value,
                    )),
                })
            }
            ModuleHomomorphismInner::FiniteField(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Polynomial(PolynomialSparseMatrixInner::FiniteField(
                        value,
                    )),
                })
            }
            ModuleHomomorphismInner::QuotientRational(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Quotient(QuotientSparseMatrixInner::Rational(value)),
                })
            }
            ModuleHomomorphismInner::QuotientFiniteField(value) => {
                value.sparse_matrix().cloned().map(|value| PySparseMatrix {
                    inner: SparseMatrixInner::Quotient(QuotientSparseMatrixInner::FiniteField(
                        value,
                    )),
                })
            }
        }
    }

    fn is_sparse(&self) -> bool {
        match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => value.is_sparse(),
            ModuleHomomorphismInner::ScalarResidue(value) => value.is_sparse(),
            ModuleHomomorphismInner::Integer(value) => value.is_sparse(),
            ModuleHomomorphismInner::Rational(value) => value.is_sparse(),
            ModuleHomomorphismInner::Residue(value) => value.is_sparse(),
            ModuleHomomorphismInner::FiniteField(value) => value.is_sparse(),
            ModuleHomomorphismInner::QuotientRational(value) => value.is_sparse(),
            ModuleHomomorphismInner::QuotientFiniteField(value) => value.is_sparse(),
        }
    }

    fn apply(&self, element: &PyModuleElement) -> PyResult<PyModuleElement> {
        let inner = match (&self.inner, &element.inner) {
            (
                ModuleHomomorphismInner::ScalarInteger(map),
                ModuleElementInner::ScalarInteger(value),
            ) => ModuleElementInner::ScalarInteger(map.apply(value).map_err(module_error)?),
            (
                ModuleHomomorphismInner::ScalarResidue(map),
                ModuleElementInner::ScalarResidue(value),
            ) => ModuleElementInner::ScalarResidue(map.apply(value).map_err(module_error)?),
            (ModuleHomomorphismInner::Integer(map), ModuleElementInner::Integer(value)) => {
                ModuleElementInner::Integer(map.apply(value).map_err(module_error)?)
            }
            (ModuleHomomorphismInner::Rational(map), ModuleElementInner::Rational(value)) => {
                ModuleElementInner::Rational(map.apply(value).map_err(module_error)?)
            }
            (ModuleHomomorphismInner::Residue(map), ModuleElementInner::Residue(value)) => {
                ModuleElementInner::Residue(map.apply(value).map_err(module_error)?)
            }
            (ModuleHomomorphismInner::FiniteField(map), ModuleElementInner::FiniteField(value)) => {
                ModuleElementInner::FiniteField(map.apply(value).map_err(module_error)?)
            }
            (
                ModuleHomomorphismInner::QuotientRational(map),
                ModuleElementInner::QuotientRational(value),
            ) => ModuleElementInner::QuotientRational(map.apply(value).map_err(module_error)?),
            (
                ModuleHomomorphismInner::QuotientFiniteField(map),
                ModuleElementInner::QuotientFiniteField(value),
            ) => ModuleElementInner::QuotientFiniteField(map.apply(value).map_err(module_error)?),
            _ => {
                return Err(PyValueError::new_err(
                    "module map and element contexts do not match",
                ))
            }
        };
        Ok(PyModuleElement { inner })
    }

    fn __call__(&self, element: &PyModuleElement) -> PyResult<PyModuleElement> {
        self.apply(element)
    }

    fn compose(&self, inner: &Self) -> PyResult<Self> {
        self.binary_map(inner, "compose")
    }

    /// Map induced on the degree-`degree` symmetric power.
    fn symmetric_power(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<super::module_power::PyModulePowerHomomorphism> {
        py.detach(|| {
            super::module_power::PyModulePowerHomomorphism::from_map(
                &self.inner,
                noether::homalg::ModulePowerKind::Symmetric,
                degree,
            )
        })
        .map_err(super::module_power::module_power_error)
    }

    /// Map induced on the degree-`degree` exterior power.
    fn exterior_power(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<super::module_power::PyModulePowerHomomorphism> {
        py.detach(|| {
            super::module_power::PyModulePowerHomomorphism::from_map(
                &self.inner,
                noether::homalg::ModulePowerKind::Exterior,
                degree,
            )
        })
        .map_err(super::module_power::module_power_error)
    }

    fn tensor(&self, rhs: &Self) -> PyResult<Self> {
        self.binary_map(rhs, "tensor")
    }
    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        self.binary_map(rhs, "add")
    }
    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        self.binary_map(rhs, "sub")
    }

    fn dual(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => {
                ModuleHomomorphismInner::ScalarInteger(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::ScalarResidue(value) => {
                ModuleHomomorphismInner::ScalarResidue(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::Integer(value) => {
                ModuleHomomorphismInner::Integer(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::Rational(value) => {
                ModuleHomomorphismInner::Rational(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::Residue(value) => {
                ModuleHomomorphismInner::Residue(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::FiniteField(value) => {
                ModuleHomomorphismInner::FiniteField(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::QuotientRational(value) => {
                ModuleHomomorphismInner::QuotientRational(value.dual().map_err(module_error)?)
            }
            ModuleHomomorphismInner::QuotientFiniteField(value) => {
                ModuleHomomorphismInner::QuotientFiniteField(value.dual().map_err(module_error)?)
            }
        };
        Ok(Self { inner })
    }

    /// Exact kernel; polynomial contexts additionally require a monomial order.
    #[pyo3(signature = (order=None))]
    fn kernel(&self, py: Python<'_>, order: Option<&PyModuleMonomialOrder>) -> PyResult<Py<PyAny>> {
        let order = order.map(|value| value.inner.clone());
        let inner = py.detach(|| match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => Ok(KernelResultInner::Integer(
                value.kernel().map_err(module_error)?,
            )),
            ModuleHomomorphismInner::ScalarResidue(value) => Ok(KernelResultInner::Residue(
                value.kernel().map_err(module_error)?,
            )),
            ModuleHomomorphismInner::Integer(value) => {
                Ok(KernelResultInner::Submodule(SubmoduleInner::Integer(
                    value
                        .kernel(order.clone().ok_or_else(|| {
                            PyTypeError::new_err(
                                "polynomial module kernels require a ModuleMonomialOrder",
                            )
                        })?)
                        .map_err(module_groebner_error)?,
                )))
            }
            ModuleHomomorphismInner::Rational(value) => {
                Ok(KernelResultInner::Submodule(SubmoduleInner::Rational(
                    value
                        .kernel(order.clone().ok_or_else(|| {
                            PyTypeError::new_err(
                                "polynomial module kernels require a ModuleMonomialOrder",
                            )
                        })?)
                        .map_err(module_groebner_error)?,
                )))
            }
            ModuleHomomorphismInner::FiniteField(value) => {
                Ok(KernelResultInner::Submodule(SubmoduleInner::FiniteField(
                    value
                        .kernel(order.clone().ok_or_else(|| {
                            PyTypeError::new_err(
                                "polynomial module kernels require a ModuleMonomialOrder",
                            )
                        })?)
                        .map_err(module_groebner_error)?,
                )))
            }
            ModuleHomomorphismInner::QuotientRational(value) => Ok(KernelResultInner::Submodule(
                SubmoduleInner::QuotientRational(
                    value
                        .kernel(order.clone().ok_or_else(|| {
                            PyTypeError::new_err(
                                "polynomial module kernels require a ModuleMonomialOrder",
                            )
                        })?)
                        .map_err(module_groebner_error)?,
                ),
            )),
            ModuleHomomorphismInner::QuotientFiniteField(value) => Ok(
                KernelResultInner::Submodule(SubmoduleInner::QuotientFiniteField(
                    value
                        .kernel(order.clone().ok_or_else(|| {
                            PyTypeError::new_err(
                                "polynomial module kernels require a ModuleMonomialOrder",
                            )
                        })?)
                        .map_err(module_groebner_error)?,
                )),
            ),
            ModuleHomomorphismInner::Residue(_) => Err(PyTypeError::new_err(
                "module kernels currently require a polynomial ring over Z, Q, or FiniteField",
            )),
        })?;
        match inner {
            KernelResultInner::Integer(inner) => Ok(Py::new(
                py,
                PyKernelSubmodule {
                    inner: KernelSubmoduleInner::Integer(inner),
                },
            )?
            .into_any()),
            KernelResultInner::Residue(inner) => Ok(Py::new(
                py,
                PyKernelSubmodule {
                    inner: KernelSubmoduleInner::Residue(inner),
                },
            )?
            .into_any()),
            KernelResultInner::Submodule(inner) => {
                Ok(Py::new(py, PySubmodule { inner })?.into_any())
            }
        }
    }

    /// Image as the canonical submodule generated by this map.
    fn image(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = py.detach(|| -> PyResult<ImageResultInner> {
            Ok(match &self.inner {
                ModuleHomomorphismInner::ScalarInteger(value) => {
                    ImageResultInner::Integer(value.image().map_err(module_error)?)
                }
                ModuleHomomorphismInner::ScalarResidue(value) => {
                    ImageResultInner::Residue(value.image().map_err(module_error)?)
                }
                ModuleHomomorphismInner::Integer(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::Integer(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
                ModuleHomomorphismInner::Rational(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::Rational(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
                ModuleHomomorphismInner::Residue(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::Residue(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
                ModuleHomomorphismInner::FiniteField(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::FiniteField(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
                ModuleHomomorphismInner::QuotientRational(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::QuotientRational(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
                ModuleHomomorphismInner::QuotientFiniteField(value) => {
                    ImageResultInner::Submodule(SubmoduleInner::QuotientFiniteField(
                        Submodule::from_generator_map(value.target().clone(), value.clone())
                            .map_err(module_error)?,
                    ))
                }
            })
        })?;
        match inner {
            ImageResultInner::Integer(inner) => Ok(Py::new(
                py,
                PyImageSubmodule {
                    inner: ImageSubmoduleInner::Integer(inner),
                },
            )?
            .into_any()),
            ImageResultInner::Residue(inner) => Ok(Py::new(
                py,
                PyImageSubmodule {
                    inner: ImageSubmoduleInner::Residue(inner),
                },
            )?
            .into_any()),
            ImageResultInner::Submodule(inner) => {
                Ok(Py::new(py, PySubmodule { inner })?.into_any())
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModuleHomomorphismInner::ScalarInteger(value) => value.to_json(),
            ModuleHomomorphismInner::ScalarResidue(value) => value.to_json(),
            ModuleHomomorphismInner::Integer(value) => value.to_json(),
            ModuleHomomorphismInner::Rational(value) => value.to_json(),
            ModuleHomomorphismInner::Residue(value) => value.to_json(),
            ModuleHomomorphismInner::FiniteField(value) => value.to_json(),
            ModuleHomomorphismInner::QuotientRational(value) => value.to_json(),
            ModuleHomomorphismInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        if let Some(kind) = serialized_scalar_module_ring(value, "/source/ring")? {
            let inner = match kind {
                SerializedScalarModuleRing::Integer => ModuleHomomorphismInner::ScalarInteger(
                    ModuleHomomorphism::<IntegerRing>::from_json(value).map_err(value_error)?,
                ),
                SerializedScalarModuleRing::Residue => ModuleHomomorphismInner::ScalarResidue(
                    ModuleHomomorphism::<ResidueRing>::from_json(value).map_err(value_error)?,
                ),
            };
            return Ok(Self { inner });
        }
        let inner = match serialized_module_ring(value, "/source/ring")? {
            SerializedModuleRing::PolynomialInteger => ModuleHomomorphismInner::Integer(
                ModuleHomomorphism::<PolynomialRing<IntegerRing>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialRational => ModuleHomomorphismInner::Rational(
                ModuleHomomorphism::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialResidue => ModuleHomomorphismInner::Residue(
                ModuleHomomorphism::<PolynomialRing<ResidueRing>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => ModuleHomomorphismInner::FiniteField(
                ModuleHomomorphism::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => ModuleHomomorphismInner::QuotientRational(
                ModuleHomomorphism::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => {
                ModuleHomomorphismInner::QuotientFiniteField(
                    ModuleHomomorphism::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                        value,
                    )
                    .map_err(value_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ModuleHomomorphism(source_rank={}, target_rank={}, sparse={})",
            self.source().rank(),
            self.target().rank(),
            self.is_sparse()
        ))
    }
}

impl PyModuleHomomorphism {
    fn binary_map(&self, rhs: &Self, operation: &str) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {{
                let value = match operation {
                    "compose" => $left.compose($right),
                    "add" => $left.add($right),
                    "sub" => $left.sub($right),
                    "tensor" => $left.tensor($right),
                    _ => return Err(PyValueError::new_err("unknown module-map operation")),
                }
                .map_err(module_error)?;
                ModuleHomomorphismInner::$variant(value)
            }};
        }
        let inner = match (&self.inner, &rhs.inner) {
            (
                ModuleHomomorphismInner::ScalarInteger(left),
                ModuleHomomorphismInner::ScalarInteger(right),
            ) => apply!(left, right, ScalarInteger),
            (
                ModuleHomomorphismInner::ScalarResidue(left),
                ModuleHomomorphismInner::ScalarResidue(right),
            ) => apply!(left, right, ScalarResidue),
            (ModuleHomomorphismInner::Integer(left), ModuleHomomorphismInner::Integer(right)) => {
                apply!(left, right, Integer)
            }
            (ModuleHomomorphismInner::Rational(left), ModuleHomomorphismInner::Rational(right)) => {
                apply!(left, right, Rational)
            }
            (ModuleHomomorphismInner::Residue(left), ModuleHomomorphismInner::Residue(right)) => {
                apply!(left, right, Residue)
            }
            (
                ModuleHomomorphismInner::FiniteField(left),
                ModuleHomomorphismInner::FiniteField(right),
            ) => apply!(left, right, FiniteField),
            (
                ModuleHomomorphismInner::QuotientRational(left),
                ModuleHomomorphismInner::QuotientRational(right),
            ) => apply!(left, right, QuotientRational),
            (
                ModuleHomomorphismInner::QuotientFiniteField(left),
                ModuleHomomorphismInner::QuotientFiniteField(right),
            ) => apply!(left, right, QuotientFiniteField),
            _ => return Err(PyValueError::new_err("module map contexts do not match")),
        };
        Ok(Self { inner })
    }
}

enum KernelResultInner {
    Integer(KernelSubmodule<IntegerRing>),
    Residue(KernelSubmodule<ResidueRing>),
    Submodule(SubmoduleInner),
}

enum ImageResultInner {
    Integer(ImageSubmodule<IntegerRing>),
    Residue(ImageSubmodule<ResidueRing>),
    Submodule(SubmoduleInner),
}

#[derive(Clone)]
enum KernelSubmoduleInner {
    Integer(KernelSubmodule<IntegerRing>),
    Residue(KernelSubmodule<ResidueRing>),
}

/// A map-bound certificate for a complete exact kernel.
#[pyclass(
    module = "noether.homalg",
    name = "KernelSubmodule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyKernelSubmodule {
    inner: KernelSubmoduleInner,
}

impl Serialize for PyKernelSubmodule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.inner {
            KernelSubmoduleInner::Integer(value) => value.serialize(serializer),
            KernelSubmoduleInner::Residue(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PyKernelSubmodule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = serde_json::Value::deserialize(deserializer)?;
        let value = serde_json::to_string(&document).map_err(serde::de::Error::custom)?;
        let kind = serialized_scalar_module_ring(&value, "/homomorphism/source/ring")
            .map_err(serde::de::Error::custom)?
            .ok_or_else(|| {
                serde::de::Error::custom("unsupported exact module coefficient context")
            })?;
        let inner = match kind {
            SerializedScalarModuleRing::Integer => KernelSubmoduleInner::Integer(
                KernelSubmodule::<IntegerRing>::from_json(&value)
                    .map_err(serde::de::Error::custom)?,
            ),
            SerializedScalarModuleRing::Residue => KernelSubmoduleInner::Residue(
                KernelSubmodule::<ResidueRing>::from_json(&value)
                    .map_err(serde::de::Error::custom)?,
            ),
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyKernelSubmodule {
    #[classattr]
    fn type_id() -> &'static str {
        KernelSubmodule::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn homomorphism(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                KernelSubmoduleInner::Integer(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.homomorphism().clone())
                }
                KernelSubmoduleInner::Residue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.homomorphism().clone())
                }
            },
        }
    }

    #[getter]
    fn submodule(&self) -> PySubmodule {
        PySubmodule {
            inner: match &self.inner {
                KernelSubmoduleInner::Integer(value) => {
                    SubmoduleInner::ScalarInteger(value.submodule().clone())
                }
                KernelSubmoduleInner::Residue(value) => {
                    SubmoduleInner::ScalarResidue(value.submodule().clone())
                }
            },
        }
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match &self.inner {
                KernelSubmoduleInner::Integer(value) => value.verify(),
                KernelSubmoduleInner::Residue(value) => value.verify(),
            }
            .map_err(super::errors::verification)
        })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            match &self.inner {
                KernelSubmoduleInner::Integer(value) => value.to_json(),
                KernelSubmoduleInner::Residue(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let kind = py
            .detach(|| serialized_scalar_module_ring(value, "/homomorphism/source/ring"))?
            .ok_or_else(|| PyTypeError::new_err("unsupported exact module coefficient context"))?;
        py.detach(|| {
            Ok(Self {
                inner: match kind {
                    SerializedScalarModuleRing::Integer => KernelSubmoduleInner::Integer(
                        KernelSubmodule::<IntegerRing>::from_json(value).map_err(value_error)?,
                    ),
                    SerializedScalarModuleRing::Residue => KernelSubmoduleInner::Residue(
                        KernelSubmodule::<ResidueRing>::from_json(value).map_err(value_error)?,
                    ),
                },
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            KernelSubmoduleInner::Integer(value) => format!(
                "KernelSubmodule(source_rank={}, generator_count={})",
                value.homomorphism().source().rank(),
                value.submodule().generator_count()
            ),
            KernelSubmoduleInner::Residue(value) => format!(
                "KernelSubmodule(source_rank={}, generator_count={})",
                value.homomorphism().source().rank(),
                value.submodule().generator_count()
            ),
        })
    }
}

#[derive(Clone)]
enum ImageSubmoduleInner {
    Integer(ImageSubmodule<IntegerRing>),
    Residue(ImageSubmodule<ResidueRing>),
}

/// A map-bound certificate for a complete exact image.
#[pyclass(
    module = "noether.homalg",
    name = "ImageSubmodule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyImageSubmodule {
    inner: ImageSubmoduleInner,
}

impl Serialize for PyImageSubmodule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.inner {
            ImageSubmoduleInner::Integer(value) => value.serialize(serializer),
            ImageSubmoduleInner::Residue(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PyImageSubmodule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = serde_json::Value::deserialize(deserializer)?;
        let value = serde_json::to_string(&document).map_err(serde::de::Error::custom)?;
        let kind = serialized_scalar_module_ring(&value, "/homomorphism/target/ring")
            .map_err(serde::de::Error::custom)?
            .ok_or_else(|| {
                serde::de::Error::custom("unsupported exact module coefficient context")
            })?;
        let inner = match kind {
            SerializedScalarModuleRing::Integer => ImageSubmoduleInner::Integer(
                ImageSubmodule::<IntegerRing>::from_json(&value)
                    .map_err(serde::de::Error::custom)?,
            ),
            SerializedScalarModuleRing::Residue => ImageSubmoduleInner::Residue(
                ImageSubmodule::<ResidueRing>::from_json(&value)
                    .map_err(serde::de::Error::custom)?,
            ),
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyImageSubmodule {
    #[classattr]
    fn type_id() -> &'static str {
        ImageSubmodule::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn homomorphism(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                ImageSubmoduleInner::Integer(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.homomorphism().clone())
                }
                ImageSubmoduleInner::Residue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.homomorphism().clone())
                }
            },
        }
    }

    #[getter]
    fn submodule(&self) -> PySubmodule {
        PySubmodule {
            inner: match &self.inner {
                ImageSubmoduleInner::Integer(value) => {
                    SubmoduleInner::ScalarInteger(value.submodule().clone())
                }
                ImageSubmoduleInner::Residue(value) => {
                    SubmoduleInner::ScalarResidue(value.submodule().clone())
                }
            },
        }
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match &self.inner {
                ImageSubmoduleInner::Integer(value) => value.verify(),
                ImageSubmoduleInner::Residue(value) => value.verify(),
            }
            .map_err(super::errors::verification)
        })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            match &self.inner {
                ImageSubmoduleInner::Integer(value) => value.to_json(),
                ImageSubmoduleInner::Residue(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let kind = py
            .detach(|| serialized_scalar_module_ring(value, "/homomorphism/target/ring"))?
            .ok_or_else(|| PyTypeError::new_err("unsupported exact module coefficient context"))?;
        py.detach(|| {
            Ok(Self {
                inner: match kind {
                    SerializedScalarModuleRing::Integer => ImageSubmoduleInner::Integer(
                        ImageSubmodule::<IntegerRing>::from_json(value).map_err(value_error)?,
                    ),
                    SerializedScalarModuleRing::Residue => ImageSubmoduleInner::Residue(
                        ImageSubmodule::<ResidueRing>::from_json(value).map_err(value_error)?,
                    ),
                },
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ImageSubmoduleInner::Integer(value) => format!(
                "ImageSubmodule(target_rank={}, generator_count={})",
                value.homomorphism().target().rank(),
                value.submodule().generator_count()
            ),
            ImageSubmoduleInner::Residue(value) => format!(
                "ImageSubmodule(target_rank={}, generator_count={})",
                value.homomorphism().target().rank(),
                value.submodule().generator_count()
            ),
        })
    }
}

#[derive(Clone)]
enum SubmoduleMembershipInner {
    Integer(SubmoduleMembership<IntegerRing>),
    Residue(SubmoduleMembership<ResidueRing>),
}

/// Replayable coefficients proving exact submodule membership.
#[pyclass(
    module = "noether.homalg",
    name = "SubmoduleMembership",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySubmoduleMembership {
    inner: SubmoduleMembershipInner,
}

#[pymethods]
impl PySubmoduleMembership {
    #[classattr]
    fn type_id() -> &'static str {
        SubmoduleMembership::<IntegerRing>::TYPE_ID
    }

    #[getter]
    fn submodule(&self) -> PySubmodule {
        PySubmodule {
            inner: match &self.inner {
                SubmoduleMembershipInner::Integer(value) => {
                    SubmoduleInner::ScalarInteger(value.submodule().clone())
                }
                SubmoduleMembershipInner::Residue(value) => {
                    SubmoduleInner::ScalarResidue(value.submodule().clone())
                }
            },
        }
    }

    #[getter]
    fn element(&self) -> PyModuleElement {
        PyModuleElement {
            inner: match &self.inner {
                SubmoduleMembershipInner::Integer(value) => {
                    ModuleElementInner::ScalarInteger(value.element().clone())
                }
                SubmoduleMembershipInner::Residue(value) => {
                    ModuleElementInner::ScalarResidue(value.element().clone())
                }
            },
        }
    }

    #[getter]
    fn coefficients(&self) -> PyVector {
        PyVector {
            inner: match &self.inner {
                SubmoduleMembershipInner::Integer(value) => {
                    VectorInner::Integer(value.coefficients().clone())
                }
                SubmoduleMembershipInner::Residue(value) => {
                    VectorInner::Residue(value.coefficients().clone())
                }
            },
        }
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match &self.inner {
                SubmoduleMembershipInner::Integer(value) => value.verify(),
                SubmoduleMembershipInner::Residue(value) => value.verify(),
            }
            .map_err(super::errors::verification)
        })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            match &self.inner {
                SubmoduleMembershipInner::Integer(value) => value.to_json(),
                SubmoduleMembershipInner::Residue(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let ring_pointer = "/submodule/generator_map/target/ring";
        let kind = py
            .detach(|| serialized_scalar_module_ring(value, ring_pointer))?
            .ok_or_else(|| {
                PyTypeError::new_err(
                    "membership witnesses are supported over IntegerRing and ResidueRing",
                )
            })?;
        py.detach(|| {
            let inner = match kind {
                SerializedScalarModuleRing::Integer => SubmoduleMembershipInner::Integer(
                    SubmoduleMembership::<IntegerRing>::from_json(value).map_err(value_error)?,
                ),
                SerializedScalarModuleRing::Residue => SubmoduleMembershipInner::Residue(
                    SubmoduleMembership::<ResidueRing>::from_json(value).map_err(value_error)?,
                ),
            };
            Ok(Self { inner })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            SubmoduleMembershipInner::Integer(value) => format!(
                "SubmoduleMembership(generator_count={})",
                value.submodule().generator_count()
            ),
            SubmoduleMembershipInner::Residue(value) => format!(
                "SubmoduleMembership(generator_count={})",
                value.submodule().generator_count()
            ),
        })
    }
}

#[derive(Clone)]
enum SubmoduleInner {
    ScalarInteger(Submodule<IntegerRing>),
    ScalarResidue(Submodule<ResidueRing>),
    Integer(Submodule<PolynomialRing<IntegerRing>>),
    Rational(Submodule<PolynomialRing<RationalField>>),
    Residue(Submodule<PolynomialRing<ResidueRing>>),
    FiniteField(Submodule<PolynomialRing<FiniteField>>),
    QuotientRational(Submodule<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(Submodule<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// A finitely generated submodule of one based free module.
#[pyclass(module = "noether.homalg", name = "Submodule", frozen, from_py_object)]
#[derive(Clone)]
pub struct PySubmodule {
    inner: SubmoduleInner,
}

#[pymethods]
impl PySubmodule {
    #[classattr]
    fn type_id() -> &'static str {
        Submodule::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[new]
    fn new(
        py: Python<'_>,
        ambient: &PyFreeModule,
        generator_map: &PyModuleHomomorphism,
    ) -> PyResult<Self> {
        let inner =
            match (&ambient.inner, &generator_map.inner) {
                (
                    FreeModuleInner::ScalarInteger(ambient),
                    ModuleHomomorphismInner::ScalarInteger(map),
                ) => {
                    let ambient = ambient.clone();
                    let map = map.clone();
                    SubmoduleInner::ScalarInteger(py.detach(|| {
                        Submodule::from_generator_map(ambient, map).map_err(module_error)
                    })?)
                }
                (
                    FreeModuleInner::ScalarResidue(ambient),
                    ModuleHomomorphismInner::ScalarResidue(map),
                ) => {
                    let ambient = ambient.clone();
                    let map = map.clone();
                    SubmoduleInner::ScalarResidue(py.detach(|| {
                        Submodule::from_generator_map(ambient, map).map_err(module_error)
                    })?)
                }
                (FreeModuleInner::Integer(ambient), ModuleHomomorphismInner::Integer(map)) => {
                    SubmoduleInner::Integer(
                        Submodule::from_generator_map(ambient.clone(), map.clone())
                            .map_err(module_error)?,
                    )
                }
                (FreeModuleInner::Rational(ambient), ModuleHomomorphismInner::Rational(map)) => {
                    SubmoduleInner::Rational(
                        Submodule::from_generator_map(ambient.clone(), map.clone())
                            .map_err(module_error)?,
                    )
                }
                (FreeModuleInner::Residue(ambient), ModuleHomomorphismInner::Residue(map)) => {
                    SubmoduleInner::Residue(
                        Submodule::from_generator_map(ambient.clone(), map.clone())
                            .map_err(module_error)?,
                    )
                }
                (
                    FreeModuleInner::FiniteField(ambient),
                    ModuleHomomorphismInner::FiniteField(map),
                ) => SubmoduleInner::FiniteField(
                    Submodule::from_generator_map(ambient.clone(), map.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientRational(ambient),
                    ModuleHomomorphismInner::QuotientRational(map),
                ) => SubmoduleInner::QuotientRational(
                    Submodule::from_generator_map(ambient.clone(), map.clone())
                        .map_err(module_error)?,
                ),
                (
                    FreeModuleInner::QuotientFiniteField(ambient),
                    ModuleHomomorphismInner::QuotientFiniteField(map),
                ) => SubmoduleInner::QuotientFiniteField(
                    Submodule::from_generator_map(ambient.clone(), map.clone())
                        .map_err(module_error)?,
                ),
                _ => {
                    return Err(PyValueError::new_err(
                        "ambient module and generator map contexts do not match",
                    ))
                }
            };
        Ok(Self { inner })
    }

    #[classmethod]
    fn from_generators(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        ambient: &PyFreeModule,
        generators: Vec<PyModuleElement>,
    ) -> PyResult<Self> {
        let inner = match &ambient.inner {
            FreeModuleInner::ScalarInteger(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::ScalarInteger(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let ambient = ambient.clone();
                SubmoduleInner::ScalarInteger(
                    py.detach(|| {
                        Submodule::from_generators(ambient, values).map_err(module_error)
                    })?,
                )
            }
            FreeModuleInner::ScalarResidue(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::ScalarResidue(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let ambient = ambient.clone();
                SubmoduleInner::ScalarResidue(
                    py.detach(|| {
                        Submodule::from_generators(ambient, values).map_err(module_error)
                    })?,
                )
            }
            FreeModuleInner::Integer(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::Integer(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::Integer(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
            FreeModuleInner::Rational(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::Rational(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::Rational(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
            FreeModuleInner::Residue(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::Residue(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::Residue(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
            FreeModuleInner::FiniteField(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::FiniteField(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::FiniteField(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
            FreeModuleInner::QuotientRational(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::QuotientRational(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::QuotientRational(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
            FreeModuleInner::QuotientFiniteField(ambient) => {
                let values = generators
                    .into_iter()
                    .map(|generator| match generator.inner {
                        ModuleElementInner::QuotientFiniteField(value) => Ok(value),
                        _ => Err(PyValueError::new_err(
                            "ambient module and generator contexts do not match",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                SubmoduleInner::QuotientFiniteField(
                    Submodule::from_generators(ambient.clone(), values).map_err(module_error)?,
                )
            }
        };
        Ok(Self { inner })
    }
    #[getter]
    fn ambient(&self) -> PyFreeModule {
        PyFreeModule {
            inner: match &self.inner {
                SubmoduleInner::ScalarInteger(value) => {
                    FreeModuleInner::ScalarInteger(value.ambient().clone())
                }
                SubmoduleInner::ScalarResidue(value) => {
                    FreeModuleInner::ScalarResidue(value.ambient().clone())
                }
                SubmoduleInner::Integer(value) => FreeModuleInner::Integer(value.ambient().clone()),
                SubmoduleInner::Rational(value) => {
                    FreeModuleInner::Rational(value.ambient().clone())
                }
                SubmoduleInner::Residue(value) => FreeModuleInner::Residue(value.ambient().clone()),
                SubmoduleInner::FiniteField(value) => {
                    FreeModuleInner::FiniteField(value.ambient().clone())
                }
                SubmoduleInner::QuotientRational(value) => {
                    FreeModuleInner::QuotientRational(value.ambient().clone())
                }
                SubmoduleInner::QuotientFiniteField(value) => {
                    FreeModuleInner::QuotientFiniteField(value.ambient().clone())
                }
            },
        }
    }
    #[getter]
    fn generator_map(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                SubmoduleInner::ScalarInteger(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.generator_map().clone())
                }
                SubmoduleInner::ScalarResidue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.generator_map().clone())
                }
                SubmoduleInner::Integer(value) => {
                    ModuleHomomorphismInner::Integer(value.generator_map().clone())
                }
                SubmoduleInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.generator_map().clone())
                }
                SubmoduleInner::Residue(value) => {
                    ModuleHomomorphismInner::Residue(value.generator_map().clone())
                }
                SubmoduleInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.generator_map().clone())
                }
                SubmoduleInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.generator_map().clone())
                }
                SubmoduleInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.generator_map().clone())
                }
            },
        }
    }
    #[getter]
    fn generator_count(&self) -> usize {
        match &self.inner {
            SubmoduleInner::ScalarInteger(value) => value.generator_count(),
            SubmoduleInner::ScalarResidue(value) => value.generator_count(),
            SubmoduleInner::Integer(value) => value.generator_count(),
            SubmoduleInner::Rational(value) => value.generator_count(),
            SubmoduleInner::Residue(value) => value.generator_count(),
            SubmoduleInner::FiniteField(value) => value.generator_count(),
            SubmoduleInner::QuotientRational(value) => value.generator_count(),
            SubmoduleInner::QuotientFiniteField(value) => value.generator_count(),
        }
    }

    /// Canonical generators for exact integer and residue submodules.
    #[getter]
    fn generators(&self, py: Python<'_>) -> PyResult<Vec<PyModuleElement>> {
        let values = py.detach(|| match &self.inner {
            SubmoduleInner::ScalarInteger(value) => Ok(value
                .generators()
                .into_iter()
                .map(ModuleElementInner::ScalarInteger)
                .collect::<Vec<_>>()),
            SubmoduleInner::ScalarResidue(value) => Ok(value
                .generators()
                .into_iter()
                .map(ModuleElementInner::ScalarResidue)
                .collect::<Vec<_>>()),
            _ => Err(PyTypeError::new_err(
                "canonical generator enumeration is exposed for IntegerRing and ResidueRing submodules",
            )),
        })?;
        Ok(values
            .into_iter()
            .map(|inner| PyModuleElement { inner })
            .collect())
    }

    /// Decide exact membership.
    fn contains(&self, py: Python<'_>, element: &PyModuleElement) -> PyResult<bool> {
        py.detach(|| match (&self.inner, &element.inner) {
            (SubmoduleInner::ScalarInteger(value), ModuleElementInner::ScalarInteger(element)) => {
                value.contains(element).map_err(module_error)
            }
            (SubmoduleInner::ScalarResidue(value), ModuleElementInner::ScalarResidue(element)) => {
                value.contains(element).map_err(module_error)
            }
            _ => Err(PyTypeError::new_err(
                "exact membership requires matching IntegerRing or ResidueRing contexts",
            )),
        })
    }

    /// Return replayable coefficients exactly when membership holds.
    fn membership_witness(
        &self,
        py: Python<'_>,
        element: &PyModuleElement,
    ) -> PyResult<Option<PySubmoduleMembership>> {
        py.detach(|| match (&self.inner, &element.inner) {
            (SubmoduleInner::ScalarInteger(value), ModuleElementInner::ScalarInteger(element)) => {
                Ok(value
                    .membership_witness(element)
                    .map_err(module_error)?
                    .map(|inner| PySubmoduleMembership {
                        inner: SubmoduleMembershipInner::Integer(inner),
                    }))
            }
            (SubmoduleInner::ScalarResidue(value), ModuleElementInner::ScalarResidue(element)) => {
                Ok(value
                    .membership_witness(element)
                    .map_err(module_error)?
                    .map(|inner| PySubmoduleMembership {
                        inner: SubmoduleMembershipInner::Residue(inner),
                    }))
            }
            _ => Err(PyTypeError::new_err(
                "membership witnesses require matching IntegerRing or ResidueRing contexts",
            )),
        })
    }

    /// Cokernel presentation of the ambient module modulo this submodule.
    fn quotient(&self) -> PyResult<PyPresentedModule> {
        let inner = match &self.inner {
            SubmoduleInner::ScalarInteger(value) => {
                PresentedModuleInner::ScalarInteger(value.quotient())
            }
            SubmoduleInner::ScalarResidue(value) => {
                PresentedModuleInner::ScalarResidue(value.quotient())
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "canonical quotient invariants are exposed for IntegerRing and ResidueRing submodules",
                ))
            }
        };
        Ok(PyPresentedModule { inner })
    }

    /// Invariant factors of the ambient quotient as an abelian group.
    fn quotient_invariants(&self, py: Python<'_>) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        let inner = py.detach(|| match &self.inner {
            SubmoduleInner::ScalarInteger(value) => Ok(value.quotient_invariants()),
            SubmoduleInner::ScalarResidue(value) => Ok(value.quotient_invariants()),
            _ => Err(PyTypeError::new_err(
                "quotient invariants are exposed for IntegerRing and ResidueRing submodules",
            )),
        })?;
        Ok(PyFinitelyGeneratedAbelianGroup { inner })
    }

    /// Finite index, rejecting positive-free-rank integer quotients.
    fn finite_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let index = py.detach(|| match &self.inner {
            SubmoduleInner::ScalarInteger(value) => value.finite_index().map_err(module_error),
            SubmoduleInner::ScalarResidue(value) => Ok(value.finite_index()),
            _ => Err(PyTypeError::new_err(
                "finite index is exposed for IntegerRing and ResidueRing submodules",
            )),
        })?;
        integer_to_python(py, &index)
    }

    /// Primitive closure inside the ambient integer lattice.
    fn saturation(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| match &self.inner {
            SubmoduleInner::ScalarInteger(value) => Ok(SubmoduleInner::ScalarInteger(
                value.saturation().map_err(module_error)?,
            )),
            _ => Err(PyTypeError::new_err(
                "primitive saturation is defined here for IntegerRing submodules",
            )),
        })?;
        Ok(Self { inner })
    }

    /// Recheck the canonical integer/Howell generator form.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match &self.inner {
                SubmoduleInner::ScalarInteger(value) => value.verify(),
                SubmoduleInner::ScalarResidue(value) => value.verify(),
                _ => return Err(PyTypeError::new_err(
                    "canonical verification is exposed for IntegerRing and ResidueRing submodules",
                )),
            }
            .map_err(super::errors::verification)
        })
    }

    fn sum(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {
                SubmoduleInner::$variant($left.sum($right).map_err(module_error)?)
            };
        }
        let inner = py.detach(|| -> PyResult<SubmoduleInner> {
            Ok(match (&self.inner, &rhs.inner) {
                (SubmoduleInner::ScalarInteger(left), SubmoduleInner::ScalarInteger(right)) => {
                    apply!(left, right, ScalarInteger)
                }
                (SubmoduleInner::ScalarResidue(left), SubmoduleInner::ScalarResidue(right)) => {
                    apply!(left, right, ScalarResidue)
                }
                (SubmoduleInner::Integer(left), SubmoduleInner::Integer(right)) => {
                    apply!(left, right, Integer)
                }
                (SubmoduleInner::Rational(left), SubmoduleInner::Rational(right)) => {
                    apply!(left, right, Rational)
                }
                (SubmoduleInner::Residue(left), SubmoduleInner::Residue(right)) => {
                    apply!(left, right, Residue)
                }
                (SubmoduleInner::FiniteField(left), SubmoduleInner::FiniteField(right)) => {
                    apply!(left, right, FiniteField)
                }
                (
                    SubmoduleInner::QuotientRational(left),
                    SubmoduleInner::QuotientRational(right),
                ) => {
                    apply!(left, right, QuotientRational)
                }
                (
                    SubmoduleInner::QuotientFiniteField(left),
                    SubmoduleInner::QuotientFiniteField(right),
                ) => apply!(left, right, QuotientFiniteField),
                _ => {
                    return Err(PyValueError::new_err(
                        "submodule coefficient contexts do not match",
                    ))
                }
            })
        })?;
        Ok(Self { inner })
    }
    #[pyo3(signature = (rhs, order=None))]
    fn is_subset_of(
        &self,
        py: Python<'_>,
        rhs: &Self,
        order: Option<&PyModuleMonomialOrder>,
    ) -> PyResult<bool> {
        let order = order.map(|value| value.inner.clone());
        py.detach(|| match (&self.inner, &rhs.inner) {
            (SubmoduleInner::ScalarInteger(left), SubmoduleInner::ScalarInteger(right)) => {
                left.is_subset_of(right).map_err(module_error)
            }
            (SubmoduleInner::ScalarResidue(left), SubmoduleInner::ScalarResidue(right)) => {
                left.is_subset_of(right).map_err(module_error)
            }
            (SubmoduleInner::Rational(left), SubmoduleInner::Rational(right)) => left
                .is_subset_of(
                    right,
                    order.clone().ok_or_else(|| {
                        PyTypeError::new_err(
                            "polynomial submodule containment requires a ModuleMonomialOrder",
                        )
                    })?,
                )
                .map_err(module_groebner_error),
            (SubmoduleInner::FiniteField(left), SubmoduleInner::FiniteField(right)) => left
                .is_subset_of(
                    right,
                    order.clone().ok_or_else(|| {
                        PyTypeError::new_err(
                            "polynomial submodule containment requires a ModuleMonomialOrder",
                        )
                    })?,
                )
                .map_err(module_groebner_error),
            (
                SubmoduleInner::Integer(_)
                | SubmoduleInner::Residue(_)
                | SubmoduleInner::QuotientRational(_)
                | SubmoduleInner::QuotientFiniteField(_),
                _,
            ) => Err(PyTypeError::new_err(
                "submodule containment currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => Err(PyValueError::new_err(
                "submodule coefficient contexts do not match",
            )),
        })
    }
    #[pyo3(signature = (rhs, order=None))]
    fn is_equal(
        &self,
        py: Python<'_>,
        rhs: &Self,
        order: Option<&PyModuleMonomialOrder>,
    ) -> PyResult<bool> {
        let order = order.map(|value| value.inner.clone());
        py.detach(|| match (&self.inner, &rhs.inner) {
            (SubmoduleInner::ScalarInteger(left), SubmoduleInner::ScalarInteger(right)) => {
                Ok(left == right)
            }
            (SubmoduleInner::ScalarResidue(left), SubmoduleInner::ScalarResidue(right)) => {
                Ok(left == right)
            }
            (SubmoduleInner::Rational(left), SubmoduleInner::Rational(right)) => left
                .is_equal(
                    right,
                    order.clone().ok_or_else(|| {
                        PyTypeError::new_err(
                            "polynomial submodule equality requires a ModuleMonomialOrder",
                        )
                    })?,
                )
                .map_err(module_groebner_error),
            (SubmoduleInner::FiniteField(left), SubmoduleInner::FiniteField(right)) => left
                .is_equal(
                    right,
                    order.clone().ok_or_else(|| {
                        PyTypeError::new_err(
                            "polynomial submodule equality requires a ModuleMonomialOrder",
                        )
                    })?,
                )
                .map_err(module_groebner_error),
            (
                SubmoduleInner::Integer(_)
                | SubmoduleInner::Residue(_)
                | SubmoduleInner::QuotientRational(_)
                | SubmoduleInner::QuotientFiniteField(_),
                _,
            ) => Err(PyTypeError::new_err(
                "submodule equality currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => Err(PyValueError::new_err(
                "submodule coefficient contexts do not match",
            )),
        })
    }
    #[pyo3(signature = (rhs, order=None))]
    fn intersection(
        &self,
        py: Python<'_>,
        rhs: &Self,
        order: Option<&PyModuleMonomialOrder>,
    ) -> PyResult<Self> {
        let order = order.map(|value| value.inner.clone());
        let inner = py.detach(|| -> PyResult<SubmoduleInner> {
            Ok(match (&self.inner, &rhs.inner) {
            (SubmoduleInner::ScalarInteger(left), SubmoduleInner::ScalarInteger(right)) => {
                SubmoduleInner::ScalarInteger(left.intersection(right).map_err(module_error)?)
            }
            (SubmoduleInner::ScalarResidue(left), SubmoduleInner::ScalarResidue(right)) => {
                SubmoduleInner::ScalarResidue(left.intersection(right).map_err(module_error)?)
            }
            (SubmoduleInner::Rational(left), SubmoduleInner::Rational(right)) => {
                SubmoduleInner::Rational(
                    left.intersection(right, order.clone().ok_or_else(|| PyTypeError::new_err("polynomial submodule intersection requires a ModuleMonomialOrder"))?)
                        .map_err(module_groebner_error)?,
                )
            }
            (SubmoduleInner::FiniteField(left), SubmoduleInner::FiniteField(right)) => {
                SubmoduleInner::FiniteField(
                    left.intersection(right, order.clone().ok_or_else(|| PyTypeError::new_err("polynomial submodule intersection requires a ModuleMonomialOrder"))?)
                        .map_err(module_groebner_error)?,
                )
            }
            (
                SubmoduleInner::Integer(_)
                | SubmoduleInner::Residue(_)
                | SubmoduleInner::QuotientRational(_)
                | SubmoduleInner::QuotientFiniteField(_),
                _,
            ) => return Err(PyTypeError::new_err(
                "submodule intersection currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => {
                return Err(PyValueError::new_err(
                    "submodule coefficient contexts do not match",
                ))
            }
        })
        })?;
        Ok(Self { inner })
    }
    fn colon_by_element(
        &self,
        py: Python<'_>,
        scalar: &PyPolynomial,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<Self> {
        let order = order.inner.clone();
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match (&self.inner, &scalar.inner) {
                (SubmoduleInner::Rational(value), PolynomialInner::Rational(scalar)) => {
                    SubmoduleInner::Rational(
                        value
                            .colon_by_element(scalar, order.clone())
                            .map_err(module_groebner_error)?,
                    )
                }
                (SubmoduleInner::FiniteField(value), PolynomialInner::FiniteField(scalar)) => {
                    SubmoduleInner::FiniteField(
                        value
                            .colon_by_element(scalar, order.clone())
                            .map_err(module_groebner_error)?,
                    )
                }
                (
                    SubmoduleInner::Integer(_)
                    | SubmoduleInner::Residue(_)
                    | SubmoduleInner::QuotientRational(_)
                    | SubmoduleInner::QuotientFiniteField(_),
                    _,
                ) => return Err(PyTypeError::new_err(
                    "submodule colons currently require a polynomial ring over Q or FiniteField",
                )),
                _ => {
                    return Err(PyValueError::new_err(
                        "submodule and scalar contexts do not match",
                    ))
                }
            })
        })?;
        Ok(Self { inner })
    }
    fn saturate_by_element(
        &self,
        py: Python<'_>,
        scalar: &PyPolynomial,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<Self> {
        let order = order.inner.clone();
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match (&self.inner, &scalar.inner) {
            (SubmoduleInner::Rational(value), PolynomialInner::Rational(scalar)) => {
                SubmoduleInner::Rational(
                    value
                        .saturate_by_element(scalar, order.clone())
                        .map_err(module_groebner_error)?,
                )
            }
            (SubmoduleInner::FiniteField(value), PolynomialInner::FiniteField(scalar)) => {
                SubmoduleInner::FiniteField(
                    value
                        .saturate_by_element(scalar, order.clone())
                        .map_err(module_groebner_error)?,
                )
            }
            (
                SubmoduleInner::Integer(_)
                | SubmoduleInner::Residue(_)
                | SubmoduleInner::QuotientRational(_)
                | SubmoduleInner::QuotientFiniteField(_),
                _,
            ) => return Err(PyTypeError::new_err(
                "submodule saturation currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => {
                return Err(PyValueError::new_err(
                    "submodule and scalar contexts do not match",
                ))
            }
            })
        })?;
        Ok(Self { inner })
    }
    fn groebner_basis(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyModuleGroebnerBasis> {
        py.detach(|| {
            let inner = match &self.inner {
            SubmoduleInner::Rational(value) => ModuleGroebnerBasisInner::Rational(
                value
                    .groebner_basis(order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::FiniteField(value) => ModuleGroebnerBasisInner::FiniteField(
                value
                    .groebner_basis(order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::QuotientRational(value) => ModuleGroebnerBasisInner::QuotientRational(
                value
                    .groebner_basis(order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::QuotientFiniteField(value) => {
                ModuleGroebnerBasisInner::QuotientFiniteField(
                    value
                        .groebner_basis(order.inner.clone())
                        .map_err(module_groebner_error)?,
                )
            }
            SubmoduleInner::ScalarInteger(_)
            | SubmoduleInner::ScalarResidue(_)
            | SubmoduleInner::Integer(_)
            | SubmoduleInner::Residue(_) => {
                return Err(PyTypeError::new_err(
                    "module Groebner bases currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PyModuleGroebnerBasis { inner })
        })
    }
    fn strong_groebner_basis(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyStrongModuleGroebnerBasis> {
        py.detach(|| {
            let SubmoduleInner::Integer(value) = &self.inner else {
                return Err(PyTypeError::new_err(
                    "strong module Groebner bases require a polynomial ring over Z",
                ));
            };
            Ok(PyStrongModuleGroebnerBasis {
                inner: value
                    .strong_groebner_basis(order.inner.clone())
                    .map_err(module_groebner_error)?,
            })
        })
    }
    fn groebner_basis_with_hilbert_series(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
        target: &PyHilbertSeries,
    ) -> PyResult<PyModuleGroebnerBasis> {
        py.detach(|| {
            let inner = match &self.inner {
            SubmoduleInner::Rational(value) => ModuleGroebnerBasisInner::Rational(
                value
                    .groebner_basis_with_hilbert_series(order.inner.clone(), &target.inner)
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::FiniteField(value) => ModuleGroebnerBasisInner::FiniteField(
                value
                    .groebner_basis_with_hilbert_series(order.inner.clone(), &target.inner)
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::ScalarInteger(_)
            | SubmoduleInner::ScalarResidue(_)
            | SubmoduleInner::Integer(_)
            | SubmoduleInner::Residue(_)
            | SubmoduleInner::QuotientRational(_)
            | SubmoduleInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "module Groebner bases currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PyModuleGroebnerBasis { inner })
        })
    }
    fn minimal_homogeneous_generators(&self, py: Python<'_>) -> PyResult<PySubmodule> {
        py.detach(|| {
            let inner = match &self.inner {
            SubmoduleInner::Rational(value) => SubmoduleInner::Rational(
                value
                    .minimal_homogeneous_generators()
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::FiniteField(value) => SubmoduleInner::FiniteField(
                value
                    .minimal_homogeneous_generators()
                    .map_err(module_groebner_error)?,
            ),
            SubmoduleInner::ScalarInteger(_)
            | SubmoduleInner::ScalarResidue(_)
            | SubmoduleInner::Integer(_)
            | SubmoduleInner::Residue(_)
            | SubmoduleInner::QuotientRational(_)
            | SubmoduleInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "minimal homogeneous module generators currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PySubmodule { inner })
        })
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            match &self.inner {
                SubmoduleInner::ScalarInteger(value) => value.to_json(),
                SubmoduleInner::ScalarResidue(value) => value.to_json(),
                SubmoduleInner::Integer(value) => value.to_json(),
                SubmoduleInner::Rational(value) => value.to_json(),
                SubmoduleInner::Residue(value) => value.to_json(),
                SubmoduleInner::FiniteField(value) => value.to_json(),
                SubmoduleInner::QuotientRational(value) => value.to_json(),
                SubmoduleInner::QuotientFiniteField(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        if let Some(ring) =
            py.detach(|| serialized_scalar_module_ring(value, "/generator_map/source/ring"))?
        {
            return py.detach(|| {
                Ok(Self {
                    inner: match ring {
                        SerializedScalarModuleRing::Integer => SubmoduleInner::ScalarInteger(
                            Submodule::<IntegerRing>::from_json(value).map_err(value_error)?,
                        ),
                        SerializedScalarModuleRing::Residue => SubmoduleInner::ScalarResidue(
                            Submodule::<ResidueRing>::from_json(value).map_err(value_error)?,
                        ),
                    },
                })
            });
        }
        let inner = match serialized_module_ring(value, "/generator_map/source/ring")? {
            SerializedModuleRing::PolynomialInteger => SubmoduleInner::Integer(
                Submodule::<PolynomialRing<IntegerRing>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialRational => SubmoduleInner::Rational(
                Submodule::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialResidue => SubmoduleInner::Residue(
                Submodule::<PolynomialRing<ResidueRing>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => SubmoduleInner::FiniteField(
                Submodule::<PolynomialRing<FiniteField>>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => SubmoduleInner::QuotientRational(
                Submodule::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => SubmoduleInner::QuotientFiniteField(
                Submodule::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
        };
        Ok(Self { inner })
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "Submodule(ambient_rank={}, generators={})",
            self.ambient().rank(),
            self.generator_count()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum PresentedModuleInner {
    ScalarInteger(PresentedModule<IntegerRing>),
    ScalarResidue(PresentedModule<ResidueRing>),
    Integer(PresentedModule<PolynomialRing<IntegerRing>>),
    Rational(PresentedModule<PolynomialRing<RationalField>>),
    Residue(PresentedModule<PolynomialRing<ResidueRing>>),
    FiniteField(PresentedModule<PolynomialRing<FiniteField>>),
    QuotientRational(PresentedModule<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(PresentedModule<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// A finitely presented module retaining an exact cokernel presentation.
#[pyclass(
    module = "noether.homalg",
    name = "PresentedModule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPresentedModule {
    pub(crate) inner: PresentedModuleInner,
}

impl PyPresentedModule {
    pub(crate) fn from_rational_polynomial(
        value: PresentedModule<PolynomialRing<RationalField>>,
    ) -> Self {
        Self {
            inner: PresentedModuleInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field_polynomial(
        value: PresentedModule<PolynomialRing<FiniteField>>,
    ) -> Self {
        Self {
            inner: PresentedModuleInner::FiniteField(value),
        }
    }

    pub(crate) fn from_rational_quotient(
        value: PresentedModule<QuotientRing<PolynomialRing<RationalField>>>,
    ) -> Self {
        Self {
            inner: PresentedModuleInner::QuotientRational(value),
        }
    }

    pub(crate) fn from_finite_field_quotient(
        value: PresentedModule<QuotientRing<PolynomialRing<FiniteField>>>,
    ) -> Self {
        Self {
            inner: PresentedModuleInner::QuotientFiniteField(value),
        }
    }
}

#[pymethods]
impl PyPresentedModule {
    #[classattr]
    fn type_id() -> &'static str {
        PresentedModule::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[new]
    fn new(presentation: &PyModuleHomomorphism) -> Self {
        Self {
            inner: match &presentation.inner {
                ModuleHomomorphismInner::ScalarInteger(value) => {
                    PresentedModuleInner::ScalarInteger(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::ScalarResidue(value) => {
                    PresentedModuleInner::ScalarResidue(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::Integer(value) => {
                    PresentedModuleInner::Integer(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleInner::Rational(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::Residue(value) => {
                    PresentedModuleInner::Residue(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(PresentedModule::cokernel(value.clone()))
                }
                ModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(PresentedModule::cokernel(
                        value.clone(),
                    ))
                }
            },
        }
    }
    #[getter]
    fn presentation(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleInner::ScalarInteger(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.presentation().clone())
                }
                PresentedModuleInner::ScalarResidue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.presentation().clone())
                }
                PresentedModuleInner::Integer(value) => {
                    ModuleHomomorphismInner::Integer(value.presentation().clone())
                }
                PresentedModuleInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.presentation().clone())
                }
                PresentedModuleInner::Residue(value) => {
                    ModuleHomomorphismInner::Residue(value.presentation().clone())
                }
                PresentedModuleInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.presentation().clone())
                }
                PresentedModuleInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.presentation().clone())
                }
                PresentedModuleInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.presentation().clone())
                }
            },
        }
    }
    #[getter]
    fn generator_module(&self) -> PyFreeModule {
        self.presentation().target()
    }
    #[getter]
    fn relation_module(&self) -> PyFreeModule {
        self.presentation().source()
    }
    #[getter]
    fn generator_count(&self) -> usize {
        match &self.inner {
            PresentedModuleInner::ScalarInteger(value) => value.generator_count(),
            PresentedModuleInner::ScalarResidue(value) => value.generator_count(),
            PresentedModuleInner::Integer(value) => value.generator_count(),
            PresentedModuleInner::Rational(value) => value.generator_count(),
            PresentedModuleInner::Residue(value) => value.generator_count(),
            PresentedModuleInner::FiniteField(value) => value.generator_count(),
            PresentedModuleInner::QuotientRational(value) => value.generator_count(),
            PresentedModuleInner::QuotientFiniteField(value) => value.generator_count(),
        }
    }
    #[getter]
    fn relation_count(&self) -> usize {
        match &self.inner {
            PresentedModuleInner::ScalarInteger(value) => value.relation_count(),
            PresentedModuleInner::ScalarResidue(value) => value.relation_count(),
            PresentedModuleInner::Integer(value) => value.relation_count(),
            PresentedModuleInner::Rational(value) => value.relation_count(),
            PresentedModuleInner::Residue(value) => value.relation_count(),
            PresentedModuleInner::FiniteField(value) => value.relation_count(),
            PresentedModuleInner::QuotientRational(value) => value.relation_count(),
            PresentedModuleInner::QuotientFiniteField(value) => value.relation_count(),
        }
    }
    fn annihilator(&self, py: Python<'_>, order: &PyModuleMonomialOrder) -> PyResult<PyIdeal> {
        // NOETHER_RELEASES_GIL: annihilator computation runs module Gröbner bases.
        py.detach(|| match &self.inner {
            PresentedModuleInner::Rational(value) => Ok(PyIdeal::from_rational_polynomial(
                value
                    .annihilator(order.inner.clone())
                    .map_err(module_invariant_error)?,
            )),
            PresentedModuleInner::FiniteField(value) => Ok(PyIdeal::from_finite_field_polynomial(
                value
                    .annihilator(order.inner.clone())
                    .map_err(module_invariant_error)?,
            )),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => Err(PyTypeError::new_err(
                "annihilators currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }
    fn associated_primes(&self, py: Python<'_>) -> PyResult<PyModuleAssociatedPrimeSet> {
        // NOETHER_RELEASES_GIL: associated-prime computation is unbounded exact work.
        py.detach(|| match &self.inner {
            PresentedModuleInner::Rational(value) => Ok(
                PyModuleAssociatedPrimeSet::from_rational(
                    value.associated_primes().map_err(associated_prime_error)?,
                ),
            ),
            PresentedModuleInner::FiniteField(value) => Ok(
                PyModuleAssociatedPrimeSet::from_finite_field(
                    value.associated_primes().map_err(associated_prime_error)?,
                ),
            ),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => Err(PyTypeError::new_err(
                "module associated primes currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }
    fn primary_decomposition(&self, py: Python<'_>) -> PyResult<PyModulePrimaryDecomposition> {
        // NOETHER_RELEASES_GIL: primary decomposition is unbounded exact work.
        py.detach(|| match &self.inner {
            PresentedModuleInner::Rational(value) => Ok(
                PyModulePrimaryDecomposition::from_rational(
                    value.primary_decomposition().map_err(module_primary_error)?,
                ),
            ),
            PresentedModuleInner::FiniteField(value) => Ok(
                PyModulePrimaryDecomposition::from_finite_field(
                    value.primary_decomposition().map_err(module_primary_error)?,
                ),
            ),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => Err(PyTypeError::new_err(
                "module primary decomposition currently requires a polynomial ring over Q or FiniteField",
            )),
        })
    }

    fn symmetric_power(&self, py: Python<'_>, degree: usize) -> PyResult<PyPresentedModulePower> {
        // NOETHER_RELEASES_GIL: relation construction can be material.
        py.detach(|| match &self.inner {
            PresentedModuleInner::ScalarInteger(value) => {
                Ok(PyPresentedModulePower::from_scalar_integer(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::ScalarResidue(value) => {
                Ok(PyPresentedModulePower::from_scalar_residue(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::Integer(value) => Ok(PyPresentedModulePower::from_integer(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::Rational(value) => Ok(PyPresentedModulePower::from_rational(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::Residue(value) => Ok(PyPresentedModulePower::from_residue(
                value.symmetric_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::FiniteField(value) => {
                Ok(PyPresentedModulePower::from_finite_field(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::QuotientRational(value) => {
                Ok(PyPresentedModulePower::from_quotient_rational(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::QuotientFiniteField(value) => {
                Ok(PyPresentedModulePower::from_quotient_finite_field(
                    value.symmetric_power(degree).map_err(module_power_error)?,
                ))
            }
        })
    }

    fn exterior_power(&self, py: Python<'_>, degree: usize) -> PyResult<PyPresentedModulePower> {
        // NOETHER_RELEASES_GIL: relation construction can be material.
        py.detach(|| match &self.inner {
            PresentedModuleInner::ScalarInteger(value) => {
                Ok(PyPresentedModulePower::from_scalar_integer(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::ScalarResidue(value) => {
                Ok(PyPresentedModulePower::from_scalar_residue(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::Integer(value) => Ok(PyPresentedModulePower::from_integer(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::Rational(value) => Ok(PyPresentedModulePower::from_rational(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::Residue(value) => Ok(PyPresentedModulePower::from_residue(
                value.exterior_power(degree).map_err(module_power_error)?,
            )),
            PresentedModuleInner::FiniteField(value) => {
                Ok(PyPresentedModulePower::from_finite_field(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::QuotientRational(value) => {
                Ok(PyPresentedModulePower::from_quotient_rational(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
            PresentedModuleInner::QuotientFiniteField(value) => {
                Ok(PyPresentedModulePower::from_quotient_finite_field(
                    value.exterior_power(degree).map_err(module_power_error)?,
                ))
            }
        })
    }
    fn fitting_ideal(&self, py: Python<'_>, index: usize) -> PyResult<PyIdeal> {
        // NOETHER_RELEASES_GIL: minors and ideal canonicalization can be material.
        py.detach(|| match &self.inner {
            PresentedModuleInner::Rational(value) => Ok(PyIdeal::from_rational_polynomial(
                value.fitting_ideal(index).map_err(module_invariant_error)?,
            )),
            PresentedModuleInner::FiniteField(value) => Ok(PyIdeal::from_finite_field_polynomial(
                value.fitting_ideal(index).map_err(module_invariant_error)?,
            )),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => Err(PyTypeError::new_err(
                "Fitting ideals currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }
    fn symmetric_algebra(&self, py: Python<'_>) -> PyResult<PySymmetricAlgebra> {
        // NOETHER_RELEASES_GIL: the quotient presentation can be material.
        py.detach(|| match &self.inner {
            PresentedModuleInner::Rational(value) => Ok(PySymmetricAlgebra::from_rational(
                value.symmetric_algebra().map_err(construction_error)?,
            )),
            PresentedModuleInner::FiniteField(value) => Ok(PySymmetricAlgebra::from_finite_field(
                value.symmetric_algebra().map_err(construction_error)?,
            )),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => Err(PyTypeError::new_err(
                "symmetric algebras currently require a polynomial ring over Q or FiniteField",
            )),
        })
    }
    fn tensor(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: tensor presentations can be material.
        py.detach(|| {
            let inner = match (&self.inner, &rhs.inner) {
                (
                    PresentedModuleInner::ScalarInteger(left),
                    PresentedModuleInner::ScalarInteger(right),
                ) => PresentedModuleInner::ScalarInteger(left.tensor(right).map_err(module_error)?),
                (
                    PresentedModuleInner::ScalarResidue(left),
                    PresentedModuleInner::ScalarResidue(right),
                ) => PresentedModuleInner::ScalarResidue(left.tensor(right).map_err(module_error)?),
                (PresentedModuleInner::Integer(left), PresentedModuleInner::Integer(right)) => {
                    PresentedModuleInner::Integer(left.tensor(right).map_err(module_error)?)
                }
                (PresentedModuleInner::Rational(left), PresentedModuleInner::Rational(right)) => {
                    PresentedModuleInner::Rational(left.tensor(right).map_err(module_error)?)
                }
                (PresentedModuleInner::Residue(left), PresentedModuleInner::Residue(right)) => {
                    PresentedModuleInner::Residue(left.tensor(right).map_err(module_error)?)
                }
                (
                    PresentedModuleInner::FiniteField(left),
                    PresentedModuleInner::FiniteField(right),
                ) => PresentedModuleInner::FiniteField(left.tensor(right).map_err(module_error)?),
                (
                    PresentedModuleInner::QuotientRational(left),
                    PresentedModuleInner::QuotientRational(right),
                ) => PresentedModuleInner::QuotientRational(
                    left.tensor(right).map_err(module_error)?,
                ),
                (
                    PresentedModuleInner::QuotientFiniteField(left),
                    PresentedModuleInner::QuotientFiniteField(right),
                ) => PresentedModuleInner::QuotientFiniteField(
                    left.tensor(right).map_err(module_error)?,
                ),
                _ => {
                    return Err(PyValueError::new_err(
                        "presented-module coefficient contexts do not match",
                    ));
                }
            };
            Ok(Self { inner })
        })
    }
    fn hom(&self, py: Python<'_>, target: &Self, order: &PyModuleMonomialOrder) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: Hom construction runs module Gröbner bases.
        py.detach(|| {
            let inner = match (&self.inner, &target.inner) {
            (PresentedModuleInner::Rational(source), PresentedModuleInner::Rational(target)) => {
                PresentedModuleInner::Rational(
                    source
                        .hom(target, order.inner.clone())
                        .map_err(module_groebner_error)?,
                )
            }
            (
                PresentedModuleInner::FiniteField(source),
                PresentedModuleInner::FiniteField(target),
            ) => PresentedModuleInner::FiniteField(
                source
                    .hom(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientRational(source),
                PresentedModuleInner::QuotientRational(target),
            ) => PresentedModuleInner::QuotientRational(
                source
                    .hom(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientFiniteField(source),
                PresentedModuleInner::QuotientFiniteField(target),
            ) => PresentedModuleInner::QuotientFiniteField(
                source
                    .hom(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (PresentedModuleInner::ScalarInteger(_), _)
            | (PresentedModuleInner::ScalarResidue(_), _)
            | (PresentedModuleInner::Integer(_), _)
            | (PresentedModuleInner::Residue(_), _) => {
                return Err(PyTypeError::new_err(
                    "Hom currently requires a polynomial ring or polynomial quotient over Q or FiniteField",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module coefficient contexts do not match",
                ));
            }
            };
            Ok(Self { inner })
        })
    }
    fn hom_inclusion(
        &self,
        py: Python<'_>,
        target: &Self,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyPresentedModuleHomomorphism> {
        // NOETHER_RELEASES_GIL: the universal inclusion reuses Hom construction.
        py.detach(|| {
            let inner = match (&self.inner, &target.inner) {
            (PresentedModuleInner::Rational(source), PresentedModuleInner::Rational(target)) => {
                PresentedModuleHomomorphismInner::Rational(
                    source
                        .hom_inclusion(target, order.inner.clone())
                        .map_err(module_groebner_error)?,
                )
            }
            (
                PresentedModuleInner::FiniteField(source),
                PresentedModuleInner::FiniteField(target),
            ) => PresentedModuleHomomorphismInner::FiniteField(
                source
                    .hom_inclusion(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientRational(source),
                PresentedModuleInner::QuotientRational(target),
            ) => PresentedModuleHomomorphismInner::QuotientRational(
                source
                    .hom_inclusion(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientFiniteField(source),
                PresentedModuleInner::QuotientFiniteField(target),
            ) => PresentedModuleHomomorphismInner::QuotientFiniteField(
                source
                    .hom_inclusion(target, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (PresentedModuleInner::ScalarInteger(_), _)
            | (PresentedModuleInner::ScalarResidue(_), _)
            | (PresentedModuleInner::Integer(_), _)
            | (PresentedModuleInner::Residue(_), _) => {
                return Err(PyTypeError::new_err(
                    "Hom currently requires a polynomial ring or polynomial quotient over Q or FiniteField",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module coefficient contexts do not match",
                ));
            }
            };
            Ok(PyPresentedModuleHomomorphism { inner })
        })
    }
    fn tor(
        &self,
        py: Python<'_>,
        rhs: &Self,
        degree: usize,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyDerivedModule> {
        // NOETHER_RELEASES_GIL: derived-module construction builds a resolution.
        py.detach(|| derived_from_presented(self, rhs, degree, order, "tor"))
    }
    fn ext(
        &self,
        py: Python<'_>,
        target: &Self,
        degree: usize,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyDerivedModule> {
        // NOETHER_RELEASES_GIL: derived-module construction builds a resolution.
        py.detach(|| derived_from_presented(self, target, degree, order, "ext"))
    }
    fn free_resolution(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyFreeResolution> {
        // NOETHER_RELEASES_GIL: complete Schreyer resolutions are unbounded.
        py.detach(|| {
            let inner = match &self.inner {
                PresentedModuleInner::Rational(value) => FreeResolutionInner::Rational(
                    value
                        .free_resolution(order.inner.clone())
                        .map_err(resolution_error)?,
                ),
                PresentedModuleInner::FiniteField(value) => FreeResolutionInner::FiniteField(
                    value
                        .free_resolution(order.inner.clone())
                        .map_err(resolution_error)?,
                ),
                PresentedModuleInner::ScalarInteger(_)
                | PresentedModuleInner::ScalarResidue(_)
                | PresentedModuleInner::Integer(_)
                | PresentedModuleInner::Residue(_)
                | PresentedModuleInner::QuotientRational(_)
                | PresentedModuleInner::QuotientFiniteField(_) => {
                    return Err(PyTypeError::new_err(
                    "free resolutions currently require a polynomial ring over Q or FiniteField",
                ));
                }
            };
            Ok(PyFreeResolution { inner })
        })
    }
    fn free_resolution_through(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
        degree: usize,
    ) -> PyResult<PyFreeResolution> {
        // NOETHER_RELEASES_GIL: bounded resolutions remain material exact work.
        py.detach(|| {
            let inner = match &self.inner {
            PresentedModuleInner::Rational(value) => FreeResolutionInner::Rational(
                value
                    .free_resolution_through(order.inner.clone(), degree)
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::FiniteField(value) => FreeResolutionInner::FiniteField(
                value
                    .free_resolution_through(order.inner.clone(), degree)
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::QuotientRational(value) => FreeResolutionInner::QuotientRational(
                value
                    .free_resolution_through(order.inner.clone(), degree)
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::QuotientFiniteField(value) => {
                FreeResolutionInner::QuotientFiniteField(
                    value
                        .free_resolution_through(order.inner.clone(), degree)
                        .map_err(resolution_error)?,
                )
            }
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_) => {
                return Err(PyTypeError::new_err(
                    "bounded free resolutions currently require a polynomial ring or polynomial quotient over Q or FiniteField",
                ));
            }
            };
            Ok(PyFreeResolution { inner })
        })
    }
    /// Compute and certify the complete minimal graded free resolution.
    fn minimal_free_resolution(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyFreeResolution> {
        // NOETHER_RELEASES_GIL: minimal resolutions are unbounded exact work.
        py.detach(|| {
            let inner = match &self.inner {
            PresentedModuleInner::Rational(value) => FreeResolutionInner::Rational(
                value
                    .minimal_free_resolution(order.inner.clone())
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::FiniteField(value) => FreeResolutionInner::FiniteField(
                value
                    .minimal_free_resolution(order.inner.clone())
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "minimal free resolutions currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PyFreeResolution { inner })
        })
    }
    fn graded_betti_table(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyGradedBettiTable> {
        // NOETHER_RELEASES_GIL: Betti tables construct a minimal resolution.
        py.detach(|| {
            let inner = match &self.inner {
                PresentedModuleInner::Rational(value) => value
                    .graded_betti_table(order.inner.clone())
                    .map_err(betti_error)?,
                PresentedModuleInner::FiniteField(value) => value
                    .graded_betti_table(order.inner.clone())
                    .map_err(betti_error)?,
                PresentedModuleInner::ScalarInteger(_)
                | PresentedModuleInner::ScalarResidue(_)
                | PresentedModuleInner::Integer(_)
                | PresentedModuleInner::Residue(_)
                | PresentedModuleInner::QuotientRational(_)
                | PresentedModuleInner::QuotientFiniteField(_) => {
                    return Err(PyTypeError::new_err(
                    "graded Betti tables currently require a polynomial ring over Q or FiniteField",
                ));
                }
            };
            Ok(PyGradedBettiTable { inner })
        })
    }
    fn graded_invariants(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyGradedModuleInvariants> {
        // NOETHER_RELEASES_GIL: graded invariants construct a minimal resolution.
        py.detach(|| {
            let inner = match &self.inner {
            PresentedModuleInner::Rational(value) => GradedModuleInvariantsInner::Rational(
                value
                    .graded_invariants(order.inner.clone())
                    .map_err(module_invariant_error)?,
            ),
            PresentedModuleInner::FiniteField(value) => GradedModuleInvariantsInner::FiniteField(
                value
                    .graded_invariants(order.inner.clone())
                    .map_err(module_invariant_error)?,
            ),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "graded structural invariants currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PyGradedModuleInvariants { inner })
        })
    }
    fn minimal_free_resolution_through(
        &self,
        py: Python<'_>,
        order: &PyModuleMonomialOrder,
        degree: usize,
    ) -> PyResult<PyFreeResolution> {
        // NOETHER_RELEASES_GIL: bounded minimal resolutions remain material exact work.
        py.detach(|| {
            let inner = match &self.inner {
            PresentedModuleInner::Rational(value) => FreeResolutionInner::Rational(
                value
                    .minimal_free_resolution_through(order.inner.clone(), degree)
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::FiniteField(value) => FreeResolutionInner::FiniteField(
                value
                    .minimal_free_resolution_through(order.inner.clone(), degree)
                    .map_err(resolution_error)?,
            ),
            PresentedModuleInner::ScalarInteger(_)
            | PresentedModuleInner::ScalarResidue(_)
            | PresentedModuleInner::Integer(_)
            | PresentedModuleInner::Residue(_)
            | PresentedModuleInner::QuotientRational(_)
            | PresentedModuleInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "minimal free resolutions currently require a polynomial ring over Q or FiniteField",
                ));
            }
            };
            Ok(PyFreeResolution { inner })
        })
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: a module presentation can be material.
        py.detach(|| {
            match &self.inner {
                PresentedModuleInner::ScalarInteger(value) => value.to_json(),
                PresentedModuleInner::ScalarResidue(value) => value.to_json(),
                PresentedModuleInner::Integer(value) => value.to_json(),
                PresentedModuleInner::Rational(value) => value.to_json(),
                PresentedModuleInner::Residue(value) => value.to_json(),
                PresentedModuleInner::FiniteField(value) => value.to_json(),
                PresentedModuleInner::QuotientRational(value) => value.to_json(),
                PresentedModuleInner::QuotientFiniteField(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Some(ring) = serialized_scalar_module_ring(&value, "/presentation/source/ring")?
            {
                return Ok(Self {
                    inner: match ring {
                        SerializedScalarModuleRing::Integer => PresentedModuleInner::ScalarInteger(
                            PresentedModule::<IntegerRing>::from_json(&value)
                                .map_err(value_error)?,
                        ),
                        SerializedScalarModuleRing::Residue => PresentedModuleInner::ScalarResidue(
                            PresentedModule::<ResidueRing>::from_json(&value)
                                .map_err(value_error)?,
                        ),
                    },
                });
            }
            let inner = match serialized_module_ring(&value, "/presentation/source/ring")? {
                SerializedModuleRing::PolynomialInteger => PresentedModuleInner::Integer(
                    PresentedModule::<PolynomialRing<IntegerRing>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::PolynomialRational => PresentedModuleInner::Rational(
                    PresentedModule::<PolynomialRing<RationalField>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::PolynomialResidue => PresentedModuleInner::Residue(
                    PresentedModule::<PolynomialRing<ResidueRing>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::PolynomialFiniteField => PresentedModuleInner::FiniteField(
                    PresentedModule::<PolynomialRing<FiniteField>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::QuotientRational => PresentedModuleInner::QuotientRational(
                    PresentedModule::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                        &value,
                    )
                    .map_err(value_error)?,
                ),
                SerializedModuleRing::QuotientFiniteField => {
                    PresentedModuleInner::QuotientFiniteField(
                        PresentedModule::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                            &value,
                        )
                        .map_err(value_error)?,
                    )
                }
            };
            Ok(Self { inner })
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PresentedModule(generators={}, relations={})",
            self.generator_count(),
            self.relation_count()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModuleAssociatedPrimeSetInner {
    Rational(ModuleAssociatedPrimeSet<RationalField>),
    FiniteField(ModuleAssociatedPrimeSet<FiniteField>),
}

/// Complete verified associated primes of a finitely presented module.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleAssociatedPrimeSet",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleAssociatedPrimeSet {
    inner: ModuleAssociatedPrimeSetInner,
}

impl PyModuleAssociatedPrimeSet {
    fn from_rational(value: ModuleAssociatedPrimeSet<RationalField>) -> Self {
        Self {
            inner: ModuleAssociatedPrimeSetInner::Rational(value),
        }
    }

    fn from_finite_field(value: ModuleAssociatedPrimeSet<FiniteField>) -> Self {
        Self {
            inner: ModuleAssociatedPrimeSetInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyModuleAssociatedPrimeSet {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/homalg/ModuleAssociatedPrimeSet@1"
    }

    #[getter]
    fn source(&self) -> PyPresentedModule {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.source().clone())
            }
            ModuleAssociatedPrimeSetInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn annihilator(&self) -> PyIdeal {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => {
                PyIdeal::from_rational_polynomial(value.annihilator().clone())
            }
            ModuleAssociatedPrimeSetInner::FiniteField(value) => {
                PyIdeal::from_finite_field_polynomial(value.annihilator().clone())
            }
        }
    }

    #[getter]
    fn resolution(&self) -> PyFreeResolution {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => {
                PyFreeResolution::from_rational_polynomial(value.resolution().clone())
            }
            ModuleAssociatedPrimeSetInner::FiniteField(value) => {
                PyFreeResolution::from_finite_field_polynomial(value.resolution().clone())
            }
        }
    }

    #[getter]
    fn stages(&self) -> Vec<PyAssociatedPrimeExtStage> {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => value
                .stages()
                .iter()
                .cloned()
                .map(PyAssociatedPrimeExtStage::from_rational)
                .collect(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value
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
            ModuleAssociatedPrimeSetInner::Rational(value) => value
                .prime_certificates()
                .iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value
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
            ModuleAssociatedPrimeSetInner::Rational(value) => value
                .primes()
                .into_iter()
                .cloned()
                .map(PyIdeal::from_rational_polynomial)
                .collect(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value
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
            ModuleAssociatedPrimeSetInner::Rational(value) => value.theorem(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => value.verify(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value.verify(),
        }
        .map_err(associated_prime_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModuleAssociatedPrimeSetInner::Rational(value) => value.to_json(),
            ModuleAssociatedPrimeSetInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_module_ring(value, "/source/presentation/source/ring")? {
            SerializedModuleRing::PolynomialRational => Ok(Self::from_rational(
                ModuleAssociatedPrimeSet::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialFiniteField => Ok(Self::from_finite_field(
                ModuleAssociatedPrimeSet::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialInteger
            | SerializedModuleRing::PolynomialResidue
            | SerializedModuleRing::QuotientRational
            | SerializedModuleRing::QuotientFiniteField => Err(PyTypeError::new_err(
                "module associated primes require a polynomial ring over Q or FiniteField",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ModuleAssociatedPrimeSet(stages={}, associated_primes={})",
            self.stages().len(),
            self.prime_certificates().len()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModulePrimaryComponentInner {
    Rational(ModulePrimaryComponent<RationalField>),
    FiniteField(ModulePrimaryComponent<FiniteField>),
}

/// One verified primary submodule of a retained presented module.
#[pyclass(
    module = "noether.homalg",
    name = "ModulePrimaryComponent",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModulePrimaryComponent {
    inner: ModulePrimaryComponentInner,
}

impl PyModulePrimaryComponent {
    fn from_rational(value: ModulePrimaryComponent<RationalField>) -> Self {
        Self {
            inner: ModulePrimaryComponentInner::Rational(value),
        }
    }

    fn from_finite_field(value: ModulePrimaryComponent<FiniteField>) -> Self {
        Self {
            inner: ModulePrimaryComponentInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyModulePrimaryComponent {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/homalg/ModulePrimaryComponent@1"
    }

    #[getter]
    fn preimage(&self) -> PySubmodule {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => PySubmodule {
                inner: SubmoduleInner::Rational(value.preimage().clone()),
            },
            ModulePrimaryComponentInner::FiniteField(value) => PySubmodule {
                inner: SubmoduleInner::FiniteField(value.preimage().clone()),
            },
        }
    }

    #[getter]
    fn inclusion(&self) -> PyPresentedModuleHomomorphism {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => PyPresentedModuleHomomorphism {
                inner: PresentedModuleHomomorphismInner::Rational(value.inclusion().clone()),
            },
            ModulePrimaryComponentInner::FiniteField(value) => PyPresentedModuleHomomorphism {
                inner: PresentedModuleHomomorphismInner::FiniteField(value.inclusion().clone()),
            },
        }
    }

    #[getter]
    fn quotient_projection(&self) -> PyPresentedModuleHomomorphism {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => PyPresentedModuleHomomorphism {
                inner: PresentedModuleHomomorphismInner::Rational(
                    value.quotient_projection().clone(),
                ),
            },
            ModulePrimaryComponentInner::FiniteField(value) => PyPresentedModuleHomomorphism {
                inner: PresentedModuleHomomorphismInner::FiniteField(
                    value.quotient_projection().clone(),
                ),
            },
        }
    }

    #[getter]
    fn quotient_associated_primes(&self) -> PyModuleAssociatedPrimeSet {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => {
                PyModuleAssociatedPrimeSet::from_rational(
                    value.quotient_associated_primes().clone(),
                )
            }
            ModulePrimaryComponentInner::FiniteField(value) => {
                PyModuleAssociatedPrimeSet::from_finite_field(
                    value.quotient_associated_primes().clone(),
                )
            }
        }
    }

    #[getter]
    fn associated_prime(&self) -> PyPolynomialPrimeCertificate {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => {
                PyPolynomialPrimeCertificate::from_rational(value.associated_prime().clone())
            }
            ModulePrimaryComponentInner::FiniteField(value) => {
                PyPolynomialPrimeCertificate::from_finite_field(value.associated_prime().clone())
            }
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => value.theorem(),
            ModulePrimaryComponentInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => value.verify(),
            ModulePrimaryComponentInner::FiniteField(value) => value.verify(),
        }
        .map_err(module_primary_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => value.to_json(),
            ModulePrimaryComponentInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_module_ring(value, "/preimage/generator_map/source/ring")? {
            SerializedModuleRing::PolynomialRational => Ok(Self::from_rational(
                ModulePrimaryComponent::<RationalField>::from_json(value).map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialFiniteField => Ok(Self::from_finite_field(
                ModulePrimaryComponent::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialInteger
            | SerializedModuleRing::PolynomialResidue
            | SerializedModuleRing::QuotientRational
            | SerializedModuleRing::QuotientFiniteField => Err(PyTypeError::new_err(
                "module primary components require a polynomial ring over Q or FiniteField",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ModulePrimaryComponentInner::Rational(value) => value.to_string(),
            ModulePrimaryComponentInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModulePrimaryDecompositionInner {
    Rational(ModulePrimaryDecomposition<RationalField>),
    FiniteField(ModulePrimaryDecomposition<FiniteField>),
}

/// Complete verified primary decomposition of a module's zero submodule.
#[pyclass(
    module = "noether.homalg",
    name = "ModulePrimaryDecomposition",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModulePrimaryDecomposition {
    inner: ModulePrimaryDecompositionInner,
}

impl PyModulePrimaryDecomposition {
    fn from_rational(value: ModulePrimaryDecomposition<RationalField>) -> Self {
        Self {
            inner: ModulePrimaryDecompositionInner::Rational(value),
        }
    }

    fn from_finite_field(value: ModulePrimaryDecomposition<FiniteField>) -> Self {
        Self {
            inner: ModulePrimaryDecompositionInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyModulePrimaryDecomposition {
    #[classattr]
    fn type_id() -> &'static str {
        "noether/homalg/ModulePrimaryDecomposition@1"
    }

    #[getter]
    fn source(&self) -> PyPresentedModule {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => {
                PyPresentedModule::from_rational_polynomial(value.source().clone())
            }
            ModulePrimaryDecompositionInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_polynomial(value.source().clone())
            }
        }
    }

    #[getter]
    fn source_associated_primes(&self) -> PyModuleAssociatedPrimeSet {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => {
                PyModuleAssociatedPrimeSet::from_rational(value.source_associated_primes().clone())
            }
            ModulePrimaryDecompositionInner::FiniteField(value) => {
                PyModuleAssociatedPrimeSet::from_finite_field(
                    value.source_associated_primes().clone(),
                )
            }
        }
    }

    #[getter]
    fn components(&self) -> Vec<PyModulePrimaryComponent> {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyModulePrimaryComponent::from_rational)
                .collect(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value
                .components()
                .iter()
                .cloned()
                .map(PyModulePrimaryComponent::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn associated_primes(&self) -> Vec<PyPolynomialPrimeCertificate> {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value
                .associated_primes()
                .into_iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_rational)
                .collect(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value
                .associated_primes()
                .into_iter()
                .cloned()
                .map(PyPolynomialPrimeCertificate::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value.theorem(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value.theorem(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value.verify(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value.verify(),
        }
        .map_err(module_primary_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value.to_json(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match serialized_module_ring(value, "/source/presentation/source/ring")? {
            SerializedModuleRing::PolynomialRational => Ok(Self::from_rational(
                ModulePrimaryDecomposition::<RationalField>::from_json(value)
                    .map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialFiniteField => Ok(Self::from_finite_field(
                ModulePrimaryDecomposition::<FiniteField>::from_json(value).map_err(value_error)?,
            )),
            SerializedModuleRing::PolynomialInteger
            | SerializedModuleRing::PolynomialResidue
            | SerializedModuleRing::QuotientRational
            | SerializedModuleRing::QuotientFiniteField => Err(PyTypeError::new_err(
                "module primary decompositions require a polynomial ring over Q or FiniteField",
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ModulePrimaryDecompositionInner::Rational(value) => value.to_string(),
            ModulePrimaryDecompositionInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum PresentedModuleHomomorphismInner {
    ScalarInteger(PresentedModuleHomomorphism<IntegerRing>),
    ScalarResidue(PresentedModuleHomomorphism<ResidueRing>),
    Integer(PresentedModuleHomomorphism<PolynomialRing<IntegerRing>>),
    Rational(PresentedModuleHomomorphism<PolynomialRing<RationalField>>),
    Residue(PresentedModuleHomomorphism<PolynomialRing<ResidueRing>>),
    FiniteField(PresentedModuleHomomorphism<PolynomialRing<FiniteField>>),
    QuotientRational(PresentedModuleHomomorphism<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(PresentedModuleHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// A checked homomorphism between retained module presentations.
#[pyclass(
    module = "noether.homalg",
    name = "PresentedModuleHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPresentedModuleHomomorphism {
    pub(super) inner: PresentedModuleHomomorphismInner,
}

impl PyPresentedModuleHomomorphism {
    pub(crate) fn from_rational_quotient(
        value: PresentedModuleHomomorphism<QuotientRing<PolynomialRing<RationalField>>>,
    ) -> Self {
        Self {
            inner: PresentedModuleHomomorphismInner::QuotientRational(value),
        }
    }

    pub(crate) fn from_finite_field_quotient(
        value: PresentedModuleHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>,
    ) -> Self {
        Self {
            inner: PresentedModuleHomomorphismInner::QuotientFiniteField(value),
        }
    }
}

#[pymethods]
impl PyPresentedModuleHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        PresentedModuleHomomorphism::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[new]
    fn new(
        source: &PyPresentedModule,
        target: &PyPresentedModule,
        generator_lift: &PyModuleHomomorphism,
        relation_lift: &PyModuleHomomorphism,
    ) -> PyResult<Self> {
        macro_rules! construct {
            ($source:expr, $target:expr, $generators:expr, $relations:expr, $variant:ident) => {
                PresentedModuleHomomorphismInner::$variant(
                    PresentedModuleHomomorphism::from_lifts(
                        $source.clone(),
                        $target.clone(),
                        $generators.clone(),
                        $relations.clone(),
                    )
                    .map_err(module_error)?,
                )
            };
        }
        let inner = match (
            &source.inner,
            &target.inner,
            &generator_lift.inner,
            &relation_lift.inner,
        ) {
            (
                PresentedModuleInner::ScalarInteger(source),
                PresentedModuleInner::ScalarInteger(target),
                ModuleHomomorphismInner::ScalarInteger(generators),
                ModuleHomomorphismInner::ScalarInteger(relations),
            ) => construct!(source, target, generators, relations, ScalarInteger),
            (
                PresentedModuleInner::ScalarResidue(source),
                PresentedModuleInner::ScalarResidue(target),
                ModuleHomomorphismInner::ScalarResidue(generators),
                ModuleHomomorphismInner::ScalarResidue(relations),
            ) => construct!(source, target, generators, relations, ScalarResidue),
            (
                PresentedModuleInner::Integer(source),
                PresentedModuleInner::Integer(target),
                ModuleHomomorphismInner::Integer(generators),
                ModuleHomomorphismInner::Integer(relations),
            ) => construct!(source, target, generators, relations, Integer),
            (
                PresentedModuleInner::Rational(source),
                PresentedModuleInner::Rational(target),
                ModuleHomomorphismInner::Rational(generators),
                ModuleHomomorphismInner::Rational(relations),
            ) => construct!(source, target, generators, relations, Rational),
            (
                PresentedModuleInner::Residue(source),
                PresentedModuleInner::Residue(target),
                ModuleHomomorphismInner::Residue(generators),
                ModuleHomomorphismInner::Residue(relations),
            ) => construct!(source, target, generators, relations, Residue),
            (
                PresentedModuleInner::FiniteField(source),
                PresentedModuleInner::FiniteField(target),
                ModuleHomomorphismInner::FiniteField(generators),
                ModuleHomomorphismInner::FiniteField(relations),
            ) => construct!(source, target, generators, relations, FiniteField),
            (
                PresentedModuleInner::QuotientRational(source),
                PresentedModuleInner::QuotientRational(target),
                ModuleHomomorphismInner::QuotientRational(generators),
                ModuleHomomorphismInner::QuotientRational(relations),
            ) => construct!(source, target, generators, relations, QuotientRational),
            (
                PresentedModuleInner::QuotientFiniteField(source),
                PresentedModuleInner::QuotientFiniteField(target),
                ModuleHomomorphismInner::QuotientFiniteField(generators),
                ModuleHomomorphismInner::QuotientFiniteField(relations),
            ) => construct!(source, target, generators, relations, QuotientFiniteField),
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module endpoints and lifts must share one exact context",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn from_generator_lift(
        _cls: &Bound<'_, PyType>,
        source: &PyPresentedModule,
        target: &PyPresentedModule,
        generator_lift: &PyModuleHomomorphism,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<Self> {
        let inner = match (&source.inner, &target.inner, &generator_lift.inner) {
            (
                PresentedModuleInner::Rational(source),
                PresentedModuleInner::Rational(target),
                ModuleHomomorphismInner::Rational(generators),
            ) => PresentedModuleHomomorphismInner::Rational(
                PresentedModuleHomomorphism::<PolynomialRing<RationalField>>::from_generator_lift(
                    source.clone(),
                    target.clone(),
                    generators.clone(),
                    order.inner.clone(),
                )
                .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::FiniteField(source),
                PresentedModuleInner::FiniteField(target),
                ModuleHomomorphismInner::FiniteField(generators),
            ) => PresentedModuleHomomorphismInner::FiniteField(
                PresentedModuleHomomorphism::<PolynomialRing<FiniteField>>::from_generator_lift(
                    source.clone(),
                    target.clone(),
                    generators.clone(),
                    order.inner.clone(),
                )
                .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientRational(source),
                PresentedModuleInner::QuotientRational(target),
                ModuleHomomorphismInner::QuotientRational(generators),
            ) => PresentedModuleHomomorphismInner::QuotientRational(
                PresentedModuleHomomorphism::<
                    QuotientRing<PolynomialRing<RationalField>>,
                >::from_generator_lift(
                    source.clone(),
                    target.clone(),
                    generators.clone(),
                    order.inner.clone(),
                )
                .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleInner::QuotientFiniteField(source),
                PresentedModuleInner::QuotientFiniteField(target),
                ModuleHomomorphismInner::QuotientFiniteField(generators),
            ) => PresentedModuleHomomorphismInner::QuotientFiniteField(
                PresentedModuleHomomorphism::<
                    QuotientRing<PolynomialRing<FiniteField>>,
                >::from_generator_lift(
                    source.clone(),
                    target.clone(),
                    generators.clone(),
                    order.inner.clone(),
                )
                .map_err(module_groebner_error)?,
            ),
            (PresentedModuleInner::ScalarInteger(_), _, _)
            | (PresentedModuleInner::ScalarResidue(_), _, _)
            | (PresentedModuleInner::Integer(_), _, _)
            | (PresentedModuleInner::Residue(_), _, _) => {
                return Err(PyTypeError::new_err(
                    "computed relation lifts require a polynomial ring or quotient over Q or FiniteField",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module endpoints and generator lift must share one exact context",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn zero(
        _cls: &Bound<'_, PyType>,
        source: &PyPresentedModule,
        target: &PyPresentedModule,
    ) -> PyResult<Self> {
        macro_rules! zero {
            ($source:expr, $target:expr, $variant:ident) => {
                PresentedModuleHomomorphismInner::$variant(
                    PresentedModuleHomomorphism::zero($source.clone(), $target.clone())
                        .map_err(module_error)?,
                )
            };
        }
        let inner = match (&source.inner, &target.inner) {
            (
                PresentedModuleInner::ScalarInteger(source),
                PresentedModuleInner::ScalarInteger(target),
            ) => zero!(source, target, ScalarInteger),
            (
                PresentedModuleInner::ScalarResidue(source),
                PresentedModuleInner::ScalarResidue(target),
            ) => zero!(source, target, ScalarResidue),
            (PresentedModuleInner::Integer(source), PresentedModuleInner::Integer(target)) => {
                zero!(source, target, Integer)
            }
            (PresentedModuleInner::Rational(source), PresentedModuleInner::Rational(target)) => {
                zero!(source, target, Rational)
            }
            (PresentedModuleInner::Residue(source), PresentedModuleInner::Residue(target)) => {
                zero!(source, target, Residue)
            }
            (
                PresentedModuleInner::FiniteField(source),
                PresentedModuleInner::FiniteField(target),
            ) => zero!(source, target, FiniteField),
            (
                PresentedModuleInner::QuotientRational(source),
                PresentedModuleInner::QuotientRational(target),
            ) => zero!(source, target, QuotientRational),
            (
                PresentedModuleInner::QuotientFiniteField(source),
                PresentedModuleInner::QuotientFiniteField(target),
            ) => zero!(source, target, QuotientFiniteField),
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module contexts do not match",
                ))
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, module: &PyPresentedModule) -> PyResult<Self> {
        let inner = match &module.inner {
            PresentedModuleInner::ScalarInteger(value) => {
                PresentedModuleHomomorphismInner::ScalarInteger(
                    PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
            PresentedModuleInner::ScalarResidue(value) => {
                PresentedModuleHomomorphismInner::ScalarResidue(
                    PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
            PresentedModuleInner::Integer(value) => PresentedModuleHomomorphismInner::Integer(
                PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            PresentedModuleInner::Rational(value) => PresentedModuleHomomorphismInner::Rational(
                PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            PresentedModuleInner::Residue(value) => PresentedModuleHomomorphismInner::Residue(
                PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
            ),
            PresentedModuleInner::FiniteField(value) => {
                PresentedModuleHomomorphismInner::FiniteField(
                    PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
            PresentedModuleInner::QuotientRational(value) => {
                PresentedModuleHomomorphismInner::QuotientRational(
                    PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
            PresentedModuleInner::QuotientFiniteField(value) => {
                PresentedModuleHomomorphismInner::QuotientFiniteField(
                    PresentedModuleHomomorphism::identity(value.clone()).map_err(module_error)?,
                )
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn source(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    PresentedModuleInner::ScalarInteger(value.source().clone())
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    PresentedModuleInner::ScalarResidue(value.source().clone())
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    PresentedModuleInner::Integer(value.source().clone())
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.source().clone())
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    PresentedModuleInner::Residue(value.source().clone())
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.source().clone())
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.source().clone())
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(value.source().clone())
                }
            },
        }
    }

    #[getter]
    fn target(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    PresentedModuleInner::ScalarInteger(value.target().clone())
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    PresentedModuleInner::ScalarResidue(value.target().clone())
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    PresentedModuleInner::Integer(value.target().clone())
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.target().clone())
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    PresentedModuleInner::Residue(value.target().clone())
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.target().clone())
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.target().clone())
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(value.target().clone())
                }
            },
        }
    }

    #[getter]
    fn generator_lift(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    ModuleHomomorphismInner::Integer(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    ModuleHomomorphismInner::Residue(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.generator_lift().clone())
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.generator_lift().clone())
                }
            },
        }
    }

    #[getter]
    fn relation_lift(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    ModuleHomomorphismInner::ScalarInteger(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    ModuleHomomorphismInner::ScalarResidue(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    ModuleHomomorphismInner::Integer(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    ModuleHomomorphismInner::Residue(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.relation_lift().clone())
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.relation_lift().clone())
                }
            },
        }
    }

    fn compose(&self, py: Python<'_>, inner: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: composition can multiply material lift matrices.
        py.detach(|| self.binary_map(inner, "compose"))
    }

    /// Map induced between degree-`degree` symmetric-power presentations.
    fn symmetric_power(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<super::module_power::PyPresentedModulePowerHomomorphism> {
        py.detach(|| {
            super::module_power::PyPresentedModulePowerHomomorphism::from_map(
                &self.inner,
                noether::homalg::ModulePowerKind::Symmetric,
                degree,
            )
        })
        .map_err(super::module_power::module_power_error)
    }

    /// Map induced between degree-`degree` exterior-power presentations.
    fn exterior_power(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<super::module_power::PyPresentedModulePowerHomomorphism> {
        py.detach(|| {
            super::module_power::PyPresentedModulePowerHomomorphism::from_map(
                &self.inner,
                noether::homalg::ModulePowerKind::Exterior,
                degree,
            )
        })
        .map_err(super::module_power::module_power_error)
    }

    fn tensor(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: tensor products can materialize large lifts.
        py.detach(|| self.binary_map(rhs, "tensor"))
    }

    fn __add__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: lift addition can be material.
        py.detach(|| self.binary_map(rhs, "add"))
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: lift subtraction can be material.
        py.detach(|| self.binary_map(rhs, "sub"))
    }

    fn cokernel(&self, py: Python<'_>) -> PyResult<PyPresentedModule> {
        // NOETHER_RELEASES_GIL: cokernel construction can retain material presentations.
        py.detach(|| {
            let inner = match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    PresentedModuleInner::ScalarInteger(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    PresentedModuleInner::ScalarResidue(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    PresentedModuleInner::Integer(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    PresentedModuleInner::Residue(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.cokernel().map_err(module_error)?)
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(
                        value.cokernel().map_err(module_error)?,
                    )
                }
            };
            Ok(PyPresentedModule { inner })
        })
    }

    fn cokernel_projection(&self, py: Python<'_>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: projection construction can retain material presentations.
        py.detach(|| {
            let inner = match &self.inner {
                PresentedModuleHomomorphismInner::ScalarInteger(value) => {
                    PresentedModuleHomomorphismInner::ScalarInteger(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::ScalarResidue(value) => {
                    PresentedModuleHomomorphismInner::ScalarResidue(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::Integer(value) => {
                    PresentedModuleHomomorphismInner::Integer(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::Residue(value) => {
                    PresentedModuleHomomorphismInner::Residue(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
                PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.cokernel_projection().map_err(module_error)?,
                    )
                }
            };
            Ok(Self { inner })
        })
    }

    fn is_equal(
        &self,
        py: Python<'_>,
        rhs: &Self,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<bool> {
        let order = order.inner.clone();
        py.detach(|| match (&self.inner, &rhs.inner) {
            (
                PresentedModuleHomomorphismInner::Rational(left),
                PresentedModuleHomomorphismInner::Rational(right),
            ) => left
                .is_equal(right, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::QuotientRational(left),
                PresentedModuleHomomorphismInner::QuotientRational(right),
            ) => left
                .is_equal(right, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::QuotientFiniteField(left),
                PresentedModuleHomomorphismInner::QuotientFiniteField(right),
            ) => left
                .is_equal(right, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::FiniteField(left),
                PresentedModuleHomomorphismInner::FiniteField(right),
            ) => left
                .is_equal(right, order.clone())
                .map_err(module_groebner_error),
            (PresentedModuleHomomorphismInner::ScalarInteger(_), _)
            | (PresentedModuleHomomorphismInner::ScalarResidue(_), _)
            | (PresentedModuleHomomorphismInner::Integer(_), _)
            | (PresentedModuleHomomorphismInner::Residue(_), _) => Err(PyTypeError::new_err(
                "induced map equality currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => Err(PyValueError::new_err(
                "presented-module map contexts do not match",
            )),
        })
    }

    fn is_zero(&self, py: Python<'_>, order: &PyModuleMonomialOrder) -> PyResult<bool> {
        let order = order.inner.clone();
        py.detach(|| match &self.inner {
            PresentedModuleHomomorphismInner::Rational(value) => {
                value.is_zero(order.clone()).map_err(module_groebner_error)
            }
            PresentedModuleHomomorphismInner::FiniteField(value) => {
                value.is_zero(order.clone()).map_err(module_groebner_error)
            }
            PresentedModuleHomomorphismInner::QuotientRational(value) => {
                value.is_zero(order.clone()).map_err(module_groebner_error)
            }
            PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                value.is_zero(order.clone()).map_err(module_groebner_error)
            }
            PresentedModuleHomomorphismInner::ScalarInteger(_)
            | PresentedModuleHomomorphismInner::ScalarResidue(_)
            | PresentedModuleHomomorphismInner::Integer(_)
            | PresentedModuleHomomorphismInner::Residue(_) => Err(PyTypeError::new_err(
                "induced zero testing currently requires a polynomial ring over Q or FiniteField",
            )),
        })
    }

    fn kernel(&self, py: Python<'_>, order: &PyModuleMonomialOrder) -> PyResult<Self> {
        let order = order.inner.clone();
        py.detach(|| self.unary_subquotient(order, "kernel"))
    }

    fn image(&self, py: Python<'_>, order: &PyModuleMonomialOrder) -> PyResult<Self> {
        let order = order.inner.clone();
        py.detach(|| self.unary_subquotient(order, "image"))
    }

    fn coimage(&self, py: Python<'_>, order: &PyModuleMonomialOrder) -> PyResult<Self> {
        let order = order.inner.clone();
        py.detach(|| self.unary_subquotient(order, "coimage"))
    }

    fn is_exact_at_source(
        &self,
        py: Python<'_>,
        incoming: &Self,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<bool> {
        let order = order.inner.clone();
        py.detach(|| match (&self.inner, &incoming.inner) {
            (
                PresentedModuleHomomorphismInner::Rational(outgoing),
                PresentedModuleHomomorphismInner::Rational(incoming),
            ) => outgoing
                .is_exact_at_source(incoming, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::QuotientRational(outgoing),
                PresentedModuleHomomorphismInner::QuotientRational(incoming),
            ) => outgoing
                .is_exact_at_source(incoming, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::QuotientFiniteField(outgoing),
                PresentedModuleHomomorphismInner::QuotientFiniteField(incoming),
            ) => outgoing
                .is_exact_at_source(incoming, order.clone())
                .map_err(module_groebner_error),
            (
                PresentedModuleHomomorphismInner::FiniteField(outgoing),
                PresentedModuleHomomorphismInner::FiniteField(incoming),
            ) => outgoing
                .is_exact_at_source(incoming, order.clone())
                .map_err(module_groebner_error),
            (PresentedModuleHomomorphismInner::ScalarInteger(_), _)
            | (PresentedModuleHomomorphismInner::ScalarResidue(_), _)
            | (PresentedModuleHomomorphismInner::Integer(_), _)
            | (PresentedModuleHomomorphismInner::Residue(_), _) => Err(PyTypeError::new_err(
                "exactness currently requires a polynomial ring over Q or FiniteField",
            )),
            _ => Err(PyValueError::new_err(
                "presented-module map contexts do not match",
            )),
        })
    }

    fn homology(
        &self,
        py: Python<'_>,
        incoming: &Self,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyPresentedModuleHomology> {
        let order = order.inner.clone();
        let inner = py.detach(|| -> PyResult<_> { Ok(match (&self.inner, &incoming.inner) {
            (
                PresentedModuleHomomorphismInner::Rational(outgoing),
                PresentedModuleHomomorphismInner::Rational(incoming),
            ) => PresentedModuleHomologyInner::Rational(
                outgoing
                    .homology(incoming, order.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomomorphismInner::FiniteField(outgoing),
                PresentedModuleHomomorphismInner::FiniteField(incoming),
            ) => PresentedModuleHomologyInner::FiniteField(
                outgoing
                    .homology(incoming, order.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomomorphismInner::QuotientRational(outgoing),
                PresentedModuleHomomorphismInner::QuotientRational(incoming),
            ) => PresentedModuleHomologyInner::QuotientRational(
                outgoing
                    .homology(incoming, order.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomomorphismInner::QuotientFiniteField(outgoing),
                PresentedModuleHomomorphismInner::QuotientFiniteField(incoming),
            ) => PresentedModuleHomologyInner::QuotientFiniteField(
                outgoing
                    .homology(incoming, order.clone())
                    .map_err(module_groebner_error)?,
            ),
            (PresentedModuleHomomorphismInner::ScalarInteger(_), _)
            | (PresentedModuleHomomorphismInner::ScalarResidue(_), _)
            | (PresentedModuleHomomorphismInner::Integer(_), _)
            | (PresentedModuleHomomorphismInner::Residue(_), _) => {
                return Err(PyTypeError::new_err(
                    "module-valued homology currently requires a polynomial ring over Q or FiniteField",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module map contexts do not match",
                ));
            }
        }) })?;
        Ok(PyPresentedModuleHomology { inner })
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // NOETHER_RELEASES_GIL: verification checks both lift matrices and their square.
        py.detach(|| match &self.inner {
            PresentedModuleHomomorphismInner::ScalarInteger(value) => value.verify(),
            PresentedModuleHomomorphismInner::ScalarResidue(value) => value.verify(),
            PresentedModuleHomomorphismInner::Integer(value) => value.verify(),
            PresentedModuleHomomorphismInner::Rational(value) => value.verify(),
            PresentedModuleHomomorphismInner::Residue(value) => value.verify(),
            PresentedModuleHomomorphismInner::FiniteField(value) => value.verify(),
            PresentedModuleHomomorphismInner::QuotientRational(value) => value.verify(),
            PresentedModuleHomomorphismInner::QuotientFiniteField(value) => value.verify(),
        })
        .map_err(module_error)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: serialization traverses complete presentation evidence.
        py.detach(|| match &self.inner {
            PresentedModuleHomomorphismInner::ScalarInteger(value) => value.to_json(),
            PresentedModuleHomomorphismInner::ScalarResidue(value) => value.to_json(),
            PresentedModuleHomomorphismInner::Integer(value) => value.to_json(),
            PresentedModuleHomomorphismInner::Rational(value) => value.to_json(),
            PresentedModuleHomomorphismInner::Residue(value) => value.to_json(),
            PresentedModuleHomomorphismInner::FiniteField(value) => value.to_json(),
            PresentedModuleHomomorphismInner::QuotientRational(value) => value.to_json(),
            PresentedModuleHomomorphismInner::QuotientFiniteField(value) => value.to_json(),
        })
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        // NOETHER_RELEASES_GIL: deserialization replays the commuting-square evidence.
        py.detach(move || {
            if let Some(ring) =
                serialized_scalar_module_ring(&value, "/source/presentation/source/ring")?
            {
                let inner = match ring {
                    SerializedScalarModuleRing::Integer => {
                        PresentedModuleHomomorphismInner::ScalarInteger(
                            PresentedModuleHomomorphism::<IntegerRing>::from_json(&value)
                                .map_err(value_error)?,
                        )
                    }
                    SerializedScalarModuleRing::Residue => {
                        PresentedModuleHomomorphismInner::ScalarResidue(
                            PresentedModuleHomomorphism::<ResidueRing>::from_json(&value)
                                .map_err(value_error)?,
                        )
                    }
                };
                return Ok(Self { inner });
            }
            let inner = match serialized_module_ring(&value, "/source/presentation/source/ring")? {
            SerializedModuleRing::PolynomialInteger => PresentedModuleHomomorphismInner::Integer(
                PresentedModuleHomomorphism::<PolynomialRing<IntegerRing>>::from_json(&value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialRational => {
                PresentedModuleHomomorphismInner::Rational(
                PresentedModuleHomomorphism::<PolynomialRing<RationalField>>::from_json(&value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialResidue => PresentedModuleHomomorphismInner::Residue(
                PresentedModuleHomomorphism::<PolynomialRing<ResidueRing>>::from_json(&value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => {
                PresentedModuleHomomorphismInner::FiniteField(
                PresentedModuleHomomorphism::<PolynomialRing<FiniteField>>::from_json(&value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientRational => {
                PresentedModuleHomomorphismInner::QuotientRational(
                    PresentedModuleHomomorphism::<
                        QuotientRing<PolynomialRing<RationalField>>,
                    >::from_json(&value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientFiniteField => {
                PresentedModuleHomomorphismInner::QuotientFiniteField(
                    PresentedModuleHomomorphism::<
                        QuotientRing<PolynomialRing<FiniteField>>,
                    >::from_json(&value)
                    .map_err(value_error)?,
                )
            }
        };
            Ok(Self { inner })
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PresentedModuleHomomorphism(source_generators={}, target_generators={})",
            self.source().generator_count(),
            self.target().generator_count()
        ))
    }
}

impl PyPresentedModuleHomomorphism {
    fn binary_map(&self, rhs: &Self, operation: &str) -> PyResult<Self> {
        macro_rules! apply {
            ($left:expr, $right:expr, $variant:ident) => {{
                let value = match operation {
                    "compose" => $left.compose($right),
                    "add" => $left.add($right),
                    "sub" => $left.sub($right),
                    "tensor" => $left.tensor($right),
                    _ => return Err(PyValueError::new_err("unknown presented-map operation")),
                }
                .map_err(module_error)?;
                PresentedModuleHomomorphismInner::$variant(value)
            }};
        }
        let inner = match (&self.inner, &rhs.inner) {
            (
                PresentedModuleHomomorphismInner::ScalarInteger(left),
                PresentedModuleHomomorphismInner::ScalarInteger(right),
            ) => apply!(left, right, ScalarInteger),
            (
                PresentedModuleHomomorphismInner::ScalarResidue(left),
                PresentedModuleHomomorphismInner::ScalarResidue(right),
            ) => apply!(left, right, ScalarResidue),
            (
                PresentedModuleHomomorphismInner::Integer(left),
                PresentedModuleHomomorphismInner::Integer(right),
            ) => apply!(left, right, Integer),
            (
                PresentedModuleHomomorphismInner::Rational(left),
                PresentedModuleHomomorphismInner::Rational(right),
            ) => apply!(left, right, Rational),
            (
                PresentedModuleHomomorphismInner::Residue(left),
                PresentedModuleHomomorphismInner::Residue(right),
            ) => apply!(left, right, Residue),
            (
                PresentedModuleHomomorphismInner::FiniteField(left),
                PresentedModuleHomomorphismInner::FiniteField(right),
            ) => apply!(left, right, FiniteField),
            (
                PresentedModuleHomomorphismInner::QuotientRational(left),
                PresentedModuleHomomorphismInner::QuotientRational(right),
            ) => apply!(left, right, QuotientRational),
            (
                PresentedModuleHomomorphismInner::QuotientFiniteField(left),
                PresentedModuleHomomorphismInner::QuotientFiniteField(right),
            ) => apply!(left, right, QuotientFiniteField),
            _ => {
                return Err(PyValueError::new_err(
                    "presented-module map contexts do not match",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn unary_subquotient(&self, order: ModuleMonomialOrder, operation: &str) -> PyResult<Self> {
        macro_rules! apply {
            ($value:expr, $variant:ident) => {{
                let result = match operation {
                    "kernel" => $value.kernel(order.clone()),
                    "image" => $value.image(order.clone()),
                    "coimage" => $value.coimage(order.clone()),
                    _ => return Err(PyValueError::new_err("unknown subquotient operation")),
                }
                .map_err(module_groebner_error)?;
                PresentedModuleHomomorphismInner::$variant(result)
            }};
        }
        let inner = match &self.inner {
            PresentedModuleHomomorphismInner::Rational(value) => apply!(value, Rational),
            PresentedModuleHomomorphismInner::FiniteField(value) => apply!(value, FiniteField),
            PresentedModuleHomomorphismInner::QuotientRational(value) => {
                apply!(value, QuotientRational)
            }
            PresentedModuleHomomorphismInner::QuotientFiniteField(value) => {
                apply!(value, QuotientFiniteField)
            }
            PresentedModuleHomomorphismInner::ScalarInteger(_)
            | PresentedModuleHomomorphismInner::ScalarResidue(_)
            | PresentedModuleHomomorphismInner::Integer(_)
            | PresentedModuleHomomorphismInner::Residue(_) => {
                return Err(PyTypeError::new_err(
                    "presented subquotients currently require a polynomial ring over Q or FiniteField",
                ));
            }
        };
        Ok(Self { inner })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PresentedModuleHomologyInner {
    Rational(PresentedModuleHomology<PolynomialRing<RationalField>>),
    FiniteField(PresentedModuleHomology<PolynomialRing<FiniteField>>),
    QuotientRational(PresentedModuleHomology<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(PresentedModuleHomology<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// Exact cycles, boundaries, and quotient evidence for module-valued homology.
#[pyclass(
    module = "noether.homalg",
    name = "PresentedModuleHomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPresentedModuleHomology {
    inner: PresentedModuleHomologyInner,
}

#[pymethods]
impl PyPresentedModuleHomology {
    #[classattr]
    fn type_id() -> &'static str {
        PresentedModuleHomology::<PolynomialRing<RationalField>>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn incoming(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomologyInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.incoming().clone())
                }
                PresentedModuleHomologyInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(value.incoming().clone())
                }
                PresentedModuleHomologyInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(value.incoming().clone())
                }
                PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(value.incoming().clone())
                }
            },
        }
    }

    #[getter]
    fn outgoing(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomologyInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.outgoing().clone())
                }
                PresentedModuleHomologyInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(value.outgoing().clone())
                }
                PresentedModuleHomologyInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(value.outgoing().clone())
                }
                PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(value.outgoing().clone())
                }
            },
        }
    }

    #[getter]
    fn cycle_inclusion(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomologyInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.cycle_inclusion().clone())
                }
                PresentedModuleHomologyInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(value.cycle_inclusion().clone())
                }
                PresentedModuleHomologyInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(
                        value.cycle_inclusion().clone(),
                    )
                }
                PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.cycle_inclusion().clone(),
                    )
                }
            },
        }
    }

    #[getter]
    fn boundary_lift(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomologyInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.boundary_lift().clone())
                }
                PresentedModuleHomologyInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(value.boundary_lift().clone())
                }
                PresentedModuleHomologyInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(
                        value.boundary_lift().clone(),
                    )
                }
                PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.boundary_lift().clone(),
                    )
                }
            },
        }
    }

    #[getter]
    fn quotient_projection(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                PresentedModuleHomologyInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.quotient_projection().clone())
                }
                PresentedModuleHomologyInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(
                        value.quotient_projection().clone(),
                    )
                }
                PresentedModuleHomologyInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(
                        value.quotient_projection().clone(),
                    )
                }
                PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.quotient_projection().clone(),
                    )
                }
            },
        }
    }

    #[getter]
    fn module(&self) -> PyPresentedModule {
        self.quotient_projection().target()
    }

    fn induced_map(
        &self,
        target: &Self,
        left: &PyPresentedModuleHomomorphism,
        middle: &PyPresentedModuleHomomorphism,
        right: &PyPresentedModuleHomomorphism,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyPresentedModuleHomomorphism> {
        let inner = match (
            &self.inner,
            &target.inner,
            &left.inner,
            &middle.inner,
            &right.inner,
        ) {
            (
                PresentedModuleHomologyInner::Rational(source),
                PresentedModuleHomologyInner::Rational(target),
                PresentedModuleHomomorphismInner::Rational(left),
                PresentedModuleHomomorphismInner::Rational(middle),
                PresentedModuleHomomorphismInner::Rational(right),
            ) => PresentedModuleHomomorphismInner::Rational(
                source
                    .induced_map(target, left, middle, right, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomologyInner::FiniteField(source),
                PresentedModuleHomologyInner::FiniteField(target),
                PresentedModuleHomomorphismInner::FiniteField(left),
                PresentedModuleHomomorphismInner::FiniteField(middle),
                PresentedModuleHomomorphismInner::FiniteField(right),
            ) => PresentedModuleHomomorphismInner::FiniteField(
                source
                    .induced_map(target, left, middle, right, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomologyInner::QuotientRational(source),
                PresentedModuleHomologyInner::QuotientRational(target),
                PresentedModuleHomomorphismInner::QuotientRational(left),
                PresentedModuleHomomorphismInner::QuotientRational(middle),
                PresentedModuleHomomorphismInner::QuotientRational(right),
            ) => PresentedModuleHomomorphismInner::QuotientRational(
                source
                    .induced_map(target, left, middle, right, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            (
                PresentedModuleHomologyInner::QuotientFiniteField(source),
                PresentedModuleHomologyInner::QuotientFiniteField(target),
                PresentedModuleHomomorphismInner::QuotientFiniteField(left),
                PresentedModuleHomomorphismInner::QuotientFiniteField(middle),
                PresentedModuleHomomorphismInner::QuotientFiniteField(right),
            ) => PresentedModuleHomomorphismInner::QuotientFiniteField(
                source
                    .induced_map(target, left, middle, right, order.inner.clone())
                    .map_err(module_groebner_error)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "the homology values and three chain maps must share matching exact contexts",
                ));
            }
        };
        Ok(PyPresentedModuleHomomorphism { inner })
    }

    fn verify(&self, order: &PyModuleMonomialOrder) -> PyResult<()> {
        match &self.inner {
            PresentedModuleHomologyInner::Rational(value) => value.verify(order.inner.clone()),
            PresentedModuleHomologyInner::FiniteField(value) => value.verify(order.inner.clone()),
            PresentedModuleHomologyInner::QuotientRational(value) => {
                value.verify(order.inner.clone())
            }
            PresentedModuleHomologyInner::QuotientFiniteField(value) => {
                value.verify(order.inner.clone())
            }
        }
        .map_err(module_groebner_error)
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            PresentedModuleHomologyInner::Rational(value) => value.to_json(),
            PresentedModuleHomologyInner::FiniteField(value) => value.to_json(),
            PresentedModuleHomologyInner::QuotientRational(value) => value.to_json(),
            PresentedModuleHomologyInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner =
            match serialized_module_ring(value, "/incoming/source/presentation/source/ring")? {
            SerializedModuleRing::PolynomialRational => PresentedModuleHomologyInner::Rational(
                PresentedModuleHomology::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => {
                PresentedModuleHomologyInner::FiniteField(
                PresentedModuleHomology::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientRational => {
                PresentedModuleHomologyInner::QuotientRational(
                    PresentedModuleHomology::<
                        QuotientRing<PolynomialRing<RationalField>>,
                    >::from_json(value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientFiniteField => {
                PresentedModuleHomologyInner::QuotientFiniteField(
                    PresentedModuleHomology::<
                        QuotientRing<PolynomialRing<FiniteField>>,
                    >::from_json(value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialInteger | SerializedModuleRing::PolynomialResidue => {
                return Err(PyTypeError::new_err(
                    "module-valued homology requires a polynomial or quotient ring over Q or FiniteField",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PresentedModuleHomology(generators={}, relations={})",
            self.module().generator_count(),
            self.module().relation_count()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum FreeResolutionInner {
    Rational(FreeResolution<PolynomialRing<RationalField>>),
    FiniteField(FreeResolution<PolynomialRing<FiniteField>>),
    QuotientRational(FreeResolution<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(FreeResolution<QuotientRing<PolynomialRing<FiniteField>>>),
}

#[derive(Clone, PartialEq, Eq)]
enum GradedModuleInvariantsInner {
    Rational(GradedModuleInvariants<RationalField>),
    FiniteField(GradedModuleInvariants<FiniteField>),
}

#[derive(Clone, PartialEq, Eq)]
enum DerivedModuleInner {
    Rational(DerivedModule<PolynomialRing<RationalField>>),
    FiniteField(DerivedModule<PolynomialRing<FiniteField>>),
    QuotientRational(DerivedModule<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(DerivedModule<QuotientRing<PolynomialRing<FiniteField>>>),
}

#[derive(Clone, PartialEq, Eq)]
enum DerivedModuleHomomorphismInner {
    Rational(DerivedModuleHomomorphism<PolynomialRing<RationalField>>),
    FiniteField(DerivedModuleHomomorphism<PolynomialRing<FiniteField>>),
    QuotientRational(DerivedModuleHomomorphism<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(DerivedModuleHomomorphism<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// One exact Tor or Ext module with its verified resolution and homology
/// evidence.
#[pyclass(
    module = "noether.homalg",
    name = "DerivedModule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDerivedModule {
    inner: DerivedModuleInner,
}

#[pymethods]
impl PyDerivedModule {
    #[classattr]
    fn type_id() -> &'static str {
        DerivedModule::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[getter]
    fn functor(&self) -> &'static str {
        match &self.inner {
            DerivedModuleInner::Rational(value) => value.functor().name(),
            DerivedModuleInner::FiniteField(value) => value.functor().name(),
            DerivedModuleInner::QuotientRational(value) => value.functor().name(),
            DerivedModuleInner::QuotientFiniteField(value) => value.functor().name(),
        }
    }
    #[getter]
    fn degree(&self) -> usize {
        match &self.inner {
            DerivedModuleInner::Rational(value) => value.degree(),
            DerivedModuleInner::FiniteField(value) => value.degree(),
            DerivedModuleInner::QuotientRational(value) => value.degree(),
            DerivedModuleInner::QuotientFiniteField(value) => value.degree(),
        }
    }
    #[getter]
    fn resolution(&self) -> PyFreeResolution {
        PyFreeResolution {
            inner: match &self.inner {
                DerivedModuleInner::Rational(value) => {
                    FreeResolutionInner::Rational(value.resolution().clone())
                }
                DerivedModuleInner::FiniteField(value) => {
                    FreeResolutionInner::FiniteField(value.resolution().clone())
                }
                DerivedModuleInner::QuotientRational(value) => {
                    FreeResolutionInner::QuotientRational(value.resolution().clone())
                }
                DerivedModuleInner::QuotientFiniteField(value) => {
                    FreeResolutionInner::QuotientFiniteField(value.resolution().clone())
                }
            },
        }
    }
    #[getter]
    fn coefficient_module(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                DerivedModuleInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.coefficient_module().clone())
                }
                DerivedModuleInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.coefficient_module().clone())
                }
                DerivedModuleInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.coefficient_module().clone())
                }
                DerivedModuleInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(value.coefficient_module().clone())
                }
            },
        }
    }
    #[getter]
    fn order(&self) -> PyModuleMonomialOrder {
        PyModuleMonomialOrder {
            inner: match &self.inner {
                DerivedModuleInner::Rational(value) => value.order(),
                DerivedModuleInner::FiniteField(value) => value.order(),
                DerivedModuleInner::QuotientRational(value) => value.order(),
                DerivedModuleInner::QuotientFiniteField(value) => value.order(),
            }
            .clone(),
        }
    }
    #[getter]
    fn homology_evidence(&self) -> PyPresentedModuleHomology {
        PyPresentedModuleHomology {
            inner: match &self.inner {
                DerivedModuleInner::Rational(value) => {
                    PresentedModuleHomologyInner::Rational(value.homology_evidence().clone())
                }
                DerivedModuleInner::FiniteField(value) => {
                    PresentedModuleHomologyInner::FiniteField(value.homology_evidence().clone())
                }
                DerivedModuleInner::QuotientRational(value) => {
                    PresentedModuleHomologyInner::QuotientRational(
                        value.homology_evidence().clone(),
                    )
                }
                DerivedModuleInner::QuotientFiniteField(value) => {
                    PresentedModuleHomologyInner::QuotientFiniteField(
                        value.homology_evidence().clone(),
                    )
                }
            },
        }
    }
    #[getter]
    fn module(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                DerivedModuleInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.module().clone())
                }
                DerivedModuleInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.module().clone())
                }
                DerivedModuleInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.module().clone())
                }
                DerivedModuleInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(value.module().clone())
                }
            },
        }
    }
    #[getter]
    fn justified_through_degree(&self) -> Option<usize> {
        match &self.inner {
            DerivedModuleInner::Rational(value) => value.justified_through_degree(),
            DerivedModuleInner::FiniteField(value) => value.justified_through_degree(),
            DerivedModuleInner::QuotientRational(value) => value.justified_through_degree(),
            DerivedModuleInner::QuotientFiniteField(value) => value.justified_through_degree(),
        }
    }
    fn map_coefficients(
        &self,
        homomorphism: &PyPresentedModuleHomomorphism,
    ) -> PyResult<PyDerivedModuleHomomorphism> {
        let inner = match (&self.inner, &homomorphism.inner) {
            (
                DerivedModuleInner::Rational(source),
                PresentedModuleHomomorphismInner::Rational(map),
            ) => DerivedModuleHomomorphismInner::Rational(
                source.map_coefficients(map).map_err(derived_error)?,
            ),
            (
                DerivedModuleInner::FiniteField(source),
                PresentedModuleHomomorphismInner::FiniteField(map),
            ) => DerivedModuleHomomorphismInner::FiniteField(
                source.map_coefficients(map).map_err(derived_error)?,
            ),
            (
                DerivedModuleInner::QuotientRational(source),
                PresentedModuleHomomorphismInner::QuotientRational(map),
            ) => DerivedModuleHomomorphismInner::QuotientRational(
                source.map_coefficients(map).map_err(derived_error)?,
            ),
            (
                DerivedModuleInner::QuotientFiniteField(source),
                PresentedModuleHomomorphismInner::QuotientFiniteField(map),
            ) => DerivedModuleHomomorphismInner::QuotientFiniteField(
                source.map_coefficients(map).map_err(derived_error)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "the derived module and coefficient homomorphism must share one exact context",
                ));
            }
        };
        Ok(PyDerivedModuleHomomorphism { inner })
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            DerivedModuleInner::Rational(value) => value.verify(),
            DerivedModuleInner::FiniteField(value) => value.verify(),
            DerivedModuleInner::QuotientRational(value) => value.verify(),
            DerivedModuleInner::QuotientFiniteField(value) => value.verify(),
        }
        .map_err(derived_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            DerivedModuleInner::Rational(value) => value.to_json(),
            DerivedModuleInner::FiniteField(value) => value.to_json(),
            DerivedModuleInner::QuotientRational(value) => value.to_json(),
            DerivedModuleInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let ring = serialized_module_ring(value, "/resolution/module/presentation/source/ring")?;
        let inner = match ring {
            SerializedModuleRing::PolynomialRational => DerivedModuleInner::Rational(
                DerivedModule::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => DerivedModuleInner::FiniteField(
                DerivedModule::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => DerivedModuleInner::QuotientRational(
                DerivedModule::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => DerivedModuleInner::QuotientFiniteField(
                DerivedModule::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialInteger | SerializedModuleRing::PolynomialResidue => {
                return Err(PyTypeError::new_err(
                    "unsupported derived-module coefficient context",
                ));
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DerivedModule(functor='{}', degree={}, generators={})",
            self.functor(),
            self.degree(),
            self.module().generator_count()
        ))
    }
}

/// A checked functorial map between two Tor or Ext modules.
#[pyclass(
    module = "noether.homalg",
    name = "DerivedModuleHomomorphism",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDerivedModuleHomomorphism {
    inner: DerivedModuleHomomorphismInner,
}

#[pymethods]
impl PyDerivedModuleHomomorphism {
    #[classattr]
    fn type_id() -> &'static str {
        DerivedModuleHomomorphism::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[getter]
    fn source(&self) -> PyDerivedModule {
        PyDerivedModule {
            inner: match &self.inner {
                DerivedModuleHomomorphismInner::Rational(value) => {
                    DerivedModuleInner::Rational(value.source().clone())
                }
                DerivedModuleHomomorphismInner::FiniteField(value) => {
                    DerivedModuleInner::FiniteField(value.source().clone())
                }
                DerivedModuleHomomorphismInner::QuotientRational(value) => {
                    DerivedModuleInner::QuotientRational(value.source().clone())
                }
                DerivedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    DerivedModuleInner::QuotientFiniteField(value.source().clone())
                }
            },
        }
    }
    #[getter]
    fn target(&self) -> PyDerivedModule {
        PyDerivedModule {
            inner: match &self.inner {
                DerivedModuleHomomorphismInner::Rational(value) => {
                    DerivedModuleInner::Rational(value.target().clone())
                }
                DerivedModuleHomomorphismInner::FiniteField(value) => {
                    DerivedModuleInner::FiniteField(value.target().clone())
                }
                DerivedModuleHomomorphismInner::QuotientRational(value) => {
                    DerivedModuleInner::QuotientRational(value.target().clone())
                }
                DerivedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    DerivedModuleInner::QuotientFiniteField(value.target().clone())
                }
            },
        }
    }
    #[getter]
    fn coefficient_homomorphism(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                DerivedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(
                        value.coefficient_homomorphism().clone(),
                    )
                }
                DerivedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(
                        value.coefficient_homomorphism().clone(),
                    )
                }
                DerivedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(
                        value.coefficient_homomorphism().clone(),
                    )
                }
                DerivedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.coefficient_homomorphism().clone(),
                    )
                }
            },
        }
    }
    #[getter]
    fn homomorphism(&self) -> PyPresentedModuleHomomorphism {
        PyPresentedModuleHomomorphism {
            inner: match &self.inner {
                DerivedModuleHomomorphismInner::Rational(value) => {
                    PresentedModuleHomomorphismInner::Rational(value.homomorphism().clone())
                }
                DerivedModuleHomomorphismInner::FiniteField(value) => {
                    PresentedModuleHomomorphismInner::FiniteField(value.homomorphism().clone())
                }
                DerivedModuleHomomorphismInner::QuotientRational(value) => {
                    PresentedModuleHomomorphismInner::QuotientRational(value.homomorphism().clone())
                }
                DerivedModuleHomomorphismInner::QuotientFiniteField(value) => {
                    PresentedModuleHomomorphismInner::QuotientFiniteField(
                        value.homomorphism().clone(),
                    )
                }
            },
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            DerivedModuleHomomorphismInner::Rational(value) => value.verify(),
            DerivedModuleHomomorphismInner::FiniteField(value) => value.verify(),
            DerivedModuleHomomorphismInner::QuotientRational(value) => value.verify(),
            DerivedModuleHomomorphismInner::QuotientFiniteField(value) => value.verify(),
        }
        .map_err(derived_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            DerivedModuleHomomorphismInner::Rational(value) => value.to_json(),
            DerivedModuleHomomorphismInner::FiniteField(value) => value.to_json(),
            DerivedModuleHomomorphismInner::QuotientRational(value) => value.to_json(),
            DerivedModuleHomomorphismInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let ring =
            serialized_module_ring(value, "/source/resolution/module/presentation/source/ring")?;
        let inner = match ring {
            SerializedModuleRing::PolynomialRational => {
                DerivedModuleHomomorphismInner::Rational(
                    DerivedModuleHomomorphism::<PolynomialRing<RationalField>>::from_json(value)
                        .map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialFiniteField => {
                DerivedModuleHomomorphismInner::FiniteField(
                    DerivedModuleHomomorphism::<PolynomialRing<FiniteField>>::from_json(value)
                        .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientRational => {
                DerivedModuleHomomorphismInner::QuotientRational(
                    DerivedModuleHomomorphism::<
                        QuotientRing<PolynomialRing<RationalField>>,
                    >::from_json(value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::QuotientFiniteField => {
                DerivedModuleHomomorphismInner::QuotientFiniteField(
                    DerivedModuleHomomorphism::<
                        QuotientRing<PolynomialRing<FiniteField>>,
                    >::from_json(value)
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialInteger | SerializedModuleRing::PolynomialResidue => {
                return Err(PyTypeError::new_err(
                    "unsupported derived-module-homomorphism coefficient context",
                ));
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DerivedModuleHomomorphism(functor='{}', degree={})",
            self.source().functor(),
            self.source().degree()
        ))
    }
}

fn derived_from_presented(
    source: &PyPresentedModule,
    coefficient_module: &PyPresentedModule,
    degree: usize,
    order: &PyModuleMonomialOrder,
    operation: &str,
) -> PyResult<PyDerivedModule> {
    let inner = match (&source.inner, &coefficient_module.inner) {
        (PresentedModuleInner::Rational(source), PresentedModuleInner::Rational(target)) => {
            let value = match operation {
                "tor" => source.tor(target, degree, order.inner.clone()),
                "ext" => source.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::Rational(value)
        }
        (PresentedModuleInner::FiniteField(source), PresentedModuleInner::FiniteField(target)) => {
            let value = match operation {
                "tor" => source.tor(target, degree, order.inner.clone()),
                "ext" => source.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::FiniteField(value)
        }
        (
            PresentedModuleInner::QuotientRational(source),
            PresentedModuleInner::QuotientRational(target),
        ) => {
            let value = match operation {
                "tor" => source.tor(target, degree, order.inner.clone()),
                "ext" => source.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::QuotientRational(value)
        }
        (
            PresentedModuleInner::QuotientFiniteField(source),
            PresentedModuleInner::QuotientFiniteField(target),
        ) => {
            let value = match operation {
                "tor" => source.tor(target, degree, order.inner.clone()),
                "ext" => source.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::QuotientFiniteField(value)
        }
        (PresentedModuleInner::Integer(_), _) | (PresentedModuleInner::Residue(_), _) => {
            return Err(PyTypeError::new_err(
                "Tor and Ext currently require a polynomial ring or polynomial quotient over Q or FiniteField",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "derived-module coefficient contexts do not match",
            ));
        }
    };
    Ok(PyDerivedModule { inner })
}

fn derived_from_resolution(
    resolution: &PyFreeResolution,
    coefficient_module: &PyPresentedModule,
    degree: usize,
    order: &PyModuleMonomialOrder,
    operation: &str,
) -> PyResult<PyDerivedModule> {
    let inner = match (&resolution.inner, &coefficient_module.inner) {
        (FreeResolutionInner::Rational(resolution), PresentedModuleInner::Rational(target)) => {
            let value = match operation {
                "tor" => resolution.tor(target, degree, order.inner.clone()),
                "ext" => resolution.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::Rational(value)
        }
        (
            FreeResolutionInner::FiniteField(resolution),
            PresentedModuleInner::FiniteField(target),
        ) => {
            let value = match operation {
                "tor" => resolution.tor(target, degree, order.inner.clone()),
                "ext" => resolution.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::FiniteField(value)
        }
        (
            FreeResolutionInner::QuotientRational(resolution),
            PresentedModuleInner::QuotientRational(target),
        ) => {
            let value = match operation {
                "tor" => resolution.tor(target, degree, order.inner.clone()),
                "ext" => resolution.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::QuotientRational(value)
        }
        (
            FreeResolutionInner::QuotientFiniteField(resolution),
            PresentedModuleInner::QuotientFiniteField(target),
        ) => {
            let value = match operation {
                "tor" => resolution.tor(target, degree, order.inner.clone()),
                "ext" => resolution.ext(target, degree, order.inner.clone()),
                _ => return Err(PyValueError::new_err("unknown derived operation")),
            }
            .map_err(derived_error)?;
            DerivedModuleInner::QuotientFiniteField(value)
        }
        _ => {
            return Err(PyValueError::new_err(
                "resolution and coefficient-module contexts do not match",
            ));
        }
    };
    Ok(PyDerivedModule { inner })
}

/// Exact completeness or homological truncation of a free resolution.
#[pyclass(
    module = "noether.homalg",
    name = "ResolutionCompleteness",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyResolutionCompleteness {
    inner: ResolutionCompleteness,
}

#[pymethods]
impl PyResolutionCompleteness {
    #[classattr]
    fn type_id() -> &'static str {
        ResolutionCompleteness::TYPE_ID
    }
    #[classmethod]
    fn complete(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ResolutionCompleteness::Complete,
        }
    }
    #[classmethod]
    fn through(_cls: &Bound<'_, PyType>, degree: usize) -> Self {
        Self {
            inner: ResolutionCompleteness::ThroughHomologicalDegree { degree },
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }
    #[getter]
    fn through_homological_degree(&self) -> Option<usize> {
        self.inner.through_homological_degree()
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ResolutionCompleteness::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match self.inner.through_homological_degree() {
            None => "ResolutionCompleteness(kind='complete')".to_owned(),
            Some(degree) => format!(
                "ResolutionCompleteness(kind='through_homological_degree', degree={degree})"
            ),
        })
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Exact minimality claim of a graded free resolution.
#[pyclass(
    module = "noether.homalg",
    name = "ResolutionMinimality",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyResolutionMinimality {
    inner: ResolutionMinimality,
}

#[pymethods]
impl PyResolutionMinimality {
    #[classattr]
    fn type_id() -> &'static str {
        ResolutionMinimality::TYPE_ID
    }
    #[classmethod]
    fn not_claimed(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ResolutionMinimality::NotClaimed,
        }
    }
    #[classmethod]
    fn minimal(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ResolutionMinimality::Minimal,
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }
    fn is_minimal(&self) -> bool {
        self.inner.is_minimal()
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ResolutionMinimality::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ResolutionMinimality(kind='{}')",
            self.inner.kind()
        ))
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Canonical exact multigraded Betti numbers.
#[pyclass(
    module = "noether.homalg",
    name = "GradedBettiTable",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedBettiTable {
    inner: GradedBettiTable,
}

#[pymethods]
impl PyGradedBettiTable {
    #[classattr]
    fn type_id() -> &'static str {
        GradedBettiTable::TYPE_ID
    }
    #[getter]
    fn grading(&self) -> PyFreeAbelianGrading {
        PyFreeAbelianGrading {
            inner: self.inner.grading().clone(),
        }
    }
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }
    #[getter]
    fn completeness(&self) -> PyResolutionCompleteness {
        PyResolutionCompleteness {
            inner: self.inner.completeness().clone(),
        }
    }
    #[getter]
    fn entries(&self) -> Vec<(usize, PyDegree, usize)> {
        self.inner
            .entries()
            .iter()
            .map(|(homological_degree, internal_degree, multiplicity)| {
                (
                    *homological_degree,
                    PyDegree {
                        inner: internal_degree.clone(),
                    },
                    *multiplicity,
                )
            })
            .collect()
    }
    fn get(&self, homological_degree: usize, internal_degree: &PyDegree) -> usize {
        self.inner.get(homological_degree, &internal_degree.inner)
    }
    fn total(&self, homological_degree: usize) -> usize {
        self.inner.total(homological_degree)
    }
    fn projective_dimension(&self) -> Option<usize> {
        self.inner.projective_dimension()
    }
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(betti_error)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GradedBettiTable::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "GradedBettiTable(length={}, entries={})",
            self.inner.length(),
            self.inner.entries().len()
        ))
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Exact structural invariants with retained minimal-resolution and Hilbert
/// evidence.
#[pyclass(
    module = "noether.homalg",
    name = "GradedModuleInvariants",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedModuleInvariants {
    inner: GradedModuleInvariantsInner,
}

#[pymethods]
impl PyGradedModuleInvariants {
    #[classattr]
    fn type_id() -> &'static str {
        GradedModuleInvariants::<RationalField>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[getter]
    fn module(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                GradedModuleInvariantsInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.module().clone())
                }
                GradedModuleInvariantsInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.module().clone())
                }
            },
        }
    }
    #[getter]
    fn order(&self) -> PyModuleMonomialOrder {
        PyModuleMonomialOrder {
            inner: match &self.inner {
                GradedModuleInvariantsInner::Rational(value) => value.order().clone(),
                GradedModuleInvariantsInner::FiniteField(value) => value.order().clone(),
            },
        }
    }
    #[getter]
    fn resolution(&self) -> PyFreeResolution {
        PyFreeResolution {
            inner: match &self.inner {
                GradedModuleInvariantsInner::Rational(value) => {
                    FreeResolutionInner::Rational(value.resolution().clone())
                }
                GradedModuleInvariantsInner::FiniteField(value) => {
                    FreeResolutionInner::FiniteField(value.resolution().clone())
                }
            },
        }
    }
    #[getter]
    fn betti_table(&self) -> PyGradedBettiTable {
        PyGradedBettiTable {
            inner: match &self.inner {
                GradedModuleInvariantsInner::Rational(value) => value.betti_table().clone(),
                GradedModuleInvariantsInner::FiniteField(value) => value.betti_table().clone(),
            },
        }
    }
    #[getter]
    fn hilbert_series(&self) -> PyHilbertSeries {
        PyHilbertSeries {
            inner: match &self.inner {
                GradedModuleInvariantsInner::Rational(value) => value.hilbert_series().clone(),
                GradedModuleInvariantsInner::FiniteField(value) => value.hilbert_series().clone(),
            },
        }
    }
    #[getter]
    fn projective_dimension(&self) -> Option<usize> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.projective_dimension(),
            GradedModuleInvariantsInner::FiniteField(value) => value.projective_dimension(),
        }
    }
    #[getter]
    fn regularity(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value
                .regularity()
                .map(|value| integer_to_python(py, value))
                .transpose(),
            GradedModuleInvariantsInner::FiniteField(value) => value
                .regularity()
                .map(|value| integer_to_python(py, value))
                .transpose(),
        }
    }
    #[getter]
    fn depth(&self) -> Option<usize> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.depth(),
            GradedModuleInvariantsInner::FiniteField(value) => value.depth(),
        }
    }
    #[getter]
    fn dimension(&self) -> Option<usize> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.dimension(),
            GradedModuleInvariantsInner::FiniteField(value) => value.dimension(),
        }
    }
    #[getter]
    fn is_cohen_macaulay(&self) -> Option<bool> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.is_cohen_macaulay(),
            GradedModuleInvariantsInner::FiniteField(value) => value.is_cohen_macaulay(),
        }
    }
    #[getter]
    fn is_perfect(&self) -> Option<bool> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.is_perfect(),
            GradedModuleInvariantsInner::FiniteField(value) => value.is_perfect(),
        }
    }
    #[getter]
    fn theorem(&self) -> &'static str {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.theorem(),
            GradedModuleInvariantsInner::FiniteField(value) => value.theorem(),
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.verify(),
            GradedModuleInvariantsInner::FiniteField(value) => value.verify(),
        }
        .map_err(module_invariant_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.to_json(),
            GradedModuleInvariantsInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_module_ring(value, "/module/presentation/source/ring")? {
            SerializedModuleRing::PolynomialRational => GradedModuleInvariantsInner::Rational(
                GradedModuleInvariants::<RationalField>::from_json(value).map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => {
                GradedModuleInvariantsInner::FiniteField(
                    GradedModuleInvariants::<FiniteField>::from_json(value).map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialInteger
            | SerializedModuleRing::PolynomialResidue
            | SerializedModuleRing::QuotientRational
            | SerializedModuleRing::QuotientFiniteField => {
                return Err(PyTypeError::new_err(
                    "unsupported graded-module-invariant coefficient context",
                ));
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            GradedModuleInvariantsInner::Rational(value) => value.to_string(),
            GradedModuleInvariantsInner::FiniteField(value) => value.to_string(),
        })
    }
}

/// A verified finite free resolution over a supported polynomial field.
#[pyclass(
    module = "noether.homalg",
    name = "FreeResolution",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFreeResolution {
    inner: FreeResolutionInner,
}

impl PyFreeResolution {
    pub(crate) fn from_rational_polynomial(
        value: FreeResolution<PolynomialRing<RationalField>>,
    ) -> Self {
        Self {
            inner: FreeResolutionInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field_polynomial(
        value: FreeResolution<PolynomialRing<FiniteField>>,
    ) -> Self {
        Self {
            inner: FreeResolutionInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyFreeResolution {
    #[classattr]
    fn type_id() -> &'static str {
        FreeResolution::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[classmethod]
    fn from_maps(
        _cls: &Bound<'_, PyType>,
        module: &PyPresentedModule,
        differentials: Vec<PyModuleHomomorphism>,
        augmentation: &PyModuleHomomorphism,
        augmentation_inverse: &PyModuleHomomorphism,
        completeness: &PyResolutionCompleteness,
        minimality: &PyResolutionMinimality,
    ) -> PyResult<Self> {
        let inner = match (
            &module.inner,
            &augmentation.inner,
            &augmentation_inverse.inner,
        ) {
            (
                PresentedModuleInner::Rational(module),
                ModuleHomomorphismInner::Rational(augmentation),
                ModuleHomomorphismInner::Rational(augmentation_inverse),
            ) => {
                let differentials = differentials
                    .into_iter()
                    .map(|value| match value.inner {
                        ModuleHomomorphismInner::Rational(value) => Ok(value),
                        _ => Err(PyTypeError::new_err(
                            "resolution maps must use one polynomial coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                FreeResolutionInner::Rational(
                    FreeResolution::<PolynomialRing<RationalField>>::from_maps(
                        module.clone(),
                        differentials,
                        augmentation.clone(),
                        augmentation_inverse.clone(),
                        completeness.inner.clone(),
                        minimality.inner.clone(),
                    )
                    .map_err(resolution_error)?,
                )
            }
            (
                PresentedModuleInner::FiniteField(module),
                ModuleHomomorphismInner::FiniteField(augmentation),
                ModuleHomomorphismInner::FiniteField(augmentation_inverse),
            ) => {
                let differentials = differentials
                    .into_iter()
                    .map(|value| match value.inner {
                        ModuleHomomorphismInner::FiniteField(value) => Ok(value),
                        _ => Err(PyTypeError::new_err(
                            "resolution maps must use one polynomial coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                FreeResolutionInner::FiniteField(
                    FreeResolution::<PolynomialRing<FiniteField>>::from_maps(
                        module.clone(),
                        differentials,
                        augmentation.clone(),
                        augmentation_inverse.clone(),
                        completeness.inner.clone(),
                        minimality.inner.clone(),
                    )
                    .map_err(resolution_error)?,
                )
            }
            (
                PresentedModuleInner::QuotientRational(module),
                ModuleHomomorphismInner::QuotientRational(augmentation),
                ModuleHomomorphismInner::QuotientRational(augmentation_inverse),
            ) => {
                let differentials = differentials
                    .into_iter()
                    .map(|value| match value.inner {
                        ModuleHomomorphismInner::QuotientRational(value) => Ok(value),
                        _ => Err(PyTypeError::new_err(
                            "resolution maps must use one polynomial-quotient coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                FreeResolutionInner::QuotientRational(
                    FreeResolution::<QuotientRing<PolynomialRing<RationalField>>>::from_maps(
                        module.clone(),
                        differentials,
                        augmentation.clone(),
                        augmentation_inverse.clone(),
                        completeness.inner.clone(),
                        minimality.inner.clone(),
                    )
                    .map_err(resolution_error)?,
                )
            }
            (
                PresentedModuleInner::QuotientFiniteField(module),
                ModuleHomomorphismInner::QuotientFiniteField(augmentation),
                ModuleHomomorphismInner::QuotientFiniteField(augmentation_inverse),
            ) => {
                let differentials = differentials
                    .into_iter()
                    .map(|value| match value.inner {
                        ModuleHomomorphismInner::QuotientFiniteField(value) => Ok(value),
                        _ => Err(PyTypeError::new_err(
                            "resolution maps must use one polynomial-quotient coefficient context",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                FreeResolutionInner::QuotientFiniteField(
                    FreeResolution::<QuotientRing<PolynomialRing<FiniteField>>>::from_maps(
                        module.clone(),
                        differentials,
                        augmentation.clone(),
                        augmentation_inverse.clone(),
                        completeness.inner.clone(),
                        minimality.inner.clone(),
                    )
                    .map_err(resolution_error)?,
                )
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "resolution module and augmentation maps must use one supported field or quotient context",
                ));
            }
        };
        Ok(Self { inner })
    }
    #[getter]
    fn module(&self) -> PyPresentedModule {
        PyPresentedModule {
            inner: match &self.inner {
                FreeResolutionInner::Rational(value) => {
                    PresentedModuleInner::Rational(value.module().clone())
                }
                FreeResolutionInner::FiniteField(value) => {
                    PresentedModuleInner::FiniteField(value.module().clone())
                }
                FreeResolutionInner::QuotientRational(value) => {
                    PresentedModuleInner::QuotientRational(value.module().clone())
                }
                FreeResolutionInner::QuotientFiniteField(value) => {
                    PresentedModuleInner::QuotientFiniteField(value.module().clone())
                }
            },
        }
    }
    #[getter]
    fn completeness(&self) -> PyResolutionCompleteness {
        PyResolutionCompleteness {
            inner: match &self.inner {
                FreeResolutionInner::Rational(value) => value.completeness(),
                FreeResolutionInner::FiniteField(value) => value.completeness(),
                FreeResolutionInner::QuotientRational(value) => value.completeness(),
                FreeResolutionInner::QuotientFiniteField(value) => value.completeness(),
            }
            .clone(),
        }
    }
    #[getter]
    fn minimality(&self) -> PyResolutionMinimality {
        PyResolutionMinimality {
            inner: match &self.inner {
                FreeResolutionInner::Rational(value) => value.minimality(),
                FreeResolutionInner::FiniteField(value) => value.minimality(),
                FreeResolutionInner::QuotientRational(value) => value.minimality(),
                FreeResolutionInner::QuotientFiniteField(value) => value.minimality(),
            }
            .clone(),
        }
    }
    #[getter]
    fn augmentation(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                FreeResolutionInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.augmentation().clone())
                }
                FreeResolutionInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.augmentation().clone())
                }
                FreeResolutionInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.augmentation().clone())
                }
                FreeResolutionInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.augmentation().clone())
                }
            },
        }
    }
    #[getter]
    fn augmentation_inverse(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                FreeResolutionInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.augmentation_inverse().clone())
                }
                FreeResolutionInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.augmentation_inverse().clone())
                }
                FreeResolutionInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.augmentation_inverse().clone())
                }
                FreeResolutionInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(
                        value.augmentation_inverse().clone(),
                    )
                }
            },
        }
    }
    #[getter]
    fn length(&self) -> usize {
        match &self.inner {
            FreeResolutionInner::Rational(value) => value.length(),
            FreeResolutionInner::FiniteField(value) => value.length(),
            FreeResolutionInner::QuotientRational(value) => value.length(),
            FreeResolutionInner::QuotientFiniteField(value) => value.length(),
        }
    }
    fn free_module(&self, degree: usize) -> Option<PyFreeModule> {
        match &self.inner {
            FreeResolutionInner::Rational(value) => {
                value
                    .free_module(degree)
                    .cloned()
                    .map(|value| PyFreeModule {
                        inner: FreeModuleInner::Rational(value),
                    })
            }
            FreeResolutionInner::FiniteField(value) => {
                value
                    .free_module(degree)
                    .cloned()
                    .map(|value| PyFreeModule {
                        inner: FreeModuleInner::FiniteField(value),
                    })
            }
            FreeResolutionInner::QuotientRational(value) => {
                value
                    .free_module(degree)
                    .cloned()
                    .map(|value| PyFreeModule {
                        inner: FreeModuleInner::QuotientRational(value),
                    })
            }
            FreeResolutionInner::QuotientFiniteField(value) => value
                .free_module(degree)
                .cloned()
                .map(|value| PyFreeModule {
                    inner: FreeModuleInner::QuotientFiniteField(value),
                }),
        }
    }
    fn differential(&self, degree: usize) -> Option<PyModuleHomomorphism> {
        match &self.inner {
            FreeResolutionInner::Rational(value) => {
                value
                    .differential(degree)
                    .cloned()
                    .map(|value| PyModuleHomomorphism {
                        inner: ModuleHomomorphismInner::Rational(value),
                    })
            }
            FreeResolutionInner::FiniteField(value) => {
                value
                    .differential(degree)
                    .cloned()
                    .map(|value| PyModuleHomomorphism {
                        inner: ModuleHomomorphismInner::FiniteField(value),
                    })
            }
            FreeResolutionInner::QuotientRational(value) => value
                .differential(degree)
                .cloned()
                .map(|value| PyModuleHomomorphism {
                    inner: ModuleHomomorphismInner::QuotientRational(value),
                }),
            FreeResolutionInner::QuotientFiniteField(value) => value
                .differential(degree)
                .cloned()
                .map(|value| PyModuleHomomorphism {
                    inner: ModuleHomomorphismInner::QuotientFiniteField(value),
                }),
        }
    }
    fn is_minimal(&self) -> bool {
        match &self.inner {
            FreeResolutionInner::Rational(value) => value.is_minimal(),
            FreeResolutionInner::FiniteField(value) => value.is_minimal(),
            FreeResolutionInner::QuotientRational(value) => value.is_minimal(),
            FreeResolutionInner::QuotientFiniteField(value) => value.is_minimal(),
        }
    }
    fn minimalize(&self, py: Python<'_>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: minimization performs exact module reduction.
        py.detach(|| {
            let inner = match &self.inner {
                FreeResolutionInner::Rational(value) => {
                    FreeResolutionInner::Rational(value.minimalize().map_err(resolution_error)?)
                }
                FreeResolutionInner::FiniteField(value) => {
                    FreeResolutionInner::FiniteField(value.minimalize().map_err(resolution_error)?)
                }
                FreeResolutionInner::QuotientRational(_)
                | FreeResolutionInner::QuotientFiniteField(_) => {
                    return Err(PyTypeError::new_err(
                        "minimalization over polynomial quotients is not yet admitted",
                    ));
                }
            };
            Ok(Self { inner })
        })
    }
    fn graded_betti_table(&self, py: Python<'_>) -> PyResult<PyGradedBettiTable> {
        // NOETHER_RELEASES_GIL: extracting a certified table verifies the resolution.
        py.detach(|| {
            let inner = match &self.inner {
                FreeResolutionInner::Rational(value) => value.graded_betti_table(),
                FreeResolutionInner::FiniteField(value) => value.graded_betti_table(),
                FreeResolutionInner::QuotientRational(_)
                | FreeResolutionInner::QuotientFiniteField(_) => {
                    return Err(PyTypeError::new_err(
                        "graded Betti tables over polynomial quotients are not yet admitted",
                    ));
                }
            }
            .map_err(betti_error)?;
            Ok(PyGradedBettiTable { inner })
        })
    }
    fn tor(
        &self,
        py: Python<'_>,
        coefficient_module: &PyPresentedModule,
        degree: usize,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyDerivedModule> {
        // NOETHER_RELEASES_GIL: derived-module construction is material exact work.
        py.detach(|| derived_from_resolution(self, coefficient_module, degree, order, "tor"))
    }
    fn ext(
        &self,
        py: Python<'_>,
        coefficient_module: &PyPresentedModule,
        degree: usize,
        order: &PyModuleMonomialOrder,
    ) -> PyResult<PyDerivedModule> {
        // NOETHER_RELEASES_GIL: derived-module construction is material exact work.
        py.detach(|| derived_from_resolution(self, coefficient_module, degree, order, "ext"))
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // NOETHER_RELEASES_GIL: verification replays every differential.
        py.detach(|| {
            match &self.inner {
                FreeResolutionInner::Rational(value) => value.verify(),
                FreeResolutionInner::FiniteField(value) => value.verify(),
                FreeResolutionInner::QuotientRational(value) => value.verify(),
                FreeResolutionInner::QuotientFiniteField(value) => value.verify(),
            }
            .map_err(resolution_error)
        })
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: a resolution payload can be material.
        py.detach(|| {
            match &self.inner {
                FreeResolutionInner::Rational(value) => value.to_json(),
                FreeResolutionInner::FiniteField(value) => value.to_json(),
                FreeResolutionInner::QuotientRational(value) => value.to_json(),
                FreeResolutionInner::QuotientFiniteField(value) => value.to_json(),
            }
            .map_err(value_error)
        })
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            let ring = serialized_module_ring(&value, "/module/presentation/source/ring")?;
            let inner = match ring {
                SerializedModuleRing::PolynomialRational => FreeResolutionInner::Rational(
                    FreeResolution::<PolynomialRing<RationalField>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::PolynomialFiniteField => FreeResolutionInner::FiniteField(
                    FreeResolution::<PolynomialRing<FiniteField>>::from_json(&value)
                        .map_err(value_error)?,
                ),
                SerializedModuleRing::QuotientRational => FreeResolutionInner::QuotientRational(
                    FreeResolution::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                        &value,
                    )
                    .map_err(value_error)?,
                ),
                SerializedModuleRing::QuotientFiniteField => {
                    FreeResolutionInner::QuotientFiniteField(
                        FreeResolution::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                            &value,
                        )
                        .map_err(value_error)?,
                    )
                }
                SerializedModuleRing::PolynomialInteger
                | SerializedModuleRing::PolynomialResidue => {
                    return Err(PyTypeError::new_err(
                        "unsupported free-resolution coefficient context",
                    ));
                }
            };
            Ok(Self { inner })
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FreeResolution(length={}, completeness='{}', minimality='{}')",
            self.length(),
            self.completeness().__str__(),
            self.minimality().__str__()
        ))
    }
    fn __str__(&self) -> String {
        match &self.inner {
            FreeResolutionInner::Rational(value) => value.to_string(),
            FreeResolutionInner::FiniteField(value) => value.to_string(),
            FreeResolutionInner::QuotientRational(value) => value.to_string(),
            FreeResolutionInner::QuotientFiniteField(value) => value.to_string(),
        }
    }
}

/// A standard module-monomial order extending a polynomial monomial order.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleMonomialOrder",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleMonomialOrder {
    inner: ModuleMonomialOrder,
}

#[pymethods]
impl PyModuleMonomialOrder {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleMonomialOrder::TYPE_ID
    }
    #[classmethod]
    fn term_over_position(_cls: &Bound<'_, PyType>, base_order: &PyMonomialOrder) -> Self {
        Self {
            inner: ModuleMonomialOrder::TermOverPosition(base_order.inner.clone()),
        }
    }
    #[classmethod]
    fn position_over_term(_cls: &Bound<'_, PyType>, base_order: &PyMonomialOrder) -> Self {
        Self {
            inner: ModuleMonomialOrder::PositionOverTerm(base_order.inner.clone()),
        }
    }
    #[classmethod]
    fn schreyer(
        _cls: &Bound<'_, PyType>,
        target_order: &PyModuleMonomialOrder,
        leading_monomials: Vec<PyModuleMonomial>,
    ) -> Self {
        Self {
            inner: ModuleMonomialOrder::schreyer(
                target_order.inner.clone(),
                leading_monomials
                    .into_iter()
                    .map(|value| value.inner)
                    .collect(),
            ),
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ModuleMonomialOrder::TermOverPosition(_) => "term_over_position",
            ModuleMonomialOrder::PositionOverTerm(_) => "position_over_term",
            ModuleMonomialOrder::Schreyer { .. } => "schreyer",
        }
    }
    #[getter]
    fn base_order(&self) -> PyMonomialOrder {
        PyMonomialOrder {
            inner: self.inner.base_order().clone(),
        }
    }
    #[getter]
    fn schreyer_target_order(&self) -> Option<Self> {
        self.inner.schreyer_data().map(|(target, _)| Self {
            inner: target.clone(),
        })
    }
    #[getter]
    fn schreyer_leading_monomials(&self) -> Option<Vec<PyModuleMonomial>> {
        self.inner.schreyer_data().map(|(_, leading)| {
            leading
                .iter()
                .cloned()
                .map(|inner| PyModuleMonomial { inner })
                .collect()
        })
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ModuleMonomialOrder::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("ModuleMonomialOrder(kind={})", self.kind()))
    }
}

/// One polynomial monomial times an ordered free-module basis vector.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleMonomial",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleMonomial {
    inner: ModuleMonomial,
}

#[pymethods]
impl PyModuleMonomial {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleMonomial::TYPE_ID
    }
    #[getter]
    fn exponent(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .exponent()
            .iter()
            .map(|power| integer_to_python(py, power))
            .collect()
    }
    #[getter]
    fn component(&self) -> usize {
        self.inner.component()
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ModuleMonomial::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ModuleMonomial(exponent={:?}, component={})",
            self.inner.exponent(),
            self.inner.component()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModuleGroebnerBasisInner {
    Rational(ModuleGroebnerBasis<PolynomialRing<RationalField>>),
    FiniteField(ModuleGroebnerBasis<PolynomialRing<FiniteField>>),
    QuotientRational(ModuleGroebnerBasis<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(ModuleGroebnerBasis<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// A complete reduced module Groebner basis with an exact change map.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleGroebnerBasis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleGroebnerBasis {
    inner: ModuleGroebnerBasisInner,
}

#[pymethods]
impl PyModuleGroebnerBasis {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleGroebnerBasis::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[getter]
    fn input(&self) -> PySubmodule {
        PySubmodule {
            inner: match &self.inner {
                ModuleGroebnerBasisInner::Rational(value) => {
                    SubmoduleInner::Rational(value.input().clone())
                }
                ModuleGroebnerBasisInner::FiniteField(value) => {
                    SubmoduleInner::FiniteField(value.input().clone())
                }
                ModuleGroebnerBasisInner::QuotientRational(value) => {
                    SubmoduleInner::QuotientRational(value.input().clone())
                }
                ModuleGroebnerBasisInner::QuotientFiniteField(value) => {
                    SubmoduleInner::QuotientFiniteField(value.input().clone())
                }
            },
        }
    }
    #[getter]
    fn order(&self) -> PyModuleMonomialOrder {
        PyModuleMonomialOrder {
            inner: match &self.inner {
                ModuleGroebnerBasisInner::Rational(value) => value.order().clone(),
                ModuleGroebnerBasisInner::FiniteField(value) => value.order().clone(),
                ModuleGroebnerBasisInner::QuotientRational(value) => value.order().clone(),
                ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.order().clone(),
            },
        }
    }
    #[getter]
    fn basis(&self) -> PySubmodule {
        PySubmodule {
            inner: match &self.inner {
                ModuleGroebnerBasisInner::Rational(value) => {
                    SubmoduleInner::Rational(value.basis().clone())
                }
                ModuleGroebnerBasisInner::FiniteField(value) => {
                    SubmoduleInner::FiniteField(value.basis().clone())
                }
                ModuleGroebnerBasisInner::QuotientRational(value) => {
                    SubmoduleInner::QuotientRational(value.basis().clone())
                }
                ModuleGroebnerBasisInner::QuotientFiniteField(value) => {
                    SubmoduleInner::QuotientFiniteField(value.basis().clone())
                }
            },
        }
    }
    #[getter]
    fn transformation(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: match &self.inner {
                ModuleGroebnerBasisInner::Rational(value) => {
                    ModuleHomomorphismInner::Rational(value.transformation().clone())
                }
                ModuleGroebnerBasisInner::FiniteField(value) => {
                    ModuleHomomorphismInner::FiniteField(value.transformation().clone())
                }
                ModuleGroebnerBasisInner::QuotientRational(value) => {
                    ModuleHomomorphismInner::QuotientRational(value.transformation().clone())
                }
                ModuleGroebnerBasisInner::QuotientFiniteField(value) => {
                    ModuleHomomorphismInner::QuotientFiniteField(value.transformation().clone())
                }
            },
        }
    }
    #[getter]
    fn leading_monomials(&self) -> Vec<PyModuleMonomial> {
        match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.leading_monomials(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.leading_monomials(),
            ModuleGroebnerBasisInner::QuotientRational(value) => value.leading_monomials(),
            ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.leading_monomials(),
        }
        .iter()
        .cloned()
        .map(|inner| PyModuleMonomial { inner })
        .collect()
    }
    fn schreyer_order(&self) -> PyModuleMonomialOrder {
        PyModuleMonomialOrder {
            inner: match &self.inner {
                ModuleGroebnerBasisInner::Rational(value) => value.schreyer_order(),
                ModuleGroebnerBasisInner::FiniteField(value) => value.schreyer_order(),
                ModuleGroebnerBasisInner::QuotientRational(value) => value.schreyer_order(),
                ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.schreyer_order(),
            },
        }
    }
    fn is_complete(&self) -> bool {
        true
    }
    fn initial_submodule(&self) -> PyResult<PySubmodule> {
        let inner = match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => {
                SubmoduleInner::Rational(value.initial_submodule().map_err(module_groebner_error)?)
            }
            ModuleGroebnerBasisInner::FiniteField(value) => SubmoduleInner::FiniteField(
                value.initial_submodule().map_err(module_groebner_error)?,
            ),
            ModuleGroebnerBasisInner::QuotientRational(_)
            | ModuleGroebnerBasisInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "initial quotient submodules are not yet projected",
                ));
            }
        };
        Ok(PySubmodule { inner })
    }
    fn multigraded_hilbert_numerator(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(PyDegree, Py<PyAny>)>> {
        let numerator = match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.multigraded_hilbert_numerator(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.multigraded_hilbert_numerator(),
            ModuleGroebnerBasisInner::QuotientRational(_)
            | ModuleGroebnerBasisInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "quotient-module Hilbert numerators are not yet projected",
                ));
            }
        }
        .map_err(module_groebner_error)?;
        numerator
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
        let inner = match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.hilbert_series(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.hilbert_series(),
            ModuleGroebnerBasisInner::QuotientRational(_)
            | ModuleGroebnerBasisInner::QuotientFiniteField(_) => {
                return Err(PyTypeError::new_err(
                    "quotient-module Hilbert series are not yet projected",
                ));
            }
        }
        .map_err(module_groebner_error)?;
        Ok(PyHilbertSeries { inner })
    }
    fn reduce(&self, value: &PyModuleElement) -> PyResult<PyModuleReduction> {
        let inner = match (&self.inner, &value.inner) {
            (ModuleGroebnerBasisInner::Rational(basis), ModuleElementInner::Rational(value)) => {
                ModuleReductionInner::Rational(basis.reduce(value).map_err(module_groebner_error)?)
            }
            (
                ModuleGroebnerBasisInner::FiniteField(basis),
                ModuleElementInner::FiniteField(value),
            ) => ModuleReductionInner::FiniteField(
                basis.reduce(value).map_err(module_groebner_error)?,
            ),
            (
                ModuleGroebnerBasisInner::QuotientRational(basis),
                ModuleElementInner::QuotientRational(value),
            ) => ModuleReductionInner::QuotientRational(
                basis.reduce(value).map_err(module_groebner_error)?,
            ),
            (
                ModuleGroebnerBasisInner::QuotientFiniteField(basis),
                ModuleElementInner::QuotientFiniteField(value),
            ) => ModuleReductionInner::QuotientFiniteField(
                basis.reduce(value).map_err(module_groebner_error)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "module Groebner basis and element contexts do not match",
                ));
            }
        };
        Ok(PyModuleReduction { inner })
    }
    fn normal_form(&self, value: &PyModuleElement) -> PyResult<PyModuleElement> {
        Ok(self.reduce(value)?.remainder())
    }
    fn contains(&self, value: &PyModuleElement) -> PyResult<bool> {
        Ok(self.reduce(value)?.is_member())
    }
    fn lift(&self, value: &PyModuleElement) -> PyResult<Option<PyModuleElement>> {
        Ok(self.reduce(value)?.lift())
    }
    fn syzygies(&self) -> PyResult<PySubmodule> {
        let inner = match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => {
                SubmoduleInner::Rational(value.syzygies().map_err(module_groebner_error)?)
            }
            ModuleGroebnerBasisInner::FiniteField(value) => {
                SubmoduleInner::FiniteField(value.syzygies().map_err(module_groebner_error)?)
            }
            ModuleGroebnerBasisInner::QuotientRational(value) => {
                SubmoduleInner::QuotientRational(value.syzygies().map_err(module_groebner_error)?)
            }
            ModuleGroebnerBasisInner::QuotientFiniteField(value) => {
                SubmoduleInner::QuotientFiniteField(
                    value.syzygies().map_err(module_groebner_error)?,
                )
            }
        };
        Ok(PySubmodule { inner })
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.verify(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.verify(),
            ModuleGroebnerBasisInner::QuotientRational(value) => value.verify(),
            ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.verify(),
        }
        .map_err(module_groebner_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.to_json(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.to_json(),
            ModuleGroebnerBasisInner::QuotientRational(value) => value.to_json(),
            ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_module_ring(value, "/input/generator_map/source/ring")? {
            SerializedModuleRing::PolynomialRational => ModuleGroebnerBasisInner::Rational(
                ModuleGroebnerBasis::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => ModuleGroebnerBasisInner::FiniteField(
                ModuleGroebnerBasis::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => ModuleGroebnerBasisInner::QuotientRational(
                ModuleGroebnerBasis::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                    value,
                )
                .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => {
                ModuleGroebnerBasisInner::QuotientFiniteField(
                    ModuleGroebnerBasis::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                        value,
                    )
                    .map_err(value_error)?,
                )
            }
            SerializedModuleRing::PolynomialInteger | SerializedModuleRing::PolynomialResidue => {
                return Err(PyTypeError::new_err(
                    "module Groebner bases require a polynomial or quotient ring over Q or FiniteField",
                ));
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ModuleGroebnerBasisInner::Rational(value) => value.to_string(),
            ModuleGroebnerBasisInner::FiniteField(value) => value.to_string(),
            ModuleGroebnerBasisInner::QuotientRational(value) => value.to_string(),
            ModuleGroebnerBasisInner::QuotientFiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModuleReductionInner {
    Rational(ModuleReduction<PolynomialRing<RationalField>>),
    FiniteField(ModuleReduction<PolynomialRing<FiniteField>>),
    QuotientRational(ModuleReduction<QuotientRing<PolynomialRing<RationalField>>>),
    QuotientFiniteField(ModuleReduction<QuotientRing<PolynomialRing<FiniteField>>>),
}

/// Exact quotient, remainder, and original-generator lift evidence.
#[pyclass(
    module = "noether.homalg",
    name = "ModuleReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyModuleReduction {
    inner: ModuleReductionInner,
}

#[pymethods]
impl PyModuleReduction {
    #[classattr]
    fn type_id() -> &'static str {
        ModuleReduction::<PolynomialRing<RationalField>>::TYPE_ID
    }
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }
    #[getter]
    fn basis(&self) -> PyModuleGroebnerBasis {
        PyModuleGroebnerBasis {
            inner: match &self.inner {
                ModuleReductionInner::Rational(value) => {
                    ModuleGroebnerBasisInner::Rational(value.basis().clone())
                }
                ModuleReductionInner::FiniteField(value) => {
                    ModuleGroebnerBasisInner::FiniteField(value.basis().clone())
                }
                ModuleReductionInner::QuotientRational(value) => {
                    ModuleGroebnerBasisInner::QuotientRational(value.basis().clone())
                }
                ModuleReductionInner::QuotientFiniteField(value) => {
                    ModuleGroebnerBasisInner::QuotientFiniteField(value.basis().clone())
                }
            },
        }
    }
    #[getter]
    fn value(&self) -> PyModuleElement {
        PyModuleElement {
            inner: match &self.inner {
                ModuleReductionInner::Rational(value) => {
                    ModuleElementInner::Rational(value.value().clone())
                }
                ModuleReductionInner::FiniteField(value) => {
                    ModuleElementInner::FiniteField(value.value().clone())
                }
                ModuleReductionInner::QuotientRational(value) => {
                    ModuleElementInner::QuotientRational(value.value().clone())
                }
                ModuleReductionInner::QuotientFiniteField(value) => {
                    ModuleElementInner::QuotientFiniteField(value.value().clone())
                }
            },
        }
    }
    #[getter]
    fn quotient(&self) -> PyModuleElement {
        PyModuleElement {
            inner: match &self.inner {
                ModuleReductionInner::Rational(value) => {
                    ModuleElementInner::Rational(value.quotient().clone())
                }
                ModuleReductionInner::FiniteField(value) => {
                    ModuleElementInner::FiniteField(value.quotient().clone())
                }
                ModuleReductionInner::QuotientRational(value) => {
                    ModuleElementInner::QuotientRational(value.quotient().clone())
                }
                ModuleReductionInner::QuotientFiniteField(value) => {
                    ModuleElementInner::QuotientFiniteField(value.quotient().clone())
                }
            },
        }
    }
    #[getter]
    fn original_combination(&self) -> PyModuleElement {
        PyModuleElement {
            inner: match &self.inner {
                ModuleReductionInner::Rational(value) => {
                    ModuleElementInner::Rational(value.original_combination().clone())
                }
                ModuleReductionInner::FiniteField(value) => {
                    ModuleElementInner::FiniteField(value.original_combination().clone())
                }
                ModuleReductionInner::QuotientRational(value) => {
                    ModuleElementInner::QuotientRational(value.original_combination().clone())
                }
                ModuleReductionInner::QuotientFiniteField(value) => {
                    ModuleElementInner::QuotientFiniteField(value.original_combination().clone())
                }
            },
        }
    }
    #[getter]
    fn remainder(&self) -> PyModuleElement {
        PyModuleElement {
            inner: match &self.inner {
                ModuleReductionInner::Rational(value) => {
                    ModuleElementInner::Rational(value.remainder().clone())
                }
                ModuleReductionInner::FiniteField(value) => {
                    ModuleElementInner::FiniteField(value.remainder().clone())
                }
                ModuleReductionInner::QuotientRational(value) => {
                    ModuleElementInner::QuotientRational(value.remainder().clone())
                }
                ModuleReductionInner::QuotientFiniteField(value) => {
                    ModuleElementInner::QuotientFiniteField(value.remainder().clone())
                }
            },
        }
    }
    fn is_member(&self) -> bool {
        match &self.inner {
            ModuleReductionInner::Rational(value) => value.is_member(),
            ModuleReductionInner::FiniteField(value) => value.is_member(),
            ModuleReductionInner::QuotientRational(value) => value.is_member(),
            ModuleReductionInner::QuotientFiniteField(value) => value.is_member(),
        }
    }
    fn lift(&self) -> Option<PyModuleElement> {
        match &self.inner {
            ModuleReductionInner::Rational(value) => {
                value.lift().cloned().map(|inner| PyModuleElement {
                    inner: ModuleElementInner::Rational(inner),
                })
            }
            ModuleReductionInner::FiniteField(value) => {
                value.lift().cloned().map(|inner| PyModuleElement {
                    inner: ModuleElementInner::FiniteField(inner),
                })
            }
            ModuleReductionInner::QuotientRational(value) => {
                value.lift().cloned().map(|inner| PyModuleElement {
                    inner: ModuleElementInner::QuotientRational(inner),
                })
            }
            ModuleReductionInner::QuotientFiniteField(value) => {
                value.lift().cloned().map(|inner| PyModuleElement {
                    inner: ModuleElementInner::QuotientFiniteField(inner),
                })
            }
        }
    }
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            ModuleReductionInner::Rational(value) => value.verify(),
            ModuleReductionInner::FiniteField(value) => value.verify(),
            ModuleReductionInner::QuotientRational(value) => value.verify(),
            ModuleReductionInner::QuotientFiniteField(value) => value.verify(),
        }
        .map_err(module_groebner_error)
    }
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ModuleReductionInner::Rational(value) => value.to_json(),
            ModuleReductionInner::FiniteField(value) => value.to_json(),
            ModuleReductionInner::QuotientRational(value) => value.to_json(),
            ModuleReductionInner::QuotientFiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let inner = match serialized_module_ring(value, "/basis/input/generator_map/source/ring")? {
            SerializedModuleRing::PolynomialRational => ModuleReductionInner::Rational(
                ModuleReduction::<PolynomialRing<RationalField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialFiniteField => ModuleReductionInner::FiniteField(
                ModuleReduction::<PolynomialRing<FiniteField>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientRational => ModuleReductionInner::QuotientRational(
                ModuleReduction::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::QuotientFiniteField => ModuleReductionInner::QuotientFiniteField(
                ModuleReduction::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                    .map_err(value_error)?,
            ),
            SerializedModuleRing::PolynomialInteger | SerializedModuleRing::PolynomialResidue => {
                return Err(PyTypeError::new_err(
                    "module reductions require a polynomial or quotient ring over Q or FiniteField",
                ));
            }
        };
        Ok(Self { inner })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            ModuleReductionInner::Rational(value) => value.to_string(),
            ModuleReductionInner::FiniteField(value) => value.to_string(),
            ModuleReductionInner::QuotientRational(value) => value.to_string(),
            ModuleReductionInner::QuotientFiniteField(value) => value.to_string(),
        })
    }
}

/// One positive coefficient-bearing leading module term over Z.
#[pyclass(
    module = "noether.homalg",
    name = "StrongModuleTerm",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStrongModuleTerm {
    inner: StrongModuleTerm,
}

#[pymethods]
impl PyStrongModuleTerm {
    #[classattr]
    fn type_id() -> &'static str {
        StrongModuleTerm::TYPE_ID
    }
    #[getter]
    fn monomial(&self) -> PyModuleMonomial {
        PyModuleMonomial {
            inner: self.inner.monomial().clone(),
        }
    }
    #[getter]
    fn coefficient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.coefficient())
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StrongModuleTerm::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "StrongModuleTerm(monomial={:?}, coefficient={})",
            self.inner.monomial(),
            self.inner.coefficient()
        ))
    }
}

/// A complete normalized strong module Groebner basis over Z.
#[pyclass(
    module = "noether.homalg",
    name = "StrongModuleGroebnerBasis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStrongModuleGroebnerBasis {
    inner: StrongModuleGroebnerBasis,
}

#[pymethods]
impl PyStrongModuleGroebnerBasis {
    #[classattr]
    fn type_id() -> &'static str {
        StrongModuleGroebnerBasis::TYPE_ID
    }
    #[getter]
    fn input(&self) -> PySubmodule {
        PySubmodule {
            inner: SubmoduleInner::Integer(self.inner.input().clone()),
        }
    }
    #[getter]
    fn order(&self) -> PyModuleMonomialOrder {
        PyModuleMonomialOrder {
            inner: self.inner.order().clone(),
        }
    }
    #[getter]
    fn basis(&self) -> PySubmodule {
        PySubmodule {
            inner: SubmoduleInner::Integer(self.inner.basis().clone()),
        }
    }
    #[getter]
    fn transformation(&self) -> PyModuleHomomorphism {
        PyModuleHomomorphism {
            inner: ModuleHomomorphismInner::Integer(self.inner.transformation().clone()),
        }
    }
    #[getter]
    fn leading_terms(&self) -> Vec<PyStrongModuleTerm> {
        self.inner
            .leading_terms()
            .iter()
            .cloned()
            .map(|inner| PyStrongModuleTerm { inner })
            .collect()
    }
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn initial_submodule(&self) -> PyResult<PySubmodule> {
        Ok(PySubmodule {
            inner: SubmoduleInner::Integer(
                self.inner
                    .initial_submodule()
                    .map_err(module_groebner_error)?,
            ),
        })
    }
    fn reduce(&self, value: &PyModuleElement) -> PyResult<PyStrongModuleReduction> {
        let ModuleElementInner::Integer(value) = &value.inner else {
            return Err(PyValueError::new_err(
                "strong module basis and element contexts do not match",
            ));
        };
        Ok(PyStrongModuleReduction {
            inner: self.inner.reduce(value).map_err(module_groebner_error)?,
        })
    }
    fn normal_form(&self, value: &PyModuleElement) -> PyResult<PyModuleElement> {
        Ok(self.reduce(value)?.remainder())
    }
    fn contains(&self, value: &PyModuleElement) -> PyResult<bool> {
        Ok(self.reduce(value)?.is_member())
    }
    fn lift(&self, value: &PyModuleElement) -> PyResult<Option<PyModuleElement>> {
        Ok(self.reduce(value)?.lift())
    }
    fn syzygies(&self) -> PyResult<PySubmodule> {
        Ok(PySubmodule {
            inner: SubmoduleInner::Integer(self.inner.syzygies().map_err(module_groebner_error)?),
        })
    }
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(module_groebner_error)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StrongModuleGroebnerBasis::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact quotient, remainder, and lift evidence over an integral strong
/// module basis.
#[pyclass(
    module = "noether.homalg",
    name = "StrongModuleReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStrongModuleReduction {
    inner: StrongModuleReduction,
}

#[pymethods]
impl PyStrongModuleReduction {
    #[classattr]
    fn type_id() -> &'static str {
        StrongModuleReduction::TYPE_ID
    }
    #[getter]
    fn basis(&self) -> PyStrongModuleGroebnerBasis {
        PyStrongModuleGroebnerBasis {
            inner: self.inner.basis().clone(),
        }
    }
    #[getter]
    fn value(&self) -> PyModuleElement {
        PyModuleElement {
            inner: ModuleElementInner::Integer(self.inner.value().clone()),
        }
    }
    #[getter]
    fn quotient(&self) -> PyModuleElement {
        PyModuleElement {
            inner: ModuleElementInner::Integer(self.inner.quotient().clone()),
        }
    }
    #[getter]
    fn original_combination(&self) -> PyModuleElement {
        PyModuleElement {
            inner: ModuleElementInner::Integer(self.inner.original_combination().clone()),
        }
    }
    #[getter]
    fn remainder(&self) -> PyModuleElement {
        PyModuleElement {
            inner: ModuleElementInner::Integer(self.inner.remainder().clone()),
        }
    }
    fn is_member(&self) -> bool {
        self.inner.is_member()
    }
    fn lift(&self) -> Option<PyModuleElement> {
        self.inner.lift().cloned().map(|inner| PyModuleElement {
            inner: ModuleElementInner::Integer(inner),
        })
    }
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(module_groebner_error)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StrongModuleReduction::from_json(value).map_err(value_error)?,
        })
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}
