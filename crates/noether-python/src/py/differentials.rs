// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python transport for exact Kähler and cotangent constructions.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::commalg::{
    ConormalSequenceEvidence, CotangentComplexAmplitude, CotangentComplexHypothesis,
    CotangentComplexPresentation, CotangentSpace, KahlerDifferential, KahlerDifferentialError,
    KahlerDifferentials,
};
use noether::numthy::{FiniteField, FiniteFieldElement};
use noether::{Rational, RationalField, SemanticType};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

use super::facade_scalar::{extract_rational, rational_to_python};
use super::linalg::{MatrixInner, PyMatrix, QuotientMatrixInner};
use super::module_facade::{PyModuleElement, PyPresentedModule, PyPresentedModuleHomomorphism};
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::polynomial_facade::{PyPolynomial, PyPolynomialRing};
use super::quotient::{PyQuotientElement, PyQuotientRing, QuotientElementInner, QuotientRingInner};
use super::root::PyRationalField;

fn differential_error(error: KahlerDifferentialError) -> PyErr {
    match error {
        KahlerDifferentialError::ContextMismatch(error) => super::errors::context_mismatch(error),
        KahlerDifferentialError::PointArity { .. }
        | KahlerDifferentialError::PointCoordinateOutsideField { .. } => {
            PyTypeError::new_err(error.to_string())
        }
        KahlerDifferentialError::DimensionOverflow => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn extract_rational_point(py: Python<'_>, point: Vec<Py<PyAny>>) -> PyResult<Vec<Rational>> {
    point
        .into_iter()
        .map(|coordinate| extract_rational(py, coordinate.bind(py)))
        .collect()
}

fn extract_finite_field_point(
    py: Python<'_>,
    point: Vec<Py<PyAny>>,
) -> PyResult<Vec<FiniteFieldElement>> {
    point
        .into_iter()
        .map(|coordinate| {
            coordinate
                .bind(py)
                .extract::<PyRef<'_, PyFiniteFieldElement>>()
                .map(|value| value.inner.clone())
                .map_err(Into::into)
        })
        .collect()
}

#[derive(Clone, PartialEq, Eq)]
enum KahlerDifferentialsInner {
    Rational(KahlerDifferentials<RationalField>),
    FiniteField(KahlerDifferentials<FiniteField>),
}

/// Exact module of Kähler differentials of an affine algebra over its field.
#[pyclass(
    module = "noether.commalg",
    name = "KahlerDifferentials",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyKahlerDifferentials {
    inner: KahlerDifferentialsInner,
}

impl PyKahlerDifferentials {
    pub(crate) fn from_rational(value: KahlerDifferentials<RationalField>) -> Self {
        Self {
            inner: KahlerDifferentialsInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: KahlerDifferentials<FiniteField>) -> Self {
        Self {
            inner: KahlerDifferentialsInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyKahlerDifferentials {
    #[classattr]
    fn type_id() -> &'static str {
        KahlerDifferentials::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    /// The affine algebra whose relative differentials this module represents.
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }

    #[getter]
    /// The exact finitely presented `A`-module `Omega_{A/k}`.
    fn module(&self) -> PyPresentedModule {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                PyPresentedModule::from_rational_quotient(value.module().clone())
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_quotient(value.module().clone())
            }
        }
    }

    #[getter]
    /// Replayable conormal/Jacobian evidence for this presentation.
    fn conormal_evidence(&self) -> PyConormalSequenceEvidence {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                PyConormalSequenceEvidence::from_rational(value.conormal_evidence().clone())
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                PyConormalSequenceEvidence::from_finite_field(value.conormal_evidence().clone())
            }
        }
    }

    #[getter]
    /// Number of ambient coordinate differentials `dx_i`.
    fn ambient_generator_count(&self) -> usize {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => value.ambient_generator_count(),
            KahlerDifferentialsInner::FiniteField(value) => value.ambient_generator_count(),
        }
    }

    /// Apply the universal derivation to a class in the affine algebra.
    fn differentiate(
        &self,
        py: Python<'_>,
        value: &PyQuotientElement,
    ) -> PyResult<PyKahlerDifferential> {
        match (&self.inner, &value.inner) {
            (
                KahlerDifferentialsInner::Rational(differentials),
                QuotientElementInner::Rational(value),
            ) => {
                let differentials = differentials.clone();
                let value = value.clone();
                py.detach(move || differentials.differentiate(&value))
                    .map(PyKahlerDifferential::from_rational)
                    .map_err(differential_error)
            }
            (
                KahlerDifferentialsInner::FiniteField(differentials),
                QuotientElementInner::FiniteField(value),
            ) => {
                let differentials = differentials.clone();
                let value = value.clone();
                py.detach(move || differentials.differentiate(&value))
                    .map(PyKahlerDifferential::from_finite_field)
                    .map_err(differential_error)
            }
            _ => Err(PyTypeError::new_err(
                "value and Kähler differential module use incompatible contexts",
            )),
        }
    }

    /// Differentiate an ambient polynomial, then descend to the quotient.
    fn differentiate_polynomial(
        &self,
        py: Python<'_>,
        value: &PyPolynomial,
    ) -> PyResult<PyKahlerDifferential> {
        match &self.inner {
            KahlerDifferentialsInner::Rational(differentials) => {
                let value = value.as_rational().cloned().ok_or_else(|| {
                    PyTypeError::new_err("differentiation requires a polynomial over Q")
                })?;
                let differentials = differentials.clone();
                py.detach(move || differentials.differentiate_polynomial(&value))
                    .map(PyKahlerDifferential::from_rational)
                    .map_err(differential_error)
            }
            KahlerDifferentialsInner::FiniteField(differentials) => {
                let value = value.as_finite_field().cloned().ok_or_else(|| {
                    PyTypeError::new_err(
                        "differentiation requires a polynomial over the algebra's finite field",
                    )
                })?;
                let differentials = differentials.clone();
                py.detach(move || differentials.differentiate_polynomial(&value))
                    .map(PyKahlerDifferential::from_finite_field)
                    .map_err(differential_error)
            }
        }
    }

    /// The zero differential in this module.
    fn zero(&self) -> PyKahlerDifferential {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                PyKahlerDifferential::from_rational(value.zero())
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                PyKahlerDifferential::from_finite_field(value.zero())
            }
        }
    }

    /// Construct the exact cotangent fiber at a checked field-valued point.
    fn cotangent_space_at(
        &self,
        py: Python<'_>,
        point: Vec<Py<PyAny>>,
    ) -> PyResult<PyCotangentSpace> {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                let point = extract_rational_point(py, point)?;
                let value = value.clone();
                py.detach(move || value.cotangent_space_at(point))
                    .map(PyCotangentSpace::from_rational)
                    .map_err(differential_error)
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                let point = extract_finite_field_point(py, point)?;
                let value = value.clone();
                py.detach(move || value.cotangent_space_at(point))
                    .map(PyCotangentSpace::from_finite_field)
                    .map_err(differential_error)
            }
        }
    }

    /// Construct the verified two-term cotangent complex for a complete intersection.
    fn cotangent_complex(&self, py: Python<'_>) -> PyResult<PyCotangentComplexPresentation> {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.cotangent_complex())
                    .map(PyCotangentComplexPresentation::from_rational)
                    .map_err(differential_error)
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.cotangent_complex())
                    .map(PyCotangentComplexPresentation::from_finite_field)
                    .map_err(differential_error)
            }
        }
    }

    /// Replay all context, conormal, and presentation checks.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
            KahlerDifferentialsInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
        }
        .map_err(differential_error)
    }

    /// Serialize the complete exact construction as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            KahlerDifferentialsInner::Rational(value) => value.to_json(),
            KahlerDifferentialsInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    /// Deserialize and verify an exact construction over `Q` or a finite field.
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Ok(inner) = KahlerDifferentials::<RationalField>::from_json(&value) {
                return Ok(Self::from_rational(inner));
            }
            if let Ok(inner) = KahlerDifferentials::<FiniteField>::from_json(&value) {
                return Ok(Self::from_finite_field(inner));
            }
            Err(value_error(
                "serialized Kähler differential module is not supported over Q or a finite field",
            ))
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            KahlerDifferentialsInner::Rational(value) => value.to_string(),
            KahlerDifferentialsInner::FiniteField(value) => value.to_string(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
enum KahlerDifferentialInner {
    Rational(KahlerDifferential<RationalField>),
    FiniteField(KahlerDifferential<FiniteField>),
}

/// One canonical element of an exact Kähler differential module.
#[pyclass(
    module = "noether.commalg",
    name = "KahlerDifferential",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyKahlerDifferential {
    inner: KahlerDifferentialInner,
}

impl PyKahlerDifferential {
    fn from_rational(value: KahlerDifferential<RationalField>) -> Self {
        Self {
            inner: KahlerDifferentialInner::Rational(value),
        }
    }

    fn from_finite_field(value: KahlerDifferential<FiniteField>) -> Self {
        Self {
            inner: KahlerDifferentialInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyKahlerDifferential {
    #[classattr]
    fn type_id() -> &'static str {
        KahlerDifferential::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    /// The differential module containing this element.
    fn differentials(&self) -> PyKahlerDifferentials {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => {
                PyKahlerDifferentials::from_rational(value.differentials().clone())
            }
            KahlerDifferentialInner::FiniteField(value) => {
                PyKahlerDifferentials::from_finite_field(value.differentials().clone())
            }
        }
    }

    #[getter]
    /// Canonical coordinates modulo the Jacobian relations.
    fn coordinates(&self) -> PyModuleElement {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => {
                PyModuleElement::from_rational_quotient(value.coordinates().clone())
            }
            KahlerDifferentialInner::FiniteField(value) => {
                PyModuleElement::from_finite_field_quotient(value.coordinates().clone())
            }
        }
    }

    /// Whether this differential is zero in `Omega_{A/k}`.
    fn is_zero(&self) -> bool {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => value.is_zero(),
            KahlerDifferentialInner::FiniteField(value) => value.is_zero(),
        }
    }

    /// Multiply this differential by a class in `A`.
    fn scalar_mul(&self, py: Python<'_>, scalar: &PyQuotientElement) -> PyResult<Self> {
        match (&self.inner, &scalar.inner) {
            (KahlerDifferentialInner::Rational(value), QuotientElementInner::Rational(scalar)) => {
                let value = value.clone();
                let scalar = scalar.clone();
                py.detach(move || value.scalar_mul(&scalar))
                    .map(Self::from_rational)
                    .map_err(differential_error)
            }
            (
                KahlerDifferentialInner::FiniteField(value),
                QuotientElementInner::FiniteField(scalar),
            ) => {
                let value = value.clone();
                let scalar = scalar.clone();
                py.detach(move || value.scalar_mul(&scalar))
                    .map(Self::from_finite_field)
                    .map_err(differential_error)
            }
            _ => Err(PyTypeError::new_err(
                "scalar and Kähler differential use incompatible contexts",
            )),
        }
    }

    /// Verify the containing module and the canonical coordinate representative.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
            KahlerDifferentialInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
        }
        .map_err(differential_error)
    }

    /// Serialize this differential and its exact context as typed JSON.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => value.to_json(),
            KahlerDifferentialInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    /// Deserialize and verify a differential over `Q` or a finite field.
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Ok(inner) = KahlerDifferential::<RationalField>::from_json(&value) {
                return Ok(Self::from_rational(inner));
            }
            if let Ok(inner) = KahlerDifferential::<FiniteField>::from_json(&value) {
                return Ok(Self::from_finite_field(inner));
            }
            Err(value_error(
                "serialized Kähler differential is not supported over Q or a finite field",
            ))
        })
    }

    fn __add__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        match (&self.inner, &rhs.inner) {
            (KahlerDifferentialInner::Rational(left), KahlerDifferentialInner::Rational(right)) => {
                let left = left.clone();
                let right = right.clone();
                py.detach(move || left.add(&right))
                    .map(Self::from_rational)
                    .map_err(differential_error)
            }
            (
                KahlerDifferentialInner::FiniteField(left),
                KahlerDifferentialInner::FiniteField(right),
            ) => {
                let left = left.clone();
                let right = right.clone();
                py.detach(move || left.add(&right))
                    .map(Self::from_finite_field)
                    .map_err(differential_error)
            }
            _ => Err(PyTypeError::new_err(
                "Kähler differentials use incompatible contexts",
            )),
        }
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        match (&self.inner, &rhs.inner) {
            (KahlerDifferentialInner::Rational(left), KahlerDifferentialInner::Rational(right)) => {
                let left = left.clone();
                let right = right.clone();
                py.detach(move || left.sub(&right))
                    .map(Self::from_rational)
                    .map_err(differential_error)
            }
            (
                KahlerDifferentialInner::FiniteField(left),
                KahlerDifferentialInner::FiniteField(right),
            ) => {
                let left = left.clone();
                let right = right.clone();
                py.detach(move || left.sub(&right))
                    .map(Self::from_finite_field)
                    .map_err(differential_error)
            }
            _ => Err(PyTypeError::new_err(
                "Kähler differentials use incompatible contexts",
            )),
        }
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        match &self.inner {
            KahlerDifferentialInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.neg())
                    .map(Self::from_rational)
                    .map_err(differential_error)
            }
            KahlerDifferentialInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.neg())
                    .map(Self::from_finite_field)
                    .map_err(differential_error)
            }
        }
    }

    fn __mul__(&self, py: Python<'_>, scalar: &PyQuotientElement) -> PyResult<Self> {
        self.scalar_mul(py, scalar)
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &PyQuotientElement) -> PyResult<Self> {
        self.scalar_mul(py, scalar)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "KahlerDifferential(coordinates={})",
            match &self.inner {
                KahlerDifferentialInner::Rational(value) => {
                    value.coordinates().coordinates().dimension()
                }
                KahlerDifferentialInner::FiniteField(value) => {
                    value.coordinates().coordinates().dimension()
                }
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ConormalSequenceEvidenceInner {
    Rational(ConormalSequenceEvidence<RationalField>),
    FiniteField(ConormalSequenceEvidence<FiniteField>),
}

/// Replayable conormal presentation `I/I² -> A^n` with its exact Jacobian.
#[pyclass(
    module = "noether.commalg",
    name = "ConormalSequenceEvidence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyConormalSequenceEvidence {
    inner: ConormalSequenceEvidenceInner,
}

impl PyConormalSequenceEvidence {
    pub(crate) fn from_rational(value: ConormalSequenceEvidence<RationalField>) -> Self {
        Self {
            inner: ConormalSequenceEvidenceInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: ConormalSequenceEvidence<FiniteField>) -> Self {
        Self {
            inner: ConormalSequenceEvidenceInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyConormalSequenceEvidence {
    #[classattr]
    fn type_id() -> &'static str {
        ConormalSequenceEvidence::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    /// The affine algebra `A = k[x_1,...,x_n]/I`.
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            ConormalSequenceEvidenceInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }

    #[getter]
    /// The exact chosen generators of `I`, including redundancies.
    fn chosen_generators(&self) -> Vec<PyPolynomial> {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => value
                .chosen_generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_rational)
                .collect(),
            ConormalSequenceEvidenceInner::FiniteField(value) => value
                .chosen_generators()
                .iter()
                .cloned()
                .map(PyPolynomial::from_finite_field)
                .collect(),
        }
    }

    #[getter]
    /// A presentation of `I/I^2` over `A` for the chosen generators.
    fn conormal_module(&self) -> PyPresentedModule {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => {
                PyPresentedModule::from_rational_quotient(value.conormal_module().clone())
            }
            ConormalSequenceEvidenceInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_quotient(value.conormal_module().clone())
            }
        }
    }

    #[getter]
    /// The free ambient module with basis `dx_1,...,dx_n`.
    fn ambient_differentials(&self) -> PyPresentedModule {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => {
                PyPresentedModule::from_rational_quotient(value.ambient_differentials().clone())
            }
            ConormalSequenceEvidenceInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_quotient(value.ambient_differentials().clone())
            }
        }
    }

    #[getter]
    /// The conormal map sending each equation to its exact differential.
    fn conormal_map(&self) -> PyPresentedModuleHomomorphism {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => {
                PyPresentedModuleHomomorphism::from_rational_quotient(value.conormal_map().clone())
            }
            ConormalSequenceEvidenceInner::FiniteField(value) => {
                PyPresentedModuleHomomorphism::from_finite_field_quotient(
                    value.conormal_map().clone(),
                )
            }
        }
    }

    #[getter]
    /// The exact Jacobian matrix over `A` for the chosen generators.
    fn jacobian(&self) -> PyMatrix {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => PyMatrix {
                inner: MatrixInner::Quotient(QuotientMatrixInner::Rational(
                    value.jacobian().clone(),
                )),
            },
            ConormalSequenceEvidenceInner::FiniteField(value) => PyMatrix {
                inner: MatrixInner::Quotient(QuotientMatrixInner::FiniteField(
                    value.jacobian().clone(),
                )),
            },
        }
    }

    /// Replay the ideal, `I/I^2`, Jacobian, and conormal-map calculations.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
            ConormalSequenceEvidenceInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
        }
        .map_err(differential_error)
    }

    /// Serialize the replayable evidence as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            ConormalSequenceEvidenceInner::Rational(value) => value.to_json(),
            ConormalSequenceEvidenceInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    /// Deserialize and independently verify conormal evidence.
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Ok(inner) = ConormalSequenceEvidence::<RationalField>::from_json(&value) {
                return Ok(Self::from_rational(inner));
            }
            if let Ok(inner) = ConormalSequenceEvidence::<FiniteField>::from_json(&value) {
                return Ok(Self::from_finite_field(inner));
            }
            Err(value_error(
                "serialized conormal evidence is not supported over Q or a finite field",
            ))
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ConormalSequenceEvidence(equations={})",
            match &self.inner {
                ConormalSequenceEvidenceInner::Rational(value) => {
                    value.chosen_generators().len()
                }
                ConormalSequenceEvidenceInner::FiniteField(value) => {
                    value.chosen_generators().len()
                }
            }
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CotangentSpaceInner {
    Rational(CotangentSpace<RationalField>),
    FiniteField(CotangentSpace<FiniteField>),
}

/// Exact cotangent fiber at a checked coefficient-field point.
#[pyclass(
    module = "noether.commalg",
    name = "CotangentSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCotangentSpace {
    inner: CotangentSpaceInner,
}

impl PyCotangentSpace {
    pub(crate) fn from_rational(value: CotangentSpace<RationalField>) -> Self {
        Self {
            inner: CotangentSpaceInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: CotangentSpace<FiniteField>) -> Self {
        Self {
            inner: CotangentSpaceInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyCotangentSpace {
    #[classattr]
    fn type_id() -> &'static str {
        CotangentSpace::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    /// The affine algebra whose point defines this cotangent fiber.
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            CotangentSpaceInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }

    #[getter]
    /// The exact coefficient field of the point and presentation matrix.
    fn coefficient_field(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            CotangentSpaceInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            CotangentSpaceInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.algebra().base_ring().coefficient_ring().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    /// Coordinates of the checked field-valued point.
    fn point(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value
                .point()
                .iter()
                .map(|coordinate| rational_to_python(py, coordinate))
                .collect(),
            CotangentSpaceInner::FiniteField(value) => value
                .point()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                .collect(),
        }
    }

    #[getter]
    /// The exact Jacobian `J(p)` evaluated at the point.
    fn evaluated_jacobian(&self) -> PyMatrix {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => PyMatrix {
                inner: MatrixInner::Rational(value.evaluated_jacobian().clone()),
            },
            CotangentSpaceInner::FiniteField(value) => PyMatrix {
                inner: MatrixInner::FiniteField(value.evaluated_jacobian().clone()),
            },
        }
    }

    /// Matrix `J(p)^T` presenting this fiber as a cokernel over its field.
    #[getter]
    fn presentation_matrix(&self) -> PyMatrix {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => PyMatrix {
                inner: MatrixInner::Rational(value.presentation_matrix()),
            },
            CotangentSpaceInner::FiniteField(value) => PyMatrix {
                inner: MatrixInner::FiniteField(value.presentation_matrix()),
            },
        }
    }

    #[getter]
    /// Rank of `J(p)` over the coefficient field.
    fn jacobian_rank(&self) -> usize {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value.jacobian_rank(),
            CotangentSpaceInner::FiniteField(value) => value.jacobian_rank(),
        }
    }

    #[getter]
    /// Number of coordinates in the chosen ambient affine space.
    fn ambient_coordinate_count(&self) -> usize {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value.ambient_coordinate_count(),
            CotangentSpaceInner::FiniteField(value) => value.ambient_coordinate_count(),
        }
    }

    #[getter]
    /// Dimension of `m_p/m_p^2`, computed as `n - rank(J(p))`.
    fn embedding_dimension(&self) -> usize {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value.embedding_dimension(),
            CotangentSpaceInner::FiniteField(value) => value.embedding_dimension(),
        }
    }

    /// Test an expected embedding dimension without making a smoothness claim.
    fn has_embedding_dimension(&self, expected: usize) -> bool {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value.has_embedding_dimension(expected),
            CotangentSpaceInner::FiniteField(value) => value.has_embedding_dimension(expected),
        }
    }

    /// Return the ambient polynomials `x_i - p_i` generating the point ideal.
    fn maximal_ideal_generators(&self, py: Python<'_>) -> PyResult<Vec<PyPolynomial>> {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.maximal_ideal_generators())
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_rational)
                            .collect()
                    })
                    .map_err(value_error)
            }
            CotangentSpaceInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.maximal_ideal_generators())
                    .map(|values| {
                        values
                            .into_iter()
                            .map(PyPolynomial::from_finite_field)
                            .collect()
                    })
                    .map_err(value_error)
            }
        }
    }

    /// Replay point membership, evaluation, rank, and dimension checks.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
            CotangentSpaceInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
        }
        .map_err(differential_error)
    }

    /// Serialize the exact point, Jacobian, and dimension evidence as typed JSON.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CotangentSpaceInner::Rational(value) => value.to_json(),
            CotangentSpaceInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    /// Deserialize and verify an exact cotangent fiber.
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Ok(inner) = CotangentSpace::<RationalField>::from_json(&value) {
                return Ok(Self::from_rational(inner));
            }
            if let Ok(inner) = CotangentSpace::<FiniteField>::from_json(&value) {
                return Ok(Self::from_finite_field(inner));
            }
            Err(value_error(
                "serialized cotangent space is not supported over Q or a finite field",
            ))
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CotangentSpace(ambient_coordinates={}, embedding_dimension={})",
            self.ambient_coordinate_count(),
            self.embedding_dimension()
        ))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CotangentComplexPresentationInner {
    Rational(CotangentComplexPresentation<RationalField>),
    FiniteField(CotangentComplexPresentation<FiniteField>),
}

/// Honest two-term cotangent complex in the proved complete-intersection lane.
#[pyclass(
    module = "noether.commalg",
    name = "CotangentComplexPresentation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCotangentComplexPresentation {
    inner: CotangentComplexPresentationInner,
}

impl PyCotangentComplexPresentation {
    pub(crate) fn from_rational(value: CotangentComplexPresentation<RationalField>) -> Self {
        Self {
            inner: CotangentComplexPresentationInner::Rational(value),
        }
    }

    pub(crate) fn from_finite_field(value: CotangentComplexPresentation<FiniteField>) -> Self {
        Self {
            inner: CotangentComplexPresentationInner::FiniteField(value),
        }
    }
}

#[pymethods]
impl PyCotangentComplexPresentation {
    #[classattr]
    fn type_id() -> &'static str {
        CotangentComplexPresentation::<RationalField>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    /// The complete-intersection algebra represented by this two-term complex.
    fn algebra(&self) -> PyQuotientRing {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                PyQuotientRing::from_rational(value.algebra().clone())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                PyQuotientRing::from_finite_field(value.algebra().clone())
            }
        }
    }

    #[getter]
    /// The degree `-1` term `I/I^2`.
    fn conormal_module(&self) -> PyPresentedModule {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                PyPresentedModule::from_rational_quotient(value.conormal_module().clone())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_quotient(value.conormal_module().clone())
            }
        }
    }

    #[getter]
    /// The degree `0` term `Omega_{P/k} tensor_P A`.
    fn ambient_differentials(&self) -> PyPresentedModule {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                PyPresentedModule::from_rational_quotient(value.ambient_differentials().clone())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                PyPresentedModule::from_finite_field_quotient(value.ambient_differentials().clone())
            }
        }
    }

    #[getter]
    /// The Jacobian/conormal differential from degree `-1` to degree `0`.
    fn differential(&self) -> PyPresentedModuleHomomorphism {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                PyPresentedModuleHomomorphism::from_rational_quotient(value.differential().clone())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                PyPresentedModuleHomomorphism::from_finite_field_quotient(
                    value.differential().clone(),
                )
            }
        }
    }

    #[getter]
    /// Proven codimension of the quotient in its ambient polynomial algebra.
    fn codimension(&self) -> usize {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => value.codimension(),
            CotangentComplexPresentationInner::FiniteField(value) => value.codimension(),
        }
    }

    #[getter]
    /// Sealed amplitude `[-1, 0]`, issued only by the verified constructor.
    fn amplitude(&self) -> PyCotangentComplexAmplitude {
        let inner = match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => value.amplitude(),
            CotangentComplexPresentationInner::FiniteField(value) => value.amplitude(),
        };
        PyCotangentComplexAmplitude { inner }
    }

    #[getter]
    /// Sealed complete-intersection hypothesis and exact dimension calculation.
    fn hypothesis(&self) -> PyCotangentComplexHypothesis {
        let inner = match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => value.hypothesis(),
            CotangentComplexPresentationInner::FiniteField(value) => value.hypothesis(),
        };
        PyCotangentComplexHypothesis { inner }
    }

    #[getter]
    /// Replayable conormal sequence and exact Jacobian evidence.
    fn evidence(&self) -> PyConormalSequenceEvidence {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                PyConormalSequenceEvidence::from_rational(value.evidence().clone())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                PyConormalSequenceEvidence::from_finite_field(value.evidence().clone())
            }
        }
    }

    /// Replay the conormal evidence and complete-intersection proof.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
            CotangentComplexPresentationInner::FiniteField(value) => {
                let value = value.clone();
                py.detach(move || value.verify())
            }
        }
        .map_err(differential_error)
    }

    /// Serialize the verified presentation as canonical typed JSON.
    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            CotangentComplexPresentationInner::Rational(value) => value.to_json(),
            CotangentComplexPresentationInner::FiniteField(value) => value.to_json(),
        }
        .map_err(value_error)
    }

    #[classmethod]
    /// Deserialize and re-prove the complete-intersection presentation.
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            if let Ok(inner) = CotangentComplexPresentation::<RationalField>::from_json(&value) {
                return Ok(Self::from_rational(inner));
            }
            if let Ok(inner) = CotangentComplexPresentation::<FiniteField>::from_json(&value) {
                return Ok(Self::from_finite_field(inner));
            }
            Err(value_error(
                "serialized cotangent-complex presentation is not supported over Q or a finite field",
            ))
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CotangentComplexPresentation(codimension={}, amplitude=[-1, 0])",
            self.codimension()
        ))
    }
}

/// Sealed amplitude issued only by a verified cotangent-complex presentation.
#[pyclass(
    module = "noether.commalg",
    name = "CotangentComplexAmplitude",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCotangentComplexAmplitude {
    inner: CotangentComplexAmplitude,
}

#[pymethods]
impl PyCotangentComplexAmplitude {
    #[getter]
    /// Lowest nonzero homological degree, always `-1` in this lane.
    fn lowest_degree(&self) -> i8 {
        self.inner.lowest_degree()
    }

    #[getter]
    /// Highest nonzero homological degree, always `0` in this lane.
    fn highest_degree(&self) -> i8 {
        self.inner.highest_degree()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CotangentComplexAmplitude({}, {})",
            self.inner.lowest_degree(),
            self.inner.highest_degree()
        ))
    }
}

/// Sealed exact hypothesis issued only after replaying a CI proof.
#[pyclass(
    module = "noether.commalg",
    name = "CotangentComplexHypothesis",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCotangentComplexHypothesis {
    inner: CotangentComplexHypothesis,
}

#[pymethods]
impl PyCotangentComplexHypothesis {
    #[getter]
    /// Krull dimension of the ambient polynomial algebra.
    fn ambient_dimension(&self) -> usize {
        self.inner.ambient_dimension()
    }

    #[getter]
    /// Proven Krull dimension of the quotient algebra.
    fn quotient_dimension(&self) -> usize {
        self.inner.quotient_dimension()
    }

    #[getter]
    /// Exact difference between ambient and quotient dimensions.
    fn codimension(&self) -> usize {
        self.inner.codimension()
    }

    #[getter]
    /// Number of equations in the verified regular-sequence presentation.
    fn equation_count(&self) -> usize {
        self.inner.equation_count()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CotangentComplexHypothesis(ambient_dimension={}, quotient_dimension={}, codimension={}, equation_count={})",
            self.inner.ambient_dimension(),
            self.inner.quotient_dimension(),
            self.inner.codimension(),
            self.inner.equation_count()
        ))
    }
}

pub(crate) fn kahler_differentials_from_polynomial_ring(
    py: Python<'_>,
    ring: &PyPolynomialRing,
) -> PyResult<PyKahlerDifferentials> {
    match &ring.inner {
        super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.kahler_differentials())
                .map(PyKahlerDifferentials::from_rational)
                .map_err(differential_error)
        }
        super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.kahler_differentials())
                .map(PyKahlerDifferentials::from_finite_field)
                .map_err(differential_error)
        }
        _ => Err(PyTypeError::new_err(
            "Kähler differentials currently require a polynomial ring over Q or FiniteField",
        )),
    }
}

pub(crate) fn cotangent_space_from_polynomial_ring(
    py: Python<'_>,
    ring: &PyPolynomialRing,
    point: Vec<Py<PyAny>>,
) -> PyResult<PyCotangentSpace> {
    match &ring.inner {
        super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
            let point = extract_rational_point(py, point)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_space_at(point))
                .map(PyCotangentSpace::from_rational)
                .map_err(differential_error)
        }
        super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
            let point = extract_finite_field_point(py, point)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_space_at(point))
                .map(PyCotangentSpace::from_finite_field)
                .map_err(differential_error)
        }
        _ => Err(PyTypeError::new_err(
            "cotangent spaces currently require a polynomial ring over Q or FiniteField",
        )),
    }
}

fn rational_generators(
    py: Python<'_>,
    generators: Vec<Py<PyPolynomial>>,
) -> PyResult<Vec<noether::commalg::Polynomial<RationalField>>> {
    generators
        .into_iter()
        .map(|generator| {
            generator
                .borrow(py)
                .as_rational()
                .cloned()
                .ok_or_else(|| PyTypeError::new_err("chosen equation must be a polynomial over Q"))
        })
        .collect()
}

fn finite_field_generators(
    py: Python<'_>,
    generators: Vec<Py<PyPolynomial>>,
) -> PyResult<Vec<noether::commalg::Polynomial<FiniteField>>> {
    generators
        .into_iter()
        .map(|generator| {
            generator
                .borrow(py)
                .as_finite_field()
                .cloned()
                .ok_or_else(|| {
                    PyTypeError::new_err(
                        "chosen equation must be a polynomial over the quotient's finite field",
                    )
                })
        })
        .collect()
}

pub(crate) fn kahler_differentials_from_quotient_ring(
    py: Python<'_>,
    ring: &PyQuotientRing,
) -> PyResult<PyKahlerDifferentials> {
    match &ring.inner {
        QuotientRingInner::Rational(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.kahler_differentials())
                .map(PyKahlerDifferentials::from_rational)
                .map_err(differential_error)
        }
        QuotientRingInner::FiniteField(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.kahler_differentials())
                .map(PyKahlerDifferentials::from_finite_field)
                .map_err(differential_error)
        }
    }
}

pub(crate) fn cotangent_space_from_quotient_ring(
    py: Python<'_>,
    ring: &PyQuotientRing,
    point: Vec<Py<PyAny>>,
) -> PyResult<PyCotangentSpace> {
    match &ring.inner {
        QuotientRingInner::Rational(ring) => {
            let point = extract_rational_point(py, point)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_space_at(point))
                .map(PyCotangentSpace::from_rational)
                .map_err(differential_error)
        }
        QuotientRingInner::FiniteField(ring) => {
            let point = extract_finite_field_point(py, point)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_space_at(point))
                .map(PyCotangentSpace::from_finite_field)
                .map_err(differential_error)
        }
    }
}

pub(crate) fn cotangent_complex_from_quotient_ring(
    py: Python<'_>,
    ring: &PyQuotientRing,
) -> PyResult<PyCotangentComplexPresentation> {
    match &ring.inner {
        QuotientRingInner::Rational(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.cotangent_complex())
                .map(PyCotangentComplexPresentation::from_rational)
                .map_err(differential_error)
        }
        QuotientRingInner::FiniteField(ring) => {
            let ring = ring.clone();
            py.detach(move || ring.cotangent_complex())
                .map(PyCotangentComplexPresentation::from_finite_field)
                .map_err(differential_error)
        }
    }
}

pub(crate) fn cotangent_complex_from_generators(
    py: Python<'_>,
    ring: &PyQuotientRing,
    generators: Vec<Py<PyPolynomial>>,
) -> PyResult<PyCotangentComplexPresentation> {
    match &ring.inner {
        QuotientRingInner::Rational(ring) => {
            let generators = rational_generators(py, generators)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_complex_from_generators(&generators))
                .map(PyCotangentComplexPresentation::from_rational)
                .map_err(differential_error)
        }
        QuotientRingInner::FiniteField(ring) => {
            let generators = finite_field_generators(py, generators)?;
            let ring = ring.clone();
            py.detach(move || ring.cotangent_complex_from_generators(&generators))
                .map(PyCotangentComplexPresentation::from_finite_field)
                .map_err(differential_error)
        }
    }
}

pub(crate) fn conormal_sequence_from_generators(
    py: Python<'_>,
    ring: &PyQuotientRing,
    generators: Vec<Py<PyPolynomial>>,
) -> PyResult<PyConormalSequenceEvidence> {
    match &ring.inner {
        QuotientRingInner::Rational(ring) => {
            let generators = rational_generators(py, generators)?;
            let ring = ring.clone();
            py.detach(move || ring.conormal_sequence_from_generators(&generators))
                .map(PyConormalSequenceEvidence::from_rational)
                .map_err(differential_error)
        }
        QuotientRingInner::FiniteField(ring) => {
            let generators = finite_field_generators(py, generators)?;
            let ring = ring.clone();
            py.detach(move || ring.conormal_sequence_from_generators(&generators))
                .map(PyConormalSequenceEvidence::from_finite_field)
                .map_err(differential_error)
        }
    }
}
