// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public `noether::linalg` facade.
//!
//! These wrappers own facade values directly. They translate Python scalars,
//! exceptions, and protocols, but contain no linear-algebra algorithms or alternate
//! mathematical identities.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::commalg::{
    Polynomial, PolynomialRing, PowerSeries, PowerSeriesRing, QuotientElement, QuotientRing,
    RationalFunction, RationalFunctionField, ResidueClass, ResidueRing,
};
use noether::linalg::{
    ClosestLatticeVectors as RustClosestLatticeVectors,
    CongruenceInconsistencyCertificate as RustCongruenceInconsistencyCertificate,
    CongruenceSolutionModule as RustCongruenceSolutionModule,
    CongruenceSystemResult as RustCongruenceSystemResult,
    FrobeniusSemilinearMap as RustFrobeniusSemilinearMap, IndexError,
    IntegerInconsistencyCertificate as RustIntegerInconsistencyCertificate,
    IntegerLinearSystemResult as RustIntegerLinearSystemResult,
    IntegerSolutionModule as RustIntegerSolutionModule,
    IntegralLineAutomorphismGroup as RustIntegralLineAutomorphismGroup,
    IntegralLineConfiguration as RustIntegralLineConfiguration,
    IntegralLineEquivalence as RustIntegralLineEquivalence,
    IntegralLineOrbitAction as RustIntegralLineOrbitAction,
    IntegralLineSubgroup as RustIntegralLineSubgroup, Lattice as RustLattice,
    LatticeEnumeration as RustLatticeEnumeration, LatticeError, LinalgError,
    LineConfigurationError, LinearOperator, Matrix, MinimumCompletion as RustMinimumCompletion,
    MinimumCompletionCertificate as RustMinimumCompletionCertificate,
    MinimumNonrealizabilityCertificate as RustMinimumNonrealizabilityCertificate,
    MinimumRealization as RustMinimumRealization,
    MinimumRealizationDiagnostic as RustMinimumRealizationDiagnostic,
    MinimumVectorConfiguration as RustMinimumVectorConfiguration,
    PLUQFactorization as RustPLUQFactorization, PadicSmithInvariants as RustPadicSmithInvariants,
    SemilinearMapError, SmithNormalForm as RustSmithNormalForm,
    SmithNormalFormWithTransforms as RustSmithNormalFormWithTransforms,
    SmithVerification as RustSmithVerification, SolutionSpace as RustSolutionSpace,
    SparseMatrix as RustSparseMatrix, Vector,
};
use noether::numthy::{
    FiniteField, FiniteFieldElement, NumberField, NumberFieldOrder, PadicElement, PadicField,
    PadicLocalField, PadicLocalFieldElement, PadicLocalFieldOrder, PadicRing, QadicElement,
    QadicField, QadicRing,
};
use noether::{
    CoefficientMappingError, CommutativeRing, ComplexBall, ComplexBallField, ContextMismatchError,
    Field, Integer, IntegerRing, Rational, RationalField, RealBall, RealBallField, Ring,
    SemanticType, UnsupportedOperationError, Q, Z,
};
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyList, PyTuple, PyType};

use super::ball::{PyComplexBall, PyComplexBallField, PyRealBall, PyRealBallField};
use super::commalg::{PyResidueClass, PyResidueRing};
use super::facade_scalar::{
    extract_integer, extract_rational, integer_to_python, rational_to_python,
};
use super::groups::{PyPermGroup, PyPermutation};
use super::number_field::{
    extract_number_field_element, extract_number_field_order_element, PyNumberField,
    PyNumberFieldElement, PyNumberFieldOrder,
};
use super::numthy::{
    PyFiniteField, PyFiniteFieldElement, PyPadicElement, PyPadicField, PyPadicLocalField,
    PyPadicLocalFieldElement, PyPadicLocalFieldOrder, PyPadicRing, PyQadicElement, PyQadicField,
    PyQadicRing,
};
use super::polynomial_facade::{PyPolynomial, PyPolynomialRing};
use super::quotient::{PyQuotientElement, PyQuotientRing, QuotientElementInner, QuotientRingInner};
use super::rational_function::{PyRationalFunction, PyRationalFunctionField};
use super::root::{PyIntegerRing, PyRationalField};
use super::series::{PowerSeriesRingInner, PyPowerSeries, PyPowerSeriesRing, SeriesInner};
use super::validity::PyMathematicalValidity;

fn linalg_error(error: LinalgError) -> PyErr {
    match error {
        LinalgError::ShapeTooLarge { .. } => PyOverflowError::new_err(error.to_string()),
        LinalgError::ContextMismatch(error) => super::errors::context_mismatch(error),
        LinalgError::Verification(error) => super::errors::verification(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn index_error(error: IndexError) -> PyErr {
    PyIndexError::new_err(error.to_string())
}

fn lattice_error(error: LatticeError) -> PyErr {
    match error {
        LatticeError::SearchRangeExceeded => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn semilinear_error(error: SemilinearMapError) -> PyErr {
    match error {
        SemilinearMapError::LinearAlgebra(error) => linalg_error(error),
        SemilinearMapError::ExponentOverflow { .. }
        | SemilinearMapError::ExpandedShapeTooLarge { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        SemilinearMapError::CoefficientMapping(CoefficientMappingError::ContextMismatch(error)) => {
            super::errors::context_mismatch(error)
        }
        SemilinearMapError::CoefficientMapping(_) | SemilinearMapError::PowerBasis(_) => {
            PyValueError::new_err(error.to_string())
        }
    }
}

fn line_configuration_error(error: LineConfigurationError) -> PyErr {
    match error {
        LineConfigurationError::AmbientRankTooLarge { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn extract_integer_rows(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<noether::Integer>>> {
    let rows: Vec<Vec<Bound<'_, PyAny>>> = value.extract()?;
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| extract_integer(&entry))
                .collect()
        })
        .collect()
}

fn extract_integer_row(value: &Bound<'_, PyAny>) -> PyResult<Vec<noether::Integer>> {
    let entries: Vec<Bound<'_, PyAny>> = value.extract()?;
    entries
        .into_iter()
        .map(|entry| extract_integer(&entry))
        .collect()
}

fn integer_rows_to_python(py: Python<'_>, rows: &[Vec<noether::Integer>]) -> PyResult<Py<PyTuple>> {
    let rows = rows
        .iter()
        .map(|row| {
            let entries = row
                .iter()
                .map(|entry| integer_to_python(py, entry))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, entries)?.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, rows)?.unbind())
}

fn integer_row_to_python(py: Python<'_>, row: &[noether::Integer]) -> PyResult<Py<PyTuple>> {
    let entries = row
        .iter()
        .map(|entry| integer_to_python(py, entry))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, entries)?.unbind())
}

/// A checked finite set of primitive integral lines in `Z^N`.
#[pyclass(
    module = "noether.linalg",
    name = "IntegralLineConfiguration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegralLineConfiguration {
    inner: RustIntegralLineConfiguration,
}

impl PyIntegralLineConfiguration {
    fn from_core(inner: RustIntegralLineConfiguration) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIntegralLineConfiguration {
    /// Construct primitive lines from arbitrary-precision integer rows.
    #[new]
    fn new(ambient_rank: usize, vectors: &Bound<'_, PyAny>) -> PyResult<Self> {
        RustIntegralLineConfiguration::new(ambient_rank, extract_integer_rows(vectors)?)
            .map(Self::from_core)
            .map_err(line_configuration_error)
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegralLineConfiguration::TYPE_ID
    }

    /// Ambient lattice rank.
    #[getter]
    fn ambient_rank(&self) -> usize {
        self.inner.ambient_rank()
    }

    /// Number of unoriented lines.
    #[getter]
    fn line_count(&self) -> usize {
        self.inner.line_count()
    }

    /// Canonically signed, sorted primitive vector representatives.
    #[getter]
    fn vectors(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        integer_rows_to_python(py, self.inner.vectors())
    }

    /// Rank of the integral vectors over the rationals.
    #[getter]
    fn span_rank(&self) -> usize {
        self.inner.span_rank()
    }

    /// Whether the lines span the ambient rational vector space.
    #[getter]
    fn is_spanning(&self) -> bool {
        self.inner.is_spanning()
    }

    /// Dimension of the span of the rank-one forms `v v^t`.
    #[getter]
    fn perfection_rank(&self) -> usize {
        self.inner.perfection_rank()
    }

    /// Whether the rank-one forms are linearly independent.
    #[getter]
    fn is_rank_one_independent(&self) -> bool {
        self.inner.is_rank_one_independent()
    }

    /// Matrix of upper-triangular coordinates of the rank-one forms.
    #[getter]
    fn rank_one_matrix(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.rank_one_matrix()),
        }
    }

    /// Exact barycenter form `sum_v v v^t`.
    #[getter]
    fn barycenter_form(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.barycenter_form()),
        }
    }

    /// Permutation induced on the unoriented lines by a preserving matrix.
    fn induced_permutation(&self, transformation: &PyMatrix) -> PyResult<PyPermutation> {
        let MatrixInner::Integer(transformation) = &transformation.inner else {
            return Err(PyTypeError::new_err(
                "line transformation requires a Matrix[IntegerRing]",
            ));
        };
        Ok(PyPermutation {
            inner: self
                .inner
                .induced_permutation(transformation)
                .map_err(line_configuration_error)?,
        })
    }

    /// Faithful permutation induced on both signs of every line.
    fn induced_signed_permutation(&self, transformation: &PyMatrix) -> PyResult<PyPermutation> {
        let MatrixInner::Integer(transformation) = &transformation.inner else {
            return Err(PyTypeError::new_err(
                "line transformation requires a Matrix[IntegerRing]",
            ));
        };
        Ok(PyPermutation {
            inner: self
                .inner
                .induced_signed_permutation(transformation)
                .map_err(line_configuration_error)?,
        })
    }

    /// Orientation character on the exact rank-one span.
    fn orientation_sign(&self, transformation: &PyMatrix) -> PyResult<i8> {
        let MatrixInner::Integer(transformation) = &transformation.inner else {
            return Err(PyTypeError::new_err(
                "line transformation requires a Matrix[IntegerRing]",
            ));
        };
        self.inner
            .orientation_sign(transformation)
            .map_err(line_configuration_error)
    }

    /// Exact finite integral stabilizer of a spanning configuration.
    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyIntegralLineAutomorphismGroup> {
        py.detach(|| {
            Ok(PyIntegralLineAutomorphismGroup {
                inner: self
                    .inner
                    .automorphism_group()
                    .map_err(line_configuration_error)?,
            })
        })
    }

    /// Find a verified equivalence, optionally reusing the target stabilizer.
    #[pyo3(signature = (target, *, target_automorphisms=None))]
    fn equivalence_to(
        &self,
        py: Python<'_>,
        target: &Self,
        target_automorphisms: Option<&PyIntegralLineAutomorphismGroup>,
    ) -> PyResult<Option<PyIntegralLineEquivalence>> {
        py.detach(|| {
            let result = if let Some(automorphisms) = target_automorphisms {
                self.inner
                    .equivalence_to_with_target_automorphisms(&target.inner, &automorphisms.inner)
            } else {
                self.inner.equivalence_to(&target.inner)
            };
            result
                .map(|result| result.map(|inner| PyIntegralLineEquivalence { inner }))
                .map_err(line_configuration_error)
        })
    }

    /// Certify realization, nonrealizability, or an inconclusive diagnostic.
    #[pyo3(signature = (*, max_iterations=50))]
    fn realize_minimum(
        &self,
        py: Python<'_>,
        max_iterations: usize,
    ) -> PyResult<PyMinimumRealization> {
        py.detach(|| {
            Ok(PyMinimumRealization {
                inner: self
                    .inner
                    .realize_minimum(max_iterations)
                    .map_err(line_configuration_error)?,
            })
        })
    }

    /// Complete to a minimum shell without increasing the perfection rank.
    #[pyo3(signature = (*, max_iterations=50))]
    fn complete_minimum(
        &self,
        py: Python<'_>,
        max_iterations: usize,
    ) -> PyResult<PyMinimumCompletion> {
        py.detach(|| {
            Ok(PyMinimumCompletion {
                inner: self
                    .inner
                    .complete_minimum(max_iterations)
                    .map_err(line_configuration_error)?,
            })
        })
    }

    /// Apply a checked unimodular matrix to all vector representatives.
    fn transform(&self, transformation: &PyMatrix) -> PyResult<Self> {
        let MatrixInner::Integer(transformation) = &transformation.inner else {
            return Err(PyTypeError::new_err(
                "line transformation requires a Matrix[IntegerRing]",
            ));
        };
        self.inner
            .transform(transformation)
            .map(Self::from_core)
            .map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustIntegralLineConfiguration::TYPE_ID)?;
        summary.set_item("ambient_rank", self.inner.ambient_rank())?;
        summary.set_item("line_count", self.inner.line_count())?;
        summary.set_item("span_rank", self.inner.span_rank())?;
        summary.set_item("perfection_rank", self.inner.perfection_rank())?;
        Ok(summary.unbind())
    }
}

/// Exact rational evidence that a line configuration is not a minimum shell.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumNonrealizabilityCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumNonrealizabilityCertificate {
    inner: RustMinimumNonrealizabilityCertificate,
}

#[pymethods]
impl PyMinimumNonrealizabilityCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumNonrealizabilityCertificate::TYPE_ID
    }

    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn obstacle_count(&self) -> usize {
        self.inner.obstacle_count()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Diagnostic information from an inconclusive minimum-realization computation.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumRealizationDiagnostic",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumRealizationDiagnostic {
    inner: RustMinimumRealizationDiagnostic,
}

#[pymethods]
impl PyMinimumRealizationDiagnostic {
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumRealizationDiagnostic::TYPE_ID
    }

    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    #[getter]
    fn obstacle_count(&self) -> usize {
        self.inner.obstacle_count()
    }

    #[getter]
    fn reason(&self) -> &str {
        self.inner.reason()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Three-way certified result of minimum-vector realization.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumRealization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumRealization {
    inner: RustMinimumRealization,
}

#[pymethods]
impl PyMinimumRealization {
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumRealization::TYPE_ID
    }

    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status()
    }

    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    #[getter]
    fn realization(&self) -> Option<PyMinimumVectorConfiguration> {
        self.inner
            .realization()
            .cloned()
            .map(|inner| PyMinimumVectorConfiguration { inner })
    }

    #[getter]
    fn certificate(&self) -> Option<PyMinimumNonrealizabilityCertificate> {
        self.inner
            .certificate()
            .cloned()
            .map(|inner| PyMinimumNonrealizabilityCertificate { inner })
    }

    #[getter]
    fn diagnostic(&self) -> Option<PyMinimumRealizationDiagnostic> {
        self.inner
            .diagnostic()
            .cloned()
            .map(|inner| PyMinimumRealizationDiagnostic { inner })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact evidence that no fixed-rank minimum completion exists.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumCompletionCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumCompletionCertificate {
    inner: RustMinimumCompletionCertificate,
}

#[pymethods]
impl PyMinimumCompletionCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumCompletionCertificate::TYPE_ID
    }

    #[getter]
    fn seed(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.seed().clone())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn obstacle_count(&self) -> usize {
        self.inner.obstacle_count()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Three-way certified result of fixed-rank minimum completion.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumCompletion",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumCompletion {
    inner: RustMinimumCompletion,
}

#[pymethods]
impl PyMinimumCompletion {
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumCompletion::TYPE_ID
    }

    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status()
    }

    #[getter]
    fn seed(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.seed().clone())
    }

    #[getter]
    fn realization(&self) -> Option<PyMinimumVectorConfiguration> {
        self.inner
            .realization()
            .cloned()
            .map(|inner| PyMinimumVectorConfiguration { inner })
    }

    #[getter]
    fn certificate(&self) -> Option<PyMinimumCompletionCertificate> {
        self.inner
            .certificate()
            .cloned()
            .map(|inner| PyMinimumCompletionCertificate { inner })
    }

    #[getter]
    fn diagnostic(&self) -> Option<PyMinimumRealizationDiagnostic> {
        self.inner
            .diagnostic()
            .cloned()
            .map(|inner| PyMinimumRealizationDiagnostic { inner })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Verified unimodular equivalence between two line configurations.
#[pyclass(
    module = "noether.linalg",
    name = "IntegralLineEquivalence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegralLineEquivalence {
    inner: RustIntegralLineEquivalence,
}

#[pymethods]
impl PyIntegralLineEquivalence {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegralLineEquivalence::TYPE_ID
    }

    /// Source configuration.
    #[getter]
    fn source(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.source().clone())
    }

    /// Target configuration.
    #[getter]
    fn target(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.target().clone())
    }

    /// Unimodular matrix carrying source to target.
    #[getter]
    fn transformation(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.transformation().clone()),
        }
    }

    /// Induced bijection of source-line and target-line indices.
    #[getter]
    fn line_permutation(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.line_permutation().clone(),
        }
    }

    /// Recheck all equivalence evidence exactly.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact finite stabilizer of a spanning integral-line configuration.
#[pyclass(
    module = "noether.linalg",
    name = "IntegralLineAutomorphismGroup",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegralLineAutomorphismGroup {
    inner: RustIntegralLineAutomorphismGroup,
}

#[pymethods]
impl PyIntegralLineAutomorphismGroup {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegralLineAutomorphismGroup::TYPE_ID
    }

    /// Stabilized integral-line configuration.
    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    /// Unimodular matrix generators acting on column vectors.
    #[getter]
    fn matrix_generators(&self) -> Vec<PyMatrix> {
        self.inner
            .matrix_generators()
            .iter()
            .cloned()
            .map(|matrix| PyMatrix {
                inner: MatrixInner::Integer(matrix),
            })
            .collect()
    }

    /// Induced permutation group on unoriented lines.
    #[getter]
    fn line_action(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.line_action().clone(),
        }
    }

    /// Faithful action on the two signed representatives of each line.
    #[getter]
    fn signed_action(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.signed_action().clone(),
        }
    }

    /// Lift a signed-action element to its verified unimodular matrix.
    fn lift_signed_action(&self, action: &PyPermutation) -> PyResult<PyMatrix> {
        Ok(PyMatrix {
            inner: MatrixInner::Integer(
                self.inner
                    .lift_signed_action(&action.inner)
                    .map_err(line_configuration_error)?,
            ),
        })
    }

    /// Apply a signed stabilizer action to one external primitive line.
    fn apply_signed_action_to_external_line(
        &self,
        py: Python<'_>,
        action: &PyPermutation,
        line: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let image = self
            .inner
            .apply_signed_action_to_external_line(&action.inner, extract_integer_row(line)?)
            .map_err(line_configuration_error)?;
        let entries = image
            .iter()
            .map(|entry| integer_to_python(py, entry))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.unbind())
    }

    /// Build the simultaneous exact action on tagged external-line orbits.
    fn action_on_line_orbits(
        &self,
        seeds: &Bound<'_, PyAny>,
    ) -> PyResult<PyIntegralLineOrbitAction> {
        Ok(PyIntegralLineOrbitAction {
            inner: self
                .inner
                .action_on_line_orbits(extract_integer_rows(seeds)?)
                .map_err(line_configuration_error)?,
        })
    }

    /// Exact subgroup stabilizing one unoriented configuration line.
    fn line_stabilizer(&self, line_index: usize) -> PyResult<PyIntegralLineSubgroup> {
        Ok(PyIntegralLineSubgroup {
            inner: self
                .inner
                .line_stabilizer(line_index)
                .map_err(line_configuration_error)?,
        })
    }

    /// Exact group order, including the kernel of the unoriented line action.
    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.order())
    }

    /// Orientation character on the matrix generators.
    #[getter]
    fn orientation_signs(&self) -> Vec<i8> {
        self.inner.orientation_signs().to_vec()
    }

    /// Whether the stabilizer contains an orientation-reversing element.
    #[getter]
    fn has_orientation_reversing(&self) -> bool {
        self.inner.has_orientation_reversing()
    }

    /// Recheck all generator actions and the exact group order.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact finite action of a line-configuration stabilizer on external lines.
#[pyclass(
    module = "noether.linalg",
    name = "IntegralLineOrbitAction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegralLineOrbitAction {
    inner: RustIntegralLineOrbitAction,
}

#[pymethods]
impl PyIntegralLineOrbitAction {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegralLineOrbitAction::TYPE_ID
    }

    /// Stabilizer inducing the action.
    #[getter]
    fn automorphisms(&self) -> PyIntegralLineAutomorphismGroup {
        PyIntegralLineAutomorphismGroup {
            inner: self.inner.automorphisms().clone(),
        }
    }

    /// Canonical tagged seed lines.
    #[getter]
    fn seeds(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        integer_rows_to_python(py, self.inner.seeds())
    }

    /// Flattened line vector at every action point.
    #[getter]
    fn lines(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        integer_rows_to_python(py, self.inner.lines())
    }

    /// Point occupied by each tagged seed.
    #[getter]
    fn seed_points(&self) -> Vec<noether::groups::Point> {
        self.inner.seed_points().to_vec()
    }

    /// Half-open point range belonging to each tagged orbit.
    #[getter]
    fn orbit_ranges(&self) -> Vec<(usize, usize)> {
        self.inner.orbit_ranges().to_vec()
    }

    /// Simultaneous finite permutation action.
    #[getter]
    fn action(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.action().clone(),
        }
    }

    /// Induced permutation action of a verified subgroup.
    fn subgroup_action(&self, subgroup: &PyIntegralLineSubgroup) -> PyResult<PyPermGroup> {
        Ok(PyPermGroup {
            inner: self
                .inner
                .subgroup_action(&subgroup.inner)
                .map_err(line_configuration_error)?,
        })
    }

    /// Canonical primitive line at one action point.
    fn line(&self, py: Python<'_>, point: noether::groups::Point) -> PyResult<Py<PyTuple>> {
        let row = self.inner.line(point).map_err(line_configuration_error)?;
        integer_row_to_python(py, row)
    }

    /// Rebuild every orbit and induced action exactly.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Verified finite subgroup of an integral-line configuration stabilizer.
#[pyclass(
    module = "noether.linalg",
    name = "IntegralLineSubgroup",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegralLineSubgroup {
    inner: RustIntegralLineSubgroup,
}

#[pymethods]
impl PyIntegralLineSubgroup {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegralLineSubgroup::TYPE_ID
    }

    /// Spanning configuration preserved by the subgroup.
    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    /// Unimodular matrix generators.
    #[getter]
    fn matrix_generators(&self) -> Vec<PyMatrix> {
        self.inner
            .matrix_generators()
            .iter()
            .cloned()
            .map(|matrix| PyMatrix {
                inner: MatrixInner::Integer(matrix),
            })
            .collect()
    }

    /// Faithful signed-line action.
    #[getter]
    fn signed_action(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.signed_action().clone(),
        }
    }

    /// Unoriented line action.
    #[getter]
    fn line_action(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.line_action().clone(),
        }
    }

    /// Exact subgroup order.
    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.order())
    }

    /// Restrict to an invariant spanning subconfiguration.
    fn restrict(
        &self,
        configuration: &PyIntegralLineConfiguration,
    ) -> PyResult<PyIntegralLineSubgroup> {
        Ok(PyIntegralLineSubgroup {
            inner: self
                .inner
                .restrict(&configuration.inner)
                .map_err(line_configuration_error)?,
        })
    }

    /// Conjugate through a unimodular coordinate transformation.
    fn conjugate(&self, transformation: &PyMatrix) -> PyResult<PyIntegralLineSubgroup> {
        let MatrixInner::Integer(transformation) = &transformation.inner else {
            return Err(PyTypeError::new_err(
                "line subgroup transport requires a Matrix[IntegerRing]",
            ));
        };
        Ok(PyIntegralLineSubgroup {
            inner: self
                .inner
                .conjugate(transformation)
                .map_err(line_configuration_error)?,
        })
    }

    /// Recheck every generator action and the exact order.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(line_configuration_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// A complete minimum-vector configuration with an exact Gram witness.
#[pyclass(
    module = "noether.linalg",
    name = "MinimumVectorConfiguration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyMinimumVectorConfiguration {
    inner: RustMinimumVectorConfiguration,
}

#[pymethods]
impl PyMinimumVectorConfiguration {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustMinimumVectorConfiguration::TYPE_ID
    }

    /// Certified primitive integral lines modulo sign.
    #[getter]
    fn configuration(&self) -> PyIntegralLineConfiguration {
        PyIntegralLineConfiguration::from_core(self.inner.configuration().clone())
    }

    /// Exact positive-definite Gram witness.
    #[getter]
    fn gram_matrix(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.gram_matrix().clone()),
        }
    }

    /// Exact lattice minimum attained by every stored line.
    #[getter]
    fn minimum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.minimum())
    }

    /// Re-enumerate the minimum shell and verify the certificate.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(lattice_error)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A positive-definite integral lattice presented by its Gram matrix.
#[pyclass(module = "noether.linalg", name = "Lattice", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyLattice {
    inner: RustLattice,
}

#[pymethods]
impl PyLattice {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustLattice::TYPE_ID
    }

    #[new]
    fn new(py: Python<'_>, gram_matrix: &PyMatrix) -> PyResult<Self> {
        let MatrixInner::Integer(gram) = &gram_matrix.inner else {
            return Err(PyTypeError::new_err(
                "Lattice requires a Matrix[IntegerRing] Gram matrix",
            ));
        };
        let gram = gram.clone();
        py.detach(move || {
            Ok(Self {
                inner: RustLattice::from_gram(gram).map_err(lattice_error)?,
            })
        })
    }

    /// Exact integral Gram matrix.
    #[getter]
    fn gram_matrix(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.gram_matrix().clone()),
        }
    }

    /// Lattice rank.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[pyo3(signature = (*, bound=None, max_pairs=None))]
    fn shortest_vectors(
        &self,
        py: Python<'_>,
        bound: Option<&Bound<'_, PyAny>>,
        max_pairs: Option<usize>,
    ) -> PyResult<PyLatticeEnumeration> {
        let bound = bound.map(extract_integer).transpose()?;
        py.detach(|| {
            Ok(PyLatticeEnumeration {
                inner: self
                    .inner
                    .shortest_vectors(bound, max_pairs)
                    .map_err(lattice_error)?,
            })
        })
    }

    /// Compute and certify the complete minimum-vector configuration.
    fn minimum_vector_configuration(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyMinimumVectorConfiguration> {
        Ok(PyMinimumVectorConfiguration {
            inner: py
                .detach(|| self.inner.minimum_vector_configuration())
                .map_err(lattice_error)?,
        })
    }

    #[pyo3(signature = (target_numerators, target_denominator=1, *, max_count=None))]
    fn closest_vectors(
        &self,
        py: Python<'_>,
        target_numerators: Vec<i64>,
        target_denominator: i64,
        max_count: Option<usize>,
    ) -> PyResult<PyClosestLatticeVectors> {
        py.detach(|| {
            Ok(PyClosestLatticeVectors {
                inner: self
                    .inner
                    .closest_vectors(&target_numerators, target_denominator, max_count)
                    .map_err(lattice_error)?,
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Exact short-vector enumeration with canonical `±`-pair representatives.
#[pyclass(
    module = "noether.linalg",
    name = "LatticeEnumeration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyLatticeEnumeration {
    inner: RustLatticeEnumeration,
}

#[pymethods]
impl PyLatticeEnumeration {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustLatticeEnumeration::TYPE_ID
    }

    #[getter]
    fn signed_count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.signed_count())
    }

    #[getter]
    fn pair_count(&self) -> usize {
        self.inner.pair_count()
    }

    #[getter]
    fn max_norm(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.max_norm())
    }

    #[getter]
    fn bound(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .bound()
            .map(|bound| integer_to_python(py, bound))
            .transpose()
    }

    #[getter]
    fn complete(&self) -> bool {
        self.inner.is_complete()
    }

    #[getter]
    fn vectors(&self) -> Vec<Vec<i64>> {
        self.inner.vectors().to_vec()
    }

    fn verify(&self, py: Python<'_>, lattice: &PyLattice) -> PyResult<()> {
        py.detach(|| self.inner.verify(&lattice.inner).map_err(lattice_error))
    }
}

/// Exact closest vectors and their common squared distance.
#[pyclass(
    module = "noether.linalg",
    name = "ClosestLatticeVectors",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyClosestLatticeVectors {
    inner: RustClosestLatticeVectors,
}

#[pymethods]
impl PyClosestLatticeVectors {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustClosestLatticeVectors::TYPE_ID
    }

    #[getter]
    fn distance_squared(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        rational_to_python(py, self.inner.distance_squared())
    }

    #[getter]
    fn vectors(&self) -> Vec<Vec<i64>> {
        self.inner.vectors().to_vec()
    }
}

fn context_mismatch(operation: &'static str, expected: String, actual: String) -> PyErr {
    super::errors::context_mismatch(ContextMismatchError {
        expected: format!("{expected} for {operation}"),
        actual,
    })
}

const FIELD_MATRIX_CONTEXTS: &[&str] = &[
    RationalField::TYPE_ID,
    FiniteField::TYPE_ID,
    NumberField::TYPE_ID,
    RationalFunctionField::<RationalField>::TYPE_ID,
];

const CHARPOLY_MATRIX_CONTEXTS: &[&str] = &[
    IntegerRing::TYPE_ID,
    RationalField::TYPE_ID,
    ResidueRing::TYPE_ID,
    FiniteField::TYPE_ID,
    PadicRing::TYPE_ID,
    PadicField::TYPE_ID,
    QadicRing::TYPE_ID,
    QadicField::TYPE_ID,
    NumberField::TYPE_ID,
    RationalFunctionField::<RationalField>::TYPE_ID,
];

fn extract_rational_function(
    py: Python<'_>,
    field: &RationalFunctionField<RationalField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<RationalFunction<RationalField>> {
    if let Ok(function) = value.extract::<PyRef<'_, PyRationalFunction>>() {
        let function = function.as_rational().ok_or_else(|| {
            PyTypeError::new_err("rational-function scalar has a different base field")
        })?;
        if field.contains(function) {
            return Ok(function.clone());
        }
        return Err(PyValueError::new_err(
            "rational-function scalar belongs to another field context",
        ));
    }
    field
        .constant(extract_rational(py, value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn unsupported_matrix_operation(
    operation: &'static str,
    context: &'static str,
    supported_contexts: &[&'static str],
) -> PyErr {
    super::errors::unsupported_operation(
        UnsupportedOperationError::new(format!("noether/linalg/Matrix/{operation}@1"), context)
            .with_supported_contexts(supported_contexts.iter().copied()),
    )
}

fn validate_python_shape(rows: usize, columns: usize) -> PyResult<()> {
    if i64::try_from(rows).is_err()
        || i64::try_from(columns).is_err()
        || rows.checked_mul(columns).is_none()
    {
        Err(PyOverflowError::new_err(format!(
            "matrix shape {rows}x{columns} is too large"
        )))
    } else {
        Ok(())
    }
}

fn bounded_coefficient(mut rendered: String) -> String {
    const CHARACTER_LIMIT: usize = 64;
    if rendered.len() > CHARACTER_LIMIT {
        rendered.truncate(CHARACTER_LIMIT);
        rendered.push_str("...");
    }
    rendered
}

fn serialized_linear_algebra_ring_types(value: &str) -> PyResult<(String, Option<String>)> {
    let payload: serde_json::Value =
        serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let ring = payload
        .get("ring")
        .ok_or_else(|| PyValueError::new_err("serialized linear-algebra value has no ring"))?;
    let ring_type = ring
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| PyValueError::new_err("serialized ring has no string type identity"))?;
    let coefficient_type = ring
        .get("coefficient_ring")
        .or_else(|| {
            ring.get("defining_ideal")
                .and_then(|ideal| ideal.get("ring"))
                .and_then(|ambient| ambient.get("coefficient_ring"))
        })
        .and_then(|coefficient| coefficient.get("type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok((ring_type, coefficient_type))
}

#[derive(Clone)]
pub(super) enum MatrixInner {
    Integer(Matrix<IntegerRing>),
    Rational(Matrix<RationalField>),
    Residue(Matrix<ResidueRing>),
    FiniteField(Matrix<FiniteField>),
    Padic(Matrix<PadicRing>),
    PadicField(Matrix<PadicField>),
    Qadic(Matrix<QadicRing>),
    QadicField(Matrix<QadicField>),
    PadicLocalField(Matrix<PadicLocalField>),
    PadicLocalFieldOrder(Matrix<PadicLocalFieldOrder>),
    NumberField(Matrix<NumberField>),
    RationalFunction(Matrix<RationalFunctionField<RationalField>>),
    NumberFieldOrder(Matrix<NumberFieldOrder>),
    RealBall(Matrix<RealBallField>),
    ComplexBall(Matrix<ComplexBallField>),
    SeriesInteger(Matrix<PowerSeriesRing<IntegerRing>>),
    SeriesRational(Matrix<PowerSeriesRing<RationalField>>),
    SeriesResidue(Matrix<PowerSeriesRing<ResidueRing>>),
    SeriesFiniteField(Matrix<PowerSeriesRing<FiniteField>>),
    SeriesPadic(Matrix<PowerSeriesRing<PadicRing>>),
    SeriesPadicField(Matrix<PowerSeriesRing<PadicField>>),
    SeriesQadic(Matrix<PowerSeriesRing<QadicRing>>),
    SeriesQadicField(Matrix<PowerSeriesRing<QadicField>>),
    SeriesPadicLocalField(Matrix<PowerSeriesRing<PadicLocalField>>),
    Polynomial(PolynomialMatrixInner),
    Quotient(QuotientMatrixInner),
}

enum DeterminantInner {
    Integer(Integer),
    Rational(Rational),
    Residue(ResidueClass),
    FiniteField(FiniteFieldElement),
    Padic(PadicElement),
    Qadic(QadicElement),
    PadicLocalField(PadicLocalFieldElement),
    NumberField(noether::numthy::NumberFieldElement),
    RationalFunction(RationalFunction<RationalField>),
    RealBall(RealBall),
    ComplexBall(ComplexBall),
    SeriesInteger(PowerSeries<IntegerRing>),
    SeriesRational(PowerSeries<RationalField>),
    SeriesResidue(PowerSeries<ResidueRing>),
    SeriesFiniteField(PowerSeries<FiniteField>),
    SeriesPadic(PowerSeries<PadicRing>),
    SeriesPadicField(PowerSeries<PadicField>),
    SeriesQadic(PowerSeries<QadicRing>),
    SeriesQadicField(PowerSeries<QadicField>),
    SeriesPadicLocalField(PowerSeries<PadicLocalField>),
    PolynomialInteger(Polynomial<IntegerRing>),
    PolynomialRational(Polynomial<RationalField>),
    PolynomialResidue(Polynomial<ResidueRing>),
    PolynomialFiniteField(Polynomial<FiniteField>),
    QuotientRational(QuotientElement<PolynomialRing<RationalField>>),
    QuotientFiniteField(QuotientElement<PolynomialRing<FiniteField>>),
}

enum ExactLinearSystemInner {
    Integer(RustIntegerLinearSystemResult),
    Congruence(RustCongruenceSystemResult),
}

#[derive(Clone)]
pub(super) enum VectorInner {
    Integer(Vector<IntegerRing>),
    Rational(Vector<RationalField>),
    Residue(Vector<ResidueRing>),
    FiniteField(Vector<FiniteField>),
    Padic(Vector<PadicRing>),
    PadicField(Vector<PadicField>),
    Qadic(Vector<QadicRing>),
    QadicField(Vector<QadicField>),
    PadicLocalField(Vector<PadicLocalField>),
    PadicLocalFieldOrder(Vector<PadicLocalFieldOrder>),
    NumberField(Vector<NumberField>),
    RationalFunction(Vector<RationalFunctionField<RationalField>>),
    NumberFieldOrder(Vector<NumberFieldOrder>),
    RealBall(Vector<RealBallField>),
    ComplexBall(Vector<ComplexBallField>),
    SeriesInteger(Vector<PowerSeriesRing<IntegerRing>>),
    SeriesRational(Vector<PowerSeriesRing<RationalField>>),
    SeriesResidue(Vector<PowerSeriesRing<ResidueRing>>),
    SeriesFiniteField(Vector<PowerSeriesRing<FiniteField>>),
    SeriesPadic(Vector<PowerSeriesRing<PadicRing>>),
    SeriesPadicField(Vector<PowerSeriesRing<PadicField>>),
    SeriesQadic(Vector<PowerSeriesRing<QadicRing>>),
    SeriesQadicField(Vector<PowerSeriesRing<QadicField>>),
    SeriesPadicLocalField(Vector<PowerSeriesRing<PadicLocalField>>),
    Polynomial(PolynomialVectorInner),
    Quotient(QuotientVectorInner),
}

#[derive(Clone)]
pub(super) enum PolynomialMatrixInner {
    Integer(Matrix<PolynomialRing<IntegerRing>>),
    Rational(Matrix<PolynomialRing<RationalField>>),
    Residue(Matrix<PolynomialRing<ResidueRing>>),
    FiniteField(Matrix<PolynomialRing<FiniteField>>),
}

#[derive(Clone)]
pub(super) enum PolynomialVectorInner {
    Integer(Vector<PolynomialRing<IntegerRing>>),
    Rational(Vector<PolynomialRing<RationalField>>),
    Residue(Vector<PolynomialRing<ResidueRing>>),
    FiniteField(Vector<PolynomialRing<FiniteField>>),
}

#[derive(Clone)]
pub(super) enum QuotientMatrixInner {
    Rational(Matrix<QuotientRing<PolynomialRing<RationalField>>>),
    FiniteField(Matrix<QuotientRing<PolynomialRing<FiniteField>>>),
}

#[derive(Clone)]
pub(super) enum QuotientVectorInner {
    Rational(Vector<QuotientRing<PolynomialRing<RationalField>>>),
    FiniteField(Vector<QuotientRing<PolynomialRing<FiniteField>>>),
}

fn polynomial_rows_to_python<R>(
    matrix: &Matrix<PolynomialRing<R>>,
    py: Python<'_>,
    wrap: fn(Polynomial<R>) -> PyPolynomial,
) -> PyResult<Vec<Py<PyAny>>>
where
    R: CommutativeRing,
{
    matrix
        .rows()
        .map(|row| {
            row.into_iter()
                .map(|entry| Ok(Py::new(py, wrap(entry))?.into_any()))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
        })
        .collect()
}

fn polynomial_values_to_python<R>(
    values: impl IntoIterator<Item = Polynomial<R>>,
    py: Python<'_>,
    wrap: fn(Polynomial<R>) -> PyPolynomial,
) -> PyResult<Py<PyAny>>
where
    R: CommutativeRing,
{
    let values = values
        .into_iter()
        .map(|entry| Ok(Py::new(py, wrap(entry))?.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, values)?.unbind().into_any())
}

fn polynomial_matrix_preview<R: CommutativeRing>(matrix: &Matrix<PolynomialRing<R>>) -> String {
    const ROW_LIMIT: usize = 4;
    const COLUMN_LIMIT: usize = 4;
    let (row_count, column_count) = matrix.shape();
    let mut rendered_rows = Vec::new();
    for row in 0..row_count.min(ROW_LIMIT) {
        let mut entries = (0..column_count.min(COLUMN_LIMIT))
            .filter_map(|column| matrix.get(row, column).ok())
            .map(|entry| bounded_coefficient(entry.to_string()))
            .collect::<Vec<_>>();
        if column_count > COLUMN_LIMIT {
            entries.push("...".to_owned());
        }
        rendered_rows.push(format!("[{}]", entries.join(", ")));
    }
    if row_count > ROW_LIMIT {
        rendered_rows.push("...".to_owned());
    }
    format!("[{}]", rendered_rows.join(", "))
}

fn polynomial_vector_preview<R: CommutativeRing>(vector: &Vector<PolynomialRing<R>>) -> String {
    const ENTRY_LIMIT: usize = 8;
    let mut entries = vector
        .values()
        .take(ENTRY_LIMIT)
        .map(|entry| bounded_coefficient(entry.to_string()))
        .collect::<Vec<_>>();
    if vector.dimension() > ENTRY_LIMIT {
        entries.push("...".to_owned());
    }
    format!("[{}]", entries.join(", "))
}

pub(super) fn extract_integer_polynomial(
    ring: &PolynomialRing<IntegerRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Polynomial<IntegerRing>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        if let Some(value) = value.as_integer().filter(|value| ring.contains(value)) {
            return Ok(value.clone());
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different polynomial context".to_owned(),
        ));
    }
    ring.constant(extract_integer(value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn extract_rational_polynomial(
    py: Python<'_>,
    ring: &PolynomialRing<RationalField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Polynomial<RationalField>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        if let Some(value) = value.as_rational().filter(|value| ring.contains(value)) {
            return Ok(value.clone());
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different polynomial context".to_owned(),
        ));
    }
    ring.constant(extract_rational(py, value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn extract_residue_polynomial(
    ring: &PolynomialRing<ResidueRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Polynomial<ResidueRing>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        if let Some(value) = value.as_residue().filter(|value| ring.contains(value)) {
            return Ok(value.clone());
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different polynomial context".to_owned(),
        ));
    }
    ring.constant(extract_residue(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn extract_finite_field_polynomial(
    ring: &PolynomialRing<FiniteField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Polynomial<FiniteField>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        if let Some(value) = value.as_finite_field().filter(|value| ring.contains(value)) {
            return Ok(value.clone());
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different polynomial context".to_owned(),
        ));
    }
    ring.constant(extract_finite_field_element(
        ring.coefficient_ring(),
        value,
    )?)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_rational_quotient(
    ring: &QuotientRing<PolynomialRing<RationalField>>,
    value: &Bound<'_, PyAny>,
) -> PyResult<QuotientElement<PolynomialRing<RationalField>>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyQuotientElement>>() {
        if let QuotientElementInner::Rational(value) = &value.inner {
            if ring.contains(value) {
                return Ok(value.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different quotient context".to_owned(),
        ));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        let value = value.as_rational().ok_or_else(|| {
            PyTypeError::new_err("quotient coefficient requires an ambient polynomial over Q")
        })?;
        return ring.element(value).map_err(|error| {
            context_mismatch(
                "linear-algebra construction",
                ring.to_string(),
                error.actual,
            )
        });
    }
    Err(PyTypeError::new_err(
        "quotient coefficient must be a QuotientElement or ambient Polynomial",
    ))
}

fn extract_finite_field_quotient(
    ring: &QuotientRing<PolynomialRing<FiniteField>>,
    value: &Bound<'_, PyAny>,
) -> PyResult<QuotientElement<PolynomialRing<FiniteField>>> {
    if let Ok(value) = value.extract::<PyRef<'_, PyQuotientElement>>() {
        if let QuotientElementInner::FiniteField(value) = &value.inner {
            if ring.contains(value) {
                return Ok(value.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different quotient context".to_owned(),
        ));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyPolynomial>>() {
        let value = value.as_finite_field().ok_or_else(|| {
            PyTypeError::new_err("quotient coefficient requires an ambient finite-field polynomial")
        })?;
        return ring.element(value).map_err(|error| {
            context_mismatch(
                "linear-algebra construction",
                ring.to_string(),
                error.actual,
            )
        });
    }
    Err(PyTypeError::new_err(
        "quotient coefficient must be a QuotientElement or ambient Polynomial",
    ))
}

impl PolynomialMatrixInner {
    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Integer(value) => value.to_json(),
            Self::Rational(value) => value.to_json(),
            Self::Residue(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn rows_to_python(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match self {
            Self::Integer(value) => {
                polynomial_rows_to_python(value, py, PyPolynomial::from_integer)
            }
            Self::Rational(value) => {
                polynomial_rows_to_python(value, py, PyPolynomial::from_rational)
            }
            Self::Residue(value) => {
                polynomial_rows_to_python(value, py, PyPolynomial::from_residue)
            }
            Self::FiniteField(value) => {
                polynomial_rows_to_python(value, py, PyPolynomial::from_finite_field)
            }
        }
    }

    fn context_name(&self) -> String {
        match self {
            Self::Integer(value) => bounded_coefficient(value.ring().to_string()),
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::Residue(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Integer(value) => value.shape(),
            Self::Rational(value) => value.shape(),
            Self::Residue(value) => value.shape(),
            Self::FiniteField(value) => value.shape(),
        }
    }

    fn transpose(&self) -> Self {
        match self {
            Self::Integer(value) => Self::Integer(value.transpose()),
            Self::Rational(value) => Self::Rational(value.transpose()),
            Self::Residue(value) => Self::Residue(value.transpose()),
            Self::FiniteField(value) => Self::FiniteField(value.transpose()),
        }
    }

    fn preview(&self) -> String {
        match self {
            Self::Integer(value) => polynomial_matrix_preview(value),
            Self::Rational(value) => polynomial_matrix_preview(value),
            Self::Residue(value) => polynomial_matrix_preview(value),
            Self::FiniteField(value) => polynomial_matrix_preview(value),
        }
    }

    fn to_sparse(&self) -> Result<PolynomialSparseMatrixInner, LinalgError> {
        Ok(match self {
            Self::Integer(value) => {
                PolynomialSparseMatrixInner::Integer(RustSparseMatrix::from_dense(value)?)
            }
            Self::Rational(value) => {
                PolynomialSparseMatrixInner::Rational(RustSparseMatrix::from_dense(value)?)
            }
            Self::Residue(value) => {
                PolynomialSparseMatrixInner::Residue(RustSparseMatrix::from_dense(value)?)
            }
            Self::FiniteField(value) => {
                PolynomialSparseMatrixInner::FiniteField(RustSparseMatrix::from_dense(value)?)
            }
        })
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Integer(value) => PyPolynomialRing::from_integer(value.ring().clone()),
            Self::Rational(value) => PyPolynomialRing::from_rational(value.ring().clone()),
            Self::Residue(value) => PyPolynomialRing::from_residue(value.ring().clone()),
            Self::FiniteField(value) => PyPolynomialRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, row: usize, column: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Integer(matrix) => {
                PyPolynomial::from_integer(matrix.get(row, column).map_err(index_error)?)
            }
            Self::Rational(matrix) => {
                PyPolynomial::from_rational(matrix.get(row, column).map_err(index_error)?)
            }
            Self::Residue(matrix) => {
                PyPolynomial::from_residue(matrix.get(row, column).map_err(index_error)?)
            }
            Self::FiniteField(matrix) => {
                PyPolynomial::from_finite_field(matrix.get(row, column).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.matmul(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix multiplication",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.add(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(value.neg().map_err(linalg_error)?),
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::Residue(value) => Self::Residue(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(
                value
                    .scalar_mul(&extract_integer_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Rational(value) => Self::Rational(
                value
                    .scalar_mul(&extract_rational_polynomial(py, value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Residue(value) => Self::Residue(
                value
                    .scalar_mul(&extract_residue_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::FiniteField(value) => Self::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
        })
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::Residue(left), Self::Residue(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

fn quotient_rows_to_python<R>(
    matrix: &Matrix<QuotientRing<PolynomialRing<R>>>,
    py: Python<'_>,
    wrap: fn(QuotientElement<PolynomialRing<R>>) -> PyQuotientElement,
) -> PyResult<Vec<Py<PyAny>>>
where
    R: Field,
{
    matrix
        .rows()
        .map(|row| {
            row.into_iter()
                .map(|entry| Ok(Py::new(py, wrap(entry))?.into_any()))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
        })
        .collect()
}

fn quotient_matrix_preview<R: Field>(matrix: &Matrix<QuotientRing<PolynomialRing<R>>>) -> String {
    const ROW_LIMIT: usize = 4;
    const COLUMN_LIMIT: usize = 4;
    let (row_count, column_count) = matrix.shape();
    let mut rendered_rows = Vec::new();
    for row in 0..row_count.min(ROW_LIMIT) {
        let mut entries = (0..column_count.min(COLUMN_LIMIT))
            .filter_map(|column| matrix.get(row, column).ok())
            .map(|entry| bounded_coefficient(entry.lift().to_string()))
            .collect::<Vec<_>>();
        if column_count > COLUMN_LIMIT {
            entries.push("...".to_owned());
        }
        rendered_rows.push(format!("[{}]", entries.join(", ")));
    }
    if row_count > ROW_LIMIT {
        rendered_rows.push("...".to_owned());
    }
    format!("[{}]", rendered_rows.join(", "))
}

impl QuotientMatrixInner {
    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Rational(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn rows_to_python(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        match self {
            Self::Rational(value) => {
                quotient_rows_to_python(value, py, PyQuotientElement::from_rational)
            }
            Self::FiniteField(value) => {
                quotient_rows_to_python(value, py, PyQuotientElement::from_finite_field)
            }
        }
    }

    fn context_name(&self) -> String {
        match self {
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Rational(value) => value.shape(),
            Self::FiniteField(value) => value.shape(),
        }
    }

    fn transpose(&self) -> Self {
        match self {
            Self::Rational(value) => Self::Rational(value.transpose()),
            Self::FiniteField(value) => Self::FiniteField(value.transpose()),
        }
    }

    fn preview(&self) -> String {
        match self {
            Self::Rational(value) => quotient_matrix_preview(value),
            Self::FiniteField(value) => quotient_matrix_preview(value),
        }
    }

    fn to_sparse(&self) -> Result<QuotientSparseMatrixInner, LinalgError> {
        Ok(match self {
            Self::Rational(value) => {
                QuotientSparseMatrixInner::Rational(RustSparseMatrix::from_dense(value)?)
            }
            Self::FiniteField(value) => {
                QuotientSparseMatrixInner::FiniteField(RustSparseMatrix::from_dense(value)?)
            }
        })
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Rational(value) => PyQuotientRing::from_rational(value.ring().clone()),
            Self::FiniteField(value) => PyQuotientRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, row: usize, column: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Rational(matrix) => {
                PyQuotientElement::from_rational(matrix.get(row, column).map_err(index_error)?)
            }
            Self::FiniteField(matrix) => {
                PyQuotientElement::from_finite_field(matrix.get(row, column).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.matmul(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix multiplication",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "matrix subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        match self {
            Self::Rational(matrix) => Ok(Self::Rational(
                matrix
                    .scalar_mul(&extract_rational_quotient(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
            Self::FiniteField(matrix) => Ok(Self::FiniteField(
                matrix
                    .scalar_mul(&extract_finite_field_quotient(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
        }
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

fn quotient_values_to_python<R>(
    values: impl IntoIterator<Item = QuotientElement<PolynomialRing<R>>>,
    py: Python<'_>,
    wrap: fn(QuotientElement<PolynomialRing<R>>) -> PyQuotientElement,
) -> PyResult<Py<PyAny>>
where
    R: Field,
{
    let values = values
        .into_iter()
        .map(|entry| Ok(Py::new(py, wrap(entry))?.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, values)?.unbind().into_any())
}

fn quotient_vector_preview<R: Field>(vector: &Vector<QuotientRing<PolynomialRing<R>>>) -> String {
    const ENTRY_LIMIT: usize = 8;
    let mut entries = vector
        .values()
        .take(ENTRY_LIMIT)
        .map(|entry| bounded_coefficient(entry.lift().to_string()))
        .collect::<Vec<_>>();
    if vector.dimension() > ENTRY_LIMIT {
        entries.push("...".to_owned());
    }
    format!("[{}]", entries.join(", "))
}

impl QuotientVectorInner {
    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Rational(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Rational(value) => {
                quotient_values_to_python(value.values(), py, PyQuotientElement::from_rational)
            }
            Self::FiniteField(value) => {
                quotient_values_to_python(value.values(), py, PyQuotientElement::from_finite_field)
            }
        }
    }

    fn context_name(&self) -> String {
        match self {
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Rational(value) => value.dimension(),
            Self::FiniteField(value) => value.dimension(),
        }
    }

    fn preview(&self) -> String {
        match self {
            Self::Rational(value) => quotient_vector_preview(value),
            Self::FiniteField(value) => quotient_vector_preview(value),
        }
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Rational(value) => PyQuotientRing::from_rational(value.ring().clone()),
            Self::FiniteField(value) => PyQuotientRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Rational(vector) => {
                PyQuotientElement::from_rational(vector.get(index).map_err(index_error)?)
            }
            Self::FiniteField(vector) => {
                PyQuotientElement::from_finite_field(vector.get(index).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn column_matrix(&self) -> QuotientMatrixInner {
        match self {
            Self::Rational(value) => QuotientMatrixInner::Rational(value.to_column_matrix()),
            Self::FiniteField(value) => QuotientMatrixInner::FiniteField(value.to_column_matrix()),
        }
    }

    fn row_matrix(&self) -> QuotientMatrixInner {
        match self {
            Self::Rational(value) => QuotientMatrixInner::Rational(value.to_row_matrix()),
            Self::FiniteField(value) => QuotientMatrixInner::FiniteField(value.to_row_matrix()),
        }
    }

    fn dot_to_python(&self, py: Python<'_>, rhs: &Self) -> PyResult<Py<PyAny>> {
        let value = match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                PyQuotientElement::from_rational(left.dot(right).map_err(linalg_error)?)
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                PyQuotientElement::from_finite_field(left.dot(right).map_err(linalg_error)?)
            }
            _ => {
                return Err(context_mismatch(
                    "vector dot product",
                    self.context_name(),
                    rhs.context_name(),
                ));
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "vector addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "vector subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        match self {
            Self::Rational(vector) => Ok(Self::Rational(
                vector
                    .scalar_mul(&extract_rational_quotient(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
            Self::FiniteField(vector) => Ok(Self::FiniteField(
                vector
                    .scalar_mul(&extract_finite_field_quotient(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
        }
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl PolynomialVectorInner {
    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Integer(value) => value.to_json(),
            Self::Rational(value) => value.to_json(),
            Self::Residue(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Integer(value) => {
                polynomial_values_to_python(value.values(), py, PyPolynomial::from_integer)
            }
            Self::Rational(value) => {
                polynomial_values_to_python(value.values(), py, PyPolynomial::from_rational)
            }
            Self::Residue(value) => {
                polynomial_values_to_python(value.values(), py, PyPolynomial::from_residue)
            }
            Self::FiniteField(value) => {
                polynomial_values_to_python(value.values(), py, PyPolynomial::from_finite_field)
            }
        }
    }

    fn context_name(&self) -> String {
        match self {
            Self::Integer(value) => bounded_coefficient(value.ring().to_string()),
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::Residue(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Integer(value) => value.dimension(),
            Self::Rational(value) => value.dimension(),
            Self::Residue(value) => value.dimension(),
            Self::FiniteField(value) => value.dimension(),
        }
    }

    fn preview(&self) -> String {
        match self {
            Self::Integer(value) => polynomial_vector_preview(value),
            Self::Rational(value) => polynomial_vector_preview(value),
            Self::Residue(value) => polynomial_vector_preview(value),
            Self::FiniteField(value) => polynomial_vector_preview(value),
        }
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Integer(value) => PyPolynomialRing::from_integer(value.ring().clone()),
            Self::Rational(value) => PyPolynomialRing::from_rational(value.ring().clone()),
            Self::Residue(value) => PyPolynomialRing::from_residue(value.ring().clone()),
            Self::FiniteField(value) => PyPolynomialRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Integer(vector) => {
                PyPolynomial::from_integer(vector.get(index).map_err(index_error)?)
            }
            Self::Rational(vector) => {
                PyPolynomial::from_rational(vector.get(index).map_err(index_error)?)
            }
            Self::Residue(vector) => {
                PyPolynomial::from_residue(vector.get(index).map_err(index_error)?)
            }
            Self::FiniteField(vector) => {
                PyPolynomial::from_finite_field(vector.get(index).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn column_matrix(&self) -> PolynomialMatrixInner {
        match self {
            Self::Integer(value) => PolynomialMatrixInner::Integer(value.to_column_matrix()),
            Self::Rational(value) => PolynomialMatrixInner::Rational(value.to_column_matrix()),
            Self::Residue(value) => PolynomialMatrixInner::Residue(value.to_column_matrix()),
            Self::FiniteField(value) => {
                PolynomialMatrixInner::FiniteField(value.to_column_matrix())
            }
        }
    }

    fn row_matrix(&self) -> PolynomialMatrixInner {
        match self {
            Self::Integer(value) => PolynomialMatrixInner::Integer(value.to_row_matrix()),
            Self::Rational(value) => PolynomialMatrixInner::Rational(value.to_row_matrix()),
            Self::Residue(value) => PolynomialMatrixInner::Residue(value.to_row_matrix()),
            Self::FiniteField(value) => PolynomialMatrixInner::FiniteField(value.to_row_matrix()),
        }
    }

    fn dot_to_python(&self, py: Python<'_>, rhs: &Self) -> PyResult<Py<PyAny>> {
        let value = match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                PyPolynomial::from_integer(left.dot(right).map_err(linalg_error)?)
            }
            (Self::Rational(left), Self::Rational(right)) => {
                PyPolynomial::from_rational(left.dot(right).map_err(linalg_error)?)
            }
            (Self::Residue(left), Self::Residue(right)) => {
                PyPolynomial::from_residue(left.dot(right).map_err(linalg_error)?)
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                PyPolynomial::from_finite_field(left.dot(right).map_err(linalg_error)?)
            }
            _ => {
                return Err(context_mismatch(
                    "vector dot product",
                    self.context_name(),
                    rhs.context_name(),
                ));
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.add(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "vector addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "vector subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(value.neg().map_err(linalg_error)?),
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::Residue(value) => Self::Residue(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(
                value
                    .scalar_mul(&extract_integer_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Rational(value) => Self::Rational(
                value
                    .scalar_mul(&extract_rational_polynomial(py, value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Residue(value) => Self::Residue(
                value
                    .scalar_mul(&extract_residue_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::FiniteField(value) => Self::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
        })
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::Residue(left), Self::Residue(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

#[derive(Clone)]
enum FrobeniusSemilinearMapInner {
    Qadic(RustFrobeniusSemilinearMap<QadicRing>),
    QadicField(RustFrobeniusSemilinearMap<QadicField>),
}

/// A map `v -> A sigma^r(v)` over a fixed-cap unramified p-adic context.
#[pyclass(
    module = "noether.linalg",
    name = "FrobeniusSemilinearMap",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFrobeniusSemilinearMap {
    inner: FrobeniusSemilinearMapInner,
}

#[pymethods]
impl PyFrobeniusSemilinearMap {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustFrobeniusSemilinearMap::<QadicRing>::TYPE_ID
    }

    #[new]
    fn new(matrix: &PyMatrix, frobenius_exponent: u64) -> PyResult<Self> {
        let inner = match &matrix.inner {
            MatrixInner::Qadic(matrix) => FrobeniusSemilinearMapInner::Qadic(
                RustFrobeniusSemilinearMap::new(matrix.clone(), frobenius_exponent),
            ),
            MatrixInner::QadicField(matrix) => FrobeniusSemilinearMapInner::QadicField(
                RustFrobeniusSemilinearMap::new(matrix.clone(), frobenius_exponent),
            ),
            _ => {
                return Err(PyTypeError::new_err(
                    "FrobeniusSemilinearMap requires a matrix over QadicRing or QadicField",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn identity(
        _cls: &Bound<'_, PyType>,
        context: &Bound<'_, PyAny>,
        dimension: usize,
    ) -> PyResult<Self> {
        let inner = if let Ok(context) = context.extract::<PyRef<'_, PyQadicRing>>() {
            FrobeniusSemilinearMapInner::Qadic(RustFrobeniusSemilinearMap::identity(
                context.inner.clone(),
                dimension,
            ))
        } else if let Ok(context) = context.extract::<PyRef<'_, PyQadicField>>() {
            FrobeniusSemilinearMapInner::QadicField(RustFrobeniusSemilinearMap::identity(
                context.inner.clone(),
                dimension,
            ))
        } else {
            return Err(PyTypeError::new_err(
                "FrobeniusSemilinearMap.identity requires QadicRing or QadicField",
            ));
        };
        Ok(Self { inner })
    }

    #[getter]
    fn context(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: map.context().clone(),
                },
            )?
            .into_any()),
            FrobeniusSemilinearMapInner::QadicField(map) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: map.context().clone(),
                },
            )?
            .into_any()),
        }
    }

    #[getter]
    fn matrix(&self) -> PyMatrix {
        let inner = match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => MatrixInner::Qadic(map.matrix().clone()),
            FrobeniusSemilinearMapInner::QadicField(map) => {
                MatrixInner::QadicField(map.matrix().clone())
            }
        };
        PyMatrix { inner }
    }

    #[getter]
    fn frobenius_exponent(&self) -> u64 {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => map.frobenius_exponent(),
            FrobeniusSemilinearMapInner::QadicField(map) => map.frobenius_exponent(),
        }
    }

    #[getter]
    fn source_dimension(&self) -> usize {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => map.source_dimension(),
            FrobeniusSemilinearMapInner::QadicField(map) => map.source_dimension(),
        }
    }

    #[getter]
    fn target_dimension(&self) -> usize {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => map.target_dimension(),
            FrobeniusSemilinearMapInner::QadicField(map) => map.target_dimension(),
        }
    }

    fn apply(&self, vector: &PyVector) -> PyResult<PyVector> {
        let inner = match (&self.inner, &vector.inner) {
            (FrobeniusSemilinearMapInner::Qadic(map), VectorInner::Qadic(vector)) => {
                VectorInner::Qadic(map.apply(vector).map_err(semilinear_error)?)
            }
            (FrobeniusSemilinearMapInner::QadicField(map), VectorInner::QadicField(vector)) => {
                VectorInner::QadicField(map.apply(vector).map_err(semilinear_error)?)
            }
            (FrobeniusSemilinearMapInner::Qadic(map), _) => {
                return Err(context_mismatch(
                    "Frobenius-semilinear application",
                    map.context().to_string(),
                    vector.context_name(),
                ));
            }
            (FrobeniusSemilinearMapInner::QadicField(map), _) => {
                return Err(context_mismatch(
                    "Frobenius-semilinear application",
                    map.context().to_string(),
                    vector.context_name(),
                ));
            }
        };
        Ok(PyVector { inner })
    }

    fn apply_block(&self, block: &PyMatrix) -> PyResult<PyMatrix> {
        let inner = match (&self.inner, &block.inner) {
            (FrobeniusSemilinearMapInner::Qadic(map), MatrixInner::Qadic(block)) => {
                MatrixInner::Qadic(map.apply_block(block).map_err(semilinear_error)?)
            }
            (FrobeniusSemilinearMapInner::QadicField(map), MatrixInner::QadicField(block)) => {
                MatrixInner::QadicField(map.apply_block(block).map_err(semilinear_error)?)
            }
            (FrobeniusSemilinearMapInner::Qadic(map), _) => {
                return Err(context_mismatch(
                    "Frobenius-semilinear block application",
                    map.context().to_string(),
                    block.context_name(),
                ));
            }
            (FrobeniusSemilinearMapInner::QadicField(map), _) => {
                return Err(context_mismatch(
                    "Frobenius-semilinear block application",
                    map.context().to_string(),
                    block.context_name(),
                ));
            }
        };
        Ok(PyMatrix { inner })
    }

    fn compose(&self, inner: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &inner.inner) {
            (
                FrobeniusSemilinearMapInner::Qadic(outer),
                FrobeniusSemilinearMapInner::Qadic(inner),
            ) => {
                FrobeniusSemilinearMapInner::Qadic(outer.compose(inner).map_err(semilinear_error)?)
            }
            (
                FrobeniusSemilinearMapInner::QadicField(outer),
                FrobeniusSemilinearMapInner::QadicField(inner),
            ) => FrobeniusSemilinearMapInner::QadicField(
                outer.compose(inner).map_err(semilinear_error)?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "context mismatch: cannot compose q-adic ring and field semilinear maps",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn restrict_scalars(&self) -> PyResult<PyMatrix> {
        let inner = match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => {
                MatrixInner::Padic(map.restrict_scalars().map_err(semilinear_error)?)
            }
            FrobeniusSemilinearMapInner::QadicField(map) => {
                MatrixInner::PadicField(map.restrict_scalars().map_err(semilinear_error)?)
            }
        };
        Ok(PyMatrix { inner })
    }

    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => {
                map.verify().map_err(super::errors::verification)
            }
            FrobeniusSemilinearMapInner::QadicField(map) => {
                map.verify().map_err(super::errors::verification)
            }
        }
    }

    fn to_json(&self) -> PyResult<String> {
        match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => map.to_json(),
            FrobeniusSemilinearMapInner::QadicField(map) => map.to_json(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        let payload: serde_json::Value = serde_json::from_str(value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let context_type = payload
            .pointer("/matrix/ring/type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PyValueError::new_err("serialized Frobenius-semilinear map has no matrix ring type")
            })?;
        let inner = match context_type {
            QadicRing::TYPE_ID => FrobeniusSemilinearMapInner::Qadic(
                RustFrobeniusSemilinearMap::<QadicRing>::from_json(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
            QadicField::TYPE_ID => FrobeniusSemilinearMapInner::QadicField(
                RustFrobeniusSemilinearMap::<QadicField>::from_json(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "FrobeniusSemilinearMap requires QadicRing or QadicField, received {context_type}"
                )));
            }
        };
        Ok(Self { inner })
    }

    fn __call__(&self, vector: &PyVector) -> PyResult<PyVector> {
        self.apply(vector)
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (
                FrobeniusSemilinearMapInner::Qadic(left),
                FrobeniusSemilinearMapInner::Qadic(right),
            ) => left == right,
            (
                FrobeniusSemilinearMapInner::QadicField(left),
                FrobeniusSemilinearMapInner::QadicField(right),
            ) => left == right,
            _ => false,
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(match &self.inner {
            FrobeniusSemilinearMapInner::Qadic(map) => map.to_string(),
            FrobeniusSemilinearMapInner::QadicField(map) => map.to_string(),
        })
    }
}

#[derive(Clone)]
pub(crate) enum SparseMatrixInner {
    Integer(RustSparseMatrix<IntegerRing>),
    Rational(RustSparseMatrix<RationalField>),
    Residue(RustSparseMatrix<ResidueRing>),
    FiniteField(RustSparseMatrix<FiniteField>),
    Padic(RustSparseMatrix<PadicRing>),
    PadicField(RustSparseMatrix<PadicField>),
    Qadic(RustSparseMatrix<QadicRing>),
    QadicField(RustSparseMatrix<QadicField>),
    PadicLocalField(RustSparseMatrix<PadicLocalField>),
    PadicLocalFieldOrder(RustSparseMatrix<PadicLocalFieldOrder>),
    RealBall(RustSparseMatrix<RealBallField>),
    ComplexBall(RustSparseMatrix<ComplexBallField>),
    Polynomial(PolynomialSparseMatrixInner),
    Quotient(QuotientSparseMatrixInner),
}

#[derive(Clone)]
pub(crate) enum PolynomialSparseMatrixInner {
    Integer(RustSparseMatrix<PolynomialRing<IntegerRing>>),
    Rational(RustSparseMatrix<PolynomialRing<RationalField>>),
    Residue(RustSparseMatrix<PolynomialRing<ResidueRing>>),
    FiniteField(RustSparseMatrix<PolynomialRing<FiniteField>>),
}

#[derive(Clone)]
pub(crate) enum QuotientSparseMatrixInner {
    Rational(RustSparseMatrix<QuotientRing<PolynomialRing<RationalField>>>),
    FiniteField(RustSparseMatrix<QuotientRing<PolynomialRing<FiniteField>>>),
}

impl PolynomialSparseMatrixInner {
    fn context_name(&self) -> String {
        match self {
            Self::Integer(value) => bounded_coefficient(value.ring().to_string()),
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::Residue(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Integer(value) => value.to_json(),
            Self::Rational(value) => value.to_json(),
            Self::Residue(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Integer(value) => value.shape(),
            Self::Rational(value) => value.shape(),
            Self::Residue(value) => value.shape(),
            Self::FiniteField(value) => value.shape(),
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Self::Integer(value) => value.nnz(),
            Self::Rational(value) => value.nnz(),
            Self::Residue(value) => value.nnz(),
            Self::FiniteField(value) => value.nnz(),
        }
    }

    fn row_offsets(&self) -> Vec<usize> {
        match self {
            Self::Integer(value) => value.row_offsets().to_vec(),
            Self::Rational(value) => value.row_offsets().to_vec(),
            Self::Residue(value) => value.row_offsets().to_vec(),
            Self::FiniteField(value) => value.row_offsets().to_vec(),
        }
    }

    fn column_indices(&self) -> Vec<usize> {
        match self {
            Self::Integer(value) => value.column_indices().to_vec(),
            Self::Rational(value) => value.column_indices().to_vec(),
            Self::Residue(value) => value.column_indices().to_vec(),
            Self::FiniteField(value) => value.column_indices().to_vec(),
        }
    }

    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Integer(value) => polynomial_values_to_python(
                value.values().iter().cloned(),
                py,
                PyPolynomial::from_integer,
            ),
            Self::Rational(value) => polynomial_values_to_python(
                value.values().iter().cloned(),
                py,
                PyPolynomial::from_rational,
            ),
            Self::Residue(value) => polynomial_values_to_python(
                value.values().iter().cloned(),
                py,
                PyPolynomial::from_residue,
            ),
            Self::FiniteField(value) => polynomial_values_to_python(
                value.values().iter().cloned(),
                py,
                PyPolynomial::from_finite_field,
            ),
        }
    }

    fn to_dense(&self) -> PolynomialMatrixInner {
        match self {
            Self::Integer(value) => PolynomialMatrixInner::Integer(value.to_dense()),
            Self::Rational(value) => PolynomialMatrixInner::Rational(value.to_dense()),
            Self::Residue(value) => PolynomialMatrixInner::Residue(value.to_dense()),
            Self::FiniteField(value) => PolynomialMatrixInner::FiniteField(value.to_dense()),
        }
    }

    fn transpose(&self) -> Self {
        match self {
            Self::Integer(value) => Self::Integer(value.transpose()),
            Self::Rational(value) => Self::Rational(value.transpose()),
            Self::Residue(value) => Self::Residue(value.transpose()),
            Self::FiniteField(value) => Self::FiniteField(value.transpose()),
        }
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Integer(value) => PyPolynomialRing::from_integer(value.ring().clone()),
            Self::Rational(value) => PyPolynomialRing::from_rational(value.ring().clone()),
            Self::Residue(value) => PyPolynomialRing::from_residue(value.ring().clone()),
            Self::FiniteField(value) => PyPolynomialRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, row: usize, column: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Integer(matrix) => {
                PyPolynomial::from_integer(matrix.get(row, column).map_err(index_error)?)
            }
            Self::Rational(matrix) => {
                PyPolynomial::from_rational(matrix.get(row, column).map_err(index_error)?)
            }
            Self::Residue(matrix) => {
                PyPolynomial::from_residue(matrix.get(row, column).map_err(index_error)?)
            }
            Self::FiniteField(matrix) => {
                PyPolynomial::from_finite_field(matrix.get(row, column).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn apply(&self, vector: &PolynomialVectorInner) -> PyResult<PolynomialVectorInner> {
        match (self, vector) {
            (Self::Integer(matrix), PolynomialVectorInner::Integer(vector)) => Ok(
                PolynomialVectorInner::Integer(matrix.apply(vector).map_err(linalg_error)?),
            ),
            (Self::Rational(matrix), PolynomialVectorInner::Rational(vector)) => Ok(
                PolynomialVectorInner::Rational(matrix.apply(vector).map_err(linalg_error)?),
            ),
            (Self::Residue(matrix), PolynomialVectorInner::Residue(vector)) => Ok(
                PolynomialVectorInner::Residue(matrix.apply(vector).map_err(linalg_error)?),
            ),
            (Self::FiniteField(matrix), PolynomialVectorInner::FiniteField(vector)) => Ok(
                PolynomialVectorInner::FiniteField(matrix.apply(vector).map_err(linalg_error)?),
            ),
            _ => Err(context_mismatch(
                "sparse matrix application",
                self.context_name(),
                vector.context_name(),
            )),
        }
    }

    fn apply_block(&self, block: &PolynomialMatrixInner) -> PyResult<PolynomialMatrixInner> {
        match (self, block) {
            (Self::Integer(matrix), PolynomialMatrixInner::Integer(block)) => Ok(
                PolynomialMatrixInner::Integer(matrix.apply_block(block).map_err(linalg_error)?),
            ),
            (Self::Rational(matrix), PolynomialMatrixInner::Rational(block)) => Ok(
                PolynomialMatrixInner::Rational(matrix.apply_block(block).map_err(linalg_error)?),
            ),
            (Self::Residue(matrix), PolynomialMatrixInner::Residue(block)) => Ok(
                PolynomialMatrixInner::Residue(matrix.apply_block(block).map_err(linalg_error)?),
            ),
            (Self::FiniteField(matrix), PolynomialMatrixInner::FiniteField(block)) => {
                Ok(PolynomialMatrixInner::FiniteField(
                    matrix.apply_block(block).map_err(linalg_error)?,
                ))
            }
            _ => Err(context_mismatch(
                "sparse block application",
                self.context_name(),
                block.context_name(),
            )),
        }
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.add(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => {
                Ok(Self::Integer(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::Residue(left), Self::Residue(right)) => {
                Ok(Self::Residue(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.matmul(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix multiplication",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(value.neg().map_err(linalg_error)?),
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::Residue(value) => Self::Residue(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match self {
            Self::Integer(value) => Self::Integer(
                value
                    .scalar_mul(&extract_integer_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Rational(value) => Self::Rational(
                value
                    .scalar_mul(&extract_rational_polynomial(py, value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::Residue(value) => Self::Residue(
                value
                    .scalar_mul(&extract_residue_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            Self::FiniteField(value) => Self::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_polynomial(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
        })
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::Residue(left), Self::Residue(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl QuotientSparseMatrixInner {
    fn context_name(&self) -> String {
        match self {
            Self::Rational(value) => bounded_coefficient(value.ring().to_string()),
            Self::FiniteField(value) => bounded_coefficient(value.ring().to_string()),
        }
    }

    fn to_json(&self) -> Result<String, noether::NoetherError> {
        match self {
            Self::Rational(value) => value.to_json(),
            Self::FiniteField(value) => value.to_json(),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Rational(value) => value.shape(),
            Self::FiniteField(value) => value.shape(),
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Self::Rational(value) => value.nnz(),
            Self::FiniteField(value) => value.nnz(),
        }
    }

    fn row_offsets(&self) -> Vec<usize> {
        match self {
            Self::Rational(value) => value.row_offsets().to_vec(),
            Self::FiniteField(value) => value.row_offsets().to_vec(),
        }
    }

    fn column_indices(&self) -> Vec<usize> {
        match self {
            Self::Rational(value) => value.column_indices().to_vec(),
            Self::FiniteField(value) => value.column_indices().to_vec(),
        }
    }

    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Rational(value) => quotient_values_to_python(
                value.values().iter().cloned(),
                py,
                PyQuotientElement::from_rational,
            ),
            Self::FiniteField(value) => quotient_values_to_python(
                value.values().iter().cloned(),
                py,
                PyQuotientElement::from_finite_field,
            ),
        }
    }

    fn to_dense(&self) -> QuotientMatrixInner {
        match self {
            Self::Rational(value) => QuotientMatrixInner::Rational(value.to_dense()),
            Self::FiniteField(value) => QuotientMatrixInner::FiniteField(value.to_dense()),
        }
    }

    fn transpose(&self) -> Self {
        match self {
            Self::Rational(value) => Self::Rational(value.transpose()),
            Self::FiniteField(value) => Self::FiniteField(value.transpose()),
        }
    }

    fn ring_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ring = match self {
            Self::Rational(value) => PyQuotientRing::from_rational(value.ring().clone()),
            Self::FiniteField(value) => PyQuotientRing::from_finite_field(value.ring().clone()),
        };
        Ok(Py::new(py, ring)?.into_any())
    }

    fn get_to_python(&self, py: Python<'_>, row: usize, column: usize) -> PyResult<Py<PyAny>> {
        let value = match self {
            Self::Rational(matrix) => {
                PyQuotientElement::from_rational(matrix.get(row, column).map_err(index_error)?)
            }
            Self::FiniteField(matrix) => {
                PyQuotientElement::from_finite_field(matrix.get(row, column).map_err(index_error)?)
            }
        };
        Ok(Py::new(py, value)?.into_any())
    }

    fn apply(&self, vector: &QuotientVectorInner) -> PyResult<QuotientVectorInner> {
        match (self, vector) {
            (Self::Rational(matrix), QuotientVectorInner::Rational(vector)) => Ok(
                QuotientVectorInner::Rational(matrix.apply(vector).map_err(linalg_error)?),
            ),
            (Self::FiniteField(matrix), QuotientVectorInner::FiniteField(vector)) => Ok(
                QuotientVectorInner::FiniteField(matrix.apply(vector).map_err(linalg_error)?),
            ),
            _ => Err(context_mismatch(
                "sparse matrix application",
                self.context_name(),
                vector.context_name(),
            )),
        }
    }

    fn apply_block(&self, block: &QuotientMatrixInner) -> PyResult<QuotientMatrixInner> {
        match (self, block) {
            (Self::Rational(matrix), QuotientMatrixInner::Rational(block)) => Ok(
                QuotientMatrixInner::Rational(matrix.apply_block(block).map_err(linalg_error)?),
            ),
            (Self::FiniteField(matrix), QuotientMatrixInner::FiniteField(block)) => Ok(
                QuotientMatrixInner::FiniteField(matrix.apply_block(block).map_err(linalg_error)?),
            ),
            _ => Err(context_mismatch(
                "sparse block application",
                self.context_name(),
                block.context_name(),
            )),
        }
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.add(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.add(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix addition",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.sub(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.sub(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix subtraction",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => {
                Ok(Self::Rational(left.matmul(right).map_err(linalg_error)?))
            }
            (Self::FiniteField(left), Self::FiniteField(right)) => {
                Ok(Self::FiniteField(left.matmul(right).map_err(linalg_error)?))
            }
            _ => Err(context_mismatch(
                "sparse matrix multiplication",
                self.context_name(),
                rhs.context_name(),
            )),
        }
    }

    fn neg(&self) -> PyResult<Self> {
        Ok(match self {
            Self::Rational(value) => Self::Rational(value.neg().map_err(linalg_error)?),
            Self::FiniteField(value) => Self::FiniteField(value.neg().map_err(linalg_error)?),
        })
    }

    fn scalar_mul(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        match self {
            Self::Rational(value) => Ok(Self::Rational(
                value
                    .scalar_mul(&extract_rational_quotient(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
            Self::FiniteField(value) => Ok(Self::FiniteField(
                value
                    .scalar_mul(&extract_finite_field_quotient(value.ring(), scalar)?)
                    .map_err(linalg_error)?,
            )),
        }
    }

    fn equals(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Rational(left), Self::Rational(right)) => left == right,
            (Self::FiniteField(left), Self::FiniteField(right)) => left == right,
            _ => false,
        }
    }
}

impl SparseMatrixInner {
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
            Self::PadicLocalFieldOrder(_) => PadicLocalFieldOrder::TYPE_ID,
            Self::RealBall(_) => RealBallField::TYPE_ID,
            Self::ComplexBall(_) => ComplexBallField::TYPE_ID,
            Self::Polynomial(_) => <PolynomialRing<IntegerRing>>::TYPE_ID,
            Self::Quotient(_) => QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID,
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
            Self::PadicLocalFieldOrder(value) => value.to_json(),
            Self::RealBall(value) => value.to_json(),
            Self::ComplexBall(value) => value.to_json(),
            Self::Polynomial(value) => value.to_json(),
            Self::Quotient(value) => value.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        macro_rules! decode {
            ($variant:ident, $ring:ty) => {
                RustSparseMatrix::<$ring>::from_json(value)
                    .map(Self::$variant)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            };
        }
        let (ring_type, coefficient_type) = serialized_linear_algebra_ring_types(value)?;
        match ring_type.as_str() {
            IntegerRing::TYPE_ID => decode!(Integer, IntegerRing),
            RationalField::TYPE_ID => decode!(Rational, RationalField),
            ResidueRing::TYPE_ID => decode!(Residue, ResidueRing),
            FiniteField::TYPE_ID => decode!(FiniteField, FiniteField),
            PadicRing::TYPE_ID => decode!(Padic, PadicRing),
            PadicField::TYPE_ID => decode!(PadicField, PadicField),
            QadicRing::TYPE_ID => decode!(Qadic, QadicRing),
            QadicField::TYPE_ID => decode!(QadicField, QadicField),
            PadicLocalField::TYPE_ID => decode!(PadicLocalField, PadicLocalField),
            PadicLocalFieldOrder::TYPE_ID => {
                decode!(PadicLocalFieldOrder, PadicLocalFieldOrder)
            }
            RealBallField::TYPE_ID => decode!(RealBall, RealBallField),
            ComplexBallField::TYPE_ID => decode!(ComplexBall, ComplexBallField),
            <PolynomialRing<IntegerRing>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(IntegerRing::TYPE_ID) => PolynomialSparseMatrixInner::Integer(
                        RustSparseMatrix::<PolynomialRing<IntegerRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(RationalField::TYPE_ID) => PolynomialSparseMatrixInner::Rational(
                        RustSparseMatrix::<PolynomialRing<RationalField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(ResidueRing::TYPE_ID) => PolynomialSparseMatrixInner::Residue(
                        RustSparseMatrix::<PolynomialRing<ResidueRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => PolynomialSparseMatrixInner::FiniteField(
                        RustSparseMatrix::<PolynomialRing<FiniteField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported sparse polynomial-matrix coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized sparse polynomial matrix has no coefficient context",
                        ));
                    }
                };
                Ok(Self::Polynomial(inner))
            }
            QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(RationalField::TYPE_ID) => QuotientSparseMatrixInner::Rational(
                        RustSparseMatrix::<QuotientRing<PolynomialRing<RationalField>>>::from_json(
                            value,
                        )
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => QuotientSparseMatrixInner::FiniteField(
                        RustSparseMatrix::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(
                            value,
                        )
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported sparse quotient-matrix coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized sparse quotient matrix has no ambient coefficient context",
                        ));
                    }
                };
                Ok(Self::Quotient(inner))
            }
            type_id => Err(PyValueError::new_err(format!(
                "unsupported sparse-matrix coefficient context {type_id:?}"
            ))),
        }
    }
}

impl MatrixInner {
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
            Self::PadicLocalFieldOrder(_) => PadicLocalFieldOrder::TYPE_ID,
            Self::NumberField(_) => NumberField::TYPE_ID,
            Self::RationalFunction(_) => RationalFunctionField::<RationalField>::TYPE_ID,
            Self::NumberFieldOrder(_) => NumberFieldOrder::TYPE_ID,
            Self::RealBall(_) => RealBallField::TYPE_ID,
            Self::ComplexBall(_) => ComplexBallField::TYPE_ID,
            Self::SeriesInteger(_)
            | Self::SeriesRational(_)
            | Self::SeriesResidue(_)
            | Self::SeriesFiniteField(_)
            | Self::SeriesPadic(_)
            | Self::SeriesPadicField(_)
            | Self::SeriesQadic(_)
            | Self::SeriesQadicField(_)
            | Self::SeriesPadicLocalField(_) => <PowerSeriesRing<IntegerRing>>::TYPE_ID,
            Self::Polynomial(_) => <PolynomialRing<IntegerRing>>::TYPE_ID,
            Self::Quotient(_) => QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID,
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
            Self::PadicLocalFieldOrder(value) => value.to_json(),
            Self::NumberField(value) => value.to_json(),
            Self::RationalFunction(value) => value.to_json(),
            Self::NumberFieldOrder(value) => value.to_json(),
            Self::RealBall(value) => value.to_json(),
            Self::ComplexBall(value) => value.to_json(),
            Self::SeriesInteger(value) => value.to_json(),
            Self::SeriesRational(value) => value.to_json(),
            Self::SeriesResidue(value) => value.to_json(),
            Self::SeriesFiniteField(value) => value.to_json(),
            Self::SeriesPadic(value) => value.to_json(),
            Self::SeriesPadicField(value) => value.to_json(),
            Self::SeriesQadic(value) => value.to_json(),
            Self::SeriesQadicField(value) => value.to_json(),
            Self::SeriesPadicLocalField(value) => value.to_json(),
            Self::Polynomial(value) => value.to_json(),
            Self::Quotient(value) => value.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        macro_rules! decode {
            ($variant:ident, $ring:ty) => {
                Matrix::<$ring>::from_json(value)
                    .map(Self::$variant)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            };
        }
        let (ring_type, coefficient_type) = serialized_linear_algebra_ring_types(value)?;
        match ring_type.as_str() {
            IntegerRing::TYPE_ID => decode!(Integer, IntegerRing),
            RationalField::TYPE_ID => decode!(Rational, RationalField),
            ResidueRing::TYPE_ID => decode!(Residue, ResidueRing),
            FiniteField::TYPE_ID => decode!(FiniteField, FiniteField),
            PadicRing::TYPE_ID => decode!(Padic, PadicRing),
            PadicField::TYPE_ID => decode!(PadicField, PadicField),
            QadicRing::TYPE_ID => decode!(Qadic, QadicRing),
            QadicField::TYPE_ID => decode!(QadicField, QadicField),
            PadicLocalField::TYPE_ID => decode!(PadicLocalField, PadicLocalField),
            PadicLocalFieldOrder::TYPE_ID => {
                decode!(PadicLocalFieldOrder, PadicLocalFieldOrder)
            }
            NumberField::TYPE_ID => decode!(NumberField, NumberField),
            RationalFunctionField::<RationalField>::TYPE_ID
                if coefficient_type.as_deref() == Some(RationalField::TYPE_ID) =>
            {
                decode!(RationalFunction, RationalFunctionField<RationalField>)
            }
            NumberFieldOrder::TYPE_ID => decode!(NumberFieldOrder, NumberFieldOrder),
            RealBallField::TYPE_ID => decode!(RealBall, RealBallField),
            ComplexBallField::TYPE_ID => decode!(ComplexBall, ComplexBallField),
            <PolynomialRing<IntegerRing>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(IntegerRing::TYPE_ID) => PolynomialMatrixInner::Integer(
                        Matrix::<PolynomialRing<IntegerRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(RationalField::TYPE_ID) => PolynomialMatrixInner::Rational(
                        Matrix::<PolynomialRing<RationalField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(ResidueRing::TYPE_ID) => PolynomialMatrixInner::Residue(
                        Matrix::<PolynomialRing<ResidueRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => PolynomialMatrixInner::FiniteField(
                        Matrix::<PolynomialRing<FiniteField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported polynomial-matrix coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized polynomial matrix has no coefficient context",
                        ));
                    }
                };
                Ok(Self::Polynomial(inner))
            }
            QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(RationalField::TYPE_ID) => QuotientMatrixInner::Rational(
                        Matrix::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => QuotientMatrixInner::FiniteField(
                        Matrix::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported quotient-matrix coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized quotient matrix has no ambient coefficient context",
                        ));
                    }
                };
                Ok(Self::Quotient(inner))
            }
            <PowerSeriesRing<IntegerRing>>::TYPE_ID => match coefficient_type.as_deref() {
                Some(IntegerRing::TYPE_ID) => {
                    decode!(SeriesInteger, PowerSeriesRing<IntegerRing>)
                }
                Some(RationalField::TYPE_ID) => {
                    decode!(SeriesRational, PowerSeriesRing<RationalField>)
                }
                Some(ResidueRing::TYPE_ID) => {
                    decode!(SeriesResidue, PowerSeriesRing<ResidueRing>)
                }
                Some(FiniteField::TYPE_ID) => {
                    decode!(SeriesFiniteField, PowerSeriesRing<FiniteField>)
                }
                Some(PadicRing::TYPE_ID) => decode!(SeriesPadic, PowerSeriesRing<PadicRing>),
                Some(PadicField::TYPE_ID) => {
                    decode!(SeriesPadicField, PowerSeriesRing<PadicField>)
                }
                Some(QadicRing::TYPE_ID) => decode!(SeriesQadic, PowerSeriesRing<QadicRing>),
                Some(QadicField::TYPE_ID) => {
                    decode!(SeriesQadicField, PowerSeriesRing<QadicField>)
                }
                Some(PadicLocalField::TYPE_ID) => {
                    decode!(SeriesPadicLocalField, PowerSeriesRing<PadicLocalField>)
                }
                Some(type_id) => Err(PyValueError::new_err(format!(
                    "unsupported matrix power-series coefficient context {type_id:?}"
                ))),
                None => Err(PyValueError::new_err(
                    "serialized matrix power-series ring has no coefficient context",
                )),
            },
            type_id => Err(PyValueError::new_err(format!(
                "unsupported matrix coefficient context {type_id:?}"
            ))),
        }
    }
}

impl VectorInner {
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
            Self::PadicLocalFieldOrder(value) => value.to_json(),
            Self::NumberField(value) => value.to_json(),
            Self::RationalFunction(value) => value.to_json(),
            Self::NumberFieldOrder(value) => value.to_json(),
            Self::RealBall(value) => value.to_json(),
            Self::ComplexBall(value) => value.to_json(),
            Self::SeriesInteger(value) => value.to_json(),
            Self::SeriesRational(value) => value.to_json(),
            Self::SeriesResidue(value) => value.to_json(),
            Self::SeriesFiniteField(value) => value.to_json(),
            Self::SeriesPadic(value) => value.to_json(),
            Self::SeriesPadicField(value) => value.to_json(),
            Self::SeriesQadic(value) => value.to_json(),
            Self::SeriesQadicField(value) => value.to_json(),
            Self::SeriesPadicLocalField(value) => value.to_json(),
            Self::Polynomial(value) => value.to_json(),
            Self::Quotient(value) => value.to_json(),
        }
    }

    fn from_json(value: &str) -> PyResult<Self> {
        macro_rules! decode {
            ($variant:ident, $ring:ty) => {
                Vector::<$ring>::from_json(value)
                    .map(Self::$variant)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            };
        }
        let (ring_type, coefficient_type) = serialized_linear_algebra_ring_types(value)?;
        match ring_type.as_str() {
            IntegerRing::TYPE_ID => decode!(Integer, IntegerRing),
            RationalField::TYPE_ID => decode!(Rational, RationalField),
            ResidueRing::TYPE_ID => decode!(Residue, ResidueRing),
            FiniteField::TYPE_ID => decode!(FiniteField, FiniteField),
            PadicRing::TYPE_ID => decode!(Padic, PadicRing),
            PadicField::TYPE_ID => decode!(PadicField, PadicField),
            QadicRing::TYPE_ID => decode!(Qadic, QadicRing),
            QadicField::TYPE_ID => decode!(QadicField, QadicField),
            PadicLocalField::TYPE_ID => decode!(PadicLocalField, PadicLocalField),
            PadicLocalFieldOrder::TYPE_ID => {
                decode!(PadicLocalFieldOrder, PadicLocalFieldOrder)
            }
            NumberField::TYPE_ID => decode!(NumberField, NumberField),
            RationalFunctionField::<RationalField>::TYPE_ID
                if coefficient_type.as_deref() == Some(RationalField::TYPE_ID) =>
            {
                decode!(RationalFunction, RationalFunctionField<RationalField>)
            }
            NumberFieldOrder::TYPE_ID => decode!(NumberFieldOrder, NumberFieldOrder),
            RealBallField::TYPE_ID => decode!(RealBall, RealBallField),
            ComplexBallField::TYPE_ID => decode!(ComplexBall, ComplexBallField),
            <PolynomialRing<IntegerRing>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(IntegerRing::TYPE_ID) => PolynomialVectorInner::Integer(
                        Vector::<PolynomialRing<IntegerRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(RationalField::TYPE_ID) => PolynomialVectorInner::Rational(
                        Vector::<PolynomialRing<RationalField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(ResidueRing::TYPE_ID) => PolynomialVectorInner::Residue(
                        Vector::<PolynomialRing<ResidueRing>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => PolynomialVectorInner::FiniteField(
                        Vector::<PolynomialRing<FiniteField>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported polynomial-vector coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized polynomial vector has no coefficient context",
                        ));
                    }
                };
                Ok(Self::Polynomial(inner))
            }
            QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID => {
                let inner = match coefficient_type.as_deref() {
                    Some(RationalField::TYPE_ID) => QuotientVectorInner::Rational(
                        Vector::<QuotientRing<PolynomialRing<RationalField>>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(FiniteField::TYPE_ID) => QuotientVectorInner::FiniteField(
                        Vector::<QuotientRing<PolynomialRing<FiniteField>>>::from_json(value)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?,
                    ),
                    Some(type_id) => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported quotient-vector coefficient context {type_id:?}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "serialized quotient vector has no ambient coefficient context",
                        ));
                    }
                };
                Ok(Self::Quotient(inner))
            }
            <PowerSeriesRing<IntegerRing>>::TYPE_ID => match coefficient_type.as_deref() {
                Some(IntegerRing::TYPE_ID) => {
                    decode!(SeriesInteger, PowerSeriesRing<IntegerRing>)
                }
                Some(RationalField::TYPE_ID) => {
                    decode!(SeriesRational, PowerSeriesRing<RationalField>)
                }
                Some(ResidueRing::TYPE_ID) => {
                    decode!(SeriesResidue, PowerSeriesRing<ResidueRing>)
                }
                Some(FiniteField::TYPE_ID) => {
                    decode!(SeriesFiniteField, PowerSeriesRing<FiniteField>)
                }
                Some(PadicRing::TYPE_ID) => decode!(SeriesPadic, PowerSeriesRing<PadicRing>),
                Some(PadicField::TYPE_ID) => {
                    decode!(SeriesPadicField, PowerSeriesRing<PadicField>)
                }
                Some(QadicRing::TYPE_ID) => decode!(SeriesQadic, PowerSeriesRing<QadicRing>),
                Some(QadicField::TYPE_ID) => {
                    decode!(SeriesQadicField, PowerSeriesRing<QadicField>)
                }
                Some(PadicLocalField::TYPE_ID) => {
                    decode!(SeriesPadicLocalField, PowerSeriesRing<PadicLocalField>)
                }
                Some(type_id) => Err(PyValueError::new_err(format!(
                    "unsupported vector power-series coefficient context {type_id:?}"
                ))),
                None => Err(PyValueError::new_err(
                    "serialized vector power-series ring has no coefficient context",
                )),
            },
            type_id => Err(PyValueError::new_err(format!(
                "unsupported vector coefficient context {type_id:?}"
            ))),
        }
    }
}

#[derive(Clone)]
pub(super) enum SolutionSpaceInner {
    Rational(RustSolutionSpace<RationalField>),
    FiniteField(RustSolutionSpace<FiniteField>),
    NumberField(RustSolutionSpace<NumberField>),
    RationalFunction(RustSolutionSpace<RationalFunctionField<RationalField>>),
}

#[derive(Clone)]
enum PLUQFactorizationInner {
    Rational(RustPLUQFactorization<RationalField>),
    FiniteField(RustPLUQFactorization<FiniteField>),
    NumberField(RustPLUQFactorization<NumberField>),
    RationalFunction(RustPLUQFactorization<RationalFunctionField<RationalField>>),
}

/// A checked, rank-revealing `P @ L @ U @ Q == source` factorization.
#[pyclass(
    module = "noether.linalg",
    name = "PLUQFactorization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPLUQFactorization {
    inner: PLUQFactorizationInner,
}

#[pymethods]
impl PyPLUQFactorization {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustPLUQFactorization::<RationalField>::TYPE_ID
    }

    /// Coefficient field.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PLUQFactorizationInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            PLUQFactorizationInner::FiniteField(value) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: value.ring().clone(),
                },
            )?
            .into_any()),
            PLUQFactorizationInner::NumberField(value) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: value.ring().clone(),
                },
            )?
            .into_any()),
            PLUQFactorizationInner::RationalFunction(value) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(value.ring().clone()),
            )?
            .into_any()),
        }
    }

    /// Matrix being factored.
    #[getter]
    fn source(&self) -> PyMatrix {
        self.component(PLUQComponent::Source)
    }

    /// Left permutation matrix.
    #[getter]
    fn p(&self) -> PyMatrix {
        self.component(PLUQComponent::P)
    }

    /// Unit lower-trapezoidal factor.
    #[getter]
    fn l(&self) -> PyMatrix {
        self.component(PLUQComponent::L)
    }

    /// Upper-trapezoidal factor.
    #[getter]
    fn u(&self) -> PyMatrix {
        self.component(PLUQComponent::U)
    }

    /// Right permutation matrix.
    #[getter]
    fn q(&self) -> PyMatrix {
        self.component(PLUQComponent::Q)
    }

    /// Exact matrix rank.
    #[getter]
    fn rank(&self) -> usize {
        match &self.inner {
            PLUQFactorizationInner::Rational(value) => value.rank(),
            PLUQFactorizationInner::FiniteField(value) => value.rank(),
            PLUQFactorizationInner::NumberField(value) => value.rank(),
            PLUQFactorizationInner::RationalFunction(value) => value.rank(),
        }
    }

    /// Recheck factor structure, rank, and the reconstruction identity.
    fn verify(&self) -> PyResult<()> {
        match &self.inner {
            PLUQFactorizationInner::Rational(value) => value.verify(),
            PLUQFactorizationInner::FiniteField(value) => value.verify(),
            PLUQFactorizationInner::NumberField(value) => value.verify(),
            PLUQFactorizationInner::RationalFunction(value) => value.verify(),
        }
        .map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            let shape = self.component(PLUQComponent::Source).dimensions();
            format!(
                "PLUQFactorization(shape={}x{}, rank={})",
                shape.0,
                shape.1,
                self.rank()
            )
        })
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let shape = self.component(PLUQComponent::Source).dimensions();
        let summary = PyDict::new(py);
        summary.set_item("type", Self::type_id())?;
        summary.set_item("shape", shape)?;
        summary.set_item("rank", self.rank())?;
        Ok(summary.into_any().unbind())
    }
}

#[derive(Clone, Copy)]
enum PLUQComponent {
    Source,
    P,
    L,
    U,
    Q,
}

impl PyPLUQFactorization {
    fn component(&self, component: PLUQComponent) -> PyMatrix {
        macro_rules! select {
            ($value:expr) => {
                match component {
                    PLUQComponent::Source => $value.source().clone(),
                    PLUQComponent::P => $value.p().clone(),
                    PLUQComponent::L => $value.l().clone(),
                    PLUQComponent::U => $value.u().clone(),
                    PLUQComponent::Q => $value.q().clone(),
                }
            };
        }
        let inner = match &self.inner {
            PLUQFactorizationInner::Rational(value) => MatrixInner::Rational(select!(value)),
            PLUQFactorizationInner::FiniteField(value) => MatrixInner::FiniteField(select!(value)),
            PLUQFactorizationInner::NumberField(value) => MatrixInner::NumberField(select!(value)),
            PLUQFactorizationInner::RationalFunction(value) => {
                MatrixInner::RationalFunction(select!(value))
            }
        };
        PyMatrix { inner }
    }
}

/// A checked affine solution set over one field context.
#[pyclass(
    module = "noether.linalg",
    name = "SolutionSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySolutionSpace {
    pub(super) inner: SolutionSpaceInner,
}

#[pymethods]
impl PySolutionSpace {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustSolutionSpace::<RationalField>::TYPE_ID
    }

    /// Coefficient field.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SolutionSpaceInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            SolutionSpaceInner::FiniteField(solution) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: solution.ring().clone(),
                },
            )?
            .into_any()),
            SolutionSpaceInner::NumberField(solution) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: solution.ring().clone(),
                },
            )?
            .into_any()),
            SolutionSpaceInner::RationalFunction(solution) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(solution.ring().clone()),
            )?
            .into_any()),
        }
    }

    /// One point in the affine solution set.
    #[getter]
    fn particular(&self) -> PyVector {
        let inner = match &self.inner {
            SolutionSpaceInner::Rational(solution) => {
                VectorInner::Rational(solution.particular().clone())
            }
            SolutionSpaceInner::FiniteField(solution) => {
                VectorInner::FiniteField(solution.particular().clone())
            }
            SolutionSpaceInner::NumberField(solution) => {
                VectorInner::NumberField(solution.particular().clone())
            }
            SolutionSpaceInner::RationalFunction(solution) => {
                VectorInner::RationalFunction(solution.particular().clone())
            }
        };
        PyVector { inner }
    }

    /// Deterministically ordered basis of the translation subspace.
    #[getter]
    fn basis(&self) -> Vec<PyVector> {
        match &self.inner {
            SolutionSpaceInner::Rational(solution) => solution
                .basis()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::Rational(vector),
                })
                .collect(),
            SolutionSpaceInner::FiniteField(solution) => solution
                .basis()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::FiniteField(vector),
                })
                .collect(),
            SolutionSpaceInner::NumberField(solution) => solution
                .basis()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::NumberField(vector),
                })
                .collect(),
            SolutionSpaceInner::RationalFunction(solution) => solution
                .basis()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::RationalFunction(vector),
                })
                .collect(),
        }
    }

    /// Dimension of the translation subspace.
    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            SolutionSpaceInner::Rational(solution) => solution.dimension(),
            SolutionSpaceInner::FiniteField(solution) => solution.dimension(),
            SolutionSpaceInner::NumberField(solution) => solution.dimension(),
            SolutionSpaceInner::RationalFunction(solution) => solution.dimension(),
        }
    }

    /// Dimension of the ambient vector space.
    #[getter]
    fn ambient_dimension(&self) -> usize {
        match &self.inner {
            SolutionSpaceInner::Rational(solution) => solution.ambient_dimension(),
            SolutionSpaceInner::FiniteField(solution) => solution.ambient_dimension(),
            SolutionSpaceInner::NumberField(solution) => solution.ambient_dimension(),
            SolutionSpaceInner::RationalFunction(solution) => solution.ambient_dimension(),
        }
    }

    /// Recheck the defining equations, completeness, and independence.
    fn verify(&self) -> PyResult<()> {
        let result = match &self.inner {
            SolutionSpaceInner::Rational(solution) => solution.verify(),
            SolutionSpaceInner::FiniteField(solution) => solution.verify(),
            SolutionSpaceInner::NumberField(solution) => solution.verify(),
            SolutionSpaceInner::RationalFunction(solution) => solution.verify(),
        };
        result.map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SolutionSpace(ambient_dimension={}, dimension={})",
                self.ambient_dimension(),
                self.dimension()
            )
        })
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustSolutionSpace::<RationalField>::TYPE_ID)?;
        summary.set_item("ambient_dimension", self.ambient_dimension())?;
        summary.set_item("dimension", self.dimension())?;
        Ok(summary.unbind())
    }
}

/// A nonempty affine lattice of integer solutions.
#[pyclass(
    module = "noether.linalg",
    name = "IntegerSolutionModule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegerSolutionModule {
    inner: RustIntegerSolutionModule,
}

#[pymethods]
impl PyIntegerSolutionModule {
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegerSolutionModule::TYPE_ID
    }
    #[getter]
    fn operator(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.operator().clone()),
        }
    }
    #[getter]
    fn rhs(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Integer(self.inner.rhs().clone()),
        }
    }
    #[getter]
    fn particular_solution(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Integer(self.inner.particular_solution().clone()),
        }
    }
    #[getter]
    fn homogeneous_generators(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.homogeneous_generators().clone()),
        }
    }
    #[getter]
    fn homogeneous_rank(&self) -> usize {
        self.inner.homogeneous_rank()
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "IntegerSolutionModule(variables={}, homogeneous_rank={})",
            self.inner.operator().shape().1,
            self.inner.homogeneous_rank()
        ))
    }
}

/// A replayable divisibility obstruction to an integer system.
#[pyclass(
    module = "noether.linalg",
    name = "IntegerInconsistencyCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegerInconsistencyCertificate {
    inner: RustIntegerInconsistencyCertificate,
}

#[pymethods]
impl PyIntegerInconsistencyCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegerInconsistencyCertificate::TYPE_ID
    }
    #[getter]
    fn operator(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.operator().clone()),
        }
    }
    #[getter]
    fn rhs(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Integer(self.inner.rhs().clone()),
        }
    }
    #[getter]
    fn row_combination(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Integer(self.inner.row_combination().clone()),
        }
    }
    #[getter]
    fn left_gcd(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = py.detach(|| self.inner.left_gcd());
        integer_to_python(py, &value)
    }
    #[getter]
    fn combined_rhs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = py.detach(|| self.inner.combined_rhs());
        integer_to_python(py, &value)
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "IntegerInconsistencyCertificate(equations={}, variables={})",
            self.inner.operator().shape().0,
            self.inner.operator().shape().1
        ))
    }
}

/// Exact consistent-or-obstructed result for an integer linear system.
#[pyclass(
    module = "noether.linalg",
    name = "IntegerLinearSystemResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyIntegerLinearSystemResult {
    inner: RustIntegerLinearSystemResult,
}

#[pymethods]
impl PyIntegerLinearSystemResult {
    #[classattr]
    fn type_id() -> &'static str {
        RustIntegerLinearSystemResult::TYPE_ID
    }
    #[getter]
    fn is_consistent(&self) -> bool {
        self.inner.is_consistent()
    }
    #[getter]
    fn solutions(&self) -> Option<PyIntegerSolutionModule> {
        self.inner
            .solutions()
            .cloned()
            .map(|inner| PyIntegerSolutionModule { inner })
    }
    #[getter]
    fn obstruction(&self) -> Option<PyIntegerInconsistencyCertificate> {
        self.inner
            .obstruction()
            .cloned()
            .map(|inner| PyIntegerInconsistencyCertificate { inner })
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .to_json()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: RustIntegerLinearSystemResult::from_json(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            })
        })
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(if self.inner.is_consistent() {
            "IntegerLinearSystemResult(consistent)".to_owned()
        } else {
            "IntegerLinearSystemResult(inconsistent)".to_owned()
        })
    }
}

/// A nonempty affine solution module over an arbitrary residue ring.
#[pyclass(
    module = "noether.linalg",
    name = "CongruenceSolutionModule",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCongruenceSolutionModule {
    inner: RustCongruenceSolutionModule,
}

#[pymethods]
impl PyCongruenceSolutionModule {
    #[classattr]
    fn type_id() -> &'static str {
        RustCongruenceSolutionModule::TYPE_ID
    }
    #[getter]
    fn operator(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Residue(self.inner.operator().clone()),
        }
    }
    #[getter]
    fn rhs(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Residue(self.inner.rhs().clone()),
        }
    }
    #[getter]
    fn particular_solution(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Residue(self.inner.particular_solution().clone()),
        }
    }
    #[getter]
    fn homogeneous_generators(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Residue(self.inner.homogeneous_generators().clone()),
        }
    }
    #[getter]
    fn homogeneous_generator_count(&self) -> usize {
        self.inner.homogeneous_generator_count()
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CongruenceSolutionModule(variables={}, homogeneous_generators={})",
            self.inner.operator().shape().1,
            self.inner.homogeneous_generator_count()
        ))
    }
}

/// A replayable obstruction to a congruence system, including composite moduli.
#[pyclass(
    module = "noether.linalg",
    name = "CongruenceInconsistencyCertificate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCongruenceInconsistencyCertificate {
    inner: RustCongruenceInconsistencyCertificate,
}

#[pymethods]
impl PyCongruenceInconsistencyCertificate {
    #[classattr]
    fn type_id() -> &'static str {
        RustCongruenceInconsistencyCertificate::TYPE_ID
    }
    #[getter]
    fn operator(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Residue(self.inner.operator().clone()),
        }
    }
    #[getter]
    fn rhs(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Residue(self.inner.rhs().clone()),
        }
    }
    #[getter]
    fn row_combination(&self) -> PyVector {
        PyVector {
            inner: VectorInner::Integer(self.inner.row_combination().clone()),
        }
    }
    #[getter]
    fn left_gcd_with_modulus(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = py.detach(|| self.inner.left_gcd_with_modulus());
        integer_to_python(py, &value)
    }
    #[getter]
    fn combined_rhs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = py.detach(|| self.inner.combined_rhs());
        integer_to_python(py, &value)
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "CongruenceInconsistencyCertificate(equations={}, variables={})",
            self.inner.operator().shape().0,
            self.inner.operator().shape().1
        ))
    }
}

/// Exact consistent-or-obstructed result for a system over `Z/nZ`.
#[pyclass(
    module = "noether.linalg",
    name = "CongruenceSystemResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCongruenceSystemResult {
    inner: RustCongruenceSystemResult,
}

#[pymethods]
impl PyCongruenceSystemResult {
    #[classattr]
    fn type_id() -> &'static str {
        RustCongruenceSystemResult::TYPE_ID
    }
    #[getter]
    fn is_consistent(&self) -> bool {
        self.inner.is_consistent()
    }
    #[getter]
    fn solutions(&self) -> Option<PyCongruenceSolutionModule> {
        self.inner
            .solutions()
            .cloned()
            .map(|inner| PyCongruenceSolutionModule { inner })
    }
    #[getter]
    fn obstruction(&self) -> Option<PyCongruenceInconsistencyCertificate> {
        self.inner
            .obstruction()
            .cloned()
            .map(|inner| PyCongruenceInconsistencyCertificate { inner })
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .to_json()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: RustCongruenceSystemResult::from_json(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            })
        })
    }
    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(if self.inner.is_consistent() {
            "CongruenceSystemResult(consistent)".to_owned()
        } else {
            "CongruenceSystemResult(inconsistent)".to_owned()
        })
    }
}

/// Smith valuations visible below a fixed p-adic precision cap.
#[pyclass(
    module = "noether.linalg",
    name = "PadicSmithInvariants",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicSmithInvariants {
    inner: RustPadicSmithInvariants,
}

#[pymethods]
impl PyPadicSmithInvariants {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustPadicSmithInvariants::TYPE_ID
    }

    /// Residue characteristic.
    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    /// Absolute p-adic precision cap.
    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision()
    }

    /// Source matrix shape.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Exact valuations of Smith factors visible below the cap.
    #[getter]
    fn exact_exponents(&self) -> Vec<u64> {
        self.inner.exact_exponents().to_vec()
    }

    /// Number of diagonal positions unresolved at the precision cap.
    #[getter]
    fn unresolved_pivots(&self) -> usize {
        self.inner.unresolved_pivots()
    }

    /// Certified lower rank bound for every compatible `Z_p` lift.
    #[getter]
    fn rank_lower_bound(&self) -> usize {
        self.inner.rank_lower_bound()
    }

    /// Upper rank bound for every compatible `Z_p` lift.
    #[getter]
    fn rank_upper_bound(&self) -> usize {
        self.inner.rank_upper_bound()
    }

    /// Recheck the result's internal precision and rank invariants.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }

    /// Recompute and compare against one dense or sparse p-adic source.
    fn verify_source(&self, py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(source) = source.extract::<PyRef<'_, PyMatrix>>() {
            return match &source.inner {
                MatrixInner::Padic(matrix) => {
                    let matrix = matrix.clone();
                    py.detach(|| {
                        self.inner
                            .verify_dense_source(&matrix)
                            .map_err(super::errors::verification)
                    })
                }
                _ => Err(PyTypeError::new_err(
                    "PadicSmithInvariants.verify_source requires a matrix over PadicRing",
                )),
            };
        }
        if let Ok(source) = source.extract::<PyRef<'_, PySparseMatrix>>() {
            return match &source.inner {
                SparseMatrixInner::Padic(matrix) => {
                    let matrix = matrix.clone();
                    py.detach(|| {
                        self.inner
                            .verify_sparse_source(&matrix)
                            .map_err(super::errors::verification)
                    })
                }
                _ => Err(PyTypeError::new_err(
                    "PadicSmithInvariants.verify_source requires a matrix over PadicRing",
                )),
            };
        }
        Err(PyTypeError::new_err(
            "PadicSmithInvariants.verify_source requires Matrix or SparseMatrix",
        ))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .to_json()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        py.detach(move || {
            Ok(Self {
                inner: RustPadicSmithInvariants::from_json(&value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        const EXPONENT_LIMIT: usize = 12;
        let summary = PyDict::new(py);
        summary.set_item("type", RustPadicSmithInvariants::TYPE_ID)?;
        summary.set_item("prime", self.prime(py)?)?;
        summary.set_item("precision", self.inner.precision())?;
        summary.set_item("shape", self.inner.shape())?;
        summary.set_item(
            "exact_exponents",
            self.inner
                .exact_exponents()
                .iter()
                .take(EXPONENT_LIMIT)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        summary.set_item(
            "exact_exponents_truncated",
            self.inner.exact_exponents().len() > EXPONENT_LIMIT,
        )?;
        summary.set_item("unresolved_pivots", self.inner.unresolved_pivots())?;
        summary.set_item("rank_lower_bound", self.inner.rank_lower_bound())?;
        summary.set_item("rank_upper_bound", self.inner.rank_upper_bound())?;
        Ok(summary.unbind())
    }
}

/// Strategy-specific evidence for a Smith normal form result.
#[pyclass(
    module = "noether.linalg",
    name = "SmithVerification",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySmithVerification {
    inner: RustSmithVerification,
}

#[pymethods]
impl PySmithVerification {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustSmithVerification::TYPE_ID
    }

    /// Mathematical strategy actually executed.
    #[getter]
    fn strategy(&self) -> &'static str {
        self.inner.strategy()
    }

    /// Inclusive smoothness bound, when local primes were swept.
    #[getter]
    fn smooth_bound(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .smooth_bound()
            .map(|bound| integer_to_python(py, bound))
            .transpose()
    }

    /// Deterministic upper bound on any torsion prime, when computed.
    #[getter]
    fn deterministic_prime_bound(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .deterministic_prime_bound()
            .map(|bound| integer_to_python(py, bound))
            .transpose()
    }

    /// Primes checked by the local strategy, in increasing order.
    #[getter]
    fn checked_primes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .checked_primes()
            .iter()
            .map(|prime| integer_to_python(py, prime))
            .collect()
    }

    /// `(prime, precision exponent, invariant valuations)` check records.
    #[getter]
    fn local_checks(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u32, Vec<u32>)>> {
        self.inner
            .local_checks()
            .map(|(prime, precision, valuations)| {
                Ok((
                    integer_to_python(py, prime)?,
                    precision,
                    valuations.to_vec(),
                ))
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustSmithVerification::TYPE_ID)?;
        summary.set_item("strategy", self.inner.strategy())?;
        summary.set_item("checked_prime_count", self.inner.checked_primes().len())?;
        summary.set_item("smooth_bound", self.smooth_bound(py)?)?;
        summary.set_item(
            "deterministic_prime_bound",
            self.deterministic_prime_bound(py)?,
        )?;
        Ok(summary.unbind())
    }
}

/// Smith normal form of an integer matrix without transformation matrices.
#[pyclass(
    module = "noether.linalg",
    name = "SmithNormalForm",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySmithNormalForm {
    inner: RustSmithNormalForm,
}

#[pymethods]
impl PySmithNormalForm {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustSmithNormalForm::TYPE_ID
    }

    /// Input matrix retained as the verification witness.
    #[getter]
    fn source(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.source().clone()),
        }
    }

    /// Diagonal Smith form with the same shape as the input.
    #[getter]
    fn form(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.form().clone()),
        }
    }

    /// Positive nonzero invariant factors in divisibility-chain order.
    #[getter]
    fn invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .invariant_factors()
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Rank over the integers.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Mathematical assumptions under which the result is valid.
    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        PyMathematicalValidity {
            inner: self.inner.validity().clone(),
        }
    }

    /// Strategy-specific checks supporting the result.
    #[getter]
    fn verification(&self) -> PySmithVerification {
        PySmithVerification {
            inner: self.inner.verification().clone(),
        }
    }

    /// Recheck the Smith conventions and retained input witness.
    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            const FACTOR_LIMIT: usize = 6;
            let mut factors = self
                .inner
                .invariant_factors()
                .iter()
                .take(FACTOR_LIMIT)
                .map(|factor| bounded_coefficient(factor.to_string()))
                .collect::<Vec<_>>();
            if self.inner.invariant_factors().len() > FACTOR_LIMIT {
                factors.push("...".to_owned());
            }
            let shape = self.inner.form().shape();
            format!(
                "SmithNormalForm(shape={}x{}, rank={}, invariants=[{}])",
                shape.0,
                shape.1,
                self.inner.rank(),
                factors.join(", ")
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        const FACTOR_LIMIT: usize = 8;
        let summary = PyDict::new(py);
        let shape = self.inner.form().shape();
        let factors = self
            .inner
            .invariant_factors()
            .iter()
            .take(FACTOR_LIMIT)
            .map(|factor| bounded_coefficient(factor.to_string()))
            .collect::<Vec<_>>();
        summary.set_item("type", RustSmithNormalForm::TYPE_ID)?;
        summary.set_item("shape", shape)?;
        summary.set_item("rank", self.inner.rank())?;
        summary.set_item("invariant_factors", factors)?;
        summary.set_item(
            "invariant_factors_truncated",
            self.inner.invariant_factors().len() > FACTOR_LIMIT,
        )?;
        summary.set_item("validity", self.inner.validity().to_string())?;
        summary.set_item("strategy", self.inner.verification().strategy())?;
        Ok(summary.unbind())
    }
}

/// Smith normal form with explicit left and right unimodular witnesses.
#[pyclass(
    module = "noether.linalg",
    name = "SmithNormalFormWithTransforms",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySmithNormalFormWithTransforms {
    inner: RustSmithNormalFormWithTransforms,
}

#[pymethods]
impl PySmithNormalFormWithTransforms {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustSmithNormalFormWithTransforms::TYPE_ID
    }

    /// The Smith result shared with the fast no-transform endpoint.
    #[getter]
    fn smith(&self) -> PySmithNormalForm {
        PySmithNormalForm {
            inner: self.inner.smith().clone(),
        }
    }

    /// Input matrix retained as the verification witness.
    #[getter]
    fn source(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.smith().source().clone()),
        }
    }

    /// Diagonal Smith form with the same shape as the input.
    #[getter]
    fn form(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.smith().form().clone()),
        }
    }

    /// Positive nonzero invariant factors in divisibility-chain order.
    #[getter]
    fn invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .smith()
            .invariant_factors()
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Rank over the integers.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.smith().rank()
    }

    /// Mathematical assumptions under which the result is valid.
    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        PyMathematicalValidity {
            inner: self.inner.smith().validity().clone(),
        }
    }

    /// Strategy-specific checks supporting the Smith result.
    #[getter]
    fn verification(&self) -> PySmithVerification {
        PySmithVerification {
            inner: self.inner.smith().verification().clone(),
        }
    }

    /// Left unimodular matrix in `left @ source @ right == form`.
    #[getter]
    fn left(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.left().clone()),
        }
    }

    /// Right unimodular matrix in `left @ source @ right == form`.
    #[getter]
    fn right(&self) -> PyMatrix {
        PyMatrix {
            inner: MatrixInner::Integer(self.inner.right().clone()),
        }
    }

    /// Verify canonical form, unimodularity, and the reconstruction identity.
    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            let shape = self.inner.smith().form().shape();
            format!(
                "SmithNormalFormWithTransforms(shape={}x{}, rank={})",
                shape.0,
                shape.1,
                self.inner.smith().rank()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let shape = self.inner.smith().form().shape();
        summary.set_item("type", RustSmithNormalFormWithTransforms::TYPE_ID)?;
        summary.set_item("shape", shape)?;
        summary.set_item("rank", self.inner.smith().rank())?;
        summary.set_item("left_shape", self.inner.left().shape())?;
        summary.set_item("right_shape", self.inner.right().shape())?;
        summary.set_item("validity", self.inner.smith().validity().to_string())?;
        summary.set_item("strategy", self.inner.smith().verification().strategy())?;
        Ok(summary.unbind())
    }
}

fn append_series_matrix_preview<R: CommutativeRing>(
    matrix: &Matrix<PowerSeriesRing<R>>,
    rendered_rows: &mut Vec<String>,
    row_count: usize,
    column_count: usize,
) {
    const ROW_LIMIT: usize = 4;
    const COLUMN_LIMIT: usize = 4;
    for row in 0..row_count.min(ROW_LIMIT) {
        let mut entries = (0..column_count.min(COLUMN_LIMIT))
            .filter_map(|column| matrix.get(row, column).ok())
            .map(|entry| bounded_coefficient(entry.to_string()))
            .collect::<Vec<_>>();
        if column_count > COLUMN_LIMIT {
            entries.push("...".to_owned());
        }
        rendered_rows.push(format!("[{}]", entries.join(", ")));
    }
}

/// A dense matrix over one explicit coefficient context.
#[pyclass(module = "noether.linalg", name = "Matrix", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMatrix {
    pub(super) inner: MatrixInner,
}

impl PyMatrix {
    #[cfg(feature = "distributed")]
    pub(crate) fn finite_field(&self) -> Option<&Matrix<FiniteField>> {
        match &self.inner {
            MatrixInner::FiniteField(matrix) => Some(matrix),
            _ => None,
        }
    }

    #[cfg(feature = "distributed")]
    pub(crate) fn from_finite_field(matrix: Matrix<FiniteField>) -> Self {
        Self {
            inner: MatrixInner::FiniteField(matrix),
        }
    }

    fn rows_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rows = match &self.inner {
            MatrixInner::Integer(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| integer_to_python(py, entry))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::Rational(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| rational_to_python(py, entry))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::Residue(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| Ok(Py::new(py, PyResidueClass { inner: entry })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::FiniteField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyFiniteFieldElement { inner: entry })?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::Padic(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| Ok(Py::new(py, PyPadicElement { inner: entry })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::PadicField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| Ok(Py::new(py, PyPadicElement { inner: entry })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::Qadic(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| Ok(Py::new(py, PyQadicElement { inner: entry })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::QadicField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| Ok(Py::new(py, PyQadicElement { inner: entry })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::PadicLocalField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyPadicLocalFieldElement { inner: entry })?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::PadicLocalFieldOrder(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyPadicLocalFieldElement { inner: entry })?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::NumberField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyNumberFieldElement { inner: entry })?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::RationalFunction(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyRationalFunction::from_rational(entry))?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::NumberFieldOrder(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            Ok(Py::new(py, PyNumberFieldElement { inner: entry })?.into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::RealBall(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| Ok(Py::new(py, PyRealBall { inner })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::ComplexBall(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| Ok(Py::new(py, PyComplexBall { inner })?.into_any()))
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesInteger(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::Integer(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesRational(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::Rational(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesResidue(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::Residue(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesFiniteField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::FiniteField(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesPadic(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::Padic(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesPadicField(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::PadicField(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesQadic(matrix) => py
                .detach(|| matrix.to_rows())
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::Qadic(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesQadicField(matrix) => matrix
                .rows()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::QadicField(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::SeriesPadicLocalField(matrix) => matrix
                .rows()
                .map(|row| {
                    row.into_iter()
                        .map(|inner| {
                            Ok(Py::new(
                                py,
                                PyPowerSeries {
                                    inner: SeriesInner::PadicLocalField(inner),
                                },
                            )?
                            .into_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .and_then(|entries| Ok(PyList::new(py, entries)?.unbind().into_any()))
                })
                .collect::<PyResult<Vec<_>>>()?,
            MatrixInner::Polynomial(matrix) => matrix.rows_to_python(py)?,
            MatrixInner::Quotient(matrix) => matrix.rows_to_python(py)?,
        };
        Ok(PyList::new(py, rows)?.unbind().into_any())
    }

    fn context_name(&self) -> String {
        match &self.inner {
            MatrixInner::Integer(_) => "Z".to_owned(),
            MatrixInner::Rational(_) => "Q".to_owned(),
            MatrixInner::Residue(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::FiniteField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::Padic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::PadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::Qadic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::QadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::PadicLocalField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::PadicLocalFieldOrder(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            MatrixInner::NumberField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::RationalFunction(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::NumberFieldOrder(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::RealBall(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::ComplexBall(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesInteger(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesRational(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesResidue(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesFiniteField(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            MatrixInner::SeriesPadic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesPadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesQadic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesQadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            MatrixInner::SeriesPadicLocalField(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            MatrixInner::Polynomial(matrix) => matrix.context_name(),
            MatrixInner::Quotient(matrix) => matrix.context_name(),
        }
    }

    fn preview(&self) -> String {
        const ROW_LIMIT: usize = 4;
        const COLUMN_LIMIT: usize = 4;
        let (row_count, column_count) = self.dimensions();
        let mut rendered_rows = Vec::new();
        match &self.inner {
            MatrixInner::Integer(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::Rational(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::Residue(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.representative().to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::FiniteField(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.representative().to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::Padic(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .filter_map(|entry| entry.representative())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::PadicField(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::Qadic(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::QadicField(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::PadicLocalField(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::PadicLocalFieldOrder(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::NumberField(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::RationalFunction(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::NumberFieldOrder(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::RealBall(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::ComplexBall(matrix) => {
                for row in 0..row_count.min(ROW_LIMIT) {
                    let mut entries = (0..column_count.min(COLUMN_LIMIT))
                        .filter_map(|column| matrix.get(row, column).ok())
                        .map(|entry| bounded_coefficient(entry.to_string()))
                        .collect::<Vec<_>>();
                    if column_count > COLUMN_LIMIT {
                        entries.push("...".to_owned());
                    }
                    rendered_rows.push(format!("[{}]", entries.join(", ")));
                }
            }
            MatrixInner::SeriesInteger(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesRational(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesResidue(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesFiniteField(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesPadic(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesPadicField(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesQadic(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesQadicField(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::SeriesPadicLocalField(matrix) => {
                append_series_matrix_preview(matrix, &mut rendered_rows, row_count, column_count)
            }
            MatrixInner::Polynomial(matrix) => return matrix.preview(),
            MatrixInner::Quotient(matrix) => return matrix.preview(),
        }
        if row_count > ROW_LIMIT {
            rendered_rows.push("...".to_owned());
        }
        format!("[{}]", rendered_rows.join(", "))
    }

    fn dimensions(&self) -> (usize, usize) {
        match &self.inner {
            MatrixInner::Integer(matrix) => matrix.shape(),
            MatrixInner::Rational(matrix) => matrix.shape(),
            MatrixInner::Residue(matrix) => matrix.shape(),
            MatrixInner::FiniteField(matrix) => matrix.shape(),
            MatrixInner::Padic(matrix) => matrix.shape(),
            MatrixInner::PadicField(matrix) => matrix.shape(),
            MatrixInner::Qadic(matrix) => matrix.shape(),
            MatrixInner::QadicField(matrix) => matrix.shape(),
            MatrixInner::PadicLocalField(matrix) => matrix.shape(),
            MatrixInner::PadicLocalFieldOrder(matrix) => matrix.shape(),
            MatrixInner::NumberField(matrix) => matrix.shape(),
            MatrixInner::RationalFunction(matrix) => matrix.shape(),
            MatrixInner::NumberFieldOrder(matrix) => matrix.shape(),
            MatrixInner::RealBall(matrix) => matrix.shape(),
            MatrixInner::ComplexBall(matrix) => matrix.shape(),
            MatrixInner::SeriesInteger(matrix) => matrix.shape(),
            MatrixInner::SeriesRational(matrix) => matrix.shape(),
            MatrixInner::SeriesResidue(matrix) => matrix.shape(),
            MatrixInner::SeriesFiniteField(matrix) => matrix.shape(),
            MatrixInner::SeriesPadic(matrix) => matrix.shape(),
            MatrixInner::SeriesPadicField(matrix) => matrix.shape(),
            MatrixInner::SeriesQadic(matrix) => matrix.shape(),
            MatrixInner::SeriesQadicField(matrix) => matrix.shape(),
            MatrixInner::SeriesPadicLocalField(matrix) => matrix.shape(),
            MatrixInner::Polynomial(matrix) => matrix.shape(),
            MatrixInner::Quotient(matrix) => matrix.shape(),
        }
    }

    fn apply_vector(&self, vector: &PyVector) -> PyResult<PyVector> {
        let inner = match (&self.inner, &vector.inner) {
            (MatrixInner::Integer(operator), VectorInner::Integer(vector)) => {
                VectorInner::Integer(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::Rational(operator), VectorInner::Rational(vector)) => {
                VectorInner::Rational(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::Residue(operator), VectorInner::Residue(vector)) => {
                VectorInner::Residue(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::FiniteField(operator), VectorInner::FiniteField(vector)) => {
                VectorInner::FiniteField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::Padic(operator), VectorInner::Padic(vector)) => {
                VectorInner::Padic(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::PadicField(operator), VectorInner::PadicField(vector)) => {
                VectorInner::PadicField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::Qadic(operator), VectorInner::Qadic(vector)) => {
                VectorInner::Qadic(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::QadicField(operator), VectorInner::QadicField(vector)) => {
                VectorInner::QadicField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::PadicLocalField(operator), VectorInner::PadicLocalField(vector)) => {
                VectorInner::PadicLocalField(operator.apply(vector).map_err(linalg_error)?)
            }
            (
                MatrixInner::PadicLocalFieldOrder(operator),
                VectorInner::PadicLocalFieldOrder(vector),
            ) => VectorInner::PadicLocalFieldOrder(operator.apply(vector).map_err(linalg_error)?),
            (MatrixInner::NumberField(operator), VectorInner::NumberField(vector)) => {
                VectorInner::NumberField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::RationalFunction(operator), VectorInner::RationalFunction(vector)) => {
                VectorInner::RationalFunction(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::NumberFieldOrder(operator), VectorInner::NumberFieldOrder(vector)) => {
                VectorInner::NumberFieldOrder(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::RealBall(operator), VectorInner::RealBall(vector)) => {
                VectorInner::RealBall(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::ComplexBall(operator), VectorInner::ComplexBall(vector)) => {
                VectorInner::ComplexBall(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesInteger(operator), VectorInner::SeriesInteger(vector)) => {
                VectorInner::SeriesInteger(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesRational(operator), VectorInner::SeriesRational(vector)) => {
                VectorInner::SeriesRational(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesResidue(operator), VectorInner::SeriesResidue(vector)) => {
                VectorInner::SeriesResidue(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesFiniteField(operator), VectorInner::SeriesFiniteField(vector)) => {
                VectorInner::SeriesFiniteField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadic(operator), VectorInner::SeriesPadic(vector)) => {
                VectorInner::SeriesPadic(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadicField(operator), VectorInner::SeriesPadicField(vector)) => {
                VectorInner::SeriesPadicField(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadic(operator), VectorInner::SeriesQadic(vector)) => {
                VectorInner::SeriesQadic(operator.apply(vector).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadicField(operator), VectorInner::SeriesQadicField(vector)) => {
                VectorInner::SeriesQadicField(operator.apply(vector).map_err(linalg_error)?)
            }
            (
                MatrixInner::SeriesPadicLocalField(operator),
                VectorInner::SeriesPadicLocalField(vector),
            ) => VectorInner::SeriesPadicLocalField(operator.apply(vector).map_err(linalg_error)?),
            (MatrixInner::Polynomial(operator), VectorInner::Polynomial(vector)) => {
                let inner = match (operator, vector) {
                    (
                        PolynomialMatrixInner::Integer(operator),
                        PolynomialVectorInner::Integer(vector),
                    ) => PolynomialVectorInner::Integer(
                        operator.apply(vector).map_err(linalg_error)?,
                    ),
                    (
                        PolynomialMatrixInner::Rational(operator),
                        PolynomialVectorInner::Rational(vector),
                    ) => PolynomialVectorInner::Rational(
                        operator.apply(vector).map_err(linalg_error)?,
                    ),
                    (
                        PolynomialMatrixInner::Residue(operator),
                        PolynomialVectorInner::Residue(vector),
                    ) => PolynomialVectorInner::Residue(
                        operator.apply(vector).map_err(linalg_error)?,
                    ),
                    (
                        PolynomialMatrixInner::FiniteField(operator),
                        PolynomialVectorInner::FiniteField(vector),
                    ) => PolynomialVectorInner::FiniteField(
                        operator.apply(vector).map_err(linalg_error)?,
                    ),
                    _ => {
                        return Err(context_mismatch(
                            "linear-operator application",
                            operator.context_name(),
                            vector.context_name(),
                        ));
                    }
                };
                VectorInner::Polynomial(inner)
            }
            (MatrixInner::Quotient(operator), VectorInner::Quotient(vector)) => {
                let inner = match (operator, vector) {
                    (
                        QuotientMatrixInner::Rational(operator),
                        QuotientVectorInner::Rational(vector),
                    ) => {
                        QuotientVectorInner::Rational(operator.apply(vector).map_err(linalg_error)?)
                    }
                    (
                        QuotientMatrixInner::FiniteField(operator),
                        QuotientVectorInner::FiniteField(vector),
                    ) => QuotientVectorInner::FiniteField(
                        operator.apply(vector).map_err(linalg_error)?,
                    ),
                    _ => {
                        return Err(context_mismatch(
                            "linear-operator application",
                            operator.context_name(),
                            vector.context_name(),
                        ));
                    }
                };
                VectorInner::Quotient(inner)
            }
            _ => {
                return Err(context_mismatch(
                    "linear-operator application",
                    self.context_name(),
                    vector.context_name(),
                ));
            }
        };
        Ok(PyVector { inner })
    }
}

/// A vector over one explicit coefficient context.
#[pyclass(module = "noether.linalg", name = "Vector", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyVector {
    pub(super) inner: VectorInner,
}

impl PyVector {
    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = match &self.inner {
            VectorInner::Integer(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| integer_to_python(py, &entry))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::Rational(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| rational_to_python(py, &entry))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::Residue(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyResidueClass { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::FiniteField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyFiniteFieldElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::Padic(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyPadicElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::PadicField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyPadicElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::Qadic(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyQadicElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::QadicField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyQadicElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::PadicLocalField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyPadicLocalFieldElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::PadicLocalFieldOrder(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyPadicLocalFieldElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::NumberField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyNumberFieldElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::RationalFunction(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyRationalFunction::from_rational(entry))?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::NumberFieldOrder(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|entry| Ok(Py::new(py, PyNumberFieldElement { inner: entry })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::RealBall(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| Ok(Py::new(py, PyRealBall { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::ComplexBall(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| Ok(Py::new(py, PyComplexBall { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesInteger(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::Integer(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesRational(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::Rational(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesResidue(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::Residue(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesFiniteField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::FiniteField(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesPadic(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::Padic(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesPadicField(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::PadicField(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesQadic(vector) => py
                .detach(|| vector.to_values())
                .into_iter()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::Qadic(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesQadicField(vector) => vector
                .values()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::QadicField(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::SeriesPadicLocalField(vector) => vector
                .values()
                .map(|inner| {
                    Ok(Py::new(
                        py,
                        PyPowerSeries {
                            inner: SeriesInner::PadicLocalField(inner),
                        },
                    )?
                    .into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
            VectorInner::Polynomial(vector) => return vector.values_to_python(py),
            VectorInner::Quotient(vector) => return vector.values_to_python(py),
        };
        Ok(PyList::new(py, values)?.unbind().into_any())
    }

    fn context_name(&self) -> String {
        match &self.inner {
            VectorInner::Integer(_) => "Z".to_owned(),
            VectorInner::Rational(_) => "Q".to_owned(),
            VectorInner::Residue(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::FiniteField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::Padic(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::PadicField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::Qadic(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::QadicField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::PadicLocalField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::PadicLocalFieldOrder(vector) => {
                bounded_coefficient(vector.ring().to_string())
            }
            VectorInner::NumberField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::RationalFunction(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::NumberFieldOrder(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::RealBall(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::ComplexBall(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesInteger(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesRational(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesResidue(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesFiniteField(vector) => {
                bounded_coefficient(vector.ring().to_string())
            }
            VectorInner::SeriesPadic(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesPadicField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesQadic(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesQadicField(vector) => bounded_coefficient(vector.ring().to_string()),
            VectorInner::SeriesPadicLocalField(vector) => {
                bounded_coefficient(vector.ring().to_string())
            }
            VectorInner::Polynomial(vector) => vector.context_name(),
            VectorInner::Quotient(vector) => vector.context_name(),
        }
    }

    fn dimension_value(&self) -> usize {
        match &self.inner {
            VectorInner::Integer(vector) => vector.dimension(),
            VectorInner::Rational(vector) => vector.dimension(),
            VectorInner::Residue(vector) => vector.dimension(),
            VectorInner::FiniteField(vector) => vector.dimension(),
            VectorInner::Padic(vector) => vector.dimension(),
            VectorInner::PadicField(vector) => vector.dimension(),
            VectorInner::Qadic(vector) => vector.dimension(),
            VectorInner::QadicField(vector) => vector.dimension(),
            VectorInner::PadicLocalField(vector) => vector.dimension(),
            VectorInner::PadicLocalFieldOrder(vector) => vector.dimension(),
            VectorInner::NumberField(vector) => vector.dimension(),
            VectorInner::RationalFunction(vector) => vector.dimension(),
            VectorInner::NumberFieldOrder(vector) => vector.dimension(),
            VectorInner::RealBall(vector) => vector.dimension(),
            VectorInner::ComplexBall(vector) => vector.dimension(),
            VectorInner::SeriesInteger(vector) => vector.dimension(),
            VectorInner::SeriesRational(vector) => vector.dimension(),
            VectorInner::SeriesResidue(vector) => vector.dimension(),
            VectorInner::SeriesFiniteField(vector) => vector.dimension(),
            VectorInner::SeriesPadic(vector) => vector.dimension(),
            VectorInner::SeriesPadicField(vector) => vector.dimension(),
            VectorInner::SeriesQadic(vector) => vector.dimension(),
            VectorInner::SeriesQadicField(vector) => vector.dimension(),
            VectorInner::SeriesPadicLocalField(vector) => vector.dimension(),
            VectorInner::Polynomial(vector) => vector.dimension(),
            VectorInner::Quotient(vector) => vector.dimension(),
        }
    }

    fn preview(&self) -> String {
        const ENTRY_LIMIT: usize = 8;
        let mut entries = match &self.inner {
            VectorInner::Integer(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::Rational(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::Residue(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.representative().to_string()))
                .collect::<Vec<_>>(),
            VectorInner::FiniteField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.representative().to_string()))
                .collect::<Vec<_>>(),
            VectorInner::Padic(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .filter_map(|entry| entry.representative())
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::PadicField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::Qadic(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::QadicField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::PadicLocalField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::PadicLocalFieldOrder(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::NumberField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::RationalFunction(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::NumberFieldOrder(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::RealBall(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::ComplexBall(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesInteger(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesRational(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesResidue(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesFiniteField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesPadic(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesPadicField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesQadic(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesQadicField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::SeriesPadicLocalField(vector) => vector
                .values()
                .take(ENTRY_LIMIT)
                .map(|entry| bounded_coefficient(entry.to_string()))
                .collect::<Vec<_>>(),
            VectorInner::Polynomial(vector) => return vector.preview(),
            VectorInner::Quotient(vector) => return vector.preview(),
        };
        if self.dimension_value() > ENTRY_LIMIT {
            entries.push("...".to_owned());
        }
        format!("[{}]", entries.join(", "))
    }
}

/// A canonical sparse matrix over one explicit coefficient context.
#[pyclass(
    module = "noether.linalg",
    name = "SparseMatrix",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySparseMatrix {
    pub(crate) inner: SparseMatrixInner,
}

impl PySparseMatrix {
    fn dense_value(&self) -> PyMatrix {
        let inner = match &self.inner {
            SparseMatrixInner::Integer(matrix) => MatrixInner::Integer(matrix.to_dense()),
            SparseMatrixInner::Rational(matrix) => MatrixInner::Rational(matrix.to_dense()),
            SparseMatrixInner::Residue(matrix) => MatrixInner::Residue(matrix.to_dense()),
            SparseMatrixInner::FiniteField(matrix) => MatrixInner::FiniteField(matrix.to_dense()),
            SparseMatrixInner::Padic(matrix) => MatrixInner::Padic(matrix.to_dense()),
            SparseMatrixInner::PadicField(matrix) => MatrixInner::PadicField(matrix.to_dense()),
            SparseMatrixInner::Qadic(matrix) => MatrixInner::Qadic(matrix.to_dense()),
            SparseMatrixInner::QadicField(matrix) => MatrixInner::QadicField(matrix.to_dense()),
            SparseMatrixInner::PadicLocalField(matrix) => {
                MatrixInner::PadicLocalField(matrix.to_dense())
            }
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => {
                MatrixInner::PadicLocalFieldOrder(matrix.to_dense())
            }
            SparseMatrixInner::RealBall(matrix) => MatrixInner::RealBall(matrix.to_dense()),
            SparseMatrixInner::ComplexBall(matrix) => MatrixInner::ComplexBall(matrix.to_dense()),
            SparseMatrixInner::Polynomial(matrix) => MatrixInner::Polynomial(matrix.to_dense()),
            SparseMatrixInner::Quotient(matrix) => MatrixInner::Quotient(matrix.to_dense()),
        };
        PyMatrix { inner }
    }

    fn dimensions(&self) -> (usize, usize) {
        match &self.inner {
            SparseMatrixInner::Integer(matrix) => matrix.shape(),
            SparseMatrixInner::Rational(matrix) => matrix.shape(),
            SparseMatrixInner::Residue(matrix) => matrix.shape(),
            SparseMatrixInner::FiniteField(matrix) => matrix.shape(),
            SparseMatrixInner::Padic(matrix) => matrix.shape(),
            SparseMatrixInner::PadicField(matrix) => matrix.shape(),
            SparseMatrixInner::Qadic(matrix) => matrix.shape(),
            SparseMatrixInner::QadicField(matrix) => matrix.shape(),
            SparseMatrixInner::PadicLocalField(matrix) => matrix.shape(),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => matrix.shape(),
            SparseMatrixInner::RealBall(matrix) => matrix.shape(),
            SparseMatrixInner::ComplexBall(matrix) => matrix.shape(),
            SparseMatrixInner::Polynomial(matrix) => matrix.shape(),
            SparseMatrixInner::Quotient(matrix) => matrix.shape(),
        }
    }

    fn nonzero_count(&self) -> usize {
        match &self.inner {
            SparseMatrixInner::Integer(matrix) => matrix.nnz(),
            SparseMatrixInner::Rational(matrix) => matrix.nnz(),
            SparseMatrixInner::Residue(matrix) => matrix.nnz(),
            SparseMatrixInner::FiniteField(matrix) => matrix.nnz(),
            SparseMatrixInner::Padic(matrix) => matrix.nnz(),
            SparseMatrixInner::PadicField(matrix) => matrix.nnz(),
            SparseMatrixInner::Qadic(matrix) => matrix.nnz(),
            SparseMatrixInner::QadicField(matrix) => matrix.nnz(),
            SparseMatrixInner::PadicLocalField(matrix) => matrix.nnz(),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => matrix.nnz(),
            SparseMatrixInner::RealBall(matrix) => matrix.nnz(),
            SparseMatrixInner::ComplexBall(matrix) => matrix.nnz(),
            SparseMatrixInner::Polynomial(matrix) => matrix.nnz(),
            SparseMatrixInner::Quotient(matrix) => matrix.nnz(),
        }
    }

    fn context_name(&self) -> String {
        match &self.inner {
            SparseMatrixInner::Integer(_) => "Z".to_owned(),
            SparseMatrixInner::Rational(_) => "Q".to_owned(),
            SparseMatrixInner::Residue(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::FiniteField(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            SparseMatrixInner::Padic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::PadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::Qadic(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::QadicField(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::PadicLocalField(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            SparseMatrixInner::RealBall(matrix) => bounded_coefficient(matrix.ring().to_string()),
            SparseMatrixInner::ComplexBall(matrix) => {
                bounded_coefficient(matrix.ring().to_string())
            }
            SparseMatrixInner::Polynomial(matrix) => matrix.context_name(),
            SparseMatrixInner::Quotient(matrix) => matrix.context_name(),
        }
    }

    fn values_to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = match &self.inner {
            SparseMatrixInner::Integer(matrix) => matrix
                .values()
                .iter()
                .map(|value| integer_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::Rational(matrix) => matrix
                .values()
                .iter()
                .map(|value| rational_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::Residue(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyResidueClass { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::FiniteField(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::Padic(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyPadicElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::PadicField(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyPadicElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::Qadic(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyQadicElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::QadicField(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyQadicElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::PadicLocalField(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyPadicLocalFieldElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyPadicLocalFieldElement { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::RealBall(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyRealBall { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::ComplexBall(matrix) => matrix
                .values()
                .iter()
                .cloned()
                .map(|inner| Ok(Py::new(py, PyComplexBall { inner })?.into_any()))
                .collect::<PyResult<Vec<_>>>()?,
            SparseMatrixInner::Polynomial(matrix) => return matrix.values_to_python(py),
            SparseMatrixInner::Quotient(matrix) => return matrix.values_to_python(py),
        };
        Ok(PyList::new(py, values)?.unbind().into_any())
    }
}

#[pymethods]
impl PySparseMatrix {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustSparseMatrix::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Construct from `(row, column, coefficient)` triplets.
    #[classmethod]
    fn from_triplets(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
        shape: (usize, usize),
        entries: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        validate_python_shape(shape.0, shape.1)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            let entries = collect_sparse_triplets(py, entries, extract_integer)?;
            SparseMatrixInner::Integer(
                RustSparseMatrix::from_triplets(Z, shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if ring.is_instance_of::<PyRationalField>() {
            let entries =
                collect_sparse_triplets(py, entries, |value| extract_rational(py, value))?;
            SparseMatrixInner::Rational(
                RustSparseMatrix::from_triplets(Q, shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_residue(&residue_ring.inner, value)
            })?;
            SparseMatrixInner::Residue(
                RustSparseMatrix::from_triplets(
                    residue_ring.inner.clone(),
                    shape.0,
                    shape.1,
                    entries,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_finite_field_element(&field.inner, value)
            })?;
            SparseMatrixInner::FiniteField(
                RustSparseMatrix::from_triplets(field.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_padic_element(&context.inner, value)
            })?;
            SparseMatrixInner::Padic(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_padic_field_element(&context.inner, value)
            })?;
            SparseMatrixInner::PadicField(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_qadic_element(&context.inner, value)
            })?;
            SparseMatrixInner::Qadic(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_qadic_field_element(&context.inner, value)
            })?;
            SparseMatrixInner::QadicField(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_padic_local_field_element(&context.inner, value)
            })?;
            SparseMatrixInner::PadicLocalField(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_padic_local_field_order_element(&context.inner, value)
            })?;
            SparseMatrixInner::PadicLocalFieldOrder(
                RustSparseMatrix::from_triplets(context.inner.clone(), shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_real_ball(py, &field.inner, value)
            })?;
            SparseMatrixInner::RealBall(
                RustSparseMatrix::from_triplets(field.inner, shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            let entries = collect_sparse_triplets(py, entries, |value| {
                extract_complex_ball(py, &field.inner, value)
            })?;
            SparseMatrixInner::ComplexBall(
                RustSparseMatrix::from_triplets(field.inner, shape.0, shape.1, entries)
                    .map_err(linalg_error)?,
            )
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            SparseMatrixInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => QuotientSparseMatrixInner::Rational(
                    RustSparseMatrix::from_triplets(
                        ring.clone(),
                        shape.0,
                        shape.1,
                        collect_sparse_triplets(py, entries, |value| {
                            extract_rational_quotient(ring, value)
                        })?,
                    )
                    .map_err(linalg_error)?,
                ),
                QuotientRingInner::FiniteField(ring) => QuotientSparseMatrixInner::FiniteField(
                    RustSparseMatrix::from_triplets(
                        ring.clone(),
                        shape.0,
                        shape.1,
                        collect_sparse_triplets(py, entries, |value| {
                            extract_finite_field_quotient(ring, value)
                        })?,
                    )
                    .map_err(linalg_error)?,
                ),
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialSparseMatrixInner::Integer(
                        RustSparseMatrix::from_triplets(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            collect_sparse_triplets(py, entries, |value| {
                                extract_integer_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialSparseMatrixInner::Rational(
                        RustSparseMatrix::from_triplets(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            collect_sparse_triplets(py, entries, |value| {
                                extract_rational_polynomial(py, ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialSparseMatrixInner::Residue(
                        RustSparseMatrix::from_triplets(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            collect_sparse_triplets(py, entries, |value| {
                                extract_residue_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialSparseMatrixInner::FiniteField(
                        RustSparseMatrix::from_triplets(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            collect_sparse_triplets(py, entries, |value| {
                                extract_finite_field_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "sparse polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            SparseMatrixInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "SparseMatrix supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, RealBallField, and ComplexBallField contexts",
            ));
        };
        Ok(Self { inner })
    }

    /// Construct from canonical CSR arrays.
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_csr(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
        shape: (usize, usize),
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        validate_python_shape(shape.0, shape.1)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            SparseMatrixInner::Integer(
                RustSparseMatrix::from_csr(
                    Z,
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, extract_integer)?,
                )
                .map_err(linalg_error)?,
            )
        } else if ring.is_instance_of::<PyRationalField>() {
            SparseMatrixInner::Rational(
                RustSparseMatrix::from_csr(
                    Q,
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| extract_rational(py, value))?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            SparseMatrixInner::Residue(
                RustSparseMatrix::from_csr(
                    residue_ring.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| extract_residue(&residue_ring.inner, value))?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            SparseMatrixInner::FiniteField(
                RustSparseMatrix::from_csr(
                    field.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_finite_field_element(&field.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            SparseMatrixInner::Padic(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| extract_padic_element(&context.inner, value))?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicField>>() {
            SparseMatrixInner::PadicField(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_padic_field_element(&context.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            SparseMatrixInner::Qadic(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| extract_qadic_element(&context.inner, value))?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicField>>() {
            SparseMatrixInner::QadicField(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_qadic_field_element(&context.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            SparseMatrixInner::PadicLocalField(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_padic_local_field_element(&context.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            SparseMatrixInner::PadicLocalFieldOrder(
                RustSparseMatrix::from_csr(
                    context.inner.clone(),
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_padic_local_field_order_element(&context.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            SparseMatrixInner::RealBall(
                RustSparseMatrix::from_csr(
                    field.inner,
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| extract_real_ball(py, &field.inner, value))?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            SparseMatrixInner::ComplexBall(
                RustSparseMatrix::from_csr(
                    field.inner,
                    shape.0,
                    shape.1,
                    row_offsets,
                    column_indices,
                    collect_values(values, |value| {
                        extract_complex_ball(py, &field.inner, value)
                    })?,
                )
                .map_err(linalg_error)?,
            )
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            SparseMatrixInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => QuotientSparseMatrixInner::Rational(
                    RustSparseMatrix::from_csr(
                        ring.clone(),
                        shape.0,
                        shape.1,
                        row_offsets,
                        column_indices,
                        collect_values(values, |value| extract_rational_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
                QuotientRingInner::FiniteField(ring) => QuotientSparseMatrixInner::FiniteField(
                    RustSparseMatrix::from_csr(
                        ring.clone(),
                        shape.0,
                        shape.1,
                        row_offsets,
                        column_indices,
                        collect_values(values, |value| extract_finite_field_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialSparseMatrixInner::Integer(
                        RustSparseMatrix::from_csr(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            row_offsets,
                            column_indices,
                            collect_values(values, |value| {
                                extract_integer_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialSparseMatrixInner::Rational(
                        RustSparseMatrix::from_csr(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            row_offsets,
                            column_indices,
                            collect_values(values, |value| {
                                extract_rational_polynomial(py, ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialSparseMatrixInner::Residue(
                        RustSparseMatrix::from_csr(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            row_offsets,
                            column_indices,
                            collect_values(values, |value| {
                                extract_residue_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialSparseMatrixInner::FiniteField(
                        RustSparseMatrix::from_csr(
                            ring.clone(),
                            shape.0,
                            shape.1,
                            row_offsets,
                            column_indices,
                            collect_values(values, |value| {
                                extract_finite_field_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "sparse polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            SparseMatrixInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "SparseMatrix supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, RealBallField, and ComplexBallField contexts",
            ));
        };
        Ok(Self { inner })
    }

    /// Construct a sparse zero matrix.
    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        ring: &Bound<'_, PyAny>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        validate_python_shape(shape.0, shape.1)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            SparseMatrixInner::Integer(RustSparseMatrix::zeros(Z, shape.0, shape.1))
        } else if ring.is_instance_of::<PyRationalField>() {
            SparseMatrixInner::Rational(RustSparseMatrix::zeros(Q, shape.0, shape.1))
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            SparseMatrixInner::Residue(RustSparseMatrix::zeros(
                residue_ring.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            SparseMatrixInner::FiniteField(RustSparseMatrix::zeros(
                field.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            SparseMatrixInner::Padic(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicField>>() {
            SparseMatrixInner::PadicField(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            SparseMatrixInner::Qadic(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyQadicField>>() {
            SparseMatrixInner::QadicField(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            SparseMatrixInner::PadicLocalField(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(context) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            SparseMatrixInner::PadicLocalFieldOrder(RustSparseMatrix::zeros(
                context.inner.clone(),
                shape.0,
                shape.1,
            ))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            SparseMatrixInner::RealBall(RustSparseMatrix::zeros(field.inner, shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            SparseMatrixInner::ComplexBall(RustSparseMatrix::zeros(field.inner, shape.0, shape.1))
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            SparseMatrixInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => QuotientSparseMatrixInner::Rational(
                    RustSparseMatrix::zeros(ring.clone(), shape.0, shape.1),
                ),
                QuotientRingInner::FiniteField(ring) => QuotientSparseMatrixInner::FiniteField(
                    RustSparseMatrix::zeros(ring.clone(), shape.0, shape.1),
                ),
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialSparseMatrixInner::Integer(RustSparseMatrix::zeros(
                        ring.clone(),
                        shape.0,
                        shape.1,
                    ))
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialSparseMatrixInner::Rational(RustSparseMatrix::zeros(
                        ring.clone(),
                        shape.0,
                        shape.1,
                    ))
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialSparseMatrixInner::Residue(RustSparseMatrix::zeros(
                        ring.clone(),
                        shape.0,
                        shape.1,
                    ))
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialSparseMatrixInner::FiniteField(RustSparseMatrix::zeros(
                        ring.clone(),
                        shape.0,
                        shape.1,
                    ))
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "sparse polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            SparseMatrixInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "SparseMatrix supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, RealBallField, and ComplexBallField contexts",
            ));
        };
        Ok(Self { inner })
    }

    /// Losslessly convert a supported dense facade matrix.
    #[classmethod]
    fn from_dense(_cls: &Bound<'_, PyType>, matrix: &PyMatrix) -> PyResult<Self> {
        let inner = match &matrix.inner {
            MatrixInner::Integer(matrix) => SparseMatrixInner::Integer(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::Rational(matrix) => SparseMatrixInner::Rational(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::Residue(matrix) => SparseMatrixInner::Residue(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::FiniteField(matrix) => SparseMatrixInner::FiniteField(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::Padic(matrix) => SparseMatrixInner::Padic(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::PadicField(matrix) => SparseMatrixInner::PadicField(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::Qadic(matrix) => SparseMatrixInner::Qadic(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::QadicField(matrix) => SparseMatrixInner::QadicField(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::PadicLocalField(matrix) => SparseMatrixInner::PadicLocalField(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::PadicLocalFieldOrder(matrix) => SparseMatrixInner::PadicLocalFieldOrder(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::RealBall(matrix) => SparseMatrixInner::RealBall(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::ComplexBall(matrix) => SparseMatrixInner::ComplexBall(
                RustSparseMatrix::from_dense(matrix).map_err(linalg_error)?,
            ),
            MatrixInner::Polynomial(matrix) => {
                SparseMatrixInner::Polynomial(matrix.to_sparse().map_err(linalg_error)?)
            }
            MatrixInner::Quotient(matrix) => {
                SparseMatrixInner::Quotient(matrix.to_sparse().map_err(linalg_error)?)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "SparseMatrix.from_dense does not yet support this coefficient context",
                ))
            }
        };
        Ok(Self { inner })
    }

    /// Serialize the coefficient context and canonical CSR arrays.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a sparse matrix.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SparseMatrixInner::from_json(value)?,
        })
    }

    /// Exact coefficient context.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SparseMatrixInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            SparseMatrixInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            SparseMatrixInner::Residue(matrix) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::FiniteField(matrix) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::Padic(matrix) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicField(matrix) => Ok(Py::new(
                py,
                PyPadicField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::Qadic(matrix) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::QadicField(matrix) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPadicLocalField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldOrder {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            SparseMatrixInner::RealBall(matrix) => Ok(Py::new(
                py,
                PyRealBallField {
                    inner: *matrix.ring(),
                },
            )?
            .into_any()),
            SparseMatrixInner::ComplexBall(matrix) => Ok(Py::new(
                py,
                PyComplexBallField {
                    inner: *matrix.ring(),
                },
            )?
            .into_any()),
            SparseMatrixInner::Polynomial(matrix) => matrix.ring_to_python(py),
            SparseMatrixInner::Quotient(matrix) => matrix.ring_to_python(py),
        }
    }

    /// `(rows, columns)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dimensions()
    }

    #[getter]
    fn input_dimension(&self) -> usize {
        self.dimensions().1
    }

    #[getter]
    fn output_dimension(&self) -> usize {
        self.dimensions().0
    }

    /// Number of stored exact nonzeros.
    #[getter]
    fn nnz(&self) -> usize {
        self.nonzero_count()
    }

    /// Fraction of cells stored explicitly.
    #[getter]
    fn density(&self) -> f64 {
        let (rows, columns) = self.dimensions();
        let cells = rows.saturating_mul(columns);
        if cells == 0 {
            0.0
        } else {
            self.nonzero_count() as f64 / cells as f64
        }
    }

    /// Canonical CSR row offsets.
    #[getter]
    fn row_offsets(&self) -> Vec<usize> {
        match &self.inner {
            SparseMatrixInner::Integer(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::Rational(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::Residue(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::FiniteField(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::Padic(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::PadicField(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::Qadic(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::QadicField(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::PadicLocalField(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::RealBall(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::ComplexBall(matrix) => matrix.row_offsets().to_vec(),
            SparseMatrixInner::Polynomial(matrix) => matrix.row_offsets(),
            SparseMatrixInner::Quotient(matrix) => matrix.row_offsets(),
        }
    }

    /// Canonical CSR column indices.
    #[getter]
    fn column_indices(&self) -> Vec<usize> {
        match &self.inner {
            SparseMatrixInner::Integer(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::Rational(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::Residue(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::FiniteField(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::Padic(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::PadicField(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::Qadic(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::QadicField(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::PadicLocalField(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::RealBall(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::ComplexBall(matrix) => matrix.column_indices().to_vec(),
            SparseMatrixInner::Polynomial(matrix) => matrix.column_indices(),
            SparseMatrixInner::Quotient(matrix) => matrix.column_indices(),
        }
    }

    /// Canonical CSR nonzero coefficients.
    #[getter]
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.values_to_python(py)
    }

    /// Canonical `(row_offsets, column_indices, values)` arrays.
    #[getter]
    fn csr(&self, py: Python<'_>) -> PyResult<(Vec<usize>, Vec<usize>, Py<PyAny>)> {
        Ok((
            self.row_offsets(),
            self.column_indices(),
            self.values_to_python(py)?,
        ))
    }

    fn __getitem__(&self, py: Python<'_>, index: (isize, isize)) -> PyResult<Py<PyAny>> {
        let shape = self.dimensions();
        let row = normalize_index(index.0, shape.0)
            .ok_or_else(|| PyIndexError::new_err("sparse matrix row index is out of range"))?;
        let column = normalize_index(index.1, shape.1)
            .ok_or_else(|| PyIndexError::new_err("sparse matrix column index is out of range"))?;
        match &self.inner {
            SparseMatrixInner::Integer(matrix) => {
                integer_to_python(py, &matrix.get(row, column).map_err(index_error)?)
            }
            SparseMatrixInner::Rational(matrix) => {
                rational_to_python(py, &matrix.get(row, column).map_err(index_error)?)
            }
            SparseMatrixInner::Residue(matrix) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::FiniteField(matrix) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::Padic(matrix) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicField(matrix) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::Qadic(matrix) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::QadicField(matrix) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::RealBall(matrix) => Ok(Py::new(
                py,
                PyRealBall {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::ComplexBall(matrix) => Ok(Py::new(
                py,
                PyComplexBall {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            SparseMatrixInner::Polynomial(matrix) => matrix.get_to_python(py, row, column),
            SparseMatrixInner::Quotient(matrix) => matrix.get_to_python(py, row, column),
        }
    }

    /// Losslessly materialize a dense matrix with the same context.
    fn to_dense(&self) -> PyMatrix {
        self.dense_value()
    }

    /// Smith invariant valuations visible below a p-adic ring's precision cap.
    fn padic_smith_invariants(&self, py: Python<'_>) -> PyResult<PyPadicSmithInvariants> {
        match &self.inner {
            SparseMatrixInner::Padic(matrix) => py.detach(|| {
                Ok(PyPadicSmithInvariants {
                    inner: matrix.padic_smith_invariants().map_err(linalg_error)?,
                })
            }),
            _ => Err(unsupported_matrix_operation(
                "padic_smith_invariants",
                self.inner.context_type_id(),
                &[PadicRing::TYPE_ID],
            )),
        }
    }

    /// Return the canonical sparse transpose.
    fn transpose(&self) -> Self {
        let inner = match &self.inner {
            SparseMatrixInner::Integer(matrix) => SparseMatrixInner::Integer(matrix.transpose()),
            SparseMatrixInner::Rational(matrix) => SparseMatrixInner::Rational(matrix.transpose()),
            SparseMatrixInner::Residue(matrix) => SparseMatrixInner::Residue(matrix.transpose()),
            SparseMatrixInner::FiniteField(matrix) => {
                SparseMatrixInner::FiniteField(matrix.transpose())
            }
            SparseMatrixInner::Padic(matrix) => SparseMatrixInner::Padic(matrix.transpose()),
            SparseMatrixInner::PadicField(matrix) => {
                SparseMatrixInner::PadicField(matrix.transpose())
            }
            SparseMatrixInner::Qadic(matrix) => SparseMatrixInner::Qadic(matrix.transpose()),
            SparseMatrixInner::QadicField(matrix) => {
                SparseMatrixInner::QadicField(matrix.transpose())
            }
            SparseMatrixInner::PadicLocalField(matrix) => {
                SparseMatrixInner::PadicLocalField(matrix.transpose())
            }
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => {
                SparseMatrixInner::PadicLocalFieldOrder(matrix.transpose())
            }
            SparseMatrixInner::RealBall(matrix) => SparseMatrixInner::RealBall(matrix.transpose()),
            SparseMatrixInner::ComplexBall(matrix) => {
                SparseMatrixInner::ComplexBall(matrix.transpose())
            }
            SparseMatrixInner::Polynomial(matrix) => {
                SparseMatrixInner::Polynomial(matrix.transpose())
            }
            SparseMatrixInner::Quotient(matrix) => SparseMatrixInner::Quotient(matrix.transpose()),
        };
        Self { inner }
    }

    /// Apply this sparse linear operator to a vector.
    fn apply(&self, py: Python<'_>, vector: &PyVector) -> PyResult<PyVector> {
        let source_context = self.context_name();
        let input_context = vector.context_name();
        let inner = py.detach(|| -> PyResult<VectorInner> {
            let inner = match (&self.inner, &vector.inner) {
                (SparseMatrixInner::Integer(matrix), VectorInner::Integer(vector)) => {
                    VectorInner::Integer(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Rational(matrix), VectorInner::Rational(vector)) => {
                    VectorInner::Rational(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Residue(matrix), VectorInner::Residue(vector)) => {
                    VectorInner::Residue(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::FiniteField(matrix), VectorInner::FiniteField(vector)) => {
                    VectorInner::FiniteField(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Padic(matrix), VectorInner::Padic(vector)) => {
                    VectorInner::Padic(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::PadicField(matrix), VectorInner::PadicField(vector)) => {
                    VectorInner::PadicField(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Qadic(matrix), VectorInner::Qadic(vector)) => {
                    VectorInner::Qadic(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::QadicField(matrix), VectorInner::QadicField(vector)) => {
                    VectorInner::QadicField(matrix.apply(vector).map_err(linalg_error)?)
                }
                (
                    SparseMatrixInner::PadicLocalField(matrix),
                    VectorInner::PadicLocalField(vector),
                ) => VectorInner::PadicLocalField(matrix.apply(vector).map_err(linalg_error)?),
                (
                    SparseMatrixInner::PadicLocalFieldOrder(matrix),
                    VectorInner::PadicLocalFieldOrder(vector),
                ) => VectorInner::PadicLocalFieldOrder(matrix.apply(vector).map_err(linalg_error)?),
                (SparseMatrixInner::RealBall(matrix), VectorInner::RealBall(vector)) => {
                    VectorInner::RealBall(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::ComplexBall(matrix), VectorInner::ComplexBall(vector)) => {
                    VectorInner::ComplexBall(matrix.apply(vector).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Polynomial(matrix), VectorInner::Polynomial(vector)) => {
                    VectorInner::Polynomial(matrix.apply(vector)?)
                }
                (SparseMatrixInner::Quotient(matrix), VectorInner::Quotient(vector)) => {
                    VectorInner::Quotient(matrix.apply(vector)?)
                }
                _ => {
                    return Err(context_mismatch(
                        "sparse matrix application",
                        source_context,
                        input_context,
                    ))
                }
            };
            Ok(inner)
        })?;
        Ok(PyVector { inner })
    }

    /// Apply this sparse linear operator to a dense matrix block.
    fn apply_block(&self, py: Python<'_>, block: &PyMatrix) -> PyResult<PyMatrix> {
        let source_context = self.context_name();
        let input_context = block.context_name();
        let inner = py.detach(|| -> PyResult<MatrixInner> {
            let inner = match (&self.inner, &block.inner) {
                (SparseMatrixInner::Integer(matrix), MatrixInner::Integer(block)) => {
                    MatrixInner::Integer(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Rational(matrix), MatrixInner::Rational(block)) => {
                    MatrixInner::Rational(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Residue(matrix), MatrixInner::Residue(block)) => {
                    MatrixInner::Residue(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::FiniteField(matrix), MatrixInner::FiniteField(block)) => {
                    MatrixInner::FiniteField(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Padic(matrix), MatrixInner::Padic(block)) => {
                    MatrixInner::Padic(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::PadicField(matrix), MatrixInner::PadicField(block)) => {
                    MatrixInner::PadicField(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Qadic(matrix), MatrixInner::Qadic(block)) => {
                    MatrixInner::Qadic(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::QadicField(matrix), MatrixInner::QadicField(block)) => {
                    MatrixInner::QadicField(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (
                    SparseMatrixInner::PadicLocalField(matrix),
                    MatrixInner::PadicLocalField(block),
                ) => MatrixInner::PadicLocalField(matrix.apply_block(block).map_err(linalg_error)?),
                (
                    SparseMatrixInner::PadicLocalFieldOrder(matrix),
                    MatrixInner::PadicLocalFieldOrder(block),
                ) => MatrixInner::PadicLocalFieldOrder(
                    matrix.apply_block(block).map_err(linalg_error)?,
                ),
                (SparseMatrixInner::RealBall(matrix), MatrixInner::RealBall(block)) => {
                    MatrixInner::RealBall(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::ComplexBall(matrix), MatrixInner::ComplexBall(block)) => {
                    MatrixInner::ComplexBall(matrix.apply_block(block).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Polynomial(matrix), MatrixInner::Polynomial(block)) => {
                    MatrixInner::Polynomial(matrix.apply_block(block)?)
                }
                (SparseMatrixInner::Quotient(matrix), MatrixInner::Quotient(block)) => {
                    MatrixInner::Quotient(matrix.apply_block(block)?)
                }
                _ => {
                    return Err(context_mismatch(
                        "sparse block application",
                        source_context,
                        input_context,
                    ))
                }
            };
            Ok(inner)
        })?;
        Ok(PyMatrix { inner })
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (SparseMatrixInner::Integer(left), SparseMatrixInner::Integer(right)) => {
                SparseMatrixInner::Integer(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Rational(left), SparseMatrixInner::Rational(right)) => {
                SparseMatrixInner::Rational(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Residue(left), SparseMatrixInner::Residue(right)) => {
                SparseMatrixInner::Residue(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::FiniteField(left), SparseMatrixInner::FiniteField(right)) => {
                SparseMatrixInner::FiniteField(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Padic(left), SparseMatrixInner::Padic(right)) => {
                SparseMatrixInner::Padic(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::PadicField(left), SparseMatrixInner::PadicField(right)) => {
                SparseMatrixInner::PadicField(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Qadic(left), SparseMatrixInner::Qadic(right)) => {
                SparseMatrixInner::Qadic(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::QadicField(left), SparseMatrixInner::QadicField(right)) => {
                SparseMatrixInner::QadicField(left.add(right).map_err(linalg_error)?)
            }
            (
                SparseMatrixInner::PadicLocalField(left),
                SparseMatrixInner::PadicLocalField(right),
            ) => SparseMatrixInner::PadicLocalField(left.add(right).map_err(linalg_error)?),
            (
                SparseMatrixInner::PadicLocalFieldOrder(left),
                SparseMatrixInner::PadicLocalFieldOrder(right),
            ) => SparseMatrixInner::PadicLocalFieldOrder(left.add(right).map_err(linalg_error)?),
            (SparseMatrixInner::RealBall(left), SparseMatrixInner::RealBall(right)) => {
                SparseMatrixInner::RealBall(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::ComplexBall(left), SparseMatrixInner::ComplexBall(right)) => {
                SparseMatrixInner::ComplexBall(left.add(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Polynomial(left), SparseMatrixInner::Polynomial(right)) => {
                SparseMatrixInner::Polynomial(left.add(right)?)
            }
            (SparseMatrixInner::Quotient(left), SparseMatrixInner::Quotient(right)) => {
                SparseMatrixInner::Quotient(left.add(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "sparse matrix addition",
                    self.context_name(),
                    other.context_name(),
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (SparseMatrixInner::Integer(left), SparseMatrixInner::Integer(right)) => {
                SparseMatrixInner::Integer(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Rational(left), SparseMatrixInner::Rational(right)) => {
                SparseMatrixInner::Rational(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Residue(left), SparseMatrixInner::Residue(right)) => {
                SparseMatrixInner::Residue(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::FiniteField(left), SparseMatrixInner::FiniteField(right)) => {
                SparseMatrixInner::FiniteField(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Padic(left), SparseMatrixInner::Padic(right)) => {
                SparseMatrixInner::Padic(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::PadicField(left), SparseMatrixInner::PadicField(right)) => {
                SparseMatrixInner::PadicField(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Qadic(left), SparseMatrixInner::Qadic(right)) => {
                SparseMatrixInner::Qadic(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::QadicField(left), SparseMatrixInner::QadicField(right)) => {
                SparseMatrixInner::QadicField(left.sub(right).map_err(linalg_error)?)
            }
            (
                SparseMatrixInner::PadicLocalField(left),
                SparseMatrixInner::PadicLocalField(right),
            ) => SparseMatrixInner::PadicLocalField(left.sub(right).map_err(linalg_error)?),
            (
                SparseMatrixInner::PadicLocalFieldOrder(left),
                SparseMatrixInner::PadicLocalFieldOrder(right),
            ) => SparseMatrixInner::PadicLocalFieldOrder(left.sub(right).map_err(linalg_error)?),
            (SparseMatrixInner::RealBall(left), SparseMatrixInner::RealBall(right)) => {
                SparseMatrixInner::RealBall(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::ComplexBall(left), SparseMatrixInner::ComplexBall(right)) => {
                SparseMatrixInner::ComplexBall(left.sub(right).map_err(linalg_error)?)
            }
            (SparseMatrixInner::Polynomial(left), SparseMatrixInner::Polynomial(right)) => {
                SparseMatrixInner::Polynomial(left.sub(right)?)
            }
            (SparseMatrixInner::Quotient(left), SparseMatrixInner::Quotient(right)) => {
                SparseMatrixInner::Quotient(left.sub(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "sparse matrix subtraction",
                    self.context_name(),
                    other.context_name(),
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            SparseMatrixInner::Integer(matrix) => {
                SparseMatrixInner::Integer(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::Rational(matrix) => {
                SparseMatrixInner::Rational(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::Residue(matrix) => {
                SparseMatrixInner::Residue(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::FiniteField(matrix) => {
                SparseMatrixInner::FiniteField(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::Padic(matrix) => {
                SparseMatrixInner::Padic(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::PadicField(matrix) => {
                SparseMatrixInner::PadicField(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::Qadic(matrix) => {
                SparseMatrixInner::Qadic(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::QadicField(matrix) => {
                SparseMatrixInner::QadicField(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::PadicLocalField(matrix) => {
                SparseMatrixInner::PadicLocalField(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => {
                SparseMatrixInner::PadicLocalFieldOrder(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::RealBall(matrix) => {
                SparseMatrixInner::RealBall(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::ComplexBall(matrix) => {
                SparseMatrixInner::ComplexBall(matrix.neg().map_err(linalg_error)?)
            }
            SparseMatrixInner::Polynomial(matrix) => SparseMatrixInner::Polynomial(matrix.neg()?),
            SparseMatrixInner::Quotient(matrix) => SparseMatrixInner::Quotient(matrix.neg()?),
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            SparseMatrixInner::Integer(matrix) => SparseMatrixInner::Integer(
                matrix
                    .scalar_mul(&extract_integer(scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::Rational(matrix) => SparseMatrixInner::Rational(
                matrix
                    .scalar_mul(&extract_rational(py, scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::Residue(matrix) => SparseMatrixInner::Residue(
                matrix
                    .scalar_mul(&extract_residue(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::FiniteField(matrix) => SparseMatrixInner::FiniteField(
                matrix
                    .scalar_mul(&extract_finite_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::Padic(matrix) => SparseMatrixInner::Padic(
                matrix
                    .scalar_mul(&extract_padic_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::PadicField(matrix) => SparseMatrixInner::PadicField(
                matrix
                    .scalar_mul(&extract_padic_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::Qadic(matrix) => SparseMatrixInner::Qadic(
                matrix
                    .scalar_mul(&extract_qadic_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::QadicField(matrix) => SparseMatrixInner::QadicField(
                matrix
                    .scalar_mul(&extract_qadic_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::PadicLocalField(matrix) => SparseMatrixInner::PadicLocalField(
                matrix
                    .scalar_mul(&extract_padic_local_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::PadicLocalFieldOrder(matrix) => {
                SparseMatrixInner::PadicLocalFieldOrder(
                    matrix
                        .scalar_mul(&extract_padic_local_field_order_element(
                            matrix.ring(),
                            scalar,
                        )?)
                        .map_err(linalg_error)?,
                )
            }
            SparseMatrixInner::RealBall(matrix) => SparseMatrixInner::RealBall(
                matrix
                    .scalar_mul(&extract_real_ball(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::ComplexBall(matrix) => SparseMatrixInner::ComplexBall(
                matrix
                    .scalar_mul(&extract_complex_ball(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            SparseMatrixInner::Polynomial(matrix) => {
                SparseMatrixInner::Polynomial(matrix.scalar_mul(py, scalar)?)
            }
            SparseMatrixInner::Quotient(matrix) => {
                SparseMatrixInner::Quotient(matrix.scalar_mul(scalar)?)
            }
        };
        Ok(Self { inner })
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, scalar)
    }

    fn __matmul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let left_context = self.context_name();
        let right_context = other.context_name();
        let inner = py.detach(|| -> PyResult<SparseMatrixInner> {
            let inner = match (&self.inner, &other.inner) {
                (SparseMatrixInner::Integer(left), SparseMatrixInner::Integer(right)) => {
                    SparseMatrixInner::Integer(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Rational(left), SparseMatrixInner::Rational(right)) => {
                    SparseMatrixInner::Rational(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Residue(left), SparseMatrixInner::Residue(right)) => {
                    SparseMatrixInner::Residue(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::FiniteField(left), SparseMatrixInner::FiniteField(right)) => {
                    SparseMatrixInner::FiniteField(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Padic(left), SparseMatrixInner::Padic(right)) => {
                    SparseMatrixInner::Padic(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::PadicField(left), SparseMatrixInner::PadicField(right)) => {
                    SparseMatrixInner::PadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Qadic(left), SparseMatrixInner::Qadic(right)) => {
                    SparseMatrixInner::Qadic(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::QadicField(left), SparseMatrixInner::QadicField(right)) => {
                    SparseMatrixInner::QadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (
                    SparseMatrixInner::PadicLocalField(left),
                    SparseMatrixInner::PadicLocalField(right),
                ) => SparseMatrixInner::PadicLocalField(left.matmul(right).map_err(linalg_error)?),
                (
                    SparseMatrixInner::PadicLocalFieldOrder(left),
                    SparseMatrixInner::PadicLocalFieldOrder(right),
                ) => SparseMatrixInner::PadicLocalFieldOrder(
                    left.matmul(right).map_err(linalg_error)?,
                ),
                (SparseMatrixInner::RealBall(left), SparseMatrixInner::RealBall(right)) => {
                    SparseMatrixInner::RealBall(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::ComplexBall(left), SparseMatrixInner::ComplexBall(right)) => {
                    SparseMatrixInner::ComplexBall(left.matmul(right).map_err(linalg_error)?)
                }
                (SparseMatrixInner::Polynomial(left), SparseMatrixInner::Polynomial(right)) => {
                    SparseMatrixInner::Polynomial(left.matmul(right)?)
                }
                (SparseMatrixInner::Quotient(left), SparseMatrixInner::Quotient(right)) => {
                    SparseMatrixInner::Quotient(left.matmul(right)?)
                }
                _ => {
                    return Err(context_mismatch(
                        "sparse matrix multiplication",
                        left_context,
                        right_context,
                    ))
                }
            };
            Ok(inner)
        })?;
        Ok(Self { inner })
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (SparseMatrixInner::Integer(left), SparseMatrixInner::Integer(right)) => left == right,
            (SparseMatrixInner::Rational(left), SparseMatrixInner::Rational(right)) => {
                left == right
            }
            (SparseMatrixInner::Residue(left), SparseMatrixInner::Residue(right)) => left == right,
            (SparseMatrixInner::FiniteField(left), SparseMatrixInner::FiniteField(right)) => {
                left == right
            }
            (SparseMatrixInner::Padic(left), SparseMatrixInner::Padic(right)) => left == right,
            (SparseMatrixInner::PadicField(left), SparseMatrixInner::PadicField(right)) => {
                left == right
            }
            (SparseMatrixInner::Qadic(left), SparseMatrixInner::Qadic(right)) => left == right,
            (SparseMatrixInner::QadicField(left), SparseMatrixInner::QadicField(right)) => {
                left == right
            }
            (
                SparseMatrixInner::PadicLocalField(left),
                SparseMatrixInner::PadicLocalField(right),
            ) => left == right,
            (
                SparseMatrixInner::PadicLocalFieldOrder(left),
                SparseMatrixInner::PadicLocalFieldOrder(right),
            ) => left == right,
            (SparseMatrixInner::RealBall(left), SparseMatrixInner::RealBall(right)) => {
                left == right
            }
            (SparseMatrixInner::ComplexBall(left), SparseMatrixInner::ComplexBall(right)) => {
                left == right
            }
            (SparseMatrixInner::Polynomial(left), SparseMatrixInner::Polynomial(right)) => {
                left.equals(right)
            }
            (SparseMatrixInner::Quotient(left), SparseMatrixInner::Quotient(right)) => {
                left.equals(right)
            }
            _ => false,
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            let shape = self.dimensions();
            format!(
                "SparseMatrix({}, shape={}x{}, nnz={})",
                self.context_name(),
                shape.0,
                shape.1,
                self.nonzero_count()
            )
        })
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", Self::type_id())?;
        summary.set_item("ring", self.context_name())?;
        summary.set_item("shape", self.dimensions())?;
        summary.set_item("nnz", self.nonzero_count())?;
        summary.set_item("density", self.density())?;
        Ok(summary.unbind())
    }
}

fn collect_values<E>(
    values: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<E>> {
    values.try_iter()?.map(|entry| extract(&entry?)).collect()
}

fn collect_sparse_triplets<E>(
    py: Python<'_>,
    entries: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<(usize, usize, E)>> {
    entries
        .try_iter()?
        .map(|entry| {
            let (row, column, value) = entry?.extract::<(usize, usize, Py<PyAny>)>()?;
            Ok((row, column, extract(value.bind(py))?))
        })
        .collect()
}

fn collect_rows<E>(
    rows: &Bound<'_, PyAny>,
    mut extract: impl FnMut(&Bound<'_, PyAny>) -> PyResult<E>,
) -> PyResult<Vec<Vec<E>>> {
    rows.try_iter()?
        .map(|row| {
            row?.try_iter()?
                .map(|entry| extract(&entry?))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

fn extract_residue(ring: &ResidueRing, value: &Bound<'_, PyAny>) -> PyResult<ResidueClass> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
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

fn extract_finite_field_element(
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
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            element.inner.field(),
            field
        )));
    }
    Err(PyTypeError::new_err(
        "finite-field coefficient must be a Python int or FiniteFieldElement",
    ))
}

fn extract_real_ball(
    py: Python<'_>,
    field: &RealBallField,
    value: &Bound<'_, PyAny>,
) -> PyResult<RealBall> {
    if let Ok(ball) = value.extract::<PyRef<'_, PyRealBall>>() {
        if field.contains(&ball.inner) {
            return Ok(ball.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            ball.inner.field(),
            field
        )));
    }
    Ok(field.element(extract_rational(py, value)?))
}

fn extract_complex_ball(
    py: Python<'_>,
    field: &ComplexBallField,
    value: &Bound<'_, PyAny>,
) -> PyResult<ComplexBall> {
    if let Ok(ball) = value.extract::<PyRef<'_, PyComplexBall>>() {
        if field.contains(&ball.inner) {
            return Ok(ball.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            ball.inner.field(),
            field
        )));
    }
    Ok(field.element(extract_rational(py, value)?))
}

fn extract_padic_element(ring: &PadicRing, value: &Bound<'_, PyAny>) -> PyResult<PadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicElement>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, ring
        )));
    }
    Err(PyTypeError::new_err(
        "p-adic coefficient must be a Python int or PadicElement",
    ))
}

fn extract_padic_field_element(
    field: &PadicField,
    value: &Bound<'_, PyAny>,
) -> PyResult<PadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(field.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, field
        )));
    }
    Err(PyTypeError::new_err(
        "p-adic coefficient must be a Python int or PadicElement",
    ))
}

fn extract_qadic_element(ring: &QadicRing, value: &Bound<'_, PyAny>) -> PyResult<QadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(ring.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyQadicElement>>() {
        if ring.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, ring
        )));
    }
    Err(PyTypeError::new_err(
        "q-adic coefficient must be a Python int or QadicElement",
    ))
}

fn extract_qadic_field_element(
    field: &QadicField,
    value: &Bound<'_, PyAny>,
) -> PyResult<QadicElement> {
    if value.is_instance_of::<PyInt>() {
        return Ok(field.element(extract_integer(value)?));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyQadicElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(PyValueError::new_err(format!(
            "coefficient belongs to {}, expected {}",
            element.inner, field
        )));
    }
    Err(PyTypeError::new_err(
        "q-adic coefficient must be a Python int or QadicElement",
    ))
}

fn extract_padic_local_field_element(
    field: &PadicLocalField,
    value: &Bound<'_, PyAny>,
) -> PyResult<PadicLocalFieldElement> {
    if value.is_instance_of::<PyInt>() {
        let scalar = field
            .coefficient_field()
            .coefficient_field()
            .element(extract_integer(value)?);
        return field
            .element_from_padic(&scalar)
            .map_err(|error| PyValueError::new_err(error.to_string()));
    }
    if let Ok(element) = value.extract::<PyRef<'_, PyPadicLocalFieldElement>>() {
        if field.contains(&element.inner) {
            return Ok(element.inner.clone());
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            field.to_string(),
            element.inner.field().to_string(),
        ));
    }
    Err(PyTypeError::new_err(
        "p-adic local coefficient must be a Python int or PadicLocalFieldElement",
    ))
}

fn extract_padic_local_field_order_element(
    order: &PadicLocalFieldOrder,
    value: &Bound<'_, PyAny>,
) -> PyResult<PadicLocalFieldElement> {
    let element = extract_padic_local_field_element(order.field(), value)?;
    if order.contains(&element) {
        Ok(element)
    } else {
        Err(PyValueError::new_err(format!(
            "coefficient {element} does not belong to {order}"
        )))
    }
}

fn extract_integer_series(
    ring: &PowerSeriesRing<IntegerRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<IntegerRing>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::Integer(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_integer(value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_rational_series(
    py: Python<'_>,
    ring: &PowerSeriesRing<RationalField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<RationalField>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::Rational(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_rational(py, value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_residue_series(
    ring: &PowerSeriesRing<ResidueRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<ResidueRing>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::Residue(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_residue(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_finite_field_series(
    ring: &PowerSeriesRing<FiniteField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<FiniteField>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::FiniteField(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_finite_field_element(
        ring.coefficient_ring(),
        value,
    )?)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_padic_series(
    ring: &PowerSeriesRing<PadicRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<PadicRing>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::Padic(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_padic_element(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_padic_field_series(
    ring: &PowerSeriesRing<PadicField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<PadicField>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::PadicField(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_padic_field_element(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_qadic_series(
    ring: &PowerSeriesRing<QadicRing>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<QadicRing>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::Qadic(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_qadic_element(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_qadic_field_series(
    ring: &PowerSeriesRing<QadicField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<QadicField>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::QadicField(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_qadic_field_element(ring.coefficient_ring(), value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn extract_padic_local_field_series(
    ring: &PowerSeriesRing<PadicLocalField>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PowerSeries<PadicLocalField>> {
    if let Ok(series) = value.extract::<PyRef<'_, PyPowerSeries>>() {
        if let SeriesInner::PadicLocalField(inner) = &series.inner {
            if ring.contains(inner) {
                return Ok(inner.clone());
            }
        }
        return Err(context_mismatch(
            "linear-algebra construction",
            ring.to_string(),
            "a different power-series context".to_owned(),
        ));
    }
    ring.constant(extract_padic_local_field_element(
        ring.coefficient_ring(),
        value,
    )?)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn normalize_index(index: isize, length: usize) -> Option<usize> {
    if index >= 0 {
        usize::try_from(index).ok().filter(|value| *value < length)
    } else {
        index
            .checked_add(isize::try_from(length).ok()?)
            .and_then(|value| usize::try_from(value).ok())
    }
}

#[pymethods]
impl PyMatrix {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Matrix::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Serialize the matrix and its complete coefficient context.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a matrix and its complete coefficient context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: MatrixInner::from_json(value)?,
        })
    }

    #[classmethod]
    fn from_rows(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if ring.is_instance_of::<PyIntegerRing>() {
            let entries = collect_rows(rows, extract_integer)?;
            return Ok(Self {
                inner: MatrixInner::Integer(Matrix::from_rows(Z, entries).map_err(linalg_error)?),
            });
        }
        if ring.is_instance_of::<PyRationalField>() {
            let entries = collect_rows(rows, |value| extract_rational(py, value))?;
            return Ok(Self {
                inner: MatrixInner::Rational(Matrix::from_rows(Q, entries).map_err(linalg_error)?),
            });
        }
        if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            let entries = collect_rows(rows, |value| extract_residue(&residue_ring.inner, value))?;
            return Ok(Self {
                inner: MatrixInner::Residue(
                    Matrix::from_rows(residue_ring.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            let entries = collect_rows(rows, |value| {
                extract_finite_field_element(&field.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::FiniteField(
                    Matrix::from_rows(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(padic_ring) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            let entries = collect_rows(rows, |value| {
                extract_padic_element(&padic_ring.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::Padic(
                    Matrix::from_rows(padic_ring.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(padic_field) = ring.extract::<PyRef<'_, PyPadicField>>() {
            let entries = collect_rows(rows, |value| {
                extract_padic_field_element(&padic_field.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::PadicField(
                    Matrix::from_rows(padic_field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(qadic_ring) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            let entries = collect_rows(rows, |value| {
                extract_qadic_element(&qadic_ring.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::Qadic(
                    Matrix::from_rows(qadic_ring.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(qadic_field) = ring.extract::<PyRef<'_, PyQadicField>>() {
            let entries = collect_rows(rows, |value| {
                extract_qadic_field_element(&qadic_field.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::QadicField(
                    Matrix::from_rows(qadic_field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            let entries = collect_rows(rows, |value| {
                extract_padic_local_field_element(&field.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::PadicLocalField(
                    Matrix::from_rows(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(order) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            let entries = collect_rows(rows, |value| {
                extract_padic_local_field_order_element(&order.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::PadicLocalFieldOrder(
                    Matrix::from_rows(order.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            let entries = collect_rows(rows, |value| {
                extract_number_field_element(py, &field.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::NumberField(
                    Matrix::from_rows(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let field = field.as_rational().ok_or_else(|| {
                PyTypeError::new_err(
                    "dense rational-function linear algebra currently requires base field Q",
                )
            })?;
            let entries = collect_rows(rows, |value| extract_rational_function(py, field, value))?;
            return Ok(Self {
                inner: MatrixInner::RationalFunction(
                    Matrix::from_rows(field.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(order) = ring.extract::<PyRef<'_, PyNumberFieldOrder>>() {
            let entries = collect_rows(rows, |value| {
                extract_number_field_order_element(py, &order.inner, value)
            })?;
            return Ok(Self {
                inner: MatrixInner::NumberFieldOrder(
                    Matrix::from_rows(order.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            let entries = collect_rows(rows, |value| extract_real_ball(py, &field.inner, value))?;
            return Ok(Self {
                inner: MatrixInner::RealBall(
                    Matrix::from_rows(field.inner, entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            let entries =
                collect_rows(rows, |value| extract_complex_ball(py, &field.inner, value))?;
            return Ok(Self {
                inner: MatrixInner::ComplexBall(
                    Matrix::from_rows(field.inner, entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(series_ring) = ring.extract::<PyRef<'_, PyPowerSeriesRing>>() {
            let inner = match &series_ring.inner {
                PowerSeriesRingInner::Integer(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_integer_series(series_ring, value))?;
                    MatrixInner::SeriesInteger(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Rational(series_ring) => {
                    let entries = collect_rows(rows, |value| {
                        extract_rational_series(py, series_ring, value)
                    })?;
                    MatrixInner::SeriesRational(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Residue(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_residue_series(series_ring, value))?;
                    MatrixInner::SeriesResidue(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::FiniteField(series_ring) => {
                    let entries = collect_rows(rows, |value| {
                        extract_finite_field_series(series_ring, value)
                    })?;
                    MatrixInner::SeriesFiniteField(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Padic(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_padic_series(series_ring, value))?;
                    MatrixInner::SeriesPadic(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::PadicField(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_padic_field_series(series_ring, value))?;
                    MatrixInner::SeriesPadicField(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Qadic(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_qadic_series(series_ring, value))?;
                    MatrixInner::SeriesQadic(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::QadicField(series_ring) => {
                    let entries =
                        collect_rows(rows, |value| extract_qadic_field_series(series_ring, value))?;
                    MatrixInner::SeriesQadicField(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::PadicLocalField(series_ring) => {
                    let entries = collect_rows(rows, |value| {
                        extract_padic_local_field_series(series_ring, value)
                    })?;
                    MatrixInner::SeriesPadicLocalField(
                        Matrix::from_rows(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
            };
            return Ok(Self { inner });
        }
        if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            let inner = match &ring.inner {
                QuotientRingInner::Rational(ring) => QuotientMatrixInner::Rational(
                    Matrix::from_rows(
                        ring.clone(),
                        collect_rows(rows, |value| extract_rational_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
                QuotientRingInner::FiniteField(ring) => QuotientMatrixInner::FiniteField(
                    Matrix::from_rows(
                        ring.clone(),
                        collect_rows(rows, |value| extract_finite_field_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
            };
            return Ok(Self {
                inner: MatrixInner::Quotient(inner),
            });
        }
        if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialMatrixInner::Integer(
                        Matrix::from_rows(
                            ring.clone(),
                            collect_rows(rows, |value| extract_integer_polynomial(ring, value))?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialMatrixInner::Rational(
                        Matrix::from_rows(
                            ring.clone(),
                            collect_rows(rows, |value| {
                                extract_rational_polynomial(py, ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialMatrixInner::Residue(
                        Matrix::from_rows(
                            ring.clone(),
                            collect_rows(rows, |value| extract_residue_polynomial(ring, value))?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialMatrixInner::FiniteField(
                        Matrix::from_rows(
                            ring.clone(),
                            collect_rows(rows, |value| {
                                extract_finite_field_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            return Ok(Self {
                inner: MatrixInner::Polynomial(inner),
            });
        }
        Err(PyTypeError::new_err(
            "Matrix.from_rows supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, NumberField, NumberFieldOrder, RationalFunctionField over Q, RealBallField, ComplexBallField, PolynomialRing, PowerSeriesRing, and QuotientRing contexts",
        ))
    }

    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        ring: &Bound<'_, PyAny>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        validate_python_shape(shape.0, shape.1)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            MatrixInner::Integer(Matrix::zeros(Z, shape.0, shape.1))
        } else if ring.is_instance_of::<PyRationalField>() {
            MatrixInner::Rational(Matrix::zeros(Q, shape.0, shape.1))
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            MatrixInner::Residue(Matrix::zeros(residue_ring.inner.clone(), shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            MatrixInner::FiniteField(Matrix::zeros(field.inner.clone(), shape.0, shape.1))
        } else if let Ok(padic_ring) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            MatrixInner::Padic(Matrix::zeros(padic_ring.inner.clone(), shape.0, shape.1))
        } else if let Ok(padic_field) = ring.extract::<PyRef<'_, PyPadicField>>() {
            MatrixInner::PadicField(Matrix::zeros(padic_field.inner.clone(), shape.0, shape.1))
        } else if let Ok(qadic_ring) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            MatrixInner::Qadic(Matrix::zeros(qadic_ring.inner.clone(), shape.0, shape.1))
        } else if let Ok(qadic_field) = ring.extract::<PyRef<'_, PyQadicField>>() {
            MatrixInner::QadicField(Matrix::zeros(qadic_field.inner.clone(), shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            MatrixInner::PadicLocalField(Matrix::zeros(field.inner.clone(), shape.0, shape.1))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            MatrixInner::PadicLocalFieldOrder(Matrix::zeros(order.inner.clone(), shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            MatrixInner::NumberField(Matrix::zeros(field.inner.clone(), shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let field = field.as_rational().ok_or_else(|| {
                PyTypeError::new_err(
                    "dense rational-function linear algebra currently requires base field Q",
                )
            })?;
            MatrixInner::RationalFunction(Matrix::zeros(field.clone(), shape.0, shape.1))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyNumberFieldOrder>>() {
            MatrixInner::NumberFieldOrder(Matrix::zeros(order.inner.clone(), shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            MatrixInner::RealBall(Matrix::zeros(field.inner, shape.0, shape.1))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            MatrixInner::ComplexBall(Matrix::zeros(field.inner, shape.0, shape.1))
        } else if let Ok(series_ring) = ring.extract::<PyRef<'_, PyPowerSeriesRing>>() {
            match &series_ring.inner {
                PowerSeriesRingInner::Integer(ring) => {
                    MatrixInner::SeriesInteger(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::Rational(ring) => {
                    MatrixInner::SeriesRational(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::Residue(ring) => {
                    MatrixInner::SeriesResidue(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::FiniteField(ring) => {
                    MatrixInner::SeriesFiniteField(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::Padic(ring) => {
                    MatrixInner::SeriesPadic(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::PadicField(ring) => {
                    MatrixInner::SeriesPadicField(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::Qadic(ring) => {
                    MatrixInner::SeriesQadic(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::QadicField(ring) => {
                    MatrixInner::SeriesQadicField(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                PowerSeriesRingInner::PadicLocalField(ring) => MatrixInner::SeriesPadicLocalField(
                    Matrix::zeros(ring.clone(), shape.0, shape.1),
                ),
            }
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            MatrixInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => {
                    QuotientMatrixInner::Rational(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                QuotientRingInner::FiniteField(ring) => {
                    QuotientMatrixInner::FiniteField(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialMatrixInner::Integer(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialMatrixInner::Rational(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialMatrixInner::Residue(Matrix::zeros(ring.clone(), shape.0, shape.1))
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialMatrixInner::FiniteField(Matrix::zeros(
                        ring.clone(),
                        shape.0,
                        shape.1,
                    ))
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            MatrixInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "Matrix.zeros supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, NumberField, NumberFieldOrder, RealBallField, ComplexBallField, PolynomialRing, PowerSeriesRing, and QuotientRing contexts",
            ));
        };
        Ok(Self { inner })
    }

    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, ring: &Bound<'_, PyAny>, size: usize) -> PyResult<Self> {
        validate_python_shape(size, size)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            MatrixInner::Integer(Matrix::identity(Z, size))
        } else if ring.is_instance_of::<PyRationalField>() {
            MatrixInner::Rational(Matrix::identity(Q, size))
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            MatrixInner::Residue(Matrix::identity(residue_ring.inner.clone(), size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            MatrixInner::FiniteField(Matrix::identity(field.inner.clone(), size))
        } else if let Ok(padic_ring) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            MatrixInner::Padic(Matrix::identity(padic_ring.inner.clone(), size))
        } else if let Ok(padic_field) = ring.extract::<PyRef<'_, PyPadicField>>() {
            MatrixInner::PadicField(Matrix::identity(padic_field.inner.clone(), size))
        } else if let Ok(qadic_ring) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            MatrixInner::Qadic(Matrix::identity(qadic_ring.inner.clone(), size))
        } else if let Ok(qadic_field) = ring.extract::<PyRef<'_, PyQadicField>>() {
            MatrixInner::QadicField(Matrix::identity(qadic_field.inner.clone(), size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            MatrixInner::PadicLocalField(Matrix::identity(field.inner.clone(), size))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            MatrixInner::PadicLocalFieldOrder(Matrix::identity(order.inner.clone(), size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            MatrixInner::NumberField(Matrix::identity(field.inner.clone(), size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let field = field.as_rational().ok_or_else(|| {
                PyTypeError::new_err(
                    "dense rational-function linear algebra currently requires base field Q",
                )
            })?;
            MatrixInner::RationalFunction(Matrix::identity(field.clone(), size))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyNumberFieldOrder>>() {
            MatrixInner::NumberFieldOrder(Matrix::identity(order.inner.clone(), size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            MatrixInner::RealBall(Matrix::identity(field.inner, size))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            MatrixInner::ComplexBall(Matrix::identity(field.inner, size))
        } else if let Ok(series_ring) = ring.extract::<PyRef<'_, PyPowerSeriesRing>>() {
            match &series_ring.inner {
                PowerSeriesRingInner::Integer(ring) => {
                    MatrixInner::SeriesInteger(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::Rational(ring) => {
                    MatrixInner::SeriesRational(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::Residue(ring) => {
                    MatrixInner::SeriesResidue(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::FiniteField(ring) => {
                    MatrixInner::SeriesFiniteField(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::Padic(ring) => {
                    MatrixInner::SeriesPadic(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::PadicField(ring) => {
                    MatrixInner::SeriesPadicField(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::Qadic(ring) => {
                    MatrixInner::SeriesQadic(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::QadicField(ring) => {
                    MatrixInner::SeriesQadicField(Matrix::identity(ring.clone(), size))
                }
                PowerSeriesRingInner::PadicLocalField(ring) => {
                    MatrixInner::SeriesPadicLocalField(Matrix::identity(ring.clone(), size))
                }
            }
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            MatrixInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => {
                    QuotientMatrixInner::Rational(Matrix::identity(ring.clone(), size))
                }
                QuotientRingInner::FiniteField(ring) => {
                    QuotientMatrixInner::FiniteField(Matrix::identity(ring.clone(), size))
                }
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialMatrixInner::Integer(Matrix::identity(ring.clone(), size))
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialMatrixInner::Rational(Matrix::identity(ring.clone(), size))
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialMatrixInner::Residue(Matrix::identity(ring.clone(), size))
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialMatrixInner::FiniteField(Matrix::identity(ring.clone(), size))
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "polynomial matrices currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            MatrixInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "Matrix.identity supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, NumberField, NumberFieldOrder, RealBallField, ComplexBallField, PolynomialRing, PowerSeriesRing, and QuotientRing contexts",
            ));
        };
        Ok(Self { inner })
    }

    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            MatrixInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            MatrixInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            MatrixInner::Residue(matrix) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::FiniteField(matrix) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::Padic(matrix) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::PadicField(matrix) => Ok(Py::new(
                py,
                PyPadicField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::Qadic(matrix) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::QadicField(matrix) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::PadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPadicLocalField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::PadicLocalFieldOrder(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldOrder {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::NumberField(matrix) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::RationalFunction(matrix) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(matrix.ring().clone()),
            )?
            .into_any()),
            MatrixInner::NumberFieldOrder(matrix) => Ok(Py::new(
                py,
                PyNumberFieldOrder {
                    inner: matrix.ring().clone(),
                },
            )?
            .into_any()),
            MatrixInner::RealBall(matrix) => Ok(Py::new(
                py,
                PyRealBallField {
                    inner: *matrix.ring(),
                },
            )?
            .into_any()),
            MatrixInner::ComplexBall(matrix) => Ok(Py::new(
                py,
                PyComplexBallField {
                    inner: *matrix.ring(),
                },
            )?
            .into_any()),
            MatrixInner::SeriesInteger(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Integer(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesRational(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Rational(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesResidue(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Residue(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesFiniteField(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::FiniteField(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadic(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Padic(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadicField(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::PadicField(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesQadic(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Qadic(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesQadicField(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::QadicField(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::PadicLocalField(matrix.ring().clone()),
                },
            )?
            .into_any()),
            MatrixInner::Polynomial(matrix) => matrix.ring_to_python(py),
            MatrixInner::Quotient(matrix) => matrix.ring_to_python(py),
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dimensions()
    }

    #[getter]
    fn input_dimension(&self) -> usize {
        self.dimensions().1
    }

    #[getter]
    fn output_dimension(&self) -> usize {
        self.dimensions().0
    }

    fn to_rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // NOETHER_RELEASES_GIL: rows_to_python detaches Rust materialization.
        self.rows_to_python(py)
    }

    fn transpose(&self) -> Self {
        let inner = match &self.inner {
            MatrixInner::Integer(matrix) => MatrixInner::Integer(matrix.transpose()),
            MatrixInner::Rational(matrix) => MatrixInner::Rational(matrix.transpose()),
            MatrixInner::Residue(matrix) => MatrixInner::Residue(matrix.transpose()),
            MatrixInner::FiniteField(matrix) => MatrixInner::FiniteField(matrix.transpose()),
            MatrixInner::Padic(matrix) => MatrixInner::Padic(matrix.transpose()),
            MatrixInner::PadicField(matrix) => MatrixInner::PadicField(matrix.transpose()),
            MatrixInner::Qadic(matrix) => MatrixInner::Qadic(matrix.transpose()),
            MatrixInner::QadicField(matrix) => MatrixInner::QadicField(matrix.transpose()),
            MatrixInner::PadicLocalField(matrix) => {
                MatrixInner::PadicLocalField(matrix.transpose())
            }
            MatrixInner::PadicLocalFieldOrder(matrix) => {
                MatrixInner::PadicLocalFieldOrder(matrix.transpose())
            }
            MatrixInner::NumberField(matrix) => MatrixInner::NumberField(matrix.transpose()),
            MatrixInner::RationalFunction(matrix) => {
                MatrixInner::RationalFunction(matrix.transpose())
            }
            MatrixInner::NumberFieldOrder(matrix) => {
                MatrixInner::NumberFieldOrder(matrix.transpose())
            }
            MatrixInner::RealBall(matrix) => MatrixInner::RealBall(matrix.transpose()),
            MatrixInner::ComplexBall(matrix) => MatrixInner::ComplexBall(matrix.transpose()),
            MatrixInner::SeriesInteger(matrix) => MatrixInner::SeriesInteger(matrix.transpose()),
            MatrixInner::SeriesRational(matrix) => MatrixInner::SeriesRational(matrix.transpose()),
            MatrixInner::SeriesResidue(matrix) => MatrixInner::SeriesResidue(matrix.transpose()),
            MatrixInner::SeriesFiniteField(matrix) => {
                MatrixInner::SeriesFiniteField(matrix.transpose())
            }
            MatrixInner::SeriesPadic(matrix) => MatrixInner::SeriesPadic(matrix.transpose()),
            MatrixInner::SeriesPadicField(matrix) => {
                MatrixInner::SeriesPadicField(matrix.transpose())
            }
            MatrixInner::SeriesQadic(matrix) => MatrixInner::SeriesQadic(matrix.transpose()),
            MatrixInner::SeriesQadicField(matrix) => {
                MatrixInner::SeriesQadicField(matrix.transpose())
            }
            MatrixInner::SeriesPadicLocalField(matrix) => {
                MatrixInner::SeriesPadicLocalField(matrix.transpose())
            }
            MatrixInner::Polynomial(matrix) => MatrixInner::Polynomial(matrix.transpose()),
            MatrixInner::Quotient(matrix) => MatrixInner::Quotient(matrix.transpose()),
        };
        Self { inner }
    }

    fn matmul(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let left_context = self.context_name();
        let right_context = other.context_name();
        let inner = py.detach(|| -> PyResult<MatrixInner> {
            let inner = match (&self.inner, &other.inner) {
                (MatrixInner::Integer(left), MatrixInner::Integer(right)) => {
                    MatrixInner::Integer(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::Rational(left), MatrixInner::Rational(right)) => {
                    MatrixInner::Rational(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::Residue(left), MatrixInner::Residue(right)) => {
                    MatrixInner::Residue(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::FiniteField(left), MatrixInner::FiniteField(right)) => {
                    MatrixInner::FiniteField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::Padic(left), MatrixInner::Padic(right)) => {
                    MatrixInner::Padic(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::PadicField(left), MatrixInner::PadicField(right)) => {
                    MatrixInner::PadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::Qadic(left), MatrixInner::Qadic(right)) => {
                    MatrixInner::Qadic(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::QadicField(left), MatrixInner::QadicField(right)) => {
                    MatrixInner::QadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::PadicLocalField(left), MatrixInner::PadicLocalField(right)) => {
                    MatrixInner::PadicLocalField(left.matmul(right).map_err(linalg_error)?)
                }
                (
                    MatrixInner::PadicLocalFieldOrder(left),
                    MatrixInner::PadicLocalFieldOrder(right),
                ) => MatrixInner::PadicLocalFieldOrder(left.matmul(right).map_err(linalg_error)?),
                (MatrixInner::NumberField(left), MatrixInner::NumberField(right)) => {
                    MatrixInner::NumberField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::RationalFunction(left), MatrixInner::RationalFunction(right)) => {
                    MatrixInner::RationalFunction(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::NumberFieldOrder(left), MatrixInner::NumberFieldOrder(right)) => {
                    MatrixInner::NumberFieldOrder(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::RealBall(left), MatrixInner::RealBall(right)) => {
                    MatrixInner::RealBall(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::ComplexBall(left), MatrixInner::ComplexBall(right)) => {
                    MatrixInner::ComplexBall(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesInteger(left), MatrixInner::SeriesInteger(right)) => {
                    MatrixInner::SeriesInteger(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesRational(left), MatrixInner::SeriesRational(right)) => {
                    MatrixInner::SeriesRational(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesResidue(left), MatrixInner::SeriesResidue(right)) => {
                    MatrixInner::SeriesResidue(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesFiniteField(left), MatrixInner::SeriesFiniteField(right)) => {
                    MatrixInner::SeriesFiniteField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesPadic(left), MatrixInner::SeriesPadic(right)) => {
                    MatrixInner::SeriesPadic(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesPadicField(left), MatrixInner::SeriesPadicField(right)) => {
                    MatrixInner::SeriesPadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesQadic(left), MatrixInner::SeriesQadic(right)) => {
                    MatrixInner::SeriesQadic(left.matmul(right).map_err(linalg_error)?)
                }
                (MatrixInner::SeriesQadicField(left), MatrixInner::SeriesQadicField(right)) => {
                    MatrixInner::SeriesQadicField(left.matmul(right).map_err(linalg_error)?)
                }
                (
                    MatrixInner::SeriesPadicLocalField(left),
                    MatrixInner::SeriesPadicLocalField(right),
                ) => MatrixInner::SeriesPadicLocalField(left.matmul(right).map_err(linalg_error)?),
                (MatrixInner::Polynomial(left), MatrixInner::Polynomial(right)) => {
                    MatrixInner::Polynomial(left.matmul(right)?)
                }
                (MatrixInner::Quotient(left), MatrixInner::Quotient(right)) => {
                    MatrixInner::Quotient(left.matmul(right)?)
                }
                _ => {
                    return Err(context_mismatch(
                        "matrix multiplication",
                        left_context,
                        right_context,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(Self { inner })
    }

    fn apply(&self, py: Python<'_>, vector: &PyVector) -> PyResult<PyVector> {
        py.detach(|| self.apply_vector(vector))
    }

    fn apply_block(&self, py: Python<'_>, block: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: matmul owns and detaches the block application.
        self.matmul(py, block)
    }

    /// Rank over a field context.
    fn rank(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| match &self.inner {
            MatrixInner::Rational(matrix) => matrix.rank().map_err(linalg_error),
            MatrixInner::FiniteField(matrix) => matrix.rank().map_err(linalg_error),
            MatrixInner::NumberField(matrix) => matrix.rank().map_err(linalg_error),
            MatrixInner::RationalFunction(matrix) => matrix.rank().map_err(linalg_error),
            _ => Err(unsupported_matrix_operation(
                "rank",
                self.inner.context_type_id(),
                FIELD_MATRIX_CONTEXTS,
            )),
        })
    }

    /// Reduced row-echelon form over a field context.
    fn rref(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| -> PyResult<MatrixInner> {
            let inner = match &self.inner {
                MatrixInner::Rational(matrix) => {
                    MatrixInner::Rational(matrix.rref().map_err(linalg_error)?)
                }
                MatrixInner::FiniteField(matrix) => {
                    MatrixInner::FiniteField(matrix.rref().map_err(linalg_error)?)
                }
                MatrixInner::NumberField(matrix) => {
                    MatrixInner::NumberField(matrix.rref().map_err(linalg_error)?)
                }
                MatrixInner::RationalFunction(matrix) => {
                    MatrixInner::RationalFunction(matrix.rref().map_err(linalg_error)?)
                }
                _ => {
                    return Err(unsupported_matrix_operation(
                        "rref",
                        self.inner.context_type_id(),
                        FIELD_MATRIX_CONTEXTS,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(Self { inner })
    }

    /// Rank-revealing exact PLUQ factorization over a field context.
    fn pluq(&self, py: Python<'_>) -> PyResult<PyPLUQFactorization> {
        let inner = py.detach(|| -> PyResult<PLUQFactorizationInner> {
            let inner = match &self.inner {
                MatrixInner::Rational(matrix) => {
                    PLUQFactorizationInner::Rational(matrix.pluq().map_err(linalg_error)?)
                }
                MatrixInner::FiniteField(matrix) => {
                    PLUQFactorizationInner::FiniteField(matrix.pluq().map_err(linalg_error)?)
                }
                MatrixInner::NumberField(matrix) => {
                    PLUQFactorizationInner::NumberField(matrix.pluq().map_err(linalg_error)?)
                }
                MatrixInner::RationalFunction(matrix) => {
                    PLUQFactorizationInner::RationalFunction(matrix.pluq().map_err(linalg_error)?)
                }
                _ => {
                    return Err(unsupported_matrix_operation(
                        "pluq",
                        self.inner.context_type_id(),
                        FIELD_MATRIX_CONTEXTS,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PyPLUQFactorization { inner })
    }

    /// Solve `self @ x = rhs` and return the complete affine solution set.
    fn solve(&self, py: Python<'_>, rhs: &PyVector) -> PyResult<PySolutionSpace> {
        let source_context = self.context_name();
        let right_context = rhs.context_name();
        let inner = py.detach(|| -> PyResult<SolutionSpaceInner> {
            let inner = match (&self.inner, &rhs.inner) {
                (MatrixInner::Rational(matrix), VectorInner::Rational(rhs)) => {
                    SolutionSpaceInner::Rational(matrix.solve(rhs).map_err(linalg_error)?)
                }
                (MatrixInner::FiniteField(matrix), VectorInner::FiniteField(rhs)) => {
                    SolutionSpaceInner::FiniteField(matrix.solve(rhs).map_err(linalg_error)?)
                }
                (MatrixInner::NumberField(matrix), VectorInner::NumberField(rhs)) => {
                    SolutionSpaceInner::NumberField(matrix.solve(rhs).map_err(linalg_error)?)
                }
                (MatrixInner::RationalFunction(matrix), VectorInner::RationalFunction(rhs)) => {
                    SolutionSpaceInner::RationalFunction(matrix.solve(rhs).map_err(linalg_error)?)
                }
                (
                    MatrixInner::Rational(_)
                    | MatrixInner::FiniteField(_)
                    | MatrixInner::NumberField(_)
                    | MatrixInner::RationalFunction(_),
                    _,
                ) => {
                    return Err(context_mismatch(
                        "linear solve",
                        source_context,
                        right_context,
                    ));
                }
                _ => {
                    return Err(unsupported_matrix_operation(
                        "solve",
                        self.inner.context_type_id(),
                        FIELD_MATRIX_CONTEXTS,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PySolutionSpace { inner })
    }

    /// Solve exactly over `Z` or arbitrary `Z/nZ`, retaining complete evidence.
    fn solve_exact(&self, py: Python<'_>, rhs: &PyVector) -> PyResult<Py<PyAny>> {
        let source_context = self.context_name();
        let right_context = rhs.context_name();
        let result = py.detach(|| -> PyResult<ExactLinearSystemInner> {
            match (&self.inner, &rhs.inner) {
                (MatrixInner::Integer(matrix), VectorInner::Integer(rhs)) => Ok(
                    ExactLinearSystemInner::Integer(matrix.solve_exact(rhs).map_err(linalg_error)?),
                ),
                (MatrixInner::Residue(matrix), VectorInner::Residue(rhs)) => {
                    Ok(ExactLinearSystemInner::Congruence(
                        matrix.solve_exact(rhs).map_err(linalg_error)?,
                    ))
                }
                (MatrixInner::Integer(_) | MatrixInner::Residue(_), _) => Err(context_mismatch(
                    "exact linear solve",
                    source_context,
                    right_context,
                )),
                _ => Err(unsupported_matrix_operation(
                    "solve_exact",
                    self.inner.context_type_id(),
                    &[IntegerRing::TYPE_ID, ResidueRing::TYPE_ID],
                )),
            }
        })?;
        match result {
            ExactLinearSystemInner::Integer(inner) => {
                Ok(Py::new(py, PyIntegerLinearSystemResult { inner })?.into_any())
            }
            ExactLinearSystemInner::Congruence(inner) => {
                Ok(Py::new(py, PyCongruenceSystemResult { inner })?.into_any())
            }
        }
    }

    /// Complete right kernel as a checked linear subspace.
    fn kernel(&self, py: Python<'_>) -> PyResult<PySolutionSpace> {
        let inner = py.detach(|| -> PyResult<SolutionSpaceInner> {
            let inner = match &self.inner {
                MatrixInner::Rational(matrix) => {
                    SolutionSpaceInner::Rational(matrix.kernel().map_err(linalg_error)?)
                }
                MatrixInner::FiniteField(matrix) => {
                    SolutionSpaceInner::FiniteField(matrix.kernel().map_err(linalg_error)?)
                }
                MatrixInner::NumberField(matrix) => {
                    SolutionSpaceInner::NumberField(matrix.kernel().map_err(linalg_error)?)
                }
                MatrixInner::RationalFunction(matrix) => {
                    SolutionSpaceInner::RationalFunction(matrix.kernel().map_err(linalg_error)?)
                }
                _ => {
                    return Err(unsupported_matrix_operation(
                        "kernel",
                        self.inner.context_type_id(),
                        FIELD_MATRIX_CONTEXTS,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PySolutionSpace { inner })
    }

    /// Column image as a checked linear subspace.
    fn image(&self, py: Python<'_>) -> PyResult<PySolutionSpace> {
        let inner = py.detach(|| -> PyResult<SolutionSpaceInner> {
            let inner = match &self.inner {
                MatrixInner::Rational(matrix) => {
                    SolutionSpaceInner::Rational(matrix.image().map_err(linalg_error)?)
                }
                MatrixInner::FiniteField(matrix) => {
                    SolutionSpaceInner::FiniteField(matrix.image().map_err(linalg_error)?)
                }
                MatrixInner::NumberField(matrix) => {
                    SolutionSpaceInner::NumberField(matrix.image().map_err(linalg_error)?)
                }
                MatrixInner::RationalFunction(matrix) => {
                    SolutionSpaceInner::RationalFunction(matrix.image().map_err(linalg_error)?)
                }
                _ => {
                    return Err(unsupported_matrix_operation(
                        "image",
                        self.inner.context_type_id(),
                        FIELD_MATRIX_CONTEXTS,
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PySolutionSpace { inner })
    }

    /// Determinant over a commutative coefficient context.
    fn det(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let determinant = py.detach(|| -> PyResult<DeterminantInner> {
            Ok(match &self.inner {
                MatrixInner::Integer(matrix) => {
                    DeterminantInner::Integer(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::Rational(matrix) => {
                    DeterminantInner::Rational(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::Residue(matrix) => {
                    DeterminantInner::Residue(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::FiniteField(matrix) => {
                    DeterminantInner::FiniteField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::Padic(matrix) => {
                    DeterminantInner::Padic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::PadicField(matrix) => {
                    DeterminantInner::Padic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::Qadic(matrix) => {
                    DeterminantInner::Qadic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::QadicField(matrix) => {
                    DeterminantInner::Qadic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::PadicLocalField(matrix) => {
                    DeterminantInner::PadicLocalField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::PadicLocalFieldOrder(matrix) => {
                    DeterminantInner::PadicLocalField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::NumberField(matrix) => {
                    DeterminantInner::NumberField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::RationalFunction(matrix) => {
                    DeterminantInner::RationalFunction(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::NumberFieldOrder(matrix) => {
                    DeterminantInner::NumberField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::RealBall(matrix) => {
                    DeterminantInner::RealBall(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::ComplexBall(matrix) => {
                    DeterminantInner::ComplexBall(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesInteger(matrix) => {
                    DeterminantInner::SeriesInteger(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesRational(matrix) => {
                    DeterminantInner::SeriesRational(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesResidue(matrix) => {
                    DeterminantInner::SeriesResidue(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesFiniteField(matrix) => {
                    DeterminantInner::SeriesFiniteField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesPadic(matrix) => {
                    DeterminantInner::SeriesPadic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesPadicField(matrix) => {
                    DeterminantInner::SeriesPadicField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesQadic(matrix) => {
                    DeterminantInner::SeriesQadic(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesQadicField(matrix) => {
                    DeterminantInner::SeriesQadicField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::SeriesPadicLocalField(matrix) => {
                    DeterminantInner::SeriesPadicLocalField(matrix.det().map_err(linalg_error)?)
                }
                MatrixInner::Polynomial(matrix) => match matrix {
                    PolynomialMatrixInner::Integer(matrix) => {
                        DeterminantInner::PolynomialInteger(matrix.det().map_err(linalg_error)?)
                    }
                    PolynomialMatrixInner::Rational(matrix) => {
                        DeterminantInner::PolynomialRational(matrix.det().map_err(linalg_error)?)
                    }
                    PolynomialMatrixInner::Residue(matrix) => {
                        DeterminantInner::PolynomialResidue(matrix.det().map_err(linalg_error)?)
                    }
                    PolynomialMatrixInner::FiniteField(matrix) => {
                        DeterminantInner::PolynomialFiniteField(matrix.det().map_err(linalg_error)?)
                    }
                },
                MatrixInner::Quotient(matrix) => match matrix {
                    QuotientMatrixInner::Rational(matrix) => {
                        DeterminantInner::QuotientRational(matrix.det().map_err(linalg_error)?)
                    }
                    QuotientMatrixInner::FiniteField(matrix) => {
                        DeterminantInner::QuotientFiniteField(matrix.det().map_err(linalg_error)?)
                    }
                },
            })
        })?;
        match determinant {
            DeterminantInner::Integer(value) => integer_to_python(py, &value),
            DeterminantInner::Rational(value) => rational_to_python(py, &value),
            DeterminantInner::Residue(inner) => {
                Ok(Py::new(py, PyResidueClass { inner })?.into_any())
            }
            DeterminantInner::FiniteField(inner) => {
                Ok(Py::new(py, PyFiniteFieldElement { inner })?.into_any())
            }
            DeterminantInner::Padic(inner) => Ok(Py::new(py, PyPadicElement { inner })?.into_any()),
            DeterminantInner::Qadic(inner) => Ok(Py::new(py, PyQadicElement { inner })?.into_any()),
            DeterminantInner::PadicLocalField(inner) => {
                Ok(Py::new(py, PyPadicLocalFieldElement { inner })?.into_any())
            }
            DeterminantInner::NumberField(inner) => {
                Ok(Py::new(py, PyNumberFieldElement { inner })?.into_any())
            }
            DeterminantInner::RationalFunction(inner) => {
                Ok(Py::new(py, PyRationalFunction::from_rational(inner))?.into_any())
            }
            DeterminantInner::RealBall(inner) => Ok(Py::new(py, PyRealBall { inner })?.into_any()),
            DeterminantInner::ComplexBall(inner) => {
                Ok(Py::new(py, PyComplexBall { inner })?.into_any())
            }
            DeterminantInner::SeriesInteger(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Integer(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesRational(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Rational(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesResidue(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Residue(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesFiniteField(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::FiniteField(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesPadic(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Padic(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesPadicField(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicField(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesQadic(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Qadic(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesQadicField(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::QadicField(inner),
                },
            )?
            .into_any()),
            DeterminantInner::SeriesPadicLocalField(inner) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicLocalField(inner),
                },
            )?
            .into_any()),
            DeterminantInner::PolynomialInteger(inner) => {
                Ok(Py::new(py, PyPolynomial::from_integer(inner))?.into_any())
            }
            DeterminantInner::PolynomialRational(inner) => {
                Ok(Py::new(py, PyPolynomial::from_rational(inner))?.into_any())
            }
            DeterminantInner::PolynomialResidue(inner) => {
                Ok(Py::new(py, PyPolynomial::from_residue(inner))?.into_any())
            }
            DeterminantInner::PolynomialFiniteField(inner) => {
                Ok(Py::new(py, PyPolynomial::from_finite_field(inner))?.into_any())
            }
            DeterminantInner::QuotientRational(inner) => {
                Ok(Py::new(py, PyQuotientElement::from_rational(inner))?.into_any())
            }
            DeterminantInner::QuotientFiniteField(inner) => {
                Ok(Py::new(py, PyQuotientElement::from_finite_field(inner))?.into_any())
            }
        }
    }

    /// Characteristic polynomial `det(tI - self)` over a supported exact
    /// commutative coefficient context.
    fn charpoly(&self, py: Python<'_>) -> PyResult<PyPolynomial> {
        py.detach(|| match &self.inner {
            MatrixInner::Integer(matrix) => Ok(PyPolynomial::from_integer(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::Rational(matrix) => Ok(PyPolynomial::from_rational(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::Residue(matrix) => Ok(PyPolynomial::from_residue(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::FiniteField(matrix) => Ok(PyPolynomial::from_finite_field(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::Padic(matrix) => Ok(PyPolynomial::from_padic(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::PadicField(matrix) => Ok(PyPolynomial::from_padic_field(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::Qadic(matrix) => Ok(PyPolynomial::from_qadic(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::QadicField(matrix) => Ok(PyPolynomial::from_qadic_field(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::NumberField(matrix) => Ok(PyPolynomial::from_number_field(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            MatrixInner::RationalFunction(matrix) => Ok(PyPolynomial::from_rational_function(
                matrix.charpoly().map_err(linalg_error)?,
            )),
            _ => Err(unsupported_matrix_operation(
                "charpoly",
                self.inner.context_type_id(),
                CHARPOLY_MATRIX_CONTEXTS,
            )),
        })
    }

    /// Smith normal form over the integers, without transformation matrices.
    ///
    /// `algorithm="auto"` and `"exact"` use the unconditional fast exact
    /// route. `algorithm="smooth"` requires an inclusive `smooth_bound` and
    /// returns its exact smoothness assumption unless the recorded
    /// deterministic bound is fully covered.
    #[pyo3(signature = (*, algorithm = "auto", smooth_bound = None))]
    fn smith_normal_form(
        &self,
        py: Python<'_>,
        algorithm: &str,
        smooth_bound: Option<u64>,
    ) -> PyResult<PySmithNormalForm> {
        let algorithm = algorithm.to_owned();
        py.detach(|| match &self.inner {
            MatrixInner::Integer(matrix) => {
                let inner = match (algorithm.as_str(), smooth_bound) {
                    ("auto" | "exact", None) => matrix.smith_normal_form(),
                    ("smooth", Some(bound)) => matrix
                        .smith_normal_form_assuming_smooth(bound)
                        .map_err(linalg_error)?,
                    ("smooth", None) => {
                        return Err(PyValueError::new_err(
                            "algorithm='smooth' requires smooth_bound",
                        ));
                    }
                    ("auto" | "exact", Some(_)) => {
                        return Err(PyValueError::new_err(
                            "smooth_bound is only valid with algorithm='smooth'",
                        ));
                    }
                    (name, _) => {
                        return Err(PyValueError::new_err(format!(
                            "unknown Smith algorithm {name:?}; expected 'auto', 'exact', or 'smooth'"
                        )));
                    }
                };
                Ok(PySmithNormalForm { inner })
            }
            _ => Err(unsupported_matrix_operation(
                "smith_normal_form",
                self.inner.context_type_id(),
                &[IntegerRing::TYPE_ID],
            )),
        })
    }

    /// Smith invariant valuations visible below a p-adic ring's precision cap.
    fn padic_smith_invariants(&self, py: Python<'_>) -> PyResult<PyPadicSmithInvariants> {
        match &self.inner {
            MatrixInner::Padic(matrix) => py.detach(|| {
                Ok(PyPadicSmithInvariants {
                    inner: matrix.padic_smith_invariants().map_err(linalg_error)?,
                })
            }),
            _ => Err(unsupported_matrix_operation(
                "padic_smith_invariants",
                self.inner.context_type_id(),
                &[PadicRing::TYPE_ID],
            )),
        }
    }

    /// Canonical row Hermite normal form over the integers.
    fn hermite_normal_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| match &self.inner {
            MatrixInner::Integer(matrix) => Ok(Self {
                inner: MatrixInner::Integer(matrix.hermite_normal_form()),
            }),
            _ => Err(unsupported_matrix_operation(
                "hermite_normal_form",
                self.inner.context_type_id(),
                &[IntegerRing::TYPE_ID],
            )),
        })
    }

    /// Canonical row Howell form over a word-modulus residue ring.
    fn howell_normal_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| match &self.inner {
            MatrixInner::Residue(matrix) => Ok(Self {
                inner: MatrixInner::Residue(matrix.howell_normal_form().map_err(linalg_error)?),
            }),
            _ => Err(unsupported_matrix_operation(
                "howell_normal_form",
                self.inner.context_type_id(),
                &[ResidueRing::TYPE_ID],
            )),
        })
    }

    /// Smith normal form with explicit unimodular transformation matrices.
    fn smith_normal_form_with_transforms(
        &self,
        py: Python<'_>,
    ) -> PyResult<PySmithNormalFormWithTransforms> {
        py.detach(|| match &self.inner {
            MatrixInner::Integer(matrix) => Ok(PySmithNormalFormWithTransforms {
                inner: matrix
                    .smith_normal_form_with_transforms()
                    .map_err(linalg_error)?,
            }),
            _ => Err(unsupported_matrix_operation(
                "smith_normal_form_with_transforms",
                self.inner.context_type_id(),
                &[IntegerRing::TYPE_ID],
            )),
        })
    }

    fn __getitem__(&self, py: Python<'_>, index: (isize, isize)) -> PyResult<Py<PyAny>> {
        let shape = self.dimensions();
        let row = normalize_index(index.0, shape.0).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "matrix index ({}, {}) is outside shape {}x{}",
                index.0, index.1, shape.0, shape.1
            ))
        })?;
        let column = normalize_index(index.1, shape.1).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "matrix index ({}, {}) is outside shape {}x{}",
                index.0, index.1, shape.0, shape.1
            ))
        })?;
        match &self.inner {
            MatrixInner::Integer(matrix) => {
                integer_to_python(py, &matrix.get(row, column).map_err(index_error)?)
            }
            MatrixInner::Rational(matrix) => {
                rational_to_python(py, &matrix.get(row, column).map_err(index_error)?)
            }
            MatrixInner::Residue(matrix) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::FiniteField(matrix) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::Padic(matrix) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::PadicField(matrix) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::Qadic(matrix) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::QadicField(matrix) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::PadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::PadicLocalFieldOrder(matrix) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::NumberField(matrix) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::RationalFunction(matrix) => Ok(Py::new(
                py,
                PyRationalFunction::from_rational(matrix.get(row, column).map_err(index_error)?),
            )?
            .into_any()),
            MatrixInner::NumberFieldOrder(matrix) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::RealBall(matrix) => Ok(Py::new(
                py,
                PyRealBall {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::ComplexBall(matrix) => Ok(Py::new(
                py,
                PyComplexBall {
                    inner: matrix.get(row, column).map_err(index_error)?,
                },
            )?
            .into_any()),
            MatrixInner::SeriesInteger(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Integer(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesRational(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Rational(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesResidue(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Residue(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesFiniteField(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::FiniteField(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadic(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Padic(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadicField(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicField(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesQadic(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Qadic(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesQadicField(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::QadicField(matrix.get(row, column).map_err(index_error)?),
                },
            )?
            .into_any()),
            MatrixInner::SeriesPadicLocalField(matrix) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicLocalField(
                        matrix.get(row, column).map_err(index_error)?,
                    ),
                },
            )?
            .into_any()),
            MatrixInner::Polynomial(matrix) => matrix.get_to_python(py, row, column),
            MatrixInner::Quotient(matrix) => matrix.get_to_python(py, row, column),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self
            .rows_to_python(py)?
            .bind(py)
            .call_method0("__iter__")?
            .unbind())
    }

    fn __len__(&self) -> usize {
        self.dimensions().0
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (MatrixInner::Integer(left), MatrixInner::Integer(right)) => {
                MatrixInner::Integer(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::Rational(left), MatrixInner::Rational(right)) => {
                MatrixInner::Rational(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::Residue(left), MatrixInner::Residue(right)) => {
                MatrixInner::Residue(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::FiniteField(left), MatrixInner::FiniteField(right)) => {
                MatrixInner::FiniteField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::Padic(left), MatrixInner::Padic(right)) => {
                MatrixInner::Padic(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicField(left), MatrixInner::PadicField(right)) => {
                MatrixInner::PadicField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::Qadic(left), MatrixInner::Qadic(right)) => {
                MatrixInner::Qadic(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::QadicField(left), MatrixInner::QadicField(right)) => {
                MatrixInner::QadicField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicLocalField(left), MatrixInner::PadicLocalField(right)) => {
                MatrixInner::PadicLocalField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicLocalFieldOrder(left), MatrixInner::PadicLocalFieldOrder(right)) => {
                MatrixInner::PadicLocalFieldOrder(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::NumberField(left), MatrixInner::NumberField(right)) => {
                MatrixInner::NumberField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::RationalFunction(left), MatrixInner::RationalFunction(right)) => {
                MatrixInner::RationalFunction(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::NumberFieldOrder(left), MatrixInner::NumberFieldOrder(right)) => {
                MatrixInner::NumberFieldOrder(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::RealBall(left), MatrixInner::RealBall(right)) => {
                MatrixInner::RealBall(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::ComplexBall(left), MatrixInner::ComplexBall(right)) => {
                MatrixInner::ComplexBall(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesInteger(left), MatrixInner::SeriesInteger(right)) => {
                MatrixInner::SeriesInteger(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesRational(left), MatrixInner::SeriesRational(right)) => {
                MatrixInner::SeriesRational(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesResidue(left), MatrixInner::SeriesResidue(right)) => {
                MatrixInner::SeriesResidue(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesFiniteField(left), MatrixInner::SeriesFiniteField(right)) => {
                MatrixInner::SeriesFiniteField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadic(left), MatrixInner::SeriesPadic(right)) => {
                MatrixInner::SeriesPadic(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadicField(left), MatrixInner::SeriesPadicField(right)) => {
                MatrixInner::SeriesPadicField(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadic(left), MatrixInner::SeriesQadic(right)) => {
                MatrixInner::SeriesQadic(left.add(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadicField(left), MatrixInner::SeriesQadicField(right)) => {
                MatrixInner::SeriesQadicField(left.add(right).map_err(linalg_error)?)
            }
            (
                MatrixInner::SeriesPadicLocalField(left),
                MatrixInner::SeriesPadicLocalField(right),
            ) => MatrixInner::SeriesPadicLocalField(left.add(right).map_err(linalg_error)?),
            (MatrixInner::Polynomial(left), MatrixInner::Polynomial(right)) => {
                MatrixInner::Polynomial(left.add(right)?)
            }
            (MatrixInner::Quotient(left), MatrixInner::Quotient(right)) => {
                MatrixInner::Quotient(left.add(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "matrix addition",
                    self.context_name(),
                    other.context_name(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (MatrixInner::Integer(left), MatrixInner::Integer(right)) => {
                MatrixInner::Integer(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::Rational(left), MatrixInner::Rational(right)) => {
                MatrixInner::Rational(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::Residue(left), MatrixInner::Residue(right)) => {
                MatrixInner::Residue(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::FiniteField(left), MatrixInner::FiniteField(right)) => {
                MatrixInner::FiniteField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::Padic(left), MatrixInner::Padic(right)) => {
                MatrixInner::Padic(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicField(left), MatrixInner::PadicField(right)) => {
                MatrixInner::PadicField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::Qadic(left), MatrixInner::Qadic(right)) => {
                MatrixInner::Qadic(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::QadicField(left), MatrixInner::QadicField(right)) => {
                MatrixInner::QadicField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicLocalField(left), MatrixInner::PadicLocalField(right)) => {
                MatrixInner::PadicLocalField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::PadicLocalFieldOrder(left), MatrixInner::PadicLocalFieldOrder(right)) => {
                MatrixInner::PadicLocalFieldOrder(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::NumberField(left), MatrixInner::NumberField(right)) => {
                MatrixInner::NumberField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::RationalFunction(left), MatrixInner::RationalFunction(right)) => {
                MatrixInner::RationalFunction(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::NumberFieldOrder(left), MatrixInner::NumberFieldOrder(right)) => {
                MatrixInner::NumberFieldOrder(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::RealBall(left), MatrixInner::RealBall(right)) => {
                MatrixInner::RealBall(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::ComplexBall(left), MatrixInner::ComplexBall(right)) => {
                MatrixInner::ComplexBall(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesInteger(left), MatrixInner::SeriesInteger(right)) => {
                MatrixInner::SeriesInteger(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesRational(left), MatrixInner::SeriesRational(right)) => {
                MatrixInner::SeriesRational(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesResidue(left), MatrixInner::SeriesResidue(right)) => {
                MatrixInner::SeriesResidue(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesFiniteField(left), MatrixInner::SeriesFiniteField(right)) => {
                MatrixInner::SeriesFiniteField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadic(left), MatrixInner::SeriesPadic(right)) => {
                MatrixInner::SeriesPadic(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesPadicField(left), MatrixInner::SeriesPadicField(right)) => {
                MatrixInner::SeriesPadicField(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadic(left), MatrixInner::SeriesQadic(right)) => {
                MatrixInner::SeriesQadic(left.sub(right).map_err(linalg_error)?)
            }
            (MatrixInner::SeriesQadicField(left), MatrixInner::SeriesQadicField(right)) => {
                MatrixInner::SeriesQadicField(left.sub(right).map_err(linalg_error)?)
            }
            (
                MatrixInner::SeriesPadicLocalField(left),
                MatrixInner::SeriesPadicLocalField(right),
            ) => MatrixInner::SeriesPadicLocalField(left.sub(right).map_err(linalg_error)?),
            (MatrixInner::Polynomial(left), MatrixInner::Polynomial(right)) => {
                MatrixInner::Polynomial(left.sub(right)?)
            }
            (MatrixInner::Quotient(left), MatrixInner::Quotient(right)) => {
                MatrixInner::Quotient(left.sub(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "matrix subtraction",
                    self.context_name(),
                    other.context_name(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            MatrixInner::Integer(matrix) => {
                MatrixInner::Integer(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::Rational(matrix) => {
                MatrixInner::Rational(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::Residue(matrix) => {
                MatrixInner::Residue(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::FiniteField(matrix) => {
                MatrixInner::FiniteField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::Padic(matrix) => MatrixInner::Padic(matrix.neg().map_err(linalg_error)?),
            MatrixInner::PadicField(matrix) => {
                MatrixInner::PadicField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::Qadic(matrix) => MatrixInner::Qadic(matrix.neg().map_err(linalg_error)?),
            MatrixInner::QadicField(matrix) => {
                MatrixInner::QadicField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::PadicLocalField(matrix) => {
                MatrixInner::PadicLocalField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::PadicLocalFieldOrder(matrix) => {
                MatrixInner::PadicLocalFieldOrder(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::NumberField(matrix) => {
                MatrixInner::NumberField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::RationalFunction(matrix) => {
                MatrixInner::RationalFunction(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::NumberFieldOrder(matrix) => {
                MatrixInner::NumberFieldOrder(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::RealBall(matrix) => {
                MatrixInner::RealBall(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::ComplexBall(matrix) => {
                MatrixInner::ComplexBall(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesInteger(matrix) => {
                MatrixInner::SeriesInteger(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesRational(matrix) => {
                MatrixInner::SeriesRational(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesResidue(matrix) => {
                MatrixInner::SeriesResidue(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesFiniteField(matrix) => {
                MatrixInner::SeriesFiniteField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesPadic(matrix) => {
                MatrixInner::SeriesPadic(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesPadicField(matrix) => {
                MatrixInner::SeriesPadicField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesQadic(matrix) => {
                MatrixInner::SeriesQadic(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesQadicField(matrix) => {
                MatrixInner::SeriesQadicField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::SeriesPadicLocalField(matrix) => {
                MatrixInner::SeriesPadicLocalField(matrix.neg().map_err(linalg_error)?)
            }
            MatrixInner::Polynomial(matrix) => MatrixInner::Polynomial(matrix.neg()?),
            MatrixInner::Quotient(matrix) => MatrixInner::Quotient(matrix.neg()?),
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            MatrixInner::Integer(matrix) => MatrixInner::Integer(
                matrix
                    .scalar_mul(&extract_integer(scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::Rational(matrix) => MatrixInner::Rational(
                matrix
                    .scalar_mul(&extract_rational(py, scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::Residue(matrix) => MatrixInner::Residue(
                matrix
                    .scalar_mul(&extract_residue(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::FiniteField(matrix) => MatrixInner::FiniteField(
                matrix
                    .scalar_mul(&extract_finite_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::Padic(matrix) => MatrixInner::Padic(
                matrix
                    .scalar_mul(&extract_padic_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::PadicField(matrix) => MatrixInner::PadicField(
                matrix
                    .scalar_mul(&extract_padic_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::Qadic(matrix) => MatrixInner::Qadic(
                matrix
                    .scalar_mul(&extract_qadic_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::QadicField(matrix) => MatrixInner::QadicField(
                matrix
                    .scalar_mul(&extract_qadic_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::PadicLocalField(matrix) => MatrixInner::PadicLocalField(
                matrix
                    .scalar_mul(&extract_padic_local_field_element(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::PadicLocalFieldOrder(matrix) => MatrixInner::PadicLocalFieldOrder(
                matrix
                    .scalar_mul(&extract_padic_local_field_order_element(
                        matrix.ring(),
                        scalar,
                    )?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::NumberField(matrix) => MatrixInner::NumberField(
                matrix
                    .scalar_mul(&extract_number_field_element(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::RationalFunction(matrix) => MatrixInner::RationalFunction(
                matrix
                    .scalar_mul(&extract_rational_function(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::NumberFieldOrder(matrix) => MatrixInner::NumberFieldOrder(
                matrix
                    .scalar_mul(&extract_number_field_order_element(
                        py,
                        matrix.ring(),
                        scalar,
                    )?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::RealBall(matrix) => MatrixInner::RealBall(
                matrix
                    .scalar_mul(&extract_real_ball(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::ComplexBall(matrix) => MatrixInner::ComplexBall(
                matrix
                    .scalar_mul(&extract_complex_ball(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesInteger(matrix) => MatrixInner::SeriesInteger(
                matrix
                    .scalar_mul(&extract_integer_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesRational(matrix) => MatrixInner::SeriesRational(
                matrix
                    .scalar_mul(&extract_rational_series(py, matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesResidue(matrix) => MatrixInner::SeriesResidue(
                matrix
                    .scalar_mul(&extract_residue_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesFiniteField(matrix) => MatrixInner::SeriesFiniteField(
                matrix
                    .scalar_mul(&extract_finite_field_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesPadic(matrix) => MatrixInner::SeriesPadic(
                matrix
                    .scalar_mul(&extract_padic_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesPadicField(matrix) => MatrixInner::SeriesPadicField(
                matrix
                    .scalar_mul(&extract_padic_field_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesQadic(matrix) => MatrixInner::SeriesQadic(
                matrix
                    .scalar_mul(&extract_qadic_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesQadicField(matrix) => MatrixInner::SeriesQadicField(
                matrix
                    .scalar_mul(&extract_qadic_field_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::SeriesPadicLocalField(matrix) => MatrixInner::SeriesPadicLocalField(
                matrix
                    .scalar_mul(&extract_padic_local_field_series(matrix.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            MatrixInner::Polynomial(matrix) => {
                MatrixInner::Polynomial(matrix.scalar_mul(py, scalar)?)
            }
            MatrixInner::Quotient(matrix) => MatrixInner::Quotient(matrix.scalar_mul(scalar)?),
        };
        Ok(Self { inner })
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, scalar)
    }

    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // NOETHER_RELEASES_GIL: both accepted branches delegate to detached Rust work.
        if let Ok(matrix) = other.extract::<PyRef<'_, PyMatrix>>() {
            return Ok(Py::new(py, self.matmul(py, &matrix)?)?.into_any());
        }
        if let Ok(vector) = other.extract::<PyRef<'_, PyVector>>() {
            return Ok(Py::new(py, self.apply(py, &vector)?)?.into_any());
        }
        Err(PyTypeError::new_err(
            "matrix multiplication requires a noether.linalg Matrix or Vector",
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (MatrixInner::Integer(left), MatrixInner::Integer(right)) => left == right,
            (MatrixInner::Rational(left), MatrixInner::Rational(right)) => left == right,
            (MatrixInner::Residue(left), MatrixInner::Residue(right)) => left == right,
            (MatrixInner::FiniteField(left), MatrixInner::FiniteField(right)) => left == right,
            (MatrixInner::Padic(left), MatrixInner::Padic(right)) => left == right,
            (MatrixInner::PadicField(left), MatrixInner::PadicField(right)) => left == right,
            (MatrixInner::Qadic(left), MatrixInner::Qadic(right)) => left == right,
            (MatrixInner::QadicField(left), MatrixInner::QadicField(right)) => left == right,
            (MatrixInner::PadicLocalField(left), MatrixInner::PadicLocalField(right)) => {
                left == right
            }
            (MatrixInner::PadicLocalFieldOrder(left), MatrixInner::PadicLocalFieldOrder(right)) => {
                left == right
            }
            (MatrixInner::NumberField(left), MatrixInner::NumberField(right)) => left == right,
            (MatrixInner::RationalFunction(left), MatrixInner::RationalFunction(right)) => {
                left == right
            }
            (MatrixInner::NumberFieldOrder(left), MatrixInner::NumberFieldOrder(right)) => {
                left == right
            }
            (MatrixInner::RealBall(left), MatrixInner::RealBall(right)) => left == right,
            (MatrixInner::ComplexBall(left), MatrixInner::ComplexBall(right)) => left == right,
            (MatrixInner::SeriesInteger(left), MatrixInner::SeriesInteger(right)) => left == right,
            (MatrixInner::SeriesRational(left), MatrixInner::SeriesRational(right)) => {
                left == right
            }
            (MatrixInner::SeriesResidue(left), MatrixInner::SeriesResidue(right)) => left == right,
            (MatrixInner::SeriesFiniteField(left), MatrixInner::SeriesFiniteField(right)) => {
                left == right
            }
            (MatrixInner::SeriesPadic(left), MatrixInner::SeriesPadic(right)) => left == right,
            (MatrixInner::SeriesPadicField(left), MatrixInner::SeriesPadicField(right)) => {
                left == right
            }
            (MatrixInner::SeriesQadic(left), MatrixInner::SeriesQadic(right)) => left == right,
            (MatrixInner::SeriesQadicField(left), MatrixInner::SeriesQadicField(right)) => {
                left == right
            }
            (
                MatrixInner::SeriesPadicLocalField(left),
                MatrixInner::SeriesPadicLocalField(right),
            ) => left == right,
            (MatrixInner::Polynomial(left), MatrixInner::Polynomial(right)) => left.equals(right),
            (MatrixInner::Quotient(left), MatrixInner::Quotient(right)) => left.equals(right),
            _ => false,
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Matrix({}, shape={:?}, preview={})",
                self.context_name(),
                self.dimensions(),
                self.preview()
            )
        })
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", "noether/linalg/Matrix@1")?;
        summary.set_item("ring", self.context_name())?;
        summary.set_item("shape", self.dimensions())?;
        summary.set_item("preview", self.preview())?;
        Ok(summary.unbind())
    }
}

#[pymethods]
impl PyVector {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Vector::<IntegerRing>::TYPE_ID
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Serialize the vector and its complete coefficient context.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Deserialize and revalidate a vector and its complete coefficient context.
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: VectorInner::from_json(value)?,
        })
    }

    #[classmethod]
    fn from_values(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if ring.is_instance_of::<PyIntegerRing>() {
            let entries = collect_values(values, extract_integer)?;
            return Ok(Self {
                inner: VectorInner::Integer(Vector::from_values(Z, entries).map_err(linalg_error)?),
            });
        }
        if ring.is_instance_of::<PyRationalField>() {
            let entries = collect_values(values, |value| extract_rational(py, value))?;
            return Ok(Self {
                inner: VectorInner::Rational(
                    Vector::from_values(Q, entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            let entries =
                collect_values(values, |value| extract_residue(&residue_ring.inner, value))?;
            return Ok(Self {
                inner: VectorInner::Residue(
                    Vector::from_values(residue_ring.inner.clone(), entries)
                        .map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            let entries = collect_values(values, |value| {
                extract_finite_field_element(&field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::FiniteField(
                    Vector::from_values(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(padic_ring) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            let entries = collect_values(values, |value| {
                extract_padic_element(&padic_ring.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::Padic(
                    Vector::from_values(padic_ring.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(padic_field) = ring.extract::<PyRef<'_, PyPadicField>>() {
            let entries = collect_values(values, |value| {
                extract_padic_field_element(&padic_field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::PadicField(
                    Vector::from_values(padic_field.inner.clone(), entries)
                        .map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(qadic_ring) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            let entries = collect_values(values, |value| {
                extract_qadic_element(&qadic_ring.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::Qadic(
                    Vector::from_values(qadic_ring.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(qadic_field) = ring.extract::<PyRef<'_, PyQadicField>>() {
            let entries = collect_values(values, |value| {
                extract_qadic_field_element(&qadic_field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::QadicField(
                    Vector::from_values(qadic_field.inner.clone(), entries)
                        .map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            let entries = collect_values(values, |value| {
                extract_padic_local_field_element(&field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::PadicLocalField(
                    Vector::from_values(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(order) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            let entries = collect_values(values, |value| {
                extract_padic_local_field_order_element(&order.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::PadicLocalFieldOrder(
                    Vector::from_values(order.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            let entries = collect_values(values, |value| {
                extract_number_field_element(py, &field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::NumberField(
                    Vector::from_values(field.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let field = field.as_rational().ok_or_else(|| {
                PyTypeError::new_err(
                    "dense rational-function linear algebra currently requires base field Q",
                )
            })?;
            let entries =
                collect_values(values, |value| extract_rational_function(py, field, value))?;
            return Ok(Self {
                inner: VectorInner::RationalFunction(
                    Vector::from_values(field.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(order) = ring.extract::<PyRef<'_, PyNumberFieldOrder>>() {
            let entries = collect_values(values, |value| {
                extract_number_field_order_element(py, &order.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::NumberFieldOrder(
                    Vector::from_values(order.inner.clone(), entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            let entries =
                collect_values(values, |value| extract_real_ball(py, &field.inner, value))?;
            return Ok(Self {
                inner: VectorInner::RealBall(
                    Vector::from_values(field.inner, entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            let entries = collect_values(values, |value| {
                extract_complex_ball(py, &field.inner, value)
            })?;
            return Ok(Self {
                inner: VectorInner::ComplexBall(
                    Vector::from_values(field.inner, entries).map_err(linalg_error)?,
                ),
            });
        }
        if let Ok(series_ring) = ring.extract::<PyRef<'_, PyPowerSeriesRing>>() {
            let inner = match &series_ring.inner {
                PowerSeriesRingInner::Integer(series_ring) => {
                    let entries =
                        collect_values(values, |value| extract_integer_series(series_ring, value))?;
                    VectorInner::SeriesInteger(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Rational(series_ring) => {
                    let entries = collect_values(values, |value| {
                        extract_rational_series(py, series_ring, value)
                    })?;
                    VectorInner::SeriesRational(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Residue(series_ring) => {
                    let entries =
                        collect_values(values, |value| extract_residue_series(series_ring, value))?;
                    VectorInner::SeriesResidue(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::FiniteField(series_ring) => {
                    let entries = collect_values(values, |value| {
                        extract_finite_field_series(series_ring, value)
                    })?;
                    VectorInner::SeriesFiniteField(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Padic(series_ring) => {
                    let entries =
                        collect_values(values, |value| extract_padic_series(series_ring, value))?;
                    VectorInner::SeriesPadic(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::PadicField(series_ring) => {
                    let entries = collect_values(values, |value| {
                        extract_padic_field_series(series_ring, value)
                    })?;
                    VectorInner::SeriesPadicField(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::Qadic(series_ring) => {
                    let entries =
                        collect_values(values, |value| extract_qadic_series(series_ring, value))?;
                    VectorInner::SeriesQadic(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::QadicField(series_ring) => {
                    let entries = collect_values(values, |value| {
                        extract_qadic_field_series(series_ring, value)
                    })?;
                    VectorInner::SeriesQadicField(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
                PowerSeriesRingInner::PadicLocalField(series_ring) => {
                    let entries = collect_values(values, |value| {
                        extract_padic_local_field_series(series_ring, value)
                    })?;
                    VectorInner::SeriesPadicLocalField(
                        Vector::from_values(series_ring.clone(), entries).map_err(linalg_error)?,
                    )
                }
            };
            return Ok(Self { inner });
        }
        if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            let inner = match &ring.inner {
                QuotientRingInner::Rational(ring) => QuotientVectorInner::Rational(
                    Vector::from_values(
                        ring.clone(),
                        collect_values(values, |value| extract_rational_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
                QuotientRingInner::FiniteField(ring) => QuotientVectorInner::FiniteField(
                    Vector::from_values(
                        ring.clone(),
                        collect_values(values, |value| extract_finite_field_quotient(ring, value))?,
                    )
                    .map_err(linalg_error)?,
                ),
            };
            return Ok(Self {
                inner: VectorInner::Quotient(inner),
            });
        }
        if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialVectorInner::Integer(
                        Vector::from_values(
                            ring.clone(),
                            collect_values(values, |value| {
                                extract_integer_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialVectorInner::Rational(
                        Vector::from_values(
                            ring.clone(),
                            collect_values(values, |value| {
                                extract_rational_polynomial(py, ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialVectorInner::Residue(
                        Vector::from_values(
                            ring.clone(),
                            collect_values(values, |value| {
                                extract_residue_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialVectorInner::FiniteField(
                        Vector::from_values(
                            ring.clone(),
                            collect_values(values, |value| {
                                extract_finite_field_polynomial(ring, value)
                            })?,
                        )
                        .map_err(linalg_error)?,
                    )
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "polynomial vectors currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            return Ok(Self {
                inner: VectorInner::Polynomial(inner),
            });
        }
        Err(PyTypeError::new_err(
            "Vector.from_values supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, NumberField, NumberFieldOrder, RationalFunctionField over Q, RealBallField, ComplexBallField, PolynomialRing, PowerSeriesRing, and QuotientRing contexts",
        ))
    }

    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        ring: &Bound<'_, PyAny>,
        dimension: usize,
    ) -> PyResult<Self> {
        validate_python_shape(dimension, 1)?;
        let inner = if ring.is_instance_of::<PyIntegerRing>() {
            VectorInner::Integer(Vector::zeros(Z, dimension))
        } else if ring.is_instance_of::<PyRationalField>() {
            VectorInner::Rational(Vector::zeros(Q, dimension))
        } else if let Ok(residue_ring) = ring.extract::<PyRef<'_, PyResidueRing>>() {
            VectorInner::Residue(Vector::zeros(residue_ring.inner.clone(), dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            VectorInner::FiniteField(Vector::zeros(field.inner.clone(), dimension))
        } else if let Ok(padic_ring) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            VectorInner::Padic(Vector::zeros(padic_ring.inner.clone(), dimension))
        } else if let Ok(padic_field) = ring.extract::<PyRef<'_, PyPadicField>>() {
            VectorInner::PadicField(Vector::zeros(padic_field.inner.clone(), dimension))
        } else if let Ok(qadic_ring) = ring.extract::<PyRef<'_, PyQadicRing>>() {
            VectorInner::Qadic(Vector::zeros(qadic_ring.inner.clone(), dimension))
        } else if let Ok(qadic_field) = ring.extract::<PyRef<'_, PyQadicField>>() {
            VectorInner::QadicField(Vector::zeros(qadic_field.inner.clone(), dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyPadicLocalField>>() {
            VectorInner::PadicLocalField(Vector::zeros(field.inner.clone(), dimension))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyPadicLocalFieldOrder>>() {
            VectorInner::PadicLocalFieldOrder(Vector::zeros(order.inner.clone(), dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            VectorInner::NumberField(Vector::zeros(field.inner.clone(), dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRationalFunctionField>>() {
            let field = field.as_rational().ok_or_else(|| {
                PyTypeError::new_err(
                    "dense rational-function linear algebra currently requires base field Q",
                )
            })?;
            VectorInner::RationalFunction(Vector::zeros(field.clone(), dimension))
        } else if let Ok(order) = ring.extract::<PyRef<'_, PyNumberFieldOrder>>() {
            VectorInner::NumberFieldOrder(Vector::zeros(order.inner.clone(), dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyRealBallField>>() {
            VectorInner::RealBall(Vector::zeros(field.inner, dimension))
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyComplexBallField>>() {
            VectorInner::ComplexBall(Vector::zeros(field.inner, dimension))
        } else if let Ok(series_ring) = ring.extract::<PyRef<'_, PyPowerSeriesRing>>() {
            match &series_ring.inner {
                PowerSeriesRingInner::Integer(ring) => {
                    VectorInner::SeriesInteger(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::Rational(ring) => {
                    VectorInner::SeriesRational(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::Residue(ring) => {
                    VectorInner::SeriesResidue(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::FiniteField(ring) => {
                    VectorInner::SeriesFiniteField(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::Padic(ring) => {
                    VectorInner::SeriesPadic(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::PadicField(ring) => {
                    VectorInner::SeriesPadicField(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::Qadic(ring) => {
                    VectorInner::SeriesQadic(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::QadicField(ring) => {
                    VectorInner::SeriesQadicField(Vector::zeros(ring.clone(), dimension))
                }
                PowerSeriesRingInner::PadicLocalField(ring) => {
                    VectorInner::SeriesPadicLocalField(Vector::zeros(ring.clone(), dimension))
                }
            }
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyQuotientRing>>() {
            VectorInner::Quotient(match &ring.inner {
                QuotientRingInner::Rational(ring) => {
                    QuotientVectorInner::Rational(Vector::zeros(ring.clone(), dimension))
                }
                QuotientRingInner::FiniteField(ring) => {
                    QuotientVectorInner::FiniteField(Vector::zeros(ring.clone(), dimension))
                }
            })
        } else if let Ok(ring) = ring.extract::<PyRef<'_, PyPolynomialRing>>() {
            let inner = match &ring.inner {
                super::polynomial_facade::PolynomialRingInner::Integer(ring) => {
                    PolynomialVectorInner::Integer(Vector::zeros(ring.clone(), dimension))
                }
                super::polynomial_facade::PolynomialRingInner::Rational(ring) => {
                    PolynomialVectorInner::Rational(Vector::zeros(ring.clone(), dimension))
                }
                super::polynomial_facade::PolynomialRingInner::Residue(ring) => {
                    PolynomialVectorInner::Residue(Vector::zeros(ring.clone(), dimension))
                }
                super::polynomial_facade::PolynomialRingInner::FiniteField(ring) => {
                    PolynomialVectorInner::FiniteField(Vector::zeros(ring.clone(), dimension))
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "polynomial vectors currently support polynomial rings over Z, Q, ResidueRing, and FiniteField",
                    ));
                }
            };
            VectorInner::Polynomial(inner)
        } else {
            return Err(PyTypeError::new_err(
                "Vector.zeros supports Z, Q, ResidueRing, FiniteField, PadicRing, PadicField, QadicRing, QadicField, PadicLocalField, PadicLocalFieldOrder, NumberField, NumberFieldOrder, RealBallField, ComplexBallField, PolynomialRing, PowerSeriesRing, and QuotientRing contexts",
            ));
        };
        Ok(Self { inner })
    }

    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            VectorInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            VectorInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            VectorInner::Residue(vector) => Ok(Py::new(
                py,
                PyResidueRing {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::FiniteField(vector) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::Padic(vector) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::PadicField(vector) => Ok(Py::new(
                py,
                PyPadicField {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::Qadic(vector) => Ok(Py::new(
                py,
                PyQadicRing {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::QadicField(vector) => Ok(Py::new(
                py,
                PyQadicField {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::PadicLocalField(vector) => Ok(Py::new(
                py,
                PyPadicLocalField {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::PadicLocalFieldOrder(vector) => Ok(Py::new(
                py,
                PyPadicLocalFieldOrder {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::NumberField(vector) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::RationalFunction(vector) => Ok(Py::new(
                py,
                PyRationalFunctionField::from_rational(vector.ring().clone()),
            )?
            .into_any()),
            VectorInner::NumberFieldOrder(vector) => Ok(Py::new(
                py,
                PyNumberFieldOrder {
                    inner: vector.ring().clone(),
                },
            )?
            .into_any()),
            VectorInner::RealBall(vector) => Ok(Py::new(
                py,
                PyRealBallField {
                    inner: *vector.ring(),
                },
            )?
            .into_any()),
            VectorInner::ComplexBall(vector) => Ok(Py::new(
                py,
                PyComplexBallField {
                    inner: *vector.ring(),
                },
            )?
            .into_any()),
            VectorInner::SeriesInteger(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Integer(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesRational(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Rational(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesResidue(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Residue(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesFiniteField(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::FiniteField(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadic(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Padic(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadicField(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::PadicField(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesQadic(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::Qadic(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesQadicField(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::QadicField(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadicLocalField(vector) => Ok(Py::new(
                py,
                PyPowerSeriesRing {
                    inner: PowerSeriesRingInner::PadicLocalField(vector.ring().clone()),
                },
            )?
            .into_any()),
            VectorInner::Polynomial(vector) => vector.ring_to_python(py),
            VectorInner::Quotient(vector) => vector.ring_to_python(py),
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension_value()
    }

    fn to_values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // NOETHER_RELEASES_GIL: values_to_python detaches Rust materialization.
        self.values_to_python(py)
    }

    fn to_column_matrix(&self) -> PyMatrix {
        let inner = match &self.inner {
            VectorInner::Integer(vector) => MatrixInner::Integer(vector.to_column_matrix()),
            VectorInner::Rational(vector) => MatrixInner::Rational(vector.to_column_matrix()),
            VectorInner::Residue(vector) => MatrixInner::Residue(vector.to_column_matrix()),
            VectorInner::FiniteField(vector) => MatrixInner::FiniteField(vector.to_column_matrix()),
            VectorInner::Padic(vector) => MatrixInner::Padic(vector.to_column_matrix()),
            VectorInner::PadicField(vector) => MatrixInner::PadicField(vector.to_column_matrix()),
            VectorInner::Qadic(vector) => MatrixInner::Qadic(vector.to_column_matrix()),
            VectorInner::QadicField(vector) => MatrixInner::QadicField(vector.to_column_matrix()),
            VectorInner::PadicLocalField(vector) => {
                MatrixInner::PadicLocalField(vector.to_column_matrix())
            }
            VectorInner::PadicLocalFieldOrder(vector) => {
                MatrixInner::PadicLocalFieldOrder(vector.to_column_matrix())
            }
            VectorInner::NumberField(vector) => MatrixInner::NumberField(vector.to_column_matrix()),
            VectorInner::RationalFunction(vector) => {
                MatrixInner::RationalFunction(vector.to_column_matrix())
            }
            VectorInner::NumberFieldOrder(vector) => {
                MatrixInner::NumberFieldOrder(vector.to_column_matrix())
            }
            VectorInner::RealBall(vector) => MatrixInner::RealBall(vector.to_column_matrix()),
            VectorInner::ComplexBall(vector) => MatrixInner::ComplexBall(vector.to_column_matrix()),
            VectorInner::SeriesInteger(vector) => {
                MatrixInner::SeriesInteger(vector.to_column_matrix())
            }
            VectorInner::SeriesRational(vector) => {
                MatrixInner::SeriesRational(vector.to_column_matrix())
            }
            VectorInner::SeriesResidue(vector) => {
                MatrixInner::SeriesResidue(vector.to_column_matrix())
            }
            VectorInner::SeriesFiniteField(vector) => {
                MatrixInner::SeriesFiniteField(vector.to_column_matrix())
            }
            VectorInner::SeriesPadic(vector) => MatrixInner::SeriesPadic(vector.to_column_matrix()),
            VectorInner::SeriesPadicField(vector) => {
                MatrixInner::SeriesPadicField(vector.to_column_matrix())
            }
            VectorInner::SeriesQadic(vector) => MatrixInner::SeriesQadic(vector.to_column_matrix()),
            VectorInner::SeriesQadicField(vector) => {
                MatrixInner::SeriesQadicField(vector.to_column_matrix())
            }
            VectorInner::SeriesPadicLocalField(vector) => {
                MatrixInner::SeriesPadicLocalField(vector.to_column_matrix())
            }
            VectorInner::Polynomial(vector) => MatrixInner::Polynomial(vector.column_matrix()),
            VectorInner::Quotient(vector) => MatrixInner::Quotient(vector.column_matrix()),
        };
        PyMatrix { inner }
    }

    fn to_row_matrix(&self) -> PyMatrix {
        let inner = match &self.inner {
            VectorInner::Integer(vector) => MatrixInner::Integer(vector.to_row_matrix()),
            VectorInner::Rational(vector) => MatrixInner::Rational(vector.to_row_matrix()),
            VectorInner::Residue(vector) => MatrixInner::Residue(vector.to_row_matrix()),
            VectorInner::FiniteField(vector) => MatrixInner::FiniteField(vector.to_row_matrix()),
            VectorInner::Padic(vector) => MatrixInner::Padic(vector.to_row_matrix()),
            VectorInner::PadicField(vector) => MatrixInner::PadicField(vector.to_row_matrix()),
            VectorInner::Qadic(vector) => MatrixInner::Qadic(vector.to_row_matrix()),
            VectorInner::QadicField(vector) => MatrixInner::QadicField(vector.to_row_matrix()),
            VectorInner::PadicLocalField(vector) => {
                MatrixInner::PadicLocalField(vector.to_row_matrix())
            }
            VectorInner::PadicLocalFieldOrder(vector) => {
                MatrixInner::PadicLocalFieldOrder(vector.to_row_matrix())
            }
            VectorInner::NumberField(vector) => MatrixInner::NumberField(vector.to_row_matrix()),
            VectorInner::RationalFunction(vector) => {
                MatrixInner::RationalFunction(vector.to_row_matrix())
            }
            VectorInner::NumberFieldOrder(vector) => {
                MatrixInner::NumberFieldOrder(vector.to_row_matrix())
            }
            VectorInner::RealBall(vector) => MatrixInner::RealBall(vector.to_row_matrix()),
            VectorInner::ComplexBall(vector) => MatrixInner::ComplexBall(vector.to_row_matrix()),
            VectorInner::SeriesInteger(vector) => {
                MatrixInner::SeriesInteger(vector.to_row_matrix())
            }
            VectorInner::SeriesRational(vector) => {
                MatrixInner::SeriesRational(vector.to_row_matrix())
            }
            VectorInner::SeriesResidue(vector) => {
                MatrixInner::SeriesResidue(vector.to_row_matrix())
            }
            VectorInner::SeriesFiniteField(vector) => {
                MatrixInner::SeriesFiniteField(vector.to_row_matrix())
            }
            VectorInner::SeriesPadic(vector) => MatrixInner::SeriesPadic(vector.to_row_matrix()),
            VectorInner::SeriesPadicField(vector) => {
                MatrixInner::SeriesPadicField(vector.to_row_matrix())
            }
            VectorInner::SeriesQadic(vector) => MatrixInner::SeriesQadic(vector.to_row_matrix()),
            VectorInner::SeriesQadicField(vector) => {
                MatrixInner::SeriesQadicField(vector.to_row_matrix())
            }
            VectorInner::SeriesPadicLocalField(vector) => {
                MatrixInner::SeriesPadicLocalField(vector.to_row_matrix())
            }
            VectorInner::Polynomial(vector) => MatrixInner::Polynomial(vector.row_matrix()),
            VectorInner::Quotient(vector) => MatrixInner::Quotient(vector.row_matrix()),
        };
        PyMatrix { inner }
    }

    fn dot(&self, py: Python<'_>, other: &Self) -> PyResult<Py<PyAny>> {
        match (&self.inner, &other.inner) {
            (VectorInner::Integer(left), VectorInner::Integer(right)) => {
                integer_to_python(py, &left.dot(right).map_err(linalg_error)?)
            }
            (VectorInner::Rational(left), VectorInner::Rational(right)) => {
                rational_to_python(py, &left.dot(right).map_err(linalg_error)?)
            }
            (VectorInner::Residue(left), VectorInner::Residue(right)) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::Padic(left), VectorInner::Padic(right)) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::PadicField(left), VectorInner::PadicField(right)) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::Qadic(left), VectorInner::Qadic(right)) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::QadicField(left), VectorInner::QadicField(right)) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::PadicLocalField(left), VectorInner::PadicLocalField(right)) => {
                Ok(Py::new(
                    py,
                    PyPadicLocalFieldElement {
                        inner: left.dot(right).map_err(linalg_error)?,
                    },
                )?
                .into_any())
            }
            (VectorInner::PadicLocalFieldOrder(left), VectorInner::PadicLocalFieldOrder(right)) => {
                Ok(Py::new(
                    py,
                    PyPadicLocalFieldElement {
                        inner: left.dot(right).map_err(linalg_error)?,
                    },
                )?
                .into_any())
            }
            (VectorInner::NumberField(left), VectorInner::NumberField(right)) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::RationalFunction(left), VectorInner::RationalFunction(right)) => {
                Ok(Py::new(
                    py,
                    PyRationalFunction::from_rational(left.dot(right).map_err(linalg_error)?),
                )?
                .into_any())
            }
            (VectorInner::NumberFieldOrder(left), VectorInner::NumberFieldOrder(right)) => {
                Ok(Py::new(
                    py,
                    PyNumberFieldElement {
                        inner: left.dot(right).map_err(linalg_error)?,
                    },
                )?
                .into_any())
            }
            (VectorInner::RealBall(left), VectorInner::RealBall(right)) => Ok(Py::new(
                py,
                PyRealBall {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::ComplexBall(left), VectorInner::ComplexBall(right)) => Ok(Py::new(
                py,
                PyComplexBall {
                    inner: left.dot(right).map_err(linalg_error)?,
                },
            )?
            .into_any()),
            (VectorInner::SeriesInteger(left), VectorInner::SeriesInteger(right)) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Integer(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::SeriesRational(left), VectorInner::SeriesRational(right)) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Rational(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::SeriesResidue(left), VectorInner::SeriesResidue(right)) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Residue(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::SeriesFiniteField(left), VectorInner::SeriesFiniteField(right)) => {
                Ok(Py::new(
                    py,
                    PyPowerSeries {
                        inner: SeriesInner::FiniteField(left.dot(right).map_err(linalg_error)?),
                    },
                )?
                .into_any())
            }
            (VectorInner::SeriesPadic(left), VectorInner::SeriesPadic(right)) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Padic(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::SeriesPadicField(left), VectorInner::SeriesPadicField(right)) => {
                Ok(Py::new(
                    py,
                    PyPowerSeries {
                        inner: SeriesInner::PadicField(left.dot(right).map_err(linalg_error)?),
                    },
                )?
                .into_any())
            }
            (VectorInner::SeriesQadic(left), VectorInner::SeriesQadic(right)) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Qadic(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::SeriesQadicField(left), VectorInner::SeriesQadicField(right)) => {
                Ok(Py::new(
                    py,
                    PyPowerSeries {
                        inner: SeriesInner::QadicField(left.dot(right).map_err(linalg_error)?),
                    },
                )?
                .into_any())
            }
            (
                VectorInner::SeriesPadicLocalField(left),
                VectorInner::SeriesPadicLocalField(right),
            ) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicLocalField(left.dot(right).map_err(linalg_error)?),
                },
            )?
            .into_any()),
            (VectorInner::Polynomial(left), VectorInner::Polynomial(right)) => {
                left.dot_to_python(py, right)
            }
            (VectorInner::Quotient(left), VectorInner::Quotient(right)) => {
                left.dot_to_python(py, right)
            }
            _ => Err(context_mismatch(
                "dot product",
                self.context_name(),
                other.context_name(),
            )),
        }
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let dimension = self.dimension_value();
        let normalized = normalize_index(index, dimension).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "vector index {index} is outside dimension {dimension}"
            ))
        })?;
        match &self.inner {
            VectorInner::Integer(vector) => {
                integer_to_python(py, &vector.get(normalized).map_err(index_error)?)
            }
            VectorInner::Rational(vector) => {
                rational_to_python(py, &vector.get(normalized).map_err(index_error)?)
            }
            VectorInner::Residue(vector) => Ok(Py::new(
                py,
                PyResidueClass {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::FiniteField(vector) => Ok(Py::new(
                py,
                PyFiniteFieldElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::Padic(vector) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::PadicField(vector) => Ok(Py::new(
                py,
                PyPadicElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::Qadic(vector) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::QadicField(vector) => Ok(Py::new(
                py,
                PyQadicElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::PadicLocalField(vector) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::PadicLocalFieldOrder(vector) => Ok(Py::new(
                py,
                PyPadicLocalFieldElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::NumberField(vector) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::RationalFunction(vector) => Ok(Py::new(
                py,
                PyRationalFunction::from_rational(vector.get(normalized).map_err(index_error)?),
            )?
            .into_any()),
            VectorInner::NumberFieldOrder(vector) => Ok(Py::new(
                py,
                PyNumberFieldElement {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::RealBall(vector) => Ok(Py::new(
                py,
                PyRealBall {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::ComplexBall(vector) => Ok(Py::new(
                py,
                PyComplexBall {
                    inner: vector.get(normalized).map_err(index_error)?,
                },
            )?
            .into_any()),
            VectorInner::SeriesInteger(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Integer(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesRational(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Rational(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesResidue(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Residue(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesFiniteField(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::FiniteField(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadic(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Padic(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadicField(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicField(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesQadic(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::Qadic(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesQadicField(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::QadicField(vector.get(normalized).map_err(index_error)?),
                },
            )?
            .into_any()),
            VectorInner::SeriesPadicLocalField(vector) => Ok(Py::new(
                py,
                PyPowerSeries {
                    inner: SeriesInner::PadicLocalField(
                        vector.get(normalized).map_err(index_error)?,
                    ),
                },
            )?
            .into_any()),
            VectorInner::Polynomial(vector) => vector.get_to_python(py, normalized),
            VectorInner::Quotient(vector) => vector.get_to_python(py, normalized),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self
            .values_to_python(py)?
            .bind(py)
            .call_method0("__iter__")?
            .unbind())
    }

    fn __len__(&self) -> usize {
        self.dimension_value()
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (VectorInner::Integer(left), VectorInner::Integer(right)) => {
                VectorInner::Integer(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::Rational(left), VectorInner::Rational(right)) => {
                VectorInner::Rational(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::Residue(left), VectorInner::Residue(right)) => {
                VectorInner::Residue(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) => {
                VectorInner::FiniteField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::Padic(left), VectorInner::Padic(right)) => {
                VectorInner::Padic(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicField(left), VectorInner::PadicField(right)) => {
                VectorInner::PadicField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::Qadic(left), VectorInner::Qadic(right)) => {
                VectorInner::Qadic(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::QadicField(left), VectorInner::QadicField(right)) => {
                VectorInner::QadicField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicLocalField(left), VectorInner::PadicLocalField(right)) => {
                VectorInner::PadicLocalField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicLocalFieldOrder(left), VectorInner::PadicLocalFieldOrder(right)) => {
                VectorInner::PadicLocalFieldOrder(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::NumberField(left), VectorInner::NumberField(right)) => {
                VectorInner::NumberField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::RationalFunction(left), VectorInner::RationalFunction(right)) => {
                VectorInner::RationalFunction(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::NumberFieldOrder(left), VectorInner::NumberFieldOrder(right)) => {
                VectorInner::NumberFieldOrder(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::RealBall(left), VectorInner::RealBall(right)) => {
                VectorInner::RealBall(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::ComplexBall(left), VectorInner::ComplexBall(right)) => {
                VectorInner::ComplexBall(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesInteger(left), VectorInner::SeriesInteger(right)) => {
                VectorInner::SeriesInteger(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesRational(left), VectorInner::SeriesRational(right)) => {
                VectorInner::SeriesRational(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesResidue(left), VectorInner::SeriesResidue(right)) => {
                VectorInner::SeriesResidue(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesFiniteField(left), VectorInner::SeriesFiniteField(right)) => {
                VectorInner::SeriesFiniteField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesPadic(left), VectorInner::SeriesPadic(right)) => {
                VectorInner::SeriesPadic(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesPadicField(left), VectorInner::SeriesPadicField(right)) => {
                VectorInner::SeriesPadicField(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesQadic(left), VectorInner::SeriesQadic(right)) => {
                VectorInner::SeriesQadic(left.add(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesQadicField(left), VectorInner::SeriesQadicField(right)) => {
                VectorInner::SeriesQadicField(left.add(right).map_err(linalg_error)?)
            }
            (
                VectorInner::SeriesPadicLocalField(left),
                VectorInner::SeriesPadicLocalField(right),
            ) => VectorInner::SeriesPadicLocalField(left.add(right).map_err(linalg_error)?),
            (VectorInner::Polynomial(left), VectorInner::Polynomial(right)) => {
                VectorInner::Polynomial(left.add(right)?)
            }
            (VectorInner::Quotient(left), VectorInner::Quotient(right)) => {
                VectorInner::Quotient(left.add(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "vector addition",
                    self.context_name(),
                    other.context_name(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &other.inner) {
            (VectorInner::Integer(left), VectorInner::Integer(right)) => {
                VectorInner::Integer(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::Rational(left), VectorInner::Rational(right)) => {
                VectorInner::Rational(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::Residue(left), VectorInner::Residue(right)) => {
                VectorInner::Residue(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) => {
                VectorInner::FiniteField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::Padic(left), VectorInner::Padic(right)) => {
                VectorInner::Padic(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicField(left), VectorInner::PadicField(right)) => {
                VectorInner::PadicField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::Qadic(left), VectorInner::Qadic(right)) => {
                VectorInner::Qadic(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::QadicField(left), VectorInner::QadicField(right)) => {
                VectorInner::QadicField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicLocalField(left), VectorInner::PadicLocalField(right)) => {
                VectorInner::PadicLocalField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::PadicLocalFieldOrder(left), VectorInner::PadicLocalFieldOrder(right)) => {
                VectorInner::PadicLocalFieldOrder(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::NumberField(left), VectorInner::NumberField(right)) => {
                VectorInner::NumberField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::RationalFunction(left), VectorInner::RationalFunction(right)) => {
                VectorInner::RationalFunction(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::NumberFieldOrder(left), VectorInner::NumberFieldOrder(right)) => {
                VectorInner::NumberFieldOrder(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::RealBall(left), VectorInner::RealBall(right)) => {
                VectorInner::RealBall(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::ComplexBall(left), VectorInner::ComplexBall(right)) => {
                VectorInner::ComplexBall(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesInteger(left), VectorInner::SeriesInteger(right)) => {
                VectorInner::SeriesInteger(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesRational(left), VectorInner::SeriesRational(right)) => {
                VectorInner::SeriesRational(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesResidue(left), VectorInner::SeriesResidue(right)) => {
                VectorInner::SeriesResidue(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesFiniteField(left), VectorInner::SeriesFiniteField(right)) => {
                VectorInner::SeriesFiniteField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesPadic(left), VectorInner::SeriesPadic(right)) => {
                VectorInner::SeriesPadic(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesPadicField(left), VectorInner::SeriesPadicField(right)) => {
                VectorInner::SeriesPadicField(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesQadic(left), VectorInner::SeriesQadic(right)) => {
                VectorInner::SeriesQadic(left.sub(right).map_err(linalg_error)?)
            }
            (VectorInner::SeriesQadicField(left), VectorInner::SeriesQadicField(right)) => {
                VectorInner::SeriesQadicField(left.sub(right).map_err(linalg_error)?)
            }
            (
                VectorInner::SeriesPadicLocalField(left),
                VectorInner::SeriesPadicLocalField(right),
            ) => VectorInner::SeriesPadicLocalField(left.sub(right).map_err(linalg_error)?),
            (VectorInner::Polynomial(left), VectorInner::Polynomial(right)) => {
                VectorInner::Polynomial(left.sub(right)?)
            }
            (VectorInner::Quotient(left), VectorInner::Quotient(right)) => {
                VectorInner::Quotient(left.sub(right)?)
            }
            _ => {
                return Err(context_mismatch(
                    "vector subtraction",
                    self.context_name(),
                    other.context_name(),
                ));
            }
        };
        Ok(Self { inner })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            VectorInner::Integer(vector) => {
                VectorInner::Integer(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::Rational(vector) => {
                VectorInner::Rational(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::Residue(vector) => {
                VectorInner::Residue(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::FiniteField(vector) => {
                VectorInner::FiniteField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::Padic(vector) => VectorInner::Padic(vector.neg().map_err(linalg_error)?),
            VectorInner::PadicField(vector) => {
                VectorInner::PadicField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::Qadic(vector) => VectorInner::Qadic(vector.neg().map_err(linalg_error)?),
            VectorInner::QadicField(vector) => {
                VectorInner::QadicField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::PadicLocalField(vector) => {
                VectorInner::PadicLocalField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::PadicLocalFieldOrder(vector) => {
                VectorInner::PadicLocalFieldOrder(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::NumberField(vector) => {
                VectorInner::NumberField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::RationalFunction(vector) => {
                VectorInner::RationalFunction(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::NumberFieldOrder(vector) => {
                VectorInner::NumberFieldOrder(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::RealBall(vector) => {
                VectorInner::RealBall(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::ComplexBall(vector) => {
                VectorInner::ComplexBall(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesInteger(vector) => {
                VectorInner::SeriesInteger(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesRational(vector) => {
                VectorInner::SeriesRational(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesResidue(vector) => {
                VectorInner::SeriesResidue(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesFiniteField(vector) => {
                VectorInner::SeriesFiniteField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesPadic(vector) => {
                VectorInner::SeriesPadic(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesPadicField(vector) => {
                VectorInner::SeriesPadicField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesQadic(vector) => {
                VectorInner::SeriesQadic(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesQadicField(vector) => {
                VectorInner::SeriesQadicField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::SeriesPadicLocalField(vector) => {
                VectorInner::SeriesPadicLocalField(vector.neg().map_err(linalg_error)?)
            }
            VectorInner::Polynomial(vector) => VectorInner::Polynomial(vector.neg()?),
            VectorInner::Quotient(vector) => VectorInner::Quotient(vector.neg()?),
        };
        Ok(Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match &self.inner {
            VectorInner::Integer(vector) => VectorInner::Integer(
                vector
                    .scalar_mul(&extract_integer(scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::Rational(vector) => VectorInner::Rational(
                vector
                    .scalar_mul(&extract_rational(py, scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::Residue(vector) => VectorInner::Residue(
                vector
                    .scalar_mul(&extract_residue(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::FiniteField(vector) => VectorInner::FiniteField(
                vector
                    .scalar_mul(&extract_finite_field_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::Padic(vector) => VectorInner::Padic(
                vector
                    .scalar_mul(&extract_padic_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::PadicField(vector) => VectorInner::PadicField(
                vector
                    .scalar_mul(&extract_padic_field_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::Qadic(vector) => VectorInner::Qadic(
                vector
                    .scalar_mul(&extract_qadic_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::QadicField(vector) => VectorInner::QadicField(
                vector
                    .scalar_mul(&extract_qadic_field_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::PadicLocalField(vector) => VectorInner::PadicLocalField(
                vector
                    .scalar_mul(&extract_padic_local_field_element(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::PadicLocalFieldOrder(vector) => VectorInner::PadicLocalFieldOrder(
                vector
                    .scalar_mul(&extract_padic_local_field_order_element(
                        vector.ring(),
                        scalar,
                    )?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::NumberField(vector) => VectorInner::NumberField(
                vector
                    .scalar_mul(&extract_number_field_element(py, vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::RationalFunction(vector) => VectorInner::RationalFunction(
                vector
                    .scalar_mul(&extract_rational_function(py, vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::NumberFieldOrder(vector) => VectorInner::NumberFieldOrder(
                vector
                    .scalar_mul(&extract_number_field_order_element(
                        py,
                        vector.ring(),
                        scalar,
                    )?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::RealBall(vector) => VectorInner::RealBall(
                vector
                    .scalar_mul(&extract_real_ball(py, vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::ComplexBall(vector) => VectorInner::ComplexBall(
                vector
                    .scalar_mul(&extract_complex_ball(py, vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesInteger(vector) => VectorInner::SeriesInteger(
                vector
                    .scalar_mul(&extract_integer_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesRational(vector) => VectorInner::SeriesRational(
                vector
                    .scalar_mul(&extract_rational_series(py, vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesResidue(vector) => VectorInner::SeriesResidue(
                vector
                    .scalar_mul(&extract_residue_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesFiniteField(vector) => VectorInner::SeriesFiniteField(
                vector
                    .scalar_mul(&extract_finite_field_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesPadic(vector) => VectorInner::SeriesPadic(
                vector
                    .scalar_mul(&extract_padic_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesPadicField(vector) => VectorInner::SeriesPadicField(
                vector
                    .scalar_mul(&extract_padic_field_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesQadic(vector) => VectorInner::SeriesQadic(
                vector
                    .scalar_mul(&extract_qadic_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesQadicField(vector) => VectorInner::SeriesQadicField(
                vector
                    .scalar_mul(&extract_qadic_field_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::SeriesPadicLocalField(vector) => VectorInner::SeriesPadicLocalField(
                vector
                    .scalar_mul(&extract_padic_local_field_series(vector.ring(), scalar)?)
                    .map_err(linalg_error)?,
            ),
            VectorInner::Polynomial(vector) => {
                VectorInner::Polynomial(vector.scalar_mul(py, scalar)?)
            }
            VectorInner::Quotient(vector) => VectorInner::Quotient(vector.scalar_mul(scalar)?),
        };
        Ok(Self { inner })
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(py, scalar)
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (VectorInner::Integer(left), VectorInner::Integer(right)) => left == right,
            (VectorInner::Rational(left), VectorInner::Rational(right)) => left == right,
            (VectorInner::Residue(left), VectorInner::Residue(right)) => left == right,
            (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) => left == right,
            (VectorInner::Padic(left), VectorInner::Padic(right)) => left == right,
            (VectorInner::PadicField(left), VectorInner::PadicField(right)) => left == right,
            (VectorInner::Qadic(left), VectorInner::Qadic(right)) => left == right,
            (VectorInner::QadicField(left), VectorInner::QadicField(right)) => left == right,
            (VectorInner::PadicLocalField(left), VectorInner::PadicLocalField(right)) => {
                left == right
            }
            (VectorInner::PadicLocalFieldOrder(left), VectorInner::PadicLocalFieldOrder(right)) => {
                left == right
            }
            (VectorInner::NumberField(left), VectorInner::NumberField(right)) => left == right,
            (VectorInner::RationalFunction(left), VectorInner::RationalFunction(right)) => {
                left == right
            }
            (VectorInner::NumberFieldOrder(left), VectorInner::NumberFieldOrder(right)) => {
                left == right
            }
            (VectorInner::RealBall(left), VectorInner::RealBall(right)) => left == right,
            (VectorInner::ComplexBall(left), VectorInner::ComplexBall(right)) => left == right,
            (VectorInner::SeriesInteger(left), VectorInner::SeriesInteger(right)) => left == right,
            (VectorInner::SeriesRational(left), VectorInner::SeriesRational(right)) => {
                left == right
            }
            (VectorInner::SeriesResidue(left), VectorInner::SeriesResidue(right)) => left == right,
            (VectorInner::SeriesFiniteField(left), VectorInner::SeriesFiniteField(right)) => {
                left == right
            }
            (VectorInner::SeriesPadic(left), VectorInner::SeriesPadic(right)) => left == right,
            (VectorInner::SeriesPadicField(left), VectorInner::SeriesPadicField(right)) => {
                left == right
            }
            (VectorInner::SeriesQadic(left), VectorInner::SeriesQadic(right)) => left == right,
            (VectorInner::SeriesQadicField(left), VectorInner::SeriesQadicField(right)) => {
                left == right
            }
            (
                VectorInner::SeriesPadicLocalField(left),
                VectorInner::SeriesPadicLocalField(right),
            ) => left == right,
            (VectorInner::Polynomial(left), VectorInner::Polynomial(right)) => left.equals(right),
            (VectorInner::Quotient(left), VectorInner::Quotient(right)) => left.equals(right),
            _ => false,
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Vector({}, dimension={}, preview={})",
                self.context_name(),
                self.dimension_value(),
                self.preview()
            )
        })
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", "noether/linalg/Vector@1")?;
        summary.set_item("ring", self.context_name())?;
        summary.set_item("dimension", self.dimension_value())?;
        summary.set_item("preview", self.preview())?;
        Ok(summary.unbind())
    }
}
