// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public `noether::homalg` facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::collections::BTreeMap;

use noether::homalg::{
    AbelianGroupError, ChainCancellation, ChainComplex, ChainReduction,
    FiniteAbelianGroupCohomologyError,
    FiniteAbelianGroupCohomologyPresentation as RustFiniteAbelianGroupCohomologyPresentation,
    FiniteAbelianGroupIntegralCohomologyPresentation as RustFiniteAbelianGroupIntegralCohomologyPresentation,
    FiniteCyclicGroupIntegralCohomologyError,
    FiniteCyclicGroupIntegralCohomologyPresentation as RustFiniteCyclicGroupIntegralCohomologyPresentation,
    FiniteGroupCohomologyAlgebraGenerator as RustFiniteGroupCohomologyAlgebraGenerator,
    FiniteGroupCohomologyAlgebraTruncation as RustFiniteGroupCohomologyAlgebraTruncation,
    FiniteGroupCohomologyCompletePresentation as RustFiniteGroupCohomologyCompletePresentation,
    FiniteGroupCohomologyElementaryAbelianRestriction as RustFiniteGroupCohomologyElementaryAbelianRestriction,
    FiniteGroupCohomologyError,
    FiniteGroupCohomologySymondsParameterCertificate as RustFiniteGroupCohomologySymondsParameterCertificate,
    FiniteGroupCohomologyTruncation as RustFiniteGroupCohomologyTruncation,
    FiniteGroupIntegralCohomologyError,
    FiniteGroupIntegralCohomologyTruncation as RustFiniteGroupIntegralCohomologyTruncation,
    FinitelyGeneratedAbelianGroup as RustFinitelyGeneratedAbelianGroup,
    GradedHomology as RustGradedHomology, HomologyError, HomologyGroup as RustHomologyGroup,
    PadicChainCancellation as RustPadicChainCancellation,
    PadicChainReduction as RustPadicChainReduction, PadicCyclicSummand as RustPadicCyclicSummand,
    PadicModuleInvariants as RustPadicModuleInvariants, RankBounds as RustRankBounds,
    UniversalUnit,
};
use noether::numthy::{FiniteField, NumberField, PadicRing};
use noether::{IntegerRing, RationalField, SemanticType};
use pyo3::exceptions::{PyAttributeError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyMapping, PyMappingMethods, PyTuple, PyType};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::commalg::{PyGradedCommutativePolynomial, PyGradedCommutativePresentation};
use super::facade_scalar::{extract_integer, integer_to_python};
use super::linalg::{
    MatrixInner, PyMatrix, PySolutionSpace, PySparseMatrix, PyVector, SolutionSpaceInner,
    SparseMatrixInner, VectorInner,
};
use super::number_field::PyNumberField;
use super::numthy::{PyFiniteField, PyFiniteFieldElement, PyPadicElement, PyPadicRing};
use super::root::{PyIntegerRing, PyRationalField};
use super::topology_facade::{
    PyAdmissibleCohomologyOperation, PyEilenbergMacLaneCohomologyGenerator,
    PyEilenbergMacLaneCohomologyOperation, PyEilenbergMacLaneIntegralCohomologyPresentation,
    PyEilenbergMacLaneModularCohomology,
};
use super::validity::PyMathematicalValidity;

fn homology_error(error: HomologyError) -> PyErr {
    match error {
        HomologyError::ContextMismatch { error, .. } => super::errors::context_mismatch(error),
        HomologyError::DegreeOverflow { .. } => PyOverflowError::new_err(error.to_string()),
        HomologyError::Verification(error) => super::errors::verification(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

pub(crate) fn abelian_group_error(error: AbelianGroupError) -> PyErr {
    match error {
        AbelianGroupError::InvalidInvariants(error) => super::errors::verification(error),
        AbelianGroupError::RankOverflow { .. } => PyOverflowError::new_err(error.to_string()),
    }
}

pub(crate) fn finite_group_cohomology_error(error: FiniteGroupCohomologyError) -> PyErr {
    match error {
        FiniteGroupCohomologyError::BarComplexTooLarge { .. }
        | FiniteGroupCohomologyError::BasisCardinalityOverflow { .. }
        | FiniteGroupCohomologyError::CompletionBoundOverflow => {
            PyOverflowError::new_err(error.to_string())
        }
        FiniteGroupCohomologyError::Verification(error) => super::errors::verification(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

pub(crate) fn finite_group_integral_cohomology_error(
    error: FiniteGroupIntegralCohomologyError,
) -> PyErr {
    match error {
        FiniteGroupIntegralCohomologyError::BasisCardinalityOverflow { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        FiniteGroupIntegralCohomologyError::Verification(error) => {
            super::errors::verification(error)
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

pub(crate) fn finite_abelian_group_cohomology_error(
    error: FiniteAbelianGroupCohomologyError,
) -> PyErr {
    match error {
        FiniteAbelianGroupCohomologyError::AbelianInvariants(error) => match error {
            noether::groups::FiniteAbelianInvariantError::EnumerationLimitExceeded { .. }
            | noether::groups::FiniteAbelianInvariantError::OrderTooLarge => {
                PyOverflowError::new_err(error.to_string())
            }
            _ => PyValueError::new_err(error.to_string()),
        },
        FiniteAbelianGroupCohomologyError::EilenbergMacLane(error) => {
            super::topology_facade::eilenberg_maclane_error(error)
        }
        FiniteAbelianGroupCohomologyError::Verification(error) => {
            super::errors::verification(error)
        }
    }
}

pub(crate) fn finite_cyclic_group_integral_cohomology_error(
    error: FiniteCyclicGroupIntegralCohomologyError,
) -> PyErr {
    match error {
        FiniteCyclicGroupIntegralCohomologyError::AbelianInvariants(error) => match error {
            noether::groups::FiniteAbelianInvariantError::EnumerationLimitExceeded { .. }
            | noether::groups::FiniteAbelianInvariantError::OrderTooLarge => {
                PyOverflowError::new_err(error.to_string())
            }
            _ => PyValueError::new_err(error.to_string()),
        },
        FiniteCyclicGroupIntegralCohomologyError::NotCyclic { .. } => {
            PyValueError::new_err(error.to_string())
        }
        FiniteCyclicGroupIntegralCohomologyError::Verification(error) => {
            super::errors::verification(error)
        }
    }
}

/// A family-theorem complete presentation of H*(BG; F_p) for finite abelian G.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteAbelianGroupCohomologyPresentation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteAbelianGroupCohomologyPresentation {
    pub(crate) inner: RustFiniteAbelianGroupCohomologyPresentation,
}

#[pymethods]
impl PyFiniteAbelianGroupCohomologyPresentation {
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteAbelianGroupCohomologyPresentation::TYPE_ID
    }

    #[getter]
    fn group(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    #[getter]
    fn invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .invariant_factors()
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    #[getter]
    fn classifying_space_cohomology(&self) -> PyEilenbergMacLaneModularCohomology {
        PyEilenbergMacLaneModularCohomology {
            inner: self.inner.classifying_space_cohomology().clone(),
        }
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn presentation(&self) -> PyGradedCommutativePresentation {
        PyGradedCommutativePresentation {
            inner: self.inner.presentation().clone(),
        }
    }

    #[getter]
    fn generators(&self) -> Vec<PyEilenbergMacLaneCohomologyGenerator> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(|inner| PyEilenbergMacLaneCohomologyGenerator { inner })
            .collect()
    }

    #[getter]
    fn operations(&self) -> Vec<PyEilenbergMacLaneCohomologyOperation> {
        self.inner
            .operations()
            .iter()
            .cloned()
            .map(|inner| PyEilenbergMacLaneCohomologyOperation { inner })
            .collect()
    }

    #[getter]
    fn completion_strategy(&self) -> &'static str {
        "finite_abelian_classifying_space"
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteAbelianGroupCohomologyPresentation(invariant_factors={:?}, field={}, generators={})",
            self.inner.invariant_factors(),
            self.inner.field(),
            self.inner.generators().len()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item(
            "type",
            RustFiniteAbelianGroupCohomologyPresentation::TYPE_ID,
        )?;
        summary.set_item("invariant_factors", self.invariant_factors(py)?)?;
        summary.set_item("field", self.inner.field().to_string())?;
        summary.set_item("completion_strategy", "finite_abelian_classifying_space")?;
        Ok(summary.unbind())
    }
}

/// A family-theorem complete presentation of H*(BG; Z) for finite abelian G.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteAbelianGroupIntegralCohomologyPresentation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteAbelianGroupIntegralCohomologyPresentation {
    pub(crate) inner: RustFiniteAbelianGroupIntegralCohomologyPresentation,
}

#[pymethods]
impl PyFiniteAbelianGroupIntegralCohomologyPresentation {
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteAbelianGroupIntegralCohomologyPresentation::TYPE_ID
    }

    #[getter]
    fn group(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    #[getter]
    fn invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .invariant_factors()
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    #[getter]
    fn classifying_space_cohomology(&self) -> PyEilenbergMacLaneIntegralCohomologyPresentation {
        PyEilenbergMacLaneIntegralCohomologyPresentation {
            inner: self.inner.classifying_space_cohomology().clone(),
        }
    }

    #[getter]
    fn completion_strategy(&self) -> &'static str {
        self.inner.completion_strategy()
    }

    fn additive_group(&self, degree: usize) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        Ok(PyFinitelyGeneratedAbelianGroup {
            inner: self
                .inner
                .additive_group(degree)
                .map_err(super::topology_facade::eilenberg_maclane_error)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteAbelianGroupIntegralCohomologyPresentation(invariant_factors={:?}, generators={})",
            self.inner.invariant_factors(),
            self.inner.classifying_space_cohomology().generators().len()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item(
            "type",
            RustFiniteAbelianGroupIntegralCohomologyPresentation::TYPE_ID,
        )?;
        summary.set_item("invariant_factors", self.invariant_factors(py)?)?;
        summary.set_item("coefficient_ring", "Z")?;
        summary.set_item("completion_strategy", self.inner.completion_strategy())?;
        Ok(summary.unbind())
    }
}

/// A periodic-resolution complete presentation of H*(BG; Z) for cyclic G.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteCyclicGroupIntegralCohomologyPresentation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteCyclicGroupIntegralCohomologyPresentation {
    pub(crate) inner: RustFiniteCyclicGroupIntegralCohomologyPresentation,
}

#[pymethods]
impl PyFiniteCyclicGroupIntegralCohomologyPresentation {
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteCyclicGroupIntegralCohomologyPresentation::TYPE_ID
    }

    #[getter]
    fn group(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.order())
    }

    #[getter]
    fn generator_id(&self) -> Option<&'static str> {
        self.inner.generator_id()
    }

    #[getter]
    fn generator_degree(&self) -> Option<u32> {
        self.inner.generator_degree()
    }

    #[getter]
    fn generator_display_name(&self) -> Option<String> {
        self.inner.generator_display_name()
    }

    #[getter]
    fn relation_coefficient(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .relation_coefficient()
            .map(|coefficient| integer_to_python(py, coefficient))
            .transpose()
    }

    #[getter]
    fn completion_strategy(&self) -> &'static str {
        self.inner.completion_strategy()
    }

    fn additive_group(&self, degree: usize) -> PyFinitelyGeneratedAbelianGroup {
        PyFinitelyGeneratedAbelianGroup {
            inner: self.inner.additive_group(degree),
        }
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteCyclicGroupIntegralCohomologyPresentation(order={}, generator_id={:?})",
            self.inner.order(),
            self.inner.generator_id()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item(
            "type",
            RustFiniteCyclicGroupIntegralCohomologyPresentation::TYPE_ID,
        )?;
        summary.set_item("order", integer_to_python(py, self.inner.order())?)?;
        summary.set_item("generator", self.inner.generator_display_name())?;
        summary.set_item("completion_strategy", self.inner.completion_strategy())?;
        Ok(summary.unbind())
    }
}

/// Exact initial segment of a finite group's mod-p cohomology ring.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologyTruncation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologyTruncation {
    pub(crate) inner: RustFiniteGroupCohomologyTruncation,
}

#[pymethods]
impl PyFiniteGroupCohomologyTruncation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteGroupCohomologyTruncation::TYPE_ID
    }

    /// Supplied concrete permutation group.
    #[getter]
    fn group(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    /// Exact prime coefficient field.
    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    /// Largest cohomological degree computed exactly.
    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    /// Stable name of the mathematical resolution strategy.
    #[getter]
    fn resolution_strategy(&self) -> &'static str {
        match self.inner.resolution_strategy() {
            noether::homalg::FiniteGroupResolutionStrategy::MinimalPGroup => "minimal_p_group",
            noether::homalg::FiniteGroupResolutionStrategy::NormalizedBar => "normalized_bar",
            noether::homalg::FiniteGroupResolutionStrategy::SylowStableElements => {
                "sylow_stable_elements"
            }
        }
    }

    /// Dimension of one mod-p cohomology group.
    fn dimension(&self, degree: usize) -> PyResult<usize> {
        self.inner
            .dimension(degree)
            .map_err(finite_group_cohomology_error)
    }

    /// Deterministically chosen model cocycles for one basis.
    fn basis_cocycles(&self, degree: usize) -> PyResult<Vec<PyVector>> {
        Ok(self
            .inner
            .basis_cocycles(degree)
            .map_err(finite_group_cohomology_error)?
            .iter()
            .cloned()
            .map(|vector| PyVector {
                inner: VectorInner::FiniteField(vector),
            })
            .collect())
    }

    /// Coordinates of the cup product of two canonical basis classes.
    fn basis_cup_product(
        &self,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
        right_index: usize,
    ) -> PyResult<PyVector> {
        Ok(PyVector {
            inner: VectorInner::FiniteField(
                self.inner
                    .basis_cup_product(left_degree, left_index, right_degree, right_index)
                    .map_err(finite_group_cohomology_error)?,
            ),
        })
    }

    /// Matrix of multiplication on the left by one canonical basis class.
    fn basis_left_multiplication(
        &self,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
    ) -> PyResult<PyMatrix> {
        Ok(PyMatrix {
            inner: MatrixInner::FiniteField(
                self.inner
                    .basis_left_multiplication(left_degree, left_index, right_degree)
                    .map_err(finite_group_cohomology_error)?,
            ),
        })
    }

    /// Apply a typed admissible Steenrod word to one basis class.
    fn apply_operation(
        &self,
        operation: &PyAdmissibleCohomologyOperation,
        source_degree: usize,
        source_index: usize,
    ) -> PyResult<PyVector> {
        Ok(PyVector {
            inner: VectorInner::FiniteField(
                self.inner
                    .apply_operation(&operation.inner, source_degree, source_index)
                    .map_err(finite_group_cohomology_error)?,
            ),
        })
    }

    /// Minimal generators and exact relations visible through this bound.
    fn algebra_truncation(&self) -> PyResult<PyFiniteGroupCohomologyAlgebraTruncation> {
        Ok(PyFiniteGroupCohomologyAlgebraTruncation {
            inner: self
                .inner
                .algebra_truncation()
                .map_err(finite_group_cohomology_error)?,
        })
    }

    /// Recompute the retained resolution, quotient bases, and evidence.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologyTruncation(group_order={}, field={}, max_degree={})",
            self.inner.group().order().to_decimal(),
            self.inner.field(),
            self.inner.max_degree()
        ))
    }

    /// Bounded semantic summary for interactive clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustFiniteGroupCohomologyTruncation::TYPE_ID)?;
        summary.set_item("group_order", self.inner.group().order().to_decimal())?;
        summary.set_item(
            "characteristic",
            self.inner.field().characteristic().to_string(),
        )?;
        summary.set_item("max_degree", self.inner.max_degree())?;
        summary.set_item("resolution_strategy", self.resolution_strategy())?;
        summary.set_item(
            "dimensions",
            (0..=self.inner.max_degree())
                .map(|degree| self.inner.dimension(degree))
                .collect::<Result<Vec<_>, _>>()
                .map_err(finite_group_cohomology_error)?,
        )?;
        Ok(summary.unbind())
    }
}

/// Exact initial segment of a finite group's integral cohomology ring.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupIntegralCohomologyTruncation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupIntegralCohomologyTruncation {
    pub(crate) inner: RustFiniteGroupIntegralCohomologyTruncation,
}

#[pymethods]
impl PyFiniteGroupIntegralCohomologyTruncation {
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteGroupIntegralCohomologyTruncation::TYPE_ID
    }

    #[getter]
    fn group(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    #[getter]
    fn resolution_strategy(&self) -> &'static str {
        match self.inner.resolution_strategy() {
            noether::homalg::FiniteGroupIntegralResolutionStrategy::UniversalLowDegree => {
                "universal_low_degree"
            }
            noether::homalg::FiniteGroupIntegralResolutionStrategy::NormalizedBar => {
                "normalized_bar"
            }
        }
    }

    fn additive_group(&self, degree: usize) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        Ok(PyFinitelyGeneratedAbelianGroup {
            inner: self
                .inner
                .additive_group(degree)
                .map_err(finite_group_integral_cohomology_error)?
                .clone(),
        })
    }

    fn basis_orders(&self, py: Python<'_>, degree: usize) -> PyResult<Vec<Option<Py<PyAny>>>> {
        self.inner
            .basis_orders(degree)
            .map_err(finite_group_integral_cohomology_error)?
            .iter()
            .map(|order| {
                order
                    .as_ref()
                    .map(|order| integer_to_python(py, order))
                    .transpose()
            })
            .collect()
    }

    fn basis_cocycles(&self, degree: usize) -> PyResult<Option<Vec<PyVector>>> {
        Ok(self
            .inner
            .basis_cocycles(degree)
            .map_err(finite_group_integral_cohomology_error)?
            .map(|cocycles| {
                cocycles
                    .iter()
                    .cloned()
                    .map(|vector| PyVector {
                        inner: VectorInner::Integer(vector),
                    })
                    .collect()
            }))
    }

    fn basis_cup_product(
        &self,
        py: Python<'_>,
        left_degree: usize,
        left_index: usize,
        right_degree: usize,
        right_index: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .basis_cup_product(left_degree, left_index, right_degree, right_index)
            .map_err(finite_group_integral_cohomology_error)?
            .iter()
            .map(|coordinate| integer_to_python(py, coordinate))
            .collect()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupIntegralCohomologyTruncation(group_order={}, max_degree={})",
            self.inner.group().order().to_decimal(),
            self.inner.max_degree()
        ))
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustFiniteGroupIntegralCohomologyTruncation::TYPE_ID)?;
        summary.set_item("group_order", self.inner.group().order().to_decimal())?;
        summary.set_item("max_degree", self.inner.max_degree())?;
        summary.set_item("resolution_strategy", self.resolution_strategy())?;
        summary.set_item(
            "additive_groups",
            (0..=self.inner.max_degree())
                .map(|degree| {
                    self.inner
                        .additive_group(degree)
                        .map(|group| group.to_string())
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(finite_group_integral_cohomology_error)?,
        )?;
        Ok(summary.unbind())
    }
}

/// One minimal homogeneous generator chosen from a canonical cohomology basis.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologyAlgebraGenerator",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologyAlgebraGenerator {
    inner: RustFiniteGroupCohomologyAlgebraGenerator,
}

#[pymethods]
impl PyFiniteGroupCohomologyAlgebraGenerator {
    /// Stable presentation-generator name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Mathematical display name, operation-adapted when certified.
    #[getter]
    fn display_name(&self) -> &str {
        self.inner.display_name()
    }

    /// Cohomological degree.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Zero-based index in the canonical basis of this degree.
    #[getter]
    fn basis_index(&self) -> usize {
        self.inner.basis_index()
    }

    /// Typed operation naming this class modulo decomposables, if available.
    #[getter]
    fn operation(&self) -> Option<PyAdmissibleCohomologyOperation> {
        self.inner
            .operation_witness()
            .map(|witness| PyAdmissibleCohomologyOperation {
                inner: witness.operation().clone(),
            })
    }

    /// Earlier generator to which `operation` applies.
    #[getter]
    fn operation_source_generator_index(&self) -> Option<usize> {
        self.inner.operation_witness().map(
            noether::homalg::FiniteGroupCohomologyGeneratorOperationWitness::source_generator_index,
        )
    }

    /// Scalar multiplying the operation image in the retained witness.
    #[getter]
    fn operation_coefficient(&self) -> Option<PyFiniteFieldElement> {
        self.inner
            .operation_witness()
            .map(|witness| PyFiniteFieldElement {
                inner: witness.operation_coefficient().clone(),
            })
    }

    /// Decomposable correction making the operation identity exact.
    #[getter]
    fn operation_decomposable_correction(&self) -> Option<PyVector> {
        self.inner.operation_witness().map(|witness| PyVector {
            inner: VectorInner::FiniteField(witness.decomposable_correction().clone()),
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologyAlgebraGenerator(name={:?}, display_name={:?}, degree={}, basis_index={})",
            self.inner.name(), self.inner.display_name(),
            self.inner.degree(),
            self.inner.basis_index()
        ))
    }
}

/// Minimal generators and all relations visible in an exact cohomology range.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologyAlgebraTruncation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologyAlgebraTruncation {
    inner: RustFiniteGroupCohomologyAlgebraTruncation,
}

#[pymethods]
impl PyFiniteGroupCohomologyAlgebraTruncation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteGroupCohomologyAlgebraTruncation::TYPE_ID
    }

    /// Source exact cohomology truncation.
    #[getter]
    fn cohomology(&self) -> PyFiniteGroupCohomologyTruncation {
        PyFiniteGroupCohomologyTruncation {
            inner: self.inner.cohomology().clone(),
        }
    }

    /// Largest degree in which the presentation is exact.
    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    /// Deterministic minimal homogeneous generators.
    #[getter]
    fn generators(&self) -> Vec<PyFiniteGroupCohomologyAlgebraGenerator> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(|inner| PyFiniteGroupCohomologyAlgebraGenerator { inner })
            .collect()
    }

    /// Quotient presentation exact through `max_degree` only.
    #[getter]
    fn presentation(&self) -> PyGradedCommutativePresentation {
        PyGradedCommutativePresentation {
            inner: self.inner.presentation().clone(),
        }
    }

    /// Promote to a complete presentation using Symonds's universal bounds.
    fn certify_symonds_universal_bound(
        &self,
    ) -> PyResult<PyFiniteGroupCohomologyCompletePresentation> {
        Ok(PyFiniteGroupCohomologyCompletePresentation {
            inner: self
                .inner
                .certify_symonds_universal_bound()
                .map_err(finite_group_cohomology_error)?,
        })
    }

    /// Promote using exact elementary-abelian restriction evidence and
    /// Symonds's parameter criterion.
    fn certify_symonds_parameter_criterion(
        &self,
    ) -> PyResult<PyFiniteGroupCohomologyCompletePresentation> {
        Ok(PyFiniteGroupCohomologyCompletePresentation {
            inner: self
                .inner
                .certify_symonds_parameter_criterion()
                .map_err(finite_group_cohomology_error)?,
        })
    }

    /// Recompute generator selection and relation extraction.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologyAlgebraTruncation(max_degree={}, generators={}, relations={})",
            self.inner.max_degree(),
            self.inner.generators().len(),
            self.inner.presentation().relations().len()
        ))
    }
}

/// One maximal elementary abelian restriction in a Symonds certificate.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologyElementaryAbelianRestriction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologyElementaryAbelianRestriction {
    inner: RustFiniteGroupCohomologyElementaryAbelianRestriction,
}

#[pymethods]
impl PyFiniteGroupCohomologyElementaryAbelianRestriction {
    #[getter]
    fn subgroup(&self) -> super::groups::PyPermGroup {
        super::groups::PyPermGroup {
            inner: self.inner.subgroup().clone(),
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn restricted_parameters(&self) -> Vec<PyGradedCommutativePolynomial> {
        self.inner
            .restricted_parameters()
            .iter()
            .cloned()
            .map(|inner| PyGradedCommutativePolynomial { inner })
            .collect()
    }

    #[getter]
    fn quotient_generator_power_bounds(&self) -> Vec<u32> {
        self.inner.quotient_generator_power_bounds().to_vec()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologyElementaryAbelianRestriction(rank={}, parameters={})",
            self.inner.rank(),
            self.inner.restricted_parameters().len()
        ))
    }
}

/// Replayable evidence for Symonds's parameter completion criterion.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologySymondsParameterCertificate",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologySymondsParameterCertificate {
    inner: RustFiniteGroupCohomologySymondsParameterCertificate,
}

#[pymethods]
impl PyFiniteGroupCohomologySymondsParameterCertificate {
    #[getter]
    fn parameter_generator_indices(&self) -> Vec<usize> {
        self.inner.parameter_generator_indices().to_vec()
    }

    #[getter]
    fn elementary_abelian_restrictions(
        &self,
    ) -> Vec<PyFiniteGroupCohomologyElementaryAbelianRestriction> {
        self.inner
            .elementary_abelian_restrictions()
            .iter()
            .cloned()
            .map(|inner| PyFiniteGroupCohomologyElementaryAbelianRestriction { inner })
            .collect()
    }

    #[getter]
    fn quotient_generator_power_bounds(&self) -> Vec<u32> {
        self.inner.quotient_generator_power_bounds().to_vec()
    }

    #[getter]
    fn maximum_module_generator_degree(&self) -> usize {
        self.inner.maximum_module_generator_degree()
    }

    #[getter]
    fn parameter_degree_bound(&self) -> usize {
        self.inner.parameter_degree_bound()
    }

    #[getter]
    fn completion_degree_bound(&self) -> usize {
        self.inner.completion_degree_bound()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologySymondsParameterCertificate(parameters={}, maximal_elementary_abelian_subgroups={}, completion_degree_bound={})",
            self.inner.parameter_generator_indices().len(),
            self.inner.elementary_abelian_restrictions().len(),
            self.inner.completion_degree_bound()
        ))
    }
}

/// A complete finite-group cohomology presentation with theorem evidence.
#[pyclass(
    module = "noether.homalg",
    name = "FiniteGroupCohomologyCompletePresentation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFiniteGroupCohomologyCompletePresentation {
    inner: RustFiniteGroupCohomologyCompletePresentation,
}

#[pymethods]
impl PyFiniteGroupCohomologyCompletePresentation {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustFiniteGroupCohomologyCompletePresentation::TYPE_ID
    }

    /// Exact algebra data promoted by the completion theorem.
    #[getter]
    fn algebra(&self) -> PyFiniteGroupCohomologyAlgebraTruncation {
        PyFiniteGroupCohomologyAlgebraTruncation {
            inner: self.inner.algebra().clone(),
        }
    }

    /// Complete graded-commutative quotient presentation.
    #[getter]
    fn presentation(&self) -> PyGradedCommutativePresentation {
        PyGradedCommutativePresentation {
            inner: self.inner.presentation().clone(),
        }
    }

    /// Stable name of the theorem used to discharge completeness.
    #[getter]
    fn completion_strategy(&self) -> &'static str {
        match self.inner.completion_strategy() {
            noether::homalg::FiniteGroupCohomologyCompletionStrategy::SymondsUniversalBound => {
                "symonds_universal_bound"
            }
            noether::homalg::FiniteGroupCohomologyCompletionStrategy::SymondsParameterCriterion => {
                "symonds_parameter_criterion"
            }
        }
    }

    /// Exact order of the finite group in the theorem certificate.
    #[getter]
    fn group_order(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.group_order())
    }

    /// Universal degree bound for ring generators.
    #[getter]
    fn generator_degree_bound(&self) -> usize {
        self.inner.generator_degree_bound()
    }

    /// Universal degree bound for defining relations.
    #[getter]
    fn relation_degree_bound(&self) -> usize {
        self.inner.relation_degree_bound()
    }

    /// Parameter certificate when the parameter criterion was selected.
    #[getter]
    fn symonds_parameter_certificate(
        &self,
    ) -> Option<PyFiniteGroupCohomologySymondsParameterCertificate> {
        self.inner
            .symonds_parameter_certificate()
            .cloned()
            .map(|inner| PyFiniteGroupCohomologySymondsParameterCertificate { inner })
    }

    /// Recompute the algebra and every completion hypothesis.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FiniteGroupCohomologyCompletePresentation(group_order={}, strategy={:?}, generator_degree_bound={}, relation_degree_bound={})",
            self.inner.group_order(),
            self.inner.completion_strategy(),
            self.inner.generator_degree_bound(),
            self.inner.relation_degree_bound()
        ))
    }
}

enum MatrixCollection<R: noether::Ring> {
    Dense(BTreeMap<i32, noether::linalg::Matrix<R>>),
    Sparse(BTreeMap<i32, noether::linalg::SparseMatrix<R>>),
}

fn collect_matrices<R, F, G>(
    values: &Bound<'_, PyMapping>,
    context_name: &str,
    extract_dense: F,
    extract_sparse: G,
) -> PyResult<MatrixCollection<R>>
where
    R: noether::Ring,
    F: Fn(&MatrixInner) -> Option<noether::linalg::Matrix<R>>,
    G: Fn(&SparseMatrixInner) -> Option<noether::linalg::SparseMatrix<R>>,
{
    let mut dense = BTreeMap::new();
    let mut sparse = BTreeMap::new();
    for item in values.items()?.iter() {
        let item = item.cast::<PyTuple>()?;
        let degree = item.get_item(0)?;
        let matrix = item.get_item(1)?;
        let degree = degree
            .extract::<i32>()
            .map_err(|_| PyTypeError::new_err("complex degrees must be signed 32-bit integers"))?;
        if let Ok(matrix) = matrix.extract::<PyRef<'_, PyMatrix>>() {
            if !sparse.is_empty() {
                return Err(PyTypeError::new_err(
                    "one complex presentation cannot mix Matrix and SparseMatrix differentials",
                ));
            }
            let matrix = extract_dense(&matrix.inner).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "differential in degree {degree} does not have coefficient context {context_name}"
                ))
            })?;
            dense.insert(degree, matrix);
            continue;
        }
        if let Ok(matrix) = matrix.extract::<PyRef<'_, PySparseMatrix>>() {
            if !dense.is_empty() {
                return Err(PyTypeError::new_err(
                    "one complex presentation cannot mix Matrix and SparseMatrix differentials",
                ));
            }
            let matrix = extract_sparse(&matrix.inner).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "differential in degree {degree} does not have coefficient context {context_name}"
                ))
            })?;
            sparse.insert(degree, matrix);
            continue;
        }
        return Err(PyTypeError::new_err(
            "complex differentials must be noether.linalg.Matrix or SparseMatrix values",
        ));
    }
    if sparse.is_empty() {
        Ok(MatrixCollection::Dense(dense))
    } else {
        Ok(MatrixCollection::Sparse(sparse))
    }
}

fn construct_complex<R: noether::Ring>(
    ring: R,
    matrices: MatrixCollection<R>,
    coboundary: bool,
) -> Result<ChainComplex<R>, HomologyError> {
    match (matrices, coboundary) {
        (MatrixCollection::Dense(matrices), false) => ChainComplex::from_boundaries(ring, matrices),
        (MatrixCollection::Dense(matrices), true) => {
            ChainComplex::from_coboundaries(ring, matrices)
        }
        (MatrixCollection::Sparse(matrices), false) => {
            ChainComplex::from_sparse_boundaries(ring, matrices)
        }
        (MatrixCollection::Sparse(matrices), true) => {
            ChainComplex::from_sparse_coboundaries(ring, matrices)
        }
    }
}

fn collect_ranks(values: &Bound<'_, PyMapping>) -> PyResult<BTreeMap<i32, usize>> {
    let mut result = BTreeMap::new();
    for item in values.items()?.iter() {
        let item = item.cast::<PyTuple>()?;
        let degree = item
            .get_item(0)?
            .extract::<i32>()
            .map_err(|_| PyTypeError::new_err("rank degrees must be signed 32-bit integers"))?;
        let rank = item
            .get_item(1)?
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("differential ranks must be nonnegative integers"))?;
        result.insert(degree, rank);
    }
    Ok(result)
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum ChainComplexInner {
    Integer(ChainComplex<IntegerRing>),
    Rational(ChainComplex<RationalField>),
    FiniteField(ChainComplex<FiniteField>),
    NumberField(ChainComplex<NumberField>),
    Padic(ChainComplex<PadicRing>),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum ChainReductionInner {
    Integer(ChainReduction<IntegerRing>),
    Rational(ChainReduction<RationalField>),
    FiniteField(ChainReduction<FiniteField>),
    NumberField(ChainReduction<NumberField>),
    Padic(ChainReduction<PadicRing>),
}

/// One universal-unit pivot removed from a based chain complex.
#[pyclass(
    module = "noether.homalg",
    name = "ChainCancellation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyChainCancellation {
    inner: ChainCancellation,
}

#[pymethods]
impl PyChainCancellation {
    #[new]
    fn new(degree: i32, row: usize, column: usize, unit: i8) -> PyResult<Self> {
        let unit = match unit {
            1 => UniversalUnit::One,
            -1 => UniversalUnit::NegativeOne,
            _ => return Err(PyValueError::new_err("unit must be 1 or -1")),
        };
        Ok(Self {
            inner: ChainCancellation::new(degree, row, column, unit),
        })
    }

    /// Displayed differential degree at the time of cancellation.
    #[getter]
    fn degree(&self) -> i32 {
        self.inner.degree()
    }

    /// Current matrix row of the pivot.
    #[getter]
    fn row(&self) -> usize {
        self.inner.row()
    }

    /// Current matrix column of the pivot.
    #[getter]
    fn column(&self) -> usize {
        self.inner.column()
    }

    /// Universal unit, either `1` or `-1`.
    #[getter]
    fn unit(&self) -> i8 {
        match self.inner.unit() {
            UniversalUnit::One => 1,
            UniversalUnit::NegativeOne => -1,
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ChainCancellation(degree={}, row={}, column={}, unit={})",
            self.degree(),
            self.row(),
            self.column(),
            self.unit()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Replayable algebraic Morse reduction of a based chain complex.
#[pyclass(
    module = "noether.homalg",
    name = "ChainReduction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyChainReduction {
    pub(crate) inner: ChainReductionInner,
}

#[pymethods]
impl PyChainReduction {
    /// Original based chain complex.
    #[getter]
    fn source(&self) -> PyChainComplex {
        let inner = match &self.inner {
            ChainReductionInner::Integer(value) => {
                ChainComplexInner::Integer(value.source().clone())
            }
            ChainReductionInner::Rational(value) => {
                ChainComplexInner::Rational(value.source().clone())
            }
            ChainReductionInner::FiniteField(value) => {
                ChainComplexInner::FiniteField(value.source().clone())
            }
            ChainReductionInner::NumberField(value) => {
                ChainComplexInner::NumberField(value.source().clone())
            }
            ChainReductionInner::Padic(value) => ChainComplexInner::Padic(value.source().clone()),
        };
        PyChainComplex { inner }
    }

    /// Chain-homotopy-equivalent reduced complex.
    #[getter]
    fn reduced(&self) -> PyChainComplex {
        let inner = match &self.inner {
            ChainReductionInner::Integer(value) => {
                ChainComplexInner::Integer(value.reduced().clone())
            }
            ChainReductionInner::Rational(value) => {
                ChainComplexInner::Rational(value.reduced().clone())
            }
            ChainReductionInner::FiniteField(value) => {
                ChainComplexInner::FiniteField(value.reduced().clone())
            }
            ChainReductionInner::NumberField(value) => {
                ChainComplexInner::NumberField(value.reduced().clone())
            }
            ChainReductionInner::Padic(value) => ChainComplexInner::Padic(value.reduced().clone()),
        };
        PyChainComplex { inner }
    }

    /// Elementary cancellations in replay order.
    #[getter]
    fn cancellations(&self) -> Vec<PyChainCancellation> {
        let cancellations = match &self.inner {
            ChainReductionInner::Integer(value) => value.cancellations(),
            ChainReductionInner::Rational(value) => value.cancellations(),
            ChainReductionInner::FiniteField(value) => value.cancellations(),
            ChainReductionInner::NumberField(value) => value.cancellations(),
            ChainReductionInner::Padic(value) => value.cancellations(),
        };
        cancellations
            .iter()
            .copied()
            .map(|inner| PyChainCancellation { inner })
            .collect()
    }

    /// Number of contractible basis pairs removed.
    #[getter]
    fn removed_pairs(&self) -> usize {
        match &self.inner {
            ChainReductionInner::Integer(value) => value.removed_pairs(),
            ChainReductionInner::Rational(value) => value.removed_pairs(),
            ChainReductionInner::FiniteField(value) => value.removed_pairs(),
            ChainReductionInner::NumberField(value) => value.removed_pairs(),
            ChainReductionInner::Padic(value) => value.removed_pairs(),
        }
    }

    /// Replay the cancellation log and recheck the reduced complex.
    fn verify(&self) -> PyResult<()> {
        let result = match &self.inner {
            ChainReductionInner::Integer(value) => value.verify(),
            ChainReductionInner::Rational(value) => value.verify(),
            ChainReductionInner::FiniteField(value) => value.verify(),
            ChainReductionInner::NumberField(value) => value.verify(),
            ChainReductionInner::Padic(value) => value.verify(),
        };
        result.map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ChainReduction(removed_pairs={})",
            self.removed_pairs()
        ))
    }
}

/// One certified p-adic-unit pivot removed from a based complex.
#[pyclass(
    module = "noether.homalg",
    name = "PadicChainCancellation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPadicChainCancellation {
    inner: RustPadicChainCancellation,
}

#[pymethods]
impl PyPadicChainCancellation {
    #[getter]
    fn degree(&self) -> i32 {
        self.inner.degree()
    }

    #[getter]
    fn row(&self) -> usize {
        self.inner.row()
    }

    #[getter]
    fn column(&self) -> usize {
        self.inner.column()
    }

    #[getter]
    fn pivot(&self) -> PyPadicElement {
        PyPadicElement {
            inner: self.inner.pivot().clone(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PadicChainCancellation(degree={}, row={}, column={}, pivot={})",
            self.degree(),
            self.row(),
            self.column(),
            self.inner.pivot()
        ))
    }
}

/// Replayable algebraic Morse reduction by certified p-adic units.
#[pyclass(
    module = "noether.homalg",
    name = "PadicChainReduction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPadicChainReduction {
    inner: RustPadicChainReduction,
}

#[pymethods]
impl PyPadicChainReduction {
    #[getter]
    fn source(&self) -> PyChainComplex {
        PyChainComplex {
            inner: ChainComplexInner::Padic(self.inner.source().clone()),
        }
    }

    #[getter]
    fn reduced(&self) -> PyChainComplex {
        PyChainComplex {
            inner: ChainComplexInner::Padic(self.inner.reduced().clone()),
        }
    }

    #[getter]
    fn cancellations(&self) -> Vec<PyPadicChainCancellation> {
        self.inner
            .cancellations()
            .iter()
            .cloned()
            .map(|inner| PyPadicChainCancellation { inner })
            .collect()
    }

    #[getter]
    fn removed_pairs(&self) -> usize {
        self.inner.removed_pairs()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PadicChainReduction(removed_pairs={})",
            self.removed_pairs()
        ))
    }
}

/// Inclusive lower and upper bounds for a finite free rank.
#[pyclass(module = "noether.homalg", name = "RankBounds", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyRankBounds {
    inner: RustRankBounds,
}

#[pymethods]
impl PyRankBounds {
    #[classattr]
    fn type_id() -> &'static str {
        RustRankBounds::TYPE_ID
    }

    #[getter]
    fn lower(&self) -> usize {
        self.inner.lower()
    }

    #[getter]
    fn upper(&self) -> usize {
        self.inner.upper()
    }

    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    #[getter]
    fn exact(&self) -> Option<usize> {
        self.inner.exact()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RustRankBounds::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "RankBounds(lower={}, upper={})",
            self.lower(),
            self.upper()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One exact or unresolved p-primary cyclic relation.
#[pyclass(
    module = "noether.homalg",
    name = "PadicCyclicSummand",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicCyclicSummand {
    inner: RustPadicCyclicSummand,
}

#[pymethods]
impl PyPadicCyclicSummand {
    #[classattr]
    fn type_id() -> &'static str {
        RustPadicCyclicSummand::TYPE_ID
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn exponent(&self) -> u64 {
        self.inner.exponent()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RustPadicCyclicSummand::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PadicCyclicSummand(kind={:?}, exponent={})",
            self.kind(),
            self.exponent()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// P-primary invariants shared by every compatible finite free `Z_p` lift.
#[pyclass(
    module = "noether.homalg",
    name = "PadicModuleInvariants",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPadicModuleInvariants {
    inner: RustPadicModuleInvariants,
}

#[pymethods]
impl PyPadicModuleInvariants {
    #[classattr]
    fn type_id() -> &'static str {
        RustPadicModuleInvariants::TYPE_ID
    }

    #[getter]
    fn prime(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, self.inner.prime())
    }

    #[getter]
    fn precision(&self) -> u64 {
        self.inner.precision()
    }

    #[getter]
    fn free_rank(&self) -> PyRankBounds {
        PyRankBounds {
            inner: *self.inner.free_rank(),
        }
    }

    #[getter]
    fn cyclic_summands(&self) -> Vec<PyPadicCyclicSummand> {
        self.inner
            .cyclic_summands()
            .iter()
            .copied()
            .map(|inner| PyPadicCyclicSummand { inner })
            .collect()
    }

    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        PyMathematicalValidity {
            inner: self.inner.validity().clone(),
        }
    }

    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        // NOETHER_RELEASES_GIL: verification replays Smith data and retained evidence.
        py.detach(|| self.inner.verify().map_err(super::errors::verification))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        // NOETHER_RELEASES_GIL: serialization materializes the retained source evidence.
        py.detach(|| {
            self.inner
                .to_json()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_owned();
        // NOETHER_RELEASES_GIL: deserialization replays all retained source evidence.
        py.detach(move || {
            Ok(Self {
                inner: RustPadicModuleInvariants::from_json(&value)
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

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", RustPadicModuleInvariants::TYPE_ID)?;
        summary.set_item("prime", self.prime(py)?)?;
        summary.set_item("precision", self.precision())?;
        summary.set_item(
            "free_rank",
            (self.free_rank().lower(), self.free_rank().upper()),
        )?;
        summary.set_item("cyclic_summand_count", self.inner.cyclic_summands().len())?;
        summary.set_item("validity", self.inner.validity().to_string())?;
        Ok(summary.unbind())
    }
}

/// One checked chain-complex type for boundary and coboundary presentations.
#[pyclass(
    module = "noether.homalg",
    name = "ChainComplex",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyChainComplex {
    pub(crate) inner: ChainComplexInner,
}

#[pymethods]
impl PyChainComplex {
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        ChainComplex::<RationalField>::TYPE_ID
    }

    /// Construct from `d_n: C_n -> C_(n-1)` matrices keyed by `n`.
    #[classmethod]
    fn from_boundaries(
        _cls: &Bound<'_, PyType>,
        ring: &Bound<'_, PyAny>,
        boundaries: &Bound<'_, PyMapping>,
    ) -> PyResult<Self> {
        Self::construct(ring, boundaries, false)
    }

    /// Construct from `delta^n: C^n -> C^(n+1)` matrices keyed by `n`.
    #[classmethod]
    fn from_coboundaries(
        _cls: &Bound<'_, PyType>,
        ring: &Bound<'_, PyAny>,
        coboundaries: &Bound<'_, PyMapping>,
    ) -> PyResult<Self> {
        Self::construct(ring, coboundaries, true)
    }

    /// Exact coefficient ring.
    #[getter]
    fn ring(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ChainComplexInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            ChainComplexInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            ChainComplexInner::FiniteField(complex) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: complex.ring().clone(),
                },
            )?
            .into_any()),
            ChainComplexInner::NumberField(complex) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: complex.ring().clone(),
                },
            )?
            .into_any()),
            ChainComplexInner::Padic(complex) => Ok(Py::new(
                py,
                PyPadicRing {
                    inner: complex.ring().clone(),
                },
            )?
            .into_any()),
        }
    }

    /// Displayed degrees carrying an inferred free module.
    #[getter]
    fn degrees(&self) -> Vec<i32> {
        match &self.inner {
            ChainComplexInner::Integer(complex) => complex.degrees(),
            ChainComplexInner::Rational(complex) => complex.degrees(),
            ChainComplexInner::FiniteField(complex) => complex.degrees(),
            ChainComplexInner::NumberField(complex) => complex.degrees(),
            ChainComplexInner::Padic(complex) => complex.degrees(),
        }
    }

    /// Rank of the free module in one displayed degree.
    fn module_dimension(&self, degree: i32) -> usize {
        match &self.inner {
            ChainComplexInner::Integer(complex) => complex.module_dimension(degree),
            ChainComplexInner::Rational(complex) => complex.module_dimension(degree),
            ChainComplexInner::FiniteField(complex) => complex.module_dimension(degree),
            ChainComplexInner::NumberField(complex) => complex.module_dimension(degree),
            ChainComplexInner::Padic(complex) => complex.module_dimension(degree),
        }
    }

    /// Retained storage layout of one represented differential.
    fn differential_layout(&self, degree: i32) -> Option<&'static str> {
        match &self.inner {
            ChainComplexInner::Integer(complex) => complex.differential_layout(degree),
            ChainComplexInner::Rational(complex) => complex.differential_layout(degree),
            ChainComplexInner::FiniteField(complex) => complex.differential_layout(degree),
            ChainComplexInner::NumberField(complex) => complex.differential_layout(degree),
            ChainComplexInner::Padic(complex) => complex.differential_layout(degree),
        }
    }

    /// Differential supplied at one displayed degree, if present.
    fn differential(&self, degree: i32) -> Option<PyMatrix> {
        let inner = match &self.inner {
            ChainComplexInner::Integer(complex) => {
                MatrixInner::Integer(complex.differential(degree)?)
            }
            ChainComplexInner::Rational(complex) => {
                MatrixInner::Rational(complex.differential(degree)?)
            }
            ChainComplexInner::FiniteField(complex) => {
                MatrixInner::FiniteField(complex.differential(degree)?)
            }
            ChainComplexInner::NumberField(complex) => {
                MatrixInner::NumberField(complex.differential(degree)?)
            }
            ChainComplexInner::Padic(complex) => MatrixInner::Padic(complex.differential(degree)?),
        };
        Some(PyMatrix { inner })
    }

    /// Cancel every `1` and `-1` pivot and return replayable evidence.
    fn algebraic_morse_reduction(&self, py: Python<'_>) -> PyResult<PyChainReduction> {
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match &self.inner {
                ChainComplexInner::Integer(complex) => ChainReductionInner::Integer(
                    complex
                        .algebraic_morse_reduction()
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::Rational(complex) => ChainReductionInner::Rational(
                    complex
                        .algebraic_morse_reduction()
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::FiniteField(complex) => ChainReductionInner::FiniteField(
                    complex
                        .algebraic_morse_reduction()
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => ChainReductionInner::NumberField(
                    complex
                        .algebraic_morse_reduction()
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(complex) => ChainReductionInner::Padic(
                    complex
                        .algebraic_morse_reduction()
                        .map_err(homology_error)?,
                ),
            })
        })?;
        Ok(PyChainReduction { inner })
    }

    /// Cancel every valuation-zero p-adic pivot with replayable evidence.
    fn padic_unit_morse_reduction(&self, py: Python<'_>) -> PyResult<PyPadicChainReduction> {
        let ChainComplexInner::Padic(complex) = &self.inner else {
            return Err(PyValueError::new_err(
                "padic_unit_morse_reduction requires PadicRing coefficients",
            ));
        };
        Ok(PyPadicChainReduction {
            inner: py
                .detach(|| complex.padic_unit_morse_reduction())
                .map_err(homology_error)?,
        })
    }

    /// Replay externally stored universal-unit cancellation evidence.
    fn reduction_from_cancellations(
        &self,
        py: Python<'_>,
        cancellations: Vec<PyChainCancellation>,
    ) -> PyResult<PyChainReduction> {
        let cancellations = cancellations
            .into_iter()
            .map(|value| value.inner)
            .collect::<Vec<_>>();
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match &self.inner {
                ChainComplexInner::Integer(complex) => ChainReductionInner::Integer(
                    complex
                        .reduction_from_cancellations(&cancellations)
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::Rational(complex) => ChainReductionInner::Rational(
                    complex
                        .reduction_from_cancellations(&cancellations)
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::FiniteField(complex) => ChainReductionInner::FiniteField(
                    complex
                        .reduction_from_cancellations(&cancellations)
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => ChainReductionInner::NumberField(
                    complex
                        .reduction_from_cancellations(&cancellations)
                        .map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(complex) => ChainReductionInner::Padic(
                    complex
                        .reduction_from_cancellations(&cancellations)
                        .map_err(homology_error)?,
                ),
            })
        })?;
        Ok(PyChainReduction { inner })
    }

    /// Return the Betti number without constructing cycle or quotient bases.
    fn betti_number(&self, py: Python<'_>, degree: i32) -> PyResult<usize> {
        py.detach(|| match &self.inner {
            ChainComplexInner::Integer(_) => Err(PyValueError::new_err(
                "betti_number requires field coefficients; construct the complex over Q or another field",
            )),
            ChainComplexInner::Padic(_) => Err(PyValueError::new_err(
                "betti_number is not exact for fixed-cap p-adic coefficients; use homology_invariants",
            )),
            ChainComplexInner::Rational(complex) => {
                complex.betti_number(degree).map_err(homology_error)
            }
            ChainComplexInner::FiniteField(complex) => {
                complex.betti_number(degree).map_err(homology_error)
            }
            ChainComplexInner::NumberField(complex) => {
                complex.betti_number(degree).map_err(homology_error)
            }
        })
    }

    /// Return lean integral or precision-aware p-adic homology invariants.
    fn homology_invariants(&self, py: Python<'_>, degree: i32) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ChainComplexInner::Integer(complex) => {
                let inner =
                    py.detach(|| complex.homology_invariants(degree).map_err(homology_error))?;
                Ok(Py::new(py, PyFinitelyGeneratedAbelianGroup { inner })?.into_any())
            }
            ChainComplexInner::Padic(complex) => {
                let inner =
                    py.detach(|| complex.homology_invariants(degree).map_err(homology_error))?;
                Ok(Py::new(py, PyPadicModuleInvariants { inner })?.into_any())
            }
            _ => Err(PyValueError::new_err(
                "homology_invariants requires Z coefficients or PadicRing coefficients",
            )),
        }
    }

    /// Resolve p-adic homology under caller-supplied exact `Q_p`-rank assumptions.
    fn homology_invariants_with_ranks(
        &self,
        py: Python<'_>,
        degree: i32,
        outgoing_rank: usize,
        incoming_rank: usize,
    ) -> PyResult<PyPadicModuleInvariants> {
        let ChainComplexInner::Padic(complex) = &self.inner else {
            return Err(PyValueError::new_err(
                "homology_invariants_with_ranks requires PadicRing coefficients",
            ));
        };
        Ok(PyPadicModuleInvariants {
            inner: py.detach(|| {
                complex
                    .homology_invariants_with_ranks(degree, outgoing_rank, incoming_rank)
                    .map_err(homology_error)
            })?,
        })
    }

    /// Certify p-adic homology using an exact integral complex lift.
    fn homology_invariants_with_integer_lift(
        &self,
        py: Python<'_>,
        degree: i32,
        integer_lift: &PyChainComplex,
    ) -> PyResult<PyPadicModuleInvariants> {
        let (ChainComplexInner::Padic(complex), ChainComplexInner::Integer(integer_lift)) =
            (&self.inner, &integer_lift.inner)
        else {
            return Err(PyValueError::new_err(
                "homology_invariants_with_integer_lift requires PadicRing coefficients and an IntegerRing lift",
            ));
        };
        Ok(PyPadicModuleInvariants {
            inner: py.detach(|| {
                complex
                    .homology_invariants_with_integer_lift(degree, integer_lift)
                    .map_err(homology_error)
            })?,
        })
    }

    /// Return lean integral or p-adic homology invariants keyed by degree.
    fn graded_homology_invariants(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        match &self.inner {
            ChainComplexInner::Integer(complex) => {
                let invariants =
                    py.detach(|| complex.graded_homology_invariants().map_err(homology_error))?;
                for (degree, inner) in invariants {
                    output.set_item(
                        degree,
                        Py::new(py, PyFinitelyGeneratedAbelianGroup { inner })?,
                    )?;
                }
            }
            ChainComplexInner::Padic(complex) => {
                let invariants =
                    py.detach(|| complex.graded_homology_invariants().map_err(homology_error))?;
                for (degree, inner) in invariants {
                    output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "graded_homology_invariants requires Z coefficients or PadicRing coefficients",
                ));
            }
        }
        Ok(output.unbind())
    }

    /// Resolve graded p-adic homology with ranks keyed by boundary degree.
    fn graded_homology_invariants_with_ranks(
        &self,
        py: Python<'_>,
        differential_ranks: &Bound<'_, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let ChainComplexInner::Padic(complex) = &self.inner else {
            return Err(PyValueError::new_err(
                "graded_homology_invariants_with_ranks requires PadicRing coefficients",
            ));
        };
        let ranks = collect_ranks(differential_ranks)?;
        let invariants = py.detach(|| {
            complex
                .graded_homology_invariants_with_ranks(&ranks)
                .map_err(homology_error)
        })?;
        let output = PyDict::new(py);
        for (degree, inner) in invariants {
            output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
        }
        Ok(output.unbind())
    }

    /// Certify graded p-adic homology using an exact integral complex lift.
    fn graded_homology_invariants_with_integer_lift(
        &self,
        py: Python<'_>,
        integer_lift: &PyChainComplex,
    ) -> PyResult<Py<PyDict>> {
        let (ChainComplexInner::Padic(complex), ChainComplexInner::Integer(integer_lift)) =
            (&self.inner, &integer_lift.inner)
        else {
            return Err(PyValueError::new_err(
                "graded_homology_invariants_with_integer_lift requires PadicRing coefficients and an IntegerRing lift",
            ));
        };
        let invariants = py.detach(|| {
            complex
                .graded_homology_invariants_with_integer_lift(integer_lift)
                .map_err(homology_error)
        })?;
        let output = PyDict::new(py);
        for (degree, inner) in invariants {
            output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
        }
        Ok(output.unbind())
    }

    /// Return lean integral or precision-aware p-adic cohomology invariants.
    fn cohomology_invariants(&self, py: Python<'_>, degree: i32) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ChainComplexInner::Integer(complex) => {
                let inner = py.detach(|| {
                    complex
                        .cohomology_invariants(degree)
                        .map_err(homology_error)
                })?;
                Ok(Py::new(py, PyFinitelyGeneratedAbelianGroup { inner })?.into_any())
            }
            ChainComplexInner::Padic(complex) => {
                let inner = py.detach(|| {
                    complex
                        .cohomology_invariants(degree)
                        .map_err(homology_error)
                })?;
                Ok(Py::new(py, PyPadicModuleInvariants { inner })?.into_any())
            }
            _ => Err(PyValueError::new_err(
                "cohomology_invariants requires Z coefficients or PadicRing coefficients",
            )),
        }
    }

    /// Resolve p-adic cohomology under caller-supplied exact `Q_p`-rank assumptions.
    fn cohomology_invariants_with_ranks(
        &self,
        py: Python<'_>,
        degree: i32,
        outgoing_rank: usize,
        incoming_rank: usize,
    ) -> PyResult<PyPadicModuleInvariants> {
        let ChainComplexInner::Padic(complex) = &self.inner else {
            return Err(PyValueError::new_err(
                "cohomology_invariants_with_ranks requires PadicRing coefficients",
            ));
        };
        Ok(PyPadicModuleInvariants {
            inner: py.detach(|| {
                complex
                    .cohomology_invariants_with_ranks(degree, outgoing_rank, incoming_rank)
                    .map_err(homology_error)
            })?,
        })
    }

    /// Certify p-adic cohomology using an exact integral cochain lift.
    fn cohomology_invariants_with_integer_lift(
        &self,
        py: Python<'_>,
        degree: i32,
        integer_lift: &PyChainComplex,
    ) -> PyResult<PyPadicModuleInvariants> {
        let (ChainComplexInner::Padic(complex), ChainComplexInner::Integer(integer_lift)) =
            (&self.inner, &integer_lift.inner)
        else {
            return Err(PyValueError::new_err(
                "cohomology_invariants_with_integer_lift requires PadicRing coefficients and an IntegerRing lift",
            ));
        };
        Ok(PyPadicModuleInvariants {
            inner: py.detach(|| {
                complex
                    .cohomology_invariants_with_integer_lift(degree, integer_lift)
                    .map_err(homology_error)
            })?,
        })
    }

    /// Return lean integral or p-adic cohomology invariants keyed by degree.
    fn graded_cohomology_invariants(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        match &self.inner {
            ChainComplexInner::Integer(complex) => {
                let invariants = py.detach(|| {
                    complex
                        .graded_cohomology_invariants()
                        .map_err(homology_error)
                })?;
                for (degree, inner) in invariants {
                    output.set_item(
                        degree,
                        Py::new(py, PyFinitelyGeneratedAbelianGroup { inner })?,
                    )?;
                }
            }
            ChainComplexInner::Padic(complex) => {
                let invariants = py.detach(|| {
                    complex
                        .graded_cohomology_invariants()
                        .map_err(homology_error)
                })?;
                for (degree, inner) in invariants {
                    output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "graded_cohomology_invariants requires Z coefficients or PadicRing coefficients",
                ));
            }
        }
        Ok(output.unbind())
    }

    /// Resolve graded p-adic cohomology with ranks keyed by coboundary degree.
    fn graded_cohomology_invariants_with_ranks(
        &self,
        py: Python<'_>,
        differential_ranks: &Bound<'_, PyMapping>,
    ) -> PyResult<Py<PyDict>> {
        let ChainComplexInner::Padic(complex) = &self.inner else {
            return Err(PyValueError::new_err(
                "graded_cohomology_invariants_with_ranks requires PadicRing coefficients",
            ));
        };
        let ranks = collect_ranks(differential_ranks)?;
        let invariants = py.detach(|| {
            complex
                .graded_cohomology_invariants_with_ranks(&ranks)
                .map_err(homology_error)
        })?;
        let output = PyDict::new(py);
        for (degree, inner) in invariants {
            output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
        }
        Ok(output.unbind())
    }

    /// Certify graded p-adic cohomology using an exact integral cochain lift.
    fn graded_cohomology_invariants_with_integer_lift(
        &self,
        py: Python<'_>,
        integer_lift: &PyChainComplex,
    ) -> PyResult<Py<PyDict>> {
        let (ChainComplexInner::Padic(complex), ChainComplexInner::Integer(integer_lift)) =
            (&self.inner, &integer_lift.inner)
        else {
            return Err(PyValueError::new_err(
                "graded_cohomology_invariants_with_integer_lift requires PadicRing coefficients and an IntegerRing lift",
            ));
        };
        let invariants = py.detach(|| {
            complex
                .graded_cohomology_invariants_with_integer_lift(integer_lift)
                .map_err(homology_error)
        })?;
        let output = PyDict::new(py);
        for (degree, inner) in invariants {
            output.set_item(degree, Py::new(py, PyPadicModuleInvariants { inner })?)?;
        }
        Ok(output.unbind())
    }

    /// Compute one checked homology group.
    fn homology(&self, py: Python<'_>, degree: i32) -> PyResult<PyHomologyGroup> {
        let inner = py.detach(|| -> PyResult<HomologyGroupInner> {
            let inner = match &self.inner {
                ChainComplexInner::Integer(complex) => {
                    HomologyGroupInner::Integer(complex.homology(degree).map_err(homology_error)?)
                }
                ChainComplexInner::Rational(complex) => {
                    HomologyGroupInner::Rational(complex.homology(degree).map_err(homology_error)?)
                }
                ChainComplexInner::FiniteField(complex) => HomologyGroupInner::FiniteField(
                    complex.homology(degree).map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => HomologyGroupInner::NumberField(
                    complex.homology(degree).map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(_) => {
                    return Err(PyValueError::new_err(
                        "fixed-cap p-adic complexes expose homology_invariants, not an exact HomologyGroup",
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PyHomologyGroup { inner })
    }

    /// Compute one checked cohomology group.
    fn cohomology(&self, py: Python<'_>, degree: i32) -> PyResult<PyHomologyGroup> {
        let inner = py.detach(|| -> PyResult<HomologyGroupInner> {
            let inner = match &self.inner {
                ChainComplexInner::Integer(complex) => {
                    HomologyGroupInner::Integer(complex.cohomology(degree).map_err(homology_error)?)
                }
                ChainComplexInner::Rational(complex) => HomologyGroupInner::Rational(
                    complex.cohomology(degree).map_err(homology_error)?,
                ),
                ChainComplexInner::FiniteField(complex) => HomologyGroupInner::FiniteField(
                    complex.cohomology(degree).map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => HomologyGroupInner::NumberField(
                    complex.cohomology(degree).map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(_) => {
                    return Err(PyValueError::new_err(
                        "fixed-cap p-adic complexes expose cohomology_invariants, not an exact HomologyGroup",
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PyHomologyGroup { inner })
    }

    /// Compute every explicitly represented homological degree.
    fn graded_homology(&self, py: Python<'_>) -> PyResult<PyGradedHomology> {
        let inner = py.detach(|| -> PyResult<GradedHomologyInner> {
            let inner = match &self.inner {
                ChainComplexInner::Integer(complex) => {
                    GradedHomologyInner::Integer(complex.graded_homology().map_err(homology_error)?)
                }
                ChainComplexInner::Rational(complex) => GradedHomologyInner::Rational(
                    complex.graded_homology().map_err(homology_error)?,
                ),
                ChainComplexInner::FiniteField(complex) => GradedHomologyInner::FiniteField(
                    complex.graded_homology().map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => GradedHomologyInner::NumberField(
                    complex.graded_homology().map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(_) => {
                    return Err(PyValueError::new_err(
                        "fixed-cap p-adic complexes expose graded_homology_invariants, not exact GradedHomology",
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PyGradedHomology { inner })
    }

    /// Compute every explicitly represented cohomological degree.
    fn graded_cohomology(&self, py: Python<'_>) -> PyResult<PyGradedHomology> {
        let inner = py.detach(|| -> PyResult<GradedHomologyInner> {
            let inner = match &self.inner {
                ChainComplexInner::Integer(complex) => GradedHomologyInner::Integer(
                    complex.graded_cohomology().map_err(homology_error)?,
                ),
                ChainComplexInner::Rational(complex) => GradedHomologyInner::Rational(
                    complex.graded_cohomology().map_err(homology_error)?,
                ),
                ChainComplexInner::FiniteField(complex) => GradedHomologyInner::FiniteField(
                    complex.graded_cohomology().map_err(homology_error)?,
                ),
                ChainComplexInner::NumberField(complex) => GradedHomologyInner::NumberField(
                    complex.graded_cohomology().map_err(homology_error)?,
                ),
                ChainComplexInner::Padic(_) => {
                    return Err(PyValueError::new_err(
                        "fixed-cap p-adic complexes expose graded_cohomology_invariants, not exact GradedHomology",
                    ));
                }
            };
            Ok(inner)
        })?;
        Ok(PyGradedHomology { inner })
    }

    /// Recheck coefficient contexts, matrix shapes, and `d^2 = 0`.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        let result = py.detach(|| match &self.inner {
            ChainComplexInner::Integer(complex) => complex.verify(),
            ChainComplexInner::Rational(complex) => complex.verify(),
            ChainComplexInner::FiniteField(complex) => complex.verify(),
            ChainComplexInner::NumberField(complex) => complex.verify(),
            ChainComplexInner::Padic(complex) => complex.verify(),
        });
        result.map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            let degrees = self.degrees();
            format!("ChainComplex(degrees={degrees:?})")
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let degrees = self.degrees();
        summary.set_item("type", ChainComplex::<RationalField>::TYPE_ID)?;
        summary.set_item(
            "degrees",
            degrees
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        if degrees.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("degrees_truncated", true)?;
        }
        Ok(summary.unbind())
    }
}

impl PyChainComplex {
    fn construct(
        ring: &Bound<'_, PyAny>,
        matrices: &Bound<'_, PyMapping>,
        coboundary: bool,
    ) -> PyResult<Self> {
        if ring.is_instance_of::<PyIntegerRing>() {
            let matrices = collect_matrices(
                matrices,
                "Z",
                |inner| match inner {
                    MatrixInner::Integer(matrix) => Some(matrix.clone()),
                    _ => None,
                },
                |inner| match inner {
                    SparseMatrixInner::Integer(matrix) => Some(matrix.clone()),
                    _ => None,
                },
            )?;
            let complex = construct_complex(noether::Z, matrices, coboundary);
            return Ok(Self {
                inner: ChainComplexInner::Integer(complex.map_err(homology_error)?),
            });
        }
        if ring.is_instance_of::<PyRationalField>() {
            let matrices = collect_matrices(
                matrices,
                "Q",
                |inner| match inner {
                    MatrixInner::Rational(matrix) => Some(matrix.clone()),
                    _ => None,
                },
                |inner| match inner {
                    SparseMatrixInner::Rational(matrix) => Some(matrix.clone()),
                    _ => None,
                },
            )?;
            let complex = construct_complex(noether::Q, matrices, coboundary);
            return Ok(Self {
                inner: ChainComplexInner::Rational(complex.map_err(homology_error)?),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            let matrices = collect_matrices(
                matrices,
                &field.inner.to_string(),
                |inner| match inner {
                    MatrixInner::FiniteField(matrix) => Some(matrix.clone()),
                    _ => None,
                },
                |inner| match inner {
                    SparseMatrixInner::FiniteField(matrix) => Some(matrix.clone()),
                    _ => None,
                },
            )?;
            let complex = construct_complex(field.inner.clone(), matrices, coboundary);
            return Ok(Self {
                inner: ChainComplexInner::FiniteField(complex.map_err(homology_error)?),
            });
        }
        if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
            let matrices = collect_matrices(
                matrices,
                &field.inner.to_string(),
                |inner| match inner {
                    MatrixInner::NumberField(matrix) => Some(matrix.clone()),
                    _ => None,
                },
                |_| None,
            )?;
            let complex = construct_complex(field.inner.clone(), matrices, coboundary);
            return Ok(Self {
                inner: ChainComplexInner::NumberField(complex.map_err(homology_error)?),
            });
        }
        if let Ok(context) = ring.extract::<PyRef<'_, PyPadicRing>>() {
            let matrices = collect_matrices(
                matrices,
                &context.inner.to_string(),
                |inner| match inner {
                    MatrixInner::Padic(matrix) => Some(matrix.clone()),
                    _ => None,
                },
                |inner| match inner {
                    SparseMatrixInner::Padic(matrix) => Some(matrix.clone()),
                    _ => None,
                },
            )?;
            let complex = construct_complex(context.inner.clone(), matrices, coboundary);
            return Ok(Self {
                inner: ChainComplexInner::Padic(complex.map_err(homology_error)?),
            });
        }
        Err(PyTypeError::new_err(
            "ChainComplex currently supports Z, Q, FiniteField, NumberField, and PadicRing coefficients",
        ))
    }
}

/// A finitely generated abelian group in canonical invariant-factor form.
#[pyclass(
    module = "noether.homalg",
    name = "FinitelyGeneratedAbelianGroup",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFinitelyGeneratedAbelianGroup {
    pub(crate) inner: RustFinitelyGeneratedAbelianGroup,
}

impl Serialize for PyFinitelyGeneratedAbelianGroup {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyFinitelyGeneratedAbelianGroup {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RustFinitelyGeneratedAbelianGroup::deserialize(deserializer).map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyFinitelyGeneratedAbelianGroup {
    #[new]
    #[pyo3(signature = (free_rank, torsion_coefficients = Vec::new()))]
    fn new(free_rank: usize, torsion_coefficients: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let torsion_coefficients = torsion_coefficients
            .iter()
            .map(extract_integer)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: RustFinitelyGeneratedAbelianGroup::new(free_rank, torsion_coefficients)
                .map_err(abelian_group_error)?,
        })
    }

    /// Construct from arbitrary positive cyclic summand orders.
    #[staticmethod]
    fn from_cyclic_summands(
        free_rank: usize,
        cyclic_orders: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let cyclic_orders = cyclic_orders
            .iter()
            .map(extract_integer)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: RustFinitelyGeneratedAbelianGroup::from_cyclic_summands(
                free_rank,
                cyclic_orders,
            )
            .map_err(abelian_group_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustFinitelyGeneratedAbelianGroup::TYPE_ID
    }

    /// Rank of the free abelian summand.
    #[getter]
    fn free_rank(&self) -> usize {
        self.inner.free_rank()
    }

    /// Positive nonunit torsion factors in divisibility order.
    #[getter]
    fn torsion_coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .torsion_coefficients()
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Whether this is the zero group.
    fn is_trivial(&self) -> bool {
        self.inner.is_trivial()
    }

    /// Canonical direct sum.
    fn direct_sum(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .direct_sum(&other.inner)
                .map_err(abelian_group_error)?,
        })
    }

    /// Tensor product over the integers.
    fn tensor_product(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .tensor_product(&other.inner)
                .map_err(abelian_group_error)?,
        })
    }

    /// First Tor group over the integers.
    fn tor_1(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.tor_1(&other.inner),
        }
    }

    /// Hom over the integers into Z.
    fn hom_to_integers(&self) -> Self {
        Self {
            inner: self.inner.hom_to_integers(),
        }
    }

    /// First Ext over the integers into Z.
    fn ext_1_to_integers(&self) -> Self {
        Self {
            inner: self.inner.ext_1_to_integers(),
        }
    }

    /// Check the canonical invariant-factor representation.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "FinitelyGeneratedAbelianGroup(free_rank={}, torsion_summands={})",
            self.inner.free_rank(),
            self.inner.torsion_coefficients().len()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let factors = self.inner.torsion_coefficients();
        let torsion_coefficients = factors
            .iter()
            .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
            .map(|factor| integer_to_python(py, factor))
            .collect::<PyResult<Vec<_>>>()?;
        summary.set_item("type", RustFinitelyGeneratedAbelianGroup::TYPE_ID)?;
        summary.set_item("free_rank", self.inner.free_rank())?;
        summary.set_item("torsion_coefficients", torsion_coefficients)?;
        if factors.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("torsion_coefficients_truncated", true)?;
        }
        Ok(summary.unbind())
    }
}

#[derive(Clone, PartialEq, Eq)]
// The enum is stored inside a heap-allocated PyO3 class; boxing individual
// mathematical variants would add another allocation to every access.
#[allow(clippy::large_enum_variant)]
pub(crate) enum HomologyGroupInner {
    Integer(RustHomologyGroup<IntegerRing>),
    Rational(RustHomologyGroup<RationalField>),
    FiniteField(RustHomologyGroup<FiniteField>),
    NumberField(RustHomologyGroup<NumberField>),
}

/// A checked single-degree homology or cohomology group.
#[pyclass(
    module = "noether.homalg",
    name = "HomologyGroup",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyHomologyGroup {
    pub(crate) inner: HomologyGroupInner,
}

#[pymethods]
impl PyHomologyGroup {
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustHomologyGroup::<RationalField>::TYPE_ID
    }

    /// Displayed degree.
    #[getter]
    fn degree(&self) -> i32 {
        match &self.inner {
            HomologyGroupInner::Integer(group) => group.degree(),
            HomologyGroupInner::Rational(group) => group.degree(),
            HomologyGroupInner::FiniteField(group) => group.degree(),
            HomologyGroupInner::NumberField(group) => group.degree(),
        }
    }

    /// Whether this result was computed from a coboundary presentation.
    #[getter]
    fn is_cohomology(&self) -> bool {
        match &self.inner {
            HomologyGroupInner::Integer(group) => group.is_cohomology(),
            HomologyGroupInner::Rational(group) => group.is_cohomology(),
            HomologyGroupInner::FiniteField(group) => group.is_cohomology(),
            HomologyGroupInner::NumberField(group) => group.is_cohomology(),
        }
    }

    /// Coefficient ring.
    #[getter]
    fn coefficient_context(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            HomologyGroupInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            HomologyGroupInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            HomologyGroupInner::FiniteField(group) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: group.coefficient_context().clone(),
                },
            )?
            .into_any()),
            HomologyGroupInner::NumberField(group) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: group.coefficient_context().clone(),
                },
            )?
            .into_any()),
        }
    }

    /// Dimension over the coefficient field.
    ///
    /// Integral homology is not a vector space; use `free_rank` and
    /// `torsion_coefficients` there.
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        self.field_dimension().ok_or_else(|| {
            PyAttributeError::new_err(
                "integral homology has no vector-space dimension; use free_rank and torsion_coefficients",
            )
        })
    }

    /// Rank of the free part; equal to `dimension` over a field.
    #[getter]
    fn free_rank(&self) -> usize {
        match &self.inner {
            HomologyGroupInner::Integer(group) => group.free_rank(),
            HomologyGroupInner::Rational(group) => group.free_rank(),
            HomologyGroupInner::FiniteField(group) => group.free_rank(),
            HomologyGroupInner::NumberField(group) => group.free_rank(),
        }
    }

    /// Positive invariant factors of the torsion subgroup.
    #[getter]
    fn torsion_coefficients(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let factors = match &self.inner {
            HomologyGroupInner::Integer(group) => group.torsion_coefficients(),
            HomologyGroupInner::Rational(group) => group.torsion_coefficients(),
            HomologyGroupInner::FiniteField(group) => group.torsion_coefficients(),
            HomologyGroupInner::NumberField(group) => group.torsion_coefficients(),
        };
        factors
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Complete checked cycle space.
    #[getter]
    fn cycles(&self) -> PyResult<PySolutionSpace> {
        let inner = match &self.inner {
            HomologyGroupInner::Integer(_) => {
                return Err(PyAttributeError::new_err(
                    "integral cycle lattices require the explicit transform-producing calculation",
                ));
            }
            HomologyGroupInner::Rational(group) => {
                SolutionSpaceInner::Rational(group.cycles().clone())
            }
            HomologyGroupInner::FiniteField(group) => {
                SolutionSpaceInner::FiniteField(group.cycles().clone())
            }
            HomologyGroupInner::NumberField(group) => {
                SolutionSpaceInner::NumberField(group.cycles().clone())
            }
        };
        Ok(PySolutionSpace { inner })
    }

    /// Complete checked boundary space.
    #[getter]
    fn boundaries(&self) -> PyResult<PySolutionSpace> {
        let inner = match &self.inner {
            HomologyGroupInner::Integer(_) => {
                return Err(PyAttributeError::new_err(
                    "integral boundary lattices require the explicit transform-producing calculation",
                ));
            }
            HomologyGroupInner::Rational(group) => {
                SolutionSpaceInner::Rational(group.boundaries().clone())
            }
            HomologyGroupInner::FiniteField(group) => {
                SolutionSpaceInner::FiniteField(group.boundaries().clone())
            }
            HomologyGroupInner::NumberField(group) => {
                SolutionSpaceInner::NumberField(group.boundaries().clone())
            }
        };
        Ok(PySolutionSpace { inner })
    }

    /// Deterministic cycle representatives forming a basis modulo boundaries.
    #[getter]
    fn representatives(&self) -> PyResult<Vec<PyVector>> {
        match &self.inner {
            HomologyGroupInner::Integer(_) => Err(PyAttributeError::new_err(
                "integral representatives require the explicit transform-producing calculation",
            )),
            HomologyGroupInner::Rational(group) => Ok(group
                .representatives()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::Rational(vector),
                })
                .collect()),
            HomologyGroupInner::FiniteField(group) => Ok(group
                .representatives()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::FiniteField(vector),
                })
                .collect()),
            HomologyGroupInner::NumberField(group) => Ok(group
                .representatives()
                .iter()
                .cloned()
                .map(|vector| PyVector {
                    inner: VectorInner::NumberField(vector),
                })
                .collect()),
        }
    }

    /// Mathematical assumptions under which the result is valid.
    #[getter]
    fn validity(&self) -> PyMathematicalValidity {
        let inner = match &self.inner {
            HomologyGroupInner::Integer(group) => group.validity().clone(),
            HomologyGroupInner::Rational(group) => group.validity().clone(),
            HomologyGroupInner::FiniteField(group) => group.validity().clone(),
            HomologyGroupInner::NumberField(group) => group.validity().clone(),
        };
        PyMathematicalValidity { inner }
    }

    /// Recheck adjacent maps, cycles, boundaries, and quotient representatives.
    fn verify(&self) -> PyResult<()> {
        let result = match &self.inner {
            HomologyGroupInner::Integer(group) => group.verify(),
            HomologyGroupInner::Rational(group) => group.verify(),
            HomologyGroupInner::FiniteField(group) => group.verify(),
            HomologyGroupInner::NumberField(group) => group.verify(),
        };
        result.map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "HomologyGroup(degree={}, free_rank={}, torsion_summands={})",
                self.degree(),
                self.free_rank(),
                match &self.inner {
                    HomologyGroupInner::Integer(group) => group.torsion_coefficients().len(),
                    HomologyGroupInner::Rational(group) => group.torsion_coefficients().len(),
                    HomologyGroupInner::FiniteField(group) => group.torsion_coefficients().len(),
                    HomologyGroupInner::NumberField(group) => group.torsion_coefficients().len(),
                }
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let factors = match &self.inner {
            HomologyGroupInner::Integer(group) => group.torsion_coefficients(),
            HomologyGroupInner::Rational(group) => group.torsion_coefficients(),
            HomologyGroupInner::FiniteField(group) => group.torsion_coefficients(),
            HomologyGroupInner::NumberField(group) => group.torsion_coefficients(),
        };
        let torsion_coefficients = factors
            .iter()
            .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
            .map(|factor| integer_to_python(py, factor))
            .collect::<PyResult<Vec<_>>>()?;
        summary.set_item("type", RustHomologyGroup::<RationalField>::TYPE_ID)?;
        summary.set_item("degree", self.degree())?;
        summary.set_item(
            "grading",
            if self.is_cohomology() {
                "cohomological"
            } else {
                "homological"
            },
        )?;
        summary.set_item("free_rank", self.free_rank())?;
        summary.set_item("torsion_coefficients", torsion_coefficients)?;
        if factors.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("torsion_coefficients_truncated", true)?;
        }
        if let Some(dimension) = self.field_dimension() {
            summary.set_item("dimension", dimension)?;
            summary.set_item("cycles_dimension", self.cycles()?.inner_dimension())?;
            summary.set_item("boundaries_dimension", self.boundaries()?.inner_dimension())?;
        }
        Ok(summary.unbind())
    }
}

impl PyHomologyGroup {
    fn field_dimension(&self) -> Option<usize> {
        match &self.inner {
            HomologyGroupInner::Integer(_) => None,
            HomologyGroupInner::Rational(group) => Some(group.dimension()),
            HomologyGroupInner::FiniteField(group) => Some(group.dimension()),
            HomologyGroupInner::NumberField(group) => Some(group.dimension()),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum GradedHomologyInner {
    Integer(RustGradedHomology<IntegerRing>),
    Rational(RustGradedHomology<RationalField>),
    FiniteField(RustGradedHomology<FiniteField>),
    NumberField(RustGradedHomology<NumberField>),
}

/// Checked homology groups in deterministic degree order.
#[pyclass(
    module = "noether.homalg",
    name = "GradedHomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyGradedHomology {
    pub(crate) inner: GradedHomologyInner,
}

#[pymethods]
impl PyGradedHomology {
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        RustGradedHomology::<RationalField>::TYPE_ID
    }

    /// Coefficient ring shared by every group.
    #[getter]
    fn coefficient_context(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            GradedHomologyInner::Integer(_) => Ok(Py::new(py, PyIntegerRing)?.into_any()),
            GradedHomologyInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            GradedHomologyInner::FiniteField(graded) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: graded.coefficient_context().clone(),
                },
            )?
            .into_any()),
            GradedHomologyInner::NumberField(graded) => Ok(Py::new(
                py,
                PyNumberField {
                    inner: graded.coefficient_context().clone(),
                },
            )?
            .into_any()),
        }
    }

    /// Groups in increasing homological degree.
    #[getter]
    fn groups(&self) -> Vec<PyHomologyGroup> {
        match &self.inner {
            GradedHomologyInner::Integer(graded) => graded
                .groups()
                .iter()
                .cloned()
                .map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::Integer(group),
                })
                .collect(),
            GradedHomologyInner::Rational(graded) => graded
                .groups()
                .iter()
                .cloned()
                .map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::Rational(group),
                })
                .collect(),
            GradedHomologyInner::FiniteField(graded) => graded
                .groups()
                .iter()
                .cloned()
                .map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::FiniteField(group),
                })
                .collect(),
            GradedHomologyInner::NumberField(graded) => graded
                .groups()
                .iter()
                .cloned()
                .map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::NumberField(group),
                })
                .collect(),
        }
    }

    /// Group in one represented degree, or `None`.
    fn get(&self, degree: i32) -> Option<PyHomologyGroup> {
        match &self.inner {
            GradedHomologyInner::Integer(graded) => {
                graded.get(degree).cloned().map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::Integer(group),
                })
            }
            GradedHomologyInner::Rational(graded) => {
                graded.get(degree).cloned().map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::Rational(group),
                })
            }
            GradedHomologyInner::FiniteField(graded) => {
                graded.get(degree).cloned().map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::FiniteField(group),
                })
            }
            GradedHomologyInner::NumberField(graded) => {
                graded.get(degree).cloned().map(|group| PyHomologyGroup {
                    inner: HomologyGroupInner::NumberField(group),
                })
            }
        }
    }

    /// Recheck ordering, coefficient context, and all group witnesses.
    fn verify(&self) -> PyResult<()> {
        let result = match &self.inner {
            GradedHomologyInner::Integer(graded) => graded.verify(),
            GradedHomologyInner::Rational(graded) => graded.verify(),
            GradedHomologyInner::FiniteField(graded) => graded.verify(),
            GradedHomologyInner::NumberField(graded) => graded.verify(),
        };
        result.map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("GradedHomology(groups={})", self.groups().len()))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let groups = self.groups();
        let displayed = groups
            .iter()
            .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
            .collect::<Vec<_>>();
        let summary = PyDict::new(py);
        summary.set_item("type", RustGradedHomology::<RationalField>::TYPE_ID)?;
        summary.set_item(
            "degrees",
            displayed
                .iter()
                .map(|group| group.degree())
                .collect::<Vec<_>>(),
        )?;
        summary.set_item(
            "free_ranks",
            displayed
                .iter()
                .map(|group| group.free_rank())
                .collect::<Vec<_>>(),
        )?;
        if displayed
            .iter()
            .all(|group| group.field_dimension().is_some())
        {
            summary.set_item(
                "dimensions",
                displayed
                    .iter()
                    .filter_map(|group| group.field_dimension())
                    .collect::<Vec<_>>(),
            )?;
        }
        if groups.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("groups_truncated", true)?;
        }
        Ok(summary.unbind())
    }
}

impl PySolutionSpace {
    fn inner_dimension(&self) -> usize {
        match &self.inner {
            SolutionSpaceInner::Rational(space) => space.dimension(),
            SolutionSpaceInner::FiniteField(space) => space.dimension(),
            SolutionSpaceInner::NumberField(space) => space.dimension(),
            SolutionSpaceInner::RationalFunction(space) => space.dimension(),
        }
    }
}
