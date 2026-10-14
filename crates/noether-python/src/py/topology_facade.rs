// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of the public `noether::topology` facade.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::numthy::{FiniteField, NumberField};
use noether::topology::{
    AdmissibleCohomologyOperation, Barcode, DominatedVertex, EilenbergMacLaneCohomologyGenerator,
    EilenbergMacLaneCohomologyOperation, EilenbergMacLaneError, EilenbergMacLaneGeneratorOrigin,
    EilenbergMacLaneHomologyStrategy, EilenbergMacLaneIntegralCohomology,
    EilenbergMacLaneIntegralCohomologyGenerator, EilenbergMacLaneIntegralCohomologyPresentation,
    EilenbergMacLaneIntegralCohomologyPresentationStrategy,
    EilenbergMacLaneIntegralCohomologyRelation, EilenbergMacLaneIntegralCohomologyRelationTerm,
    EilenbergMacLaneIntegralGeneratorOrigin, EilenbergMacLaneIntegralHomology,
    EilenbergMacLaneModularCohomology, EilenbergMacLaneModularCohomologyStrategy,
    EilenbergMacLaneModularCohomologyTruncation,
    EilenbergMacLaneModularCohomologyTruncationStrategy, EilenbergMacLaneSpace, ElementaryCollapse,
    FilteredComplex, Simplex, SimplicialCohomologyBasisElement, SimplicialCohomologyProduct,
    SimplicialCohomologyProductTerm, SimplicialCohomologyRing, SimplicialCollapseSequence,
    SimplicialComplex, SimplicialFace, SimplicialMorseReduction, SimplicialSet,
    SimplicialSetAnodyneMove, SimplicialSetAnodyneZigzag, SimplicialSetCohomologyRing,
    SimplicialSetElementaryHornCollapse, SimplicialSetHornCollapseSequence,
    SimplicialSimpleHomotopyMove, SimplicialSimpleHomotopyZigzag, StrongCoreReduction,
    TopologyError,
};
use noether::{Field, IntegerRing, RationalField, Ring, SemanticType, Q, Z};
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::commalg::{PyGradedCommutativePolynomial, PyGradedCommutativePresentation};
use super::geometry_facade::{PyPointCloud, PyTriangulation};
use super::homalg::{
    ChainComplexInner, ChainReductionInner, GradedHomologyInner, HomologyGroupInner,
    PyChainComplex, PyChainReduction, PyFinitelyGeneratedAbelianGroup, PyGradedHomology,
    PyHomologyGroup,
};
use super::linalg::{PyVector, VectorInner};
use super::number_field::PyNumberField;
use super::numthy::{PyFiniteField, PyFiniteFieldElement};
use super::root::{PyIntegerRing, PyRationalField};
use crate::py::groups::PyPermGroup;

pub(crate) fn eilenberg_maclane_error(error: EilenbergMacLaneError) -> PyErr {
    match error {
        EilenbergMacLaneError::ZeroDeloopingDegree => PyValueError::new_err(error.to_string()),
        EilenbergMacLaneError::DegreeBoundTooLarge { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        EilenbergMacLaneError::CharacteristicTooLarge { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        EilenbergMacLaneError::AbelianGroup(error) => super::homalg::abelian_group_error(error),
        EilenbergMacLaneError::GradedCommutative(error) => PyValueError::new_err(error.to_string()),
        EilenbergMacLaneError::UnsupportedOperation(error) => {
            super::errors::unsupported_operation(error)
        }
        EilenbergMacLaneError::Verification(error) => super::errors::verification(error),
    }
}

/// A checked admissible word in the mod-p Steenrod algebra.
#[pyclass(
    module = "noether.topology",
    name = "AdmissibleCohomologyOperation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAdmissibleCohomologyOperation {
    pub(crate) inner: AdmissibleCohomologyOperation,
}

#[pymethods]
impl PyAdmissibleCohomologyOperation {
    #[staticmethod]
    fn mod_two(sequence: Vec<u32>) -> PyResult<Self> {
        Ok(Self {
            inner: AdmissibleCohomologyOperation::mod_two(sequence).map_err(topology_error)?,
        })
    }

    #[staticmethod]
    fn odd_primary(prime: u64, bocksteins: Vec<bool>, reduced_powers: Vec<u32>) -> PyResult<Self> {
        Ok(Self {
            inner: AdmissibleCohomologyOperation::odd_primary(prime, bocksteins, reduced_powers)
                .map_err(topology_error)?,
        })
    }

    #[staticmethod]
    fn identity(prime: u64) -> PyResult<Self> {
        Ok(Self {
            inner: AdmissibleCohomologyOperation::identity(prime).map_err(topology_error)?,
        })
    }

    #[staticmethod]
    fn steenrod_square(square: u32) -> Self {
        Self {
            inner: AdmissibleCohomologyOperation::steenrod_square(square),
        }
    }

    #[staticmethod]
    fn reduced_power(prime: u64, power: u32) -> PyResult<Self> {
        Ok(Self {
            inner: AdmissibleCohomologyOperation::reduced_power(prime, power)
                .map_err(topology_error)?,
        })
    }

    #[staticmethod]
    fn bockstein_reduced_power(prime: u64, power: u32) -> PyResult<Self> {
        Ok(Self {
            inner: AdmissibleCohomologyOperation::bockstein_reduced_power(prime, power)
                .map_err(topology_error)?,
        })
    }

    #[getter]
    fn prime(&self) -> u64 {
        self.inner.prime()
    }

    #[getter]
    fn square_sequence(&self) -> Option<Vec<u32>> {
        self.inner.square_sequence().map(<[u32]>::to_vec)
    }

    #[getter]
    fn bocksteins(&self) -> Option<Vec<bool>> {
        self.inner.bocksteins().map(<[bool]>::to_vec)
    }

    #[getter]
    fn reduced_powers(&self) -> Option<Vec<u32>> {
        self.inner.reduced_powers().map(<[u32]>::to_vec)
    }

    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    #[getter]
    fn degree_shift(&self) -> u32 {
        self.inner.degree_shift()
    }

    #[getter]
    fn excess(&self) -> u32 {
        self.inner.excess()
    }

    #[getter]
    fn display_name(&self) -> String {
        self.inner.display_name()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: serde_json::from_str(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __str__(&self) -> String {
        self.inner.display_name()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "AdmissibleCohomologyOperation(prime={}, word={:?})",
            self.inner.prime(),
            self.inner.display_name()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

fn eilenberg_maclane_modular_strategy(
    strategy: EilenbergMacLaneModularCohomologyStrategy,
) -> &'static str {
    match strategy {
        EilenbergMacLaneModularCohomologyStrategy::TrivialSpace => "trivial_space",
        EilenbergMacLaneModularCohomologyStrategy::ClassifyingSpaceDecomposition => {
            "classifying_space_decomposition"
        }
        EilenbergMacLaneModularCohomologyStrategy::FreeAbelianSecondDelooping => {
            "free_abelian_second_delooping"
        }
    }
}

fn eilenberg_maclane_modular_truncation_strategy(
    strategy: EilenbergMacLaneModularCohomologyTruncationStrategy,
) -> &'static str {
    match strategy {
        EilenbergMacLaneModularCohomologyTruncationStrategy::TrivialSpace => "trivial_space",
        EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleSquares => {
            "cartan_serre_admissible_squares"
        }
        EilenbergMacLaneModularCohomologyTruncationStrategy::CartanSerreAdmissibleOddPrimaryOperations => {
            "cartan_serre_admissible_odd_primary_operations"
        }
    }
}

fn eilenberg_maclane_strategy(strategy: EilenbergMacLaneHomologyStrategy) -> &'static str {
    match strategy {
        EilenbergMacLaneHomologyStrategy::TrivialSpace => "trivial_space",
        EilenbergMacLaneHomologyStrategy::ClassifyingSpaceKunneth => "classifying_space_kunneth",
        EilenbergMacLaneHomologyStrategy::FreeAbelianSecondDelooping => {
            "free_abelian_second_delooping"
        }
    }
}

fn eilenberg_maclane_integral_presentation_strategy(
    strategy: EilenbergMacLaneIntegralCohomologyPresentationStrategy,
) -> &'static str {
    match strategy {
        EilenbergMacLaneIntegralCohomologyPresentationStrategy::TrivialSpace => "trivial_space",
        EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndCyclicClassifyingSpace => {
            "free_and_cyclic_classifying_space"
        }
        EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAndRankTwoAbelianClassifyingSpace => {
            "free_and_rank_two_abelian_classifying_space"
        }
        EilenbergMacLaneIntegralCohomologyPresentationStrategy::FreeAbelianSecondDelooping => {
            "free_abelian_second_delooping"
        }
    }
}

/// A symbolic Eilenberg--Mac Lane space K(A,n).
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneSpace",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneSpace {
    pub(crate) inner: EilenbergMacLaneSpace,
}

impl Serialize for PyEilenbergMacLaneSpace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneSpace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneSpace::deserialize(deserializer).map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneSpace {
    #[new]
    fn new(group: &PyFinitelyGeneratedAbelianGroup, delooping_degree: usize) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneSpace::new(group.inner.clone(), delooping_degree)
                .map_err(eilenberg_maclane_error)?,
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneSpace::TYPE_ID
    }

    /// The sole nonzero homotopy group.
    #[getter]
    fn group(&self) -> PyFinitelyGeneratedAbelianGroup {
        PyFinitelyGeneratedAbelianGroup {
            inner: self.inner.group().clone(),
        }
    }

    /// Degree containing the sole nonzero homotopy group.
    #[getter]
    fn delooping_degree(&self) -> usize {
        self.inner.delooping_degree()
    }

    /// Exact integral homology through an inclusive degree bound.
    fn integral_homology(&self, max_degree: usize) -> PyResult<PyEilenbergMacLaneIntegralHomology> {
        Ok(PyEilenbergMacLaneIntegralHomology {
            inner: self
                .inner
                .integral_homology(max_degree)
                .map_err(eilenberg_maclane_error)?,
        })
    }

    /// Exact additive integral cohomology through an inclusive degree bound.
    fn integral_cohomology(
        &self,
        max_degree: usize,
    ) -> PyResult<PyEilenbergMacLaneIntegralCohomology> {
        Ok(PyEilenbergMacLaneIntegralCohomology {
            inner: self
                .inner
                .integral_cohomology(max_degree)
                .map_err(eilenberg_maclane_error)?,
        })
    }

    /// Complete closed-form integral cohomology-ring presentation.
    fn integral_cohomology_presentation(
        &self,
    ) -> PyResult<PyEilenbergMacLaneIntegralCohomologyPresentation> {
        Ok(PyEilenbergMacLaneIntegralCohomologyPresentation {
            inner: self
                .inner
                .integral_cohomology_presentation()
                .map_err(eilenberg_maclane_error)?,
        })
    }

    /// Complete closed-form cohomology algebra over a prime field.
    fn modular_cohomology(
        &self,
        field: &PyFiniteField,
    ) -> PyResult<PyEilenbergMacLaneModularCohomology> {
        Ok(PyEilenbergMacLaneModularCohomology {
            inner: self
                .inner
                .modular_cohomology(field.inner.clone())
                .map_err(eilenberg_maclane_error)?,
        })
    }

    /// Cohomology algebra exact through an inclusive degree bound.
    fn modular_cohomology_through_degree(
        &self,
        field: &PyFiniteField,
        max_degree: usize,
    ) -> PyResult<PyEilenbergMacLaneModularCohomologyTruncation> {
        Ok(PyEilenbergMacLaneModularCohomologyTruncation {
            inner: self
                .inner
                .modular_cohomology_through_degree(field.inner.clone(), max_degree)
                .map_err(eilenberg_maclane_error)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(eilenberg_maclane_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneSpace::from_json(value).map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneSpace(group={}, delooping_degree={})",
            self.inner.group(),
            self.inner.delooping_degree()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EilenbergMacLaneSpace::TYPE_ID)?;
        summary.set_item("group", self.inner.group().to_string())?;
        summary.set_item("delooping_degree", self.inner.delooping_degree())?;
        Ok(summary.unbind())
    }
}

/// Exact bounded integral homology of a symbolic K(A,n).
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralHomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralHomology {
    pub(crate) inner: EilenbergMacLaneIntegralHomology,
}

impl Serialize for PyEilenbergMacLaneIntegralHomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneIntegralHomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneIntegralHomology::deserialize(deserializer).map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneIntegralHomology {
    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneIntegralHomology::TYPE_ID
    }

    #[getter]
    fn space(&self) -> PyEilenbergMacLaneSpace {
        PyEilenbergMacLaneSpace {
            inner: self.inner.space().clone(),
        }
    }

    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    #[getter]
    fn groups(&self) -> Vec<PyFinitelyGeneratedAbelianGroup> {
        self.inner
            .groups()
            .iter()
            .cloned()
            .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
            .collect()
    }

    fn group(&self, degree: usize) -> Option<PyFinitelyGeneratedAbelianGroup> {
        self.inner
            .group(degree)
            .cloned()
            .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
    }

    #[getter]
    fn strategy(&self) -> &'static str {
        eilenberg_maclane_strategy(self.inner.strategy())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneIntegralHomology::from_json(value)
                .map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralHomology(space={}, max_degree={}, strategy={})",
            self.inner.space(),
            self.inner.max_degree(),
            eilenberg_maclane_strategy(self.inner.strategy())
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EilenbergMacLaneIntegralHomology::TYPE_ID)?;
        summary.set_item("space", self.inner.space().to_string())?;
        summary.set_item("max_degree", self.inner.max_degree())?;
        summary.set_item(
            "strategy",
            eilenberg_maclane_strategy(self.inner.strategy()),
        )?;
        Ok(summary.unbind())
    }
}

/// Exact bounded additive integral cohomology of a symbolic K(A,n).
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralCohomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralCohomology {
    pub(crate) inner: EilenbergMacLaneIntegralCohomology,
}

impl Serialize for PyEilenbergMacLaneIntegralCohomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneIntegralCohomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneIntegralCohomology::deserialize(deserializer).map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneIntegralCohomology {
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneIntegralCohomology::TYPE_ID
    }

    #[getter]
    fn space(&self) -> PyEilenbergMacLaneSpace {
        PyEilenbergMacLaneSpace {
            inner: self.inner.space().clone(),
        }
    }

    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    #[getter]
    fn homology(&self) -> PyEilenbergMacLaneIntegralHomology {
        PyEilenbergMacLaneIntegralHomology {
            inner: self.inner.homology().clone(),
        }
    }

    #[getter]
    fn groups(&self) -> Vec<PyFinitelyGeneratedAbelianGroup> {
        self.inner
            .groups()
            .iter()
            .cloned()
            .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
            .collect()
    }

    fn group(&self, degree: usize) -> Option<PyFinitelyGeneratedAbelianGroup> {
        self.inner
            .group(degree)
            .cloned()
            .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneIntegralCohomology::from_json(value)
                .map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralCohomology(space={}, max_degree={})",
            self.inner.space(),
            self.inner.max_degree()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EilenbergMacLaneIntegralCohomology::TYPE_ID)?;
        summary.set_item("space", self.inner.space().to_string())?;
        summary.set_item("max_degree", self.inner.max_degree())?;
        summary.set_item("method", "integral_universal_coefficient")?;
        Ok(summary.unbind())
    }
}

/// One generator in a complete integral cohomology presentation.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralCohomologyGenerator",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralCohomologyGenerator {
    inner: EilenbergMacLaneIntegralCohomologyGenerator,
}

#[pymethods]
impl PyEilenbergMacLaneIntegralCohomologyGenerator {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    #[getter]
    fn degree(&self) -> u32 {
        self.inner.degree()
    }

    #[getter]
    fn display_name(&self) -> &str {
        self.inner.display_name()
    }

    #[getter]
    fn origin(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let origin = PyDict::new(py);
        match self.inner.origin() {
            EilenbergMacLaneIntegralGeneratorOrigin::FreeFundamentalClass { component } => {
                origin.set_item("kind", "free_fundamental_class")?;
                origin.set_item("component", component)?;
            }
            EilenbergMacLaneIntegralGeneratorOrigin::IntegralBockstein { component, modulus } => {
                origin.set_item("kind", "integral_bockstein")?;
                origin.set_item("component", component)?;
                origin.set_item(
                    "modulus",
                    super::facade_scalar::integer_to_python(py, modulus)?,
                )?;
            }
            EilenbergMacLaneIntegralGeneratorOrigin::CyclicTorClass {
                left_component,
                right_component,
                order,
            } => {
                origin.set_item("kind", "cyclic_tor_class")?;
                origin.set_item("left_component", left_component)?;
                origin.set_item("right_component", right_component)?;
                origin.set_item("order", super::facade_scalar::integer_to_python(py, order)?)?;
            }
            EilenbergMacLaneIntegralGeneratorOrigin::SecondDeloopingFundamentalClass {
                component,
            } => {
                origin.set_item("kind", "second_delooping_fundamental_class")?;
                origin.set_item("component", component)?;
            }
        }
        Ok(origin.unbind())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralCohomologyGenerator(id={:?}, degree={}, display_name={:?})",
            self.inner.id(),
            self.inner.degree(),
            self.inner.display_name()
        ))
    }
}

/// One sparse term in an integral cohomology relation.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralCohomologyRelationTerm",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralCohomologyRelationTerm {
    inner: EilenbergMacLaneIntegralCohomologyRelationTerm,
}

#[pymethods]
impl PyEilenbergMacLaneIntegralCohomologyRelationTerm {
    #[getter]
    fn coefficient(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        super::facade_scalar::integer_to_python(py, self.inner.coefficient())
    }

    #[getter]
    fn exponents(&self) -> Vec<u32> {
        self.inner.exponents().to_vec()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralCohomologyRelationTerm(coefficient={}, exponents={:?})",
            self.inner.coefficient(),
            self.inner.exponents()
        ))
    }
}

/// One homogeneous relation in an integral cohomology presentation.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralCohomologyRelation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralCohomologyRelation {
    inner: EilenbergMacLaneIntegralCohomologyRelation,
}

#[pymethods]
impl PyEilenbergMacLaneIntegralCohomologyRelation {
    #[getter]
    fn terms(&self) -> Vec<PyEilenbergMacLaneIntegralCohomologyRelationTerm> {
        self.inner
            .terms()
            .iter()
            .cloned()
            .map(|inner| PyEilenbergMacLaneIntegralCohomologyRelationTerm { inner })
            .collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralCohomologyRelation(terms={:?})",
            self.inner.terms()
        ))
    }
}

/// Complete integral cohomology-ring presentation of a symbolic K(A,n).
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneIntegralCohomologyPresentation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneIntegralCohomologyPresentation {
    pub(crate) inner: EilenbergMacLaneIntegralCohomologyPresentation,
}

impl Serialize for PyEilenbergMacLaneIntegralCohomologyPresentation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneIntegralCohomologyPresentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneIntegralCohomologyPresentation::deserialize(deserializer)
            .map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneIntegralCohomologyPresentation {
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneIntegralCohomologyPresentation::TYPE_ID
    }

    #[getter]
    fn space(&self) -> PyEilenbergMacLaneSpace {
        PyEilenbergMacLaneSpace {
            inner: self.inner.space().clone(),
        }
    }

    #[getter]
    fn coefficient_ring(&self) -> &'static str {
        self.inner.coefficient_ring()
    }

    #[getter]
    fn graded_commutative(&self) -> bool {
        true
    }

    #[getter]
    fn generators(&self) -> Vec<PyEilenbergMacLaneIntegralCohomologyGenerator> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(|inner| PyEilenbergMacLaneIntegralCohomologyGenerator { inner })
            .collect()
    }

    #[getter]
    fn relations(&self) -> Vec<PyEilenbergMacLaneIntegralCohomologyRelation> {
        self.inner
            .relations()
            .iter()
            .cloned()
            .map(|inner| PyEilenbergMacLaneIntegralCohomologyRelation { inner })
            .collect()
    }

    #[getter]
    fn strategy(&self) -> &'static str {
        eilenberg_maclane_integral_presentation_strategy(self.inner.strategy())
    }

    fn additive_group(&self, degree: usize) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        Ok(PyFinitelyGeneratedAbelianGroup {
            inner: self
                .inner
                .additive_group(degree)
                .map_err(eilenberg_maclane_error)?,
        })
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneIntegralCohomologyPresentation::from_json(value)
                .map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneIntegralCohomologyPresentation(space={}, generators={}, relations={}, strategy={})",
            self.inner.space(),
            self.inner.generators().len(),
            self.inner.relations().len(),
            eilenberg_maclane_integral_presentation_strategy(self.inner.strategy())
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item(
            "type",
            EilenbergMacLaneIntegralCohomologyPresentation::TYPE_ID,
        )?;
        summary.set_item("space", self.inner.space().to_string())?;
        summary.set_item("coefficient_ring", self.inner.coefficient_ring())?;
        summary.set_item("generator_count", self.inner.generators().len())?;
        summary.set_item("relation_count", self.inner.relations().len())?;
        summary.set_item(
            "strategy",
            eilenberg_maclane_integral_presentation_strategy(self.inner.strategy()),
        )?;
        Ok(summary.unbind())
    }
}

/// One named generator in a complete modular cohomology presentation.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneCohomologyGenerator",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneCohomologyGenerator {
    pub(crate) inner: EilenbergMacLaneCohomologyGenerator,
}

#[pymethods]
impl PyEilenbergMacLaneCohomologyGenerator {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    #[getter]
    fn degree(&self) -> u32 {
        self.inner.degree()
    }

    #[getter]
    fn display_name(&self) -> &str {
        self.inner.display_name()
    }

    #[getter]
    fn origin(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let origin = PyDict::new(py);
        match self.inner.origin() {
            EilenbergMacLaneGeneratorOrigin::FundamentalClass { component } => {
                origin.set_item("kind", "fundamental_class")?;
                origin.set_item("component", component)?;
            }
            EilenbergMacLaneGeneratorOrigin::Bockstein { source_generator } => {
                origin.set_item("kind", "bockstein")?;
                origin.set_item("source_generator", source_generator)?;
            }
            EilenbergMacLaneGeneratorOrigin::PeriodicCyclicClass {
                component,
                primary_exponent,
            } => {
                origin.set_item("kind", "periodic_cyclic_class")?;
                origin.set_item("component", component)?;
                origin.set_item("primary_exponent", primary_exponent)?;
            }
            EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquare {
                component,
                operation,
                fundamental_class_lifts_integrally,
            } => {
                origin.set_item("kind", "admissible_steenrod_square")?;
                origin.set_item("component", component)?;
                origin.set_item(
                    "operation",
                    PyAdmissibleCohomologyOperation {
                        inner: operation.clone(),
                    },
                )?;
                origin.set_item(
                    "fundamental_class_lifts_integrally",
                    fundamental_class_lifts_integrally,
                )?;
            }
            EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperation {
                component,
                operation,
                fundamental_class_lifts_integrally,
            } => {
                origin.set_item("kind", "admissible_odd_primary_operation")?;
                origin.set_item("component", component)?;
                origin.set_item(
                    "operation",
                    PyAdmissibleCohomologyOperation {
                        inner: operation.clone(),
                    },
                )?;
                origin.set_item(
                    "fundamental_class_lifts_integrally",
                    fundamental_class_lifts_integrally,
                )?;
            }
            EilenbergMacLaneGeneratorOrigin::HigherBockstein {
                component,
                primary_exponent,
            } => {
                origin.set_item("kind", "higher_bockstein")?;
                origin.set_item("component", component)?;
                origin.set_item("primary_exponent", primary_exponent)?;
            }
            EilenbergMacLaneGeneratorOrigin::AdmissibleSteenrodSquareOnHigherPrimaryClass {
                component,
                primary_exponent,
                source_is_higher_bockstein,
                operation,
            } => {
                origin.set_item(
                    "kind",
                    "admissible_steenrod_square_on_higher_primary_class",
                )?;
                origin.set_item("component", component)?;
                origin.set_item("primary_exponent", primary_exponent)?;
                origin.set_item(
                    "source_is_higher_bockstein",
                    source_is_higher_bockstein,
                )?;
                origin.set_item(
                    "operation",
                    PyAdmissibleCohomologyOperation {
                        inner: operation.clone(),
                    },
                )?;
            }
            EilenbergMacLaneGeneratorOrigin::AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
                component,
                primary_exponent,
                source_is_higher_bockstein,
                operation,
            } => {
                origin.set_item(
                    "kind",
                    "admissible_odd_primary_operation_on_higher_primary_class",
                )?;
                origin.set_item("component", component)?;
                origin.set_item("primary_exponent", primary_exponent)?;
                origin.set_item(
                    "source_is_higher_bockstein",
                    source_is_higher_bockstein,
                )?;
                origin.set_item(
                    "operation",
                    PyAdmissibleCohomologyOperation {
                        inner: operation.clone(),
                    },
                )?;
            }
        }
        Ok(origin.unbind())
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneCohomologyGenerator(id={:?}, degree={}, display_name={:?})",
            self.inner.id(),
            self.inner.degree(),
            self.inner.display_name()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One named primary operation on a modular cohomology generator.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneCohomologyOperation",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneCohomologyOperation {
    pub(crate) inner: EilenbergMacLaneCohomologyOperation,
}

#[pymethods]
impl PyEilenbergMacLaneCohomologyOperation {
    #[getter]
    fn operation(&self) -> PyAdmissibleCohomologyOperation {
        PyAdmissibleCohomologyOperation {
            inner: self.inner.operation().clone(),
        }
    }

    #[getter]
    fn source_generator(&self) -> usize {
        self.inner.source_generator()
    }

    #[getter]
    fn target(&self) -> PyGradedCommutativePolynomial {
        PyGradedCommutativePolynomial {
            inner: self.inner.target().clone(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneCohomologyOperation(operation={:?}, source_generator={}, target={})",
            self.inner.operation().display_name(),
            self.inner.source_generator(),
            self.inner.target()
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A complete prime-field cohomology presentation of a symbolic K(A,n).
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneModularCohomology",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneModularCohomology {
    pub(crate) inner: EilenbergMacLaneModularCohomology,
}

impl Serialize for PyEilenbergMacLaneModularCohomology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneModularCohomology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneModularCohomology::deserialize(deserializer).map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneModularCohomology {
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneModularCohomology::TYPE_ID
    }

    #[getter]
    fn space(&self) -> PyEilenbergMacLaneSpace {
        PyEilenbergMacLaneSpace {
            inner: self.inner.space().clone(),
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
    fn strategy(&self) -> &'static str {
        eilenberg_maclane_modular_strategy(self.inner.strategy())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneModularCohomology::from_json(value)
                .map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneModularCohomology(space={}, field={}, generators={}, strategy={})",
            self.inner.space(),
            self.inner.field(),
            self.inner.generators().len(),
            eilenberg_maclane_modular_strategy(self.inner.strategy())
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EilenbergMacLaneModularCohomology::TYPE_ID)?;
        summary.set_item("space", self.inner.space().to_string())?;
        summary.set_item("field", self.inner.field().to_string())?;
        summary.set_item("generators", self.inner.generators().len())?;
        summary.set_item(
            "strategy",
            eilenberg_maclane_modular_strategy(self.inner.strategy()),
        )?;
        Ok(summary.unbind())
    }
}

/// A prime-field cohomology presentation exact through one degree.
#[pyclass(
    module = "noether.topology",
    name = "EilenbergMacLaneModularCohomologyTruncation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEilenbergMacLaneModularCohomologyTruncation {
    pub(crate) inner: EilenbergMacLaneModularCohomologyTruncation,
}

impl Serialize for PyEilenbergMacLaneModularCohomologyTruncation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyEilenbergMacLaneModularCohomologyTruncation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EilenbergMacLaneModularCohomologyTruncation::deserialize(deserializer)
            .map(|inner| Self { inner })
    }
}

#[pymethods]
impl PyEilenbergMacLaneModularCohomologyTruncation {
    #[classattr]
    fn type_id() -> &'static str {
        EilenbergMacLaneModularCohomologyTruncation::TYPE_ID
    }

    #[getter]
    fn space(&self) -> PyEilenbergMacLaneSpace {
        PyEilenbergMacLaneSpace {
            inner: self.inner.space().clone(),
        }
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn exact_through_degree(&self) -> usize {
        self.inner.exact_through_degree()
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
    fn strategy(&self) -> &'static str {
        eilenberg_maclane_modular_truncation_strategy(self.inner.strategy())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(super::errors::verification)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(super::errors::noether_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EilenbergMacLaneModularCohomologyTruncation::from_json(value)
                .map_err(super::errors::noether_error)?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "EilenbergMacLaneModularCohomologyTruncation(space={}, field={}, exact_through_degree={}, generators={}, strategy={})",
            self.inner.space(),
            self.inner.field(),
            self.inner.exact_through_degree(),
            self.inner.generators().len(),
            eilenberg_maclane_modular_truncation_strategy(self.inner.strategy())
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", EilenbergMacLaneModularCohomologyTruncation::TYPE_ID)?;
        summary.set_item("space", self.inner.space().to_string())?;
        summary.set_item("field", self.inner.field().to_string())?;
        summary.set_item("exact_through_degree", self.inner.exact_through_degree())?;
        summary.set_item("generators", self.inner.generators().len())?;
        summary.set_item(
            "strategy",
            eilenberg_maclane_modular_truncation_strategy(self.inner.strategy()),
        )?;
        Ok(summary.unbind())
    }
}

/// One canonical cohomology class and its simplicial cocycle.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialCohomologyBasisElement",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialCohomologyBasisElement {
    inner: SimplicialCohomologyBasisElement,
}

#[pymethods]
impl PySimplicialCohomologyBasisElement {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    #[getter]
    fn degree(&self) -> u32 {
        self.inner.degree()
    }

    #[getter]
    fn cocycle(&self) -> PyVector {
        PyVector {
            inner: VectorInner::FiniteField(self.inner.cocycle().clone()),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialCohomologyBasisElement(id={:?}, degree={})",
            self.inner.id(),
            self.inner.degree()
        ))
    }
}

/// One nonzero coefficient in a cohomology basis product.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialCohomologyProductTerm",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialCohomologyProductTerm {
    inner: SimplicialCohomologyProductTerm,
}

#[pymethods]
impl PySimplicialCohomologyProductTerm {
    #[getter]
    fn basis(&self) -> usize {
        self.inner.basis()
    }

    #[getter]
    fn coefficient(&self) -> PyFiniteFieldElement {
        PyFiniteFieldElement {
            inner: self.inner.coefficient().clone(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialCohomologyProductTerm(basis={}, coefficient={})",
            self.inner.basis(),
            self.inner.coefficient()
        ))
    }
}

/// One ordered basis product in a finite simplicial cohomology ring.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialCohomologyProduct",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialCohomologyProduct {
    inner: SimplicialCohomologyProduct,
}

#[pymethods]
impl PySimplicialCohomologyProduct {
    #[getter]
    fn left(&self) -> usize {
        self.inner.left()
    }

    #[getter]
    fn right(&self) -> usize {
        self.inner.right()
    }

    #[getter]
    fn terms(&self) -> Vec<PySimplicialCohomologyProductTerm> {
        self.inner
            .terms()
            .iter()
            .cloned()
            .map(|inner| PySimplicialCohomologyProductTerm { inner })
            .collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialCohomologyProduct(left={}, right={}, terms={})",
            self.inner.left(),
            self.inner.right(),
            self.inner.terms().len()
        ))
    }
}

/// Complete finite-dimensional cohomology algebra over a finite field.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialCohomologyRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialCohomologyRing {
    inner: SimplicialCohomologyRing,
}

#[pymethods]
impl PySimplicialCohomologyRing {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialCohomologyRing::TYPE_ID
    }

    #[getter]
    fn complex(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.complex().clone())
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn basis(&self) -> Vec<PySimplicialCohomologyBasisElement> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(|inner| PySimplicialCohomologyBasisElement { inner })
            .collect()
    }

    #[getter]
    fn products(&self) -> Vec<PySimplicialCohomologyProduct> {
        self.inner
            .products()
            .iter()
            .cloned()
            .map(|inner| PySimplicialCohomologyProduct { inner })
            .collect()
    }

    #[getter]
    fn presentation(&self) -> Option<PyGradedCommutativePresentation> {
        self.inner
            .presentation()
            .cloned()
            .map(|inner| PyGradedCommutativePresentation { inner })
    }

    fn product(
        &self,
        left: usize,
        right: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .product(left, right)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn steenrod_square(
        &self,
        square: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .steenrod_square(square, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn steenrod_reduced_power(
        &self,
        power: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .steenrod_reduced_power(power, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn bockstein_reduced_power(
        &self,
        power: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .bockstein_reduced_power(power, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn apply_operation(
        &self,
        operation: &PyAdmissibleCohomologyOperation,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .apply_operation(&operation.inner, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialCohomologyRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// Complete normalized cohomology algebra of a finite simplicial set.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSetCohomologyRing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSetCohomologyRing {
    inner: SimplicialSetCohomologyRing,
}

#[pymethods]
impl PySimplicialSetCohomologyRing {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialSetCohomologyRing::TYPE_ID
    }

    #[getter]
    fn simplicial_set(&self) -> PySimplicialSet {
        PySimplicialSet {
            inner: self.inner.simplicial_set().clone(),
        }
    }

    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    #[getter]
    fn basis(&self) -> Vec<PySimplicialCohomologyBasisElement> {
        self.inner
            .basis()
            .iter()
            .cloned()
            .map(|inner| PySimplicialCohomologyBasisElement { inner })
            .collect()
    }

    #[getter]
    fn products(&self) -> Vec<PySimplicialCohomologyProduct> {
        self.inner
            .products()
            .iter()
            .cloned()
            .map(|inner| PySimplicialCohomologyProduct { inner })
            .collect()
    }

    #[getter]
    fn presentation(&self) -> Option<PyGradedCommutativePresentation> {
        self.inner
            .presentation()
            .cloned()
            .map(|inner| PyGradedCommutativePresentation { inner })
    }

    fn product(
        &self,
        left: usize,
        right: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .product(left, right)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn steenrod_square(
        &self,
        square: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .steenrod_square(square, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn steenrod_reduced_power(
        &self,
        power: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .steenrod_reduced_power(power, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn bockstein_reduced_power(
        &self,
        power: usize,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .bockstein_reduced_power(power, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn apply_operation(
        &self,
        operation: &PyAdmissibleCohomologyOperation,
        source: usize,
    ) -> PyResult<Vec<PySimplicialCohomologyProductTerm>> {
        self.inner
            .apply_operation(&operation.inner, source)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|inner| PySimplicialCohomologyProductTerm { inner })
                    .collect()
            })
            .map_err(topology_error)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSetCohomologyRing::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One elementary simplicial-set collapse inverse to a horn attachment.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSetElementaryHornCollapse",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSetElementaryHornCollapse {
    inner: SimplicialSetElementaryHornCollapse,
}

#[pymethods]
impl PySimplicialSetElementaryHornCollapse {
    #[new]
    fn new(simplex_dimension: usize, simplex: usize, face: usize, free_face: usize) -> Self {
        Self {
            inner: SimplicialSetElementaryHornCollapse::new(
                simplex_dimension,
                simplex,
                face,
                free_face,
            ),
        }
    }

    #[getter]
    fn simplex_dimension(&self) -> usize {
        self.inner.simplex_dimension()
    }

    #[getter]
    fn simplex(&self) -> usize {
        self.inner.simplex()
    }

    #[getter]
    fn face(&self) -> usize {
        self.inner.face()
    }

    #[getter]
    fn free_face(&self) -> usize {
        self.inner.free_face()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialSetElementaryHornCollapse(simplex_dimension={}, simplex={}, face={}, free_face={})",
            self.inner.simplex_dimension(),
            self.inner.simplex(),
            self.inner.face(),
            self.inner.free_face()
        ))
    }
}

/// Replayable finite sequence of elementary simplicial-set horn collapses.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSetHornCollapseSequence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSetHornCollapseSequence {
    inner: SimplicialSetHornCollapseSequence,
}

#[pymethods]
impl PySimplicialSetHornCollapseSequence {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialSetHornCollapseSequence::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PySimplicialSet {
        PySimplicialSet {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn steps(&self) -> Vec<PySimplicialSetElementaryHornCollapse> {
        self.inner
            .steps()
            .iter()
            .cloned()
            .map(|inner| PySimplicialSetElementaryHornCollapse { inner })
            .collect()
    }

    #[getter]
    fn target(&self) -> PySimplicialSet {
        PySimplicialSet {
            inner: self.inner.target().clone(),
        }
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSetHornCollapseSequence::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One horn attachment or collapse in an anodyne zigzag.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSetAnodyneMove",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSetAnodyneMove {
    inner: SimplicialSetAnodyneMove,
}

#[pymethods]
impl PySimplicialSetAnodyneMove {
    #[staticmethod]
    fn collapse(step: &PySimplicialSetElementaryHornCollapse) -> Self {
        Self {
            inner: SimplicialSetAnodyneMove::Collapse {
                step: step.inner.clone(),
            },
        }
    }

    #[staticmethod]
    fn expansion(
        expanded: &PySimplicialSet,
        inverse_collapse: &PySimplicialSetElementaryHornCollapse,
    ) -> Self {
        Self {
            inner: SimplicialSetAnodyneMove::Expansion {
                expanded: expanded.inner.clone(),
                inverse_collapse: inverse_collapse.inner.clone(),
            },
        }
    }

    #[getter]
    fn direction(&self) -> &'static str {
        match &self.inner {
            SimplicialSetAnodyneMove::Collapse { .. } => "collapse",
            SimplicialSetAnodyneMove::Expansion { .. } => "expansion",
        }
    }

    #[getter]
    fn step(&self) -> PySimplicialSetElementaryHornCollapse {
        let inner = match &self.inner {
            SimplicialSetAnodyneMove::Collapse { step } => step.clone(),
            SimplicialSetAnodyneMove::Expansion {
                inverse_collapse, ..
            } => inverse_collapse.clone(),
        };
        PySimplicialSetElementaryHornCollapse { inner }
    }

    #[getter]
    fn expanded(&self) -> Option<PySimplicialSet> {
        match &self.inner {
            SimplicialSetAnodyneMove::Expansion { expanded, .. } => Some(PySimplicialSet {
                inner: expanded.clone(),
            }),
            SimplicialSetAnodyneMove::Collapse { .. } => None,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialSetAnodyneMove(direction={:?}, step={:?})",
            self.direction(),
            self.step().inner
        ))
    }
}

/// A verified zigzag of horn attachments and collapses.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSetAnodyneZigzag",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSetAnodyneZigzag {
    inner: SimplicialSetAnodyneZigzag,
}

#[pymethods]
impl PySimplicialSetAnodyneZigzag {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialSetAnodyneZigzag::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PySimplicialSet {
        PySimplicialSet {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    fn moves(&self) -> Vec<PySimplicialSetAnodyneMove> {
        self.inner
            .moves()
            .iter()
            .cloned()
            .map(|inner| PySimplicialSetAnodyneMove { inner })
            .collect()
    }

    #[getter]
    fn target(&self) -> PySimplicialSet {
        PySimplicialSet {
            inner: self.inner.target().clone(),
        }
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSetAnodyneZigzag::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One replayable elementary free-face collapse.
#[pyclass(
    module = "noether.topology",
    name = "ElementaryCollapse",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyElementaryCollapse {
    inner: ElementaryCollapse,
}

#[pymethods]
impl PyElementaryCollapse {
    #[new]
    fn new(free_face: Vec<u32>, coface: Vec<u32>) -> Self {
        Self {
            inner: ElementaryCollapse::new(free_face, coface),
        }
    }

    #[getter]
    fn free_face(&self) -> Vec<u32> {
        self.inner.free_face().to_vec()
    }

    #[getter]
    fn coface(&self) -> Vec<u32> {
        self.inner.coface().to_vec()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "ElementaryCollapse(free_face={:?}, coface={:?})",
            self.inner.free_face(),
            self.inner.coface()
        ))
    }
}

/// A verified sequence of elementary simplicial collapses.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialCollapseSequence",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialCollapseSequence {
    inner: SimplicialCollapseSequence,
}

#[pymethods]
impl PySimplicialCollapseSequence {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialCollapseSequence::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.source().clone())
    }

    #[getter]
    fn steps(&self) -> Vec<PyElementaryCollapse> {
        self.inner
            .steps()
            .iter()
            .cloned()
            .map(|inner| PyElementaryCollapse { inner })
            .collect()
    }

    #[getter]
    fn target(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.target().clone())
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialCollapseSequence::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One expansion or collapse in a simplicial simple-homotopy zigzag.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSimpleHomotopyMove",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSimpleHomotopyMove {
    inner: SimplicialSimpleHomotopyMove,
}

#[pymethods]
impl PySimplicialSimpleHomotopyMove {
    #[staticmethod]
    fn collapse(step: &PyElementaryCollapse) -> Self {
        Self {
            inner: SimplicialSimpleHomotopyMove::Collapse {
                step: step.inner.clone(),
            },
        }
    }

    #[staticmethod]
    fn expansion(expanded: &PySimplicialComplex, inverse_collapse: &PyElementaryCollapse) -> Self {
        Self {
            inner: SimplicialSimpleHomotopyMove::Expansion {
                expanded: expanded.inner.clone(),
                inverse_collapse: inverse_collapse.inner.clone(),
            },
        }
    }

    #[getter]
    fn direction(&self) -> &'static str {
        match &self.inner {
            SimplicialSimpleHomotopyMove::Collapse { .. } => "collapse",
            SimplicialSimpleHomotopyMove::Expansion { .. } => "expansion",
        }
    }

    #[getter]
    fn step(&self) -> PyElementaryCollapse {
        let inner = match &self.inner {
            SimplicialSimpleHomotopyMove::Collapse { step } => step.clone(),
            SimplicialSimpleHomotopyMove::Expansion {
                inverse_collapse, ..
            } => inverse_collapse.clone(),
        };
        PyElementaryCollapse { inner }
    }

    #[getter]
    fn expanded(&self) -> Option<PySimplicialComplex> {
        match &self.inner {
            SimplicialSimpleHomotopyMove::Expansion { expanded, .. } => {
                Some(PySimplicialComplex::from_core(expanded.clone()))
            }
            SimplicialSimpleHomotopyMove::Collapse { .. } => None,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialSimpleHomotopyMove(direction={:?}, step={:?})",
            self.direction(),
            self.step().inner
        ))
    }
}

/// A verified zigzag of elementary simplicial expansions and collapses.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSimpleHomotopyZigzag",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSimpleHomotopyZigzag {
    inner: SimplicialSimpleHomotopyZigzag,
}

#[pymethods]
impl PySimplicialSimpleHomotopyZigzag {
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialSimpleHomotopyZigzag::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.source().clone())
    }

    #[getter]
    fn moves(&self) -> Vec<PySimplicialSimpleHomotopyMove> {
        self.inner
            .moves()
            .iter()
            .cloned()
            .map(|inner| PySimplicialSimpleHomotopyMove { inner })
            .collect()
    }

    #[getter]
    fn target(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.target().clone())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSimpleHomotopyZigzag::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

/// One replayable dominated-vertex deletion.
#[pyclass(
    module = "noether.topology",
    name = "DominatedVertex",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyDominatedVertex {
    inner: DominatedVertex,
}

#[pymethods]
impl PyDominatedVertex {
    #[new]
    fn new(vertex: u32, dominator: u32) -> Self {
        Self {
            inner: DominatedVertex::new(vertex, dominator),
        }
    }

    #[getter]
    fn vertex(&self) -> u32 {
        self.inner.vertex()
    }

    #[getter]
    fn dominator(&self) -> u32 {
        self.inner.dominator()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DominatedVertex(vertex={}, dominator={})",
            self.inner.vertex(),
            self.inner.dominator()
        ))
    }
}

/// A verified reduction of a finite complex to its strong-homotopy core.
#[pyclass(
    module = "noether.topology",
    name = "StrongCoreReduction",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStrongCoreReduction {
    inner: StrongCoreReduction,
}

#[pymethods]
impl PyStrongCoreReduction {
    #[classattr]
    fn type_id() -> &'static str {
        StrongCoreReduction::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.source().clone())
    }

    #[getter]
    fn steps(&self) -> Vec<PyDominatedVertex> {
        self.inner
            .steps()
            .iter()
            .copied()
            .map(|inner| PyDominatedVertex { inner })
            .collect()
    }

    #[getter]
    fn reduced(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.reduced().clone())
    }

    #[getter]
    fn core(&self) -> PySimplicialComplex {
        PySimplicialComplex::from_core(self.inner.core().clone())
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(topology_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: StrongCoreReduction::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

fn topology_error(error: TopologyError) -> PyErr {
    match error {
        TopologyError::DimensionTooLarge { .. }
        | TopologyError::SymmetrySearchTooLarge { .. }
        | TopologyError::SymmetryDegreeTooLarge { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        TopologyError::ContextMismatch(error) => super::errors::context_mismatch(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn parse_faces(
    py: Python<'_>,
    faces: Vec<Vec<Vec<Py<PyAny>>>>,
) -> PyResult<Vec<Vec<Vec<SimplicialFace>>>> {
    faces
        .into_iter()
        .enumerate()
        .map(|(dimension, level)| {
            level
                .into_iter()
                .enumerate()
                .map(|(simplex, faces)| {
                    faces
                        .into_iter()
                        .enumerate()
                        .map(|(face, value)| parse_face(py, value, dimension, simplex, face))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn parse_face(
    py: Python<'_>,
    value: Py<PyAny>,
    dimension: usize,
    simplex: usize,
    face: usize,
) -> PyResult<SimplicialFace> {
    let value = value.bind(py);
    if let Ok(index) = value.extract::<usize>() {
        return Ok(SimplicialFace::Cell(index));
    }
    let mapping = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "face {face} of {dimension}-simplex {simplex} must be an integer cell index or a degeneracy mapping"
        ))
    })?;
    let item = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        mapping.get_item(key)?.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "degenerate face {face} of {dimension}-simplex {simplex} is missing '{key}'"
            ))
        })
    };
    let simplex_dimension = item("simplex_dimension")?.extract::<usize>()?;
    let simplex = item("simplex")?.extract::<usize>()?;
    let surjection = item("surjection")?.extract::<Vec<usize>>()?;
    Ok(SimplicialFace::degenerate(
        simplex_dimension,
        simplex,
        surjection,
    ))
}

fn face_to_object(py: Python<'_>, face: &SimplicialFace) -> PyResult<Py<PyAny>> {
    match face {
        SimplicialFace::Cell(index) => Ok(index.into_pyobject(py)?.into_any().unbind()),
        SimplicialFace::Degenerate {
            simplex_dimension,
            simplex,
            surjection,
        } => {
            let mapping = PyDict::new(py);
            mapping.set_item("simplex_dimension", simplex_dimension)?;
            mapping.set_item("simplex", simplex)?;
            mapping.set_item("surjection", surjection)?;
            Ok(mapping.into_any().unbind())
        }
    }
}

fn faces_to_python(py: Python<'_>, faces: &[SimplicialFace]) -> PyResult<Vec<Py<PyAny>>> {
    faces.iter().map(|face| face_to_object(py, face)).collect()
}

fn nonnegative_degree_count(dimension: i64) -> usize {
    usize::try_from(dimension.saturating_add(1)).unwrap_or(0)
}

trait SimplicialChains {
    fn chain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError>;

    fn cochain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError>;

    fn homology<R: Field>(
        &self,
        field: R,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<R>, TopologyError>;

    fn betti_number<R: Field>(&self, field: R, degree: usize) -> Result<usize, TopologyError>;

    fn graded_homology<R: Field>(
        &self,
        field: R,
    ) -> Result<noether::homalg::GradedHomology<R>, TopologyError>;

    fn integral_homology(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<IntegerRing>, TopologyError>;

    fn homology_invariants(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::FinitelyGeneratedAbelianGroup, TopologyError>;

    fn graded_integral_homology(
        &self,
    ) -> Result<noether::homalg::GradedHomology<IntegerRing>, TopologyError>;
}

impl SimplicialChains for SimplicialComplex {
    fn chain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError> {
        SimplicialComplex::chain_complex(self, ring)
    }

    fn cochain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError> {
        SimplicialComplex::cochain_complex(self, ring)
    }

    fn homology<R: Field>(
        &self,
        field: R,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<R>, TopologyError> {
        SimplicialComplex::homology(self, field, degree)
    }

    fn betti_number<R: Field>(&self, field: R, degree: usize) -> Result<usize, TopologyError> {
        SimplicialComplex::betti_number(self, field, degree)
    }

    fn graded_homology<R: Field>(
        &self,
        field: R,
    ) -> Result<noether::homalg::GradedHomology<R>, TopologyError> {
        SimplicialComplex::graded_homology(self, field)
    }

    fn integral_homology(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<IntegerRing>, TopologyError> {
        SimplicialComplex::integral_homology(self, degree)
    }

    fn homology_invariants(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::FinitelyGeneratedAbelianGroup, TopologyError> {
        SimplicialComplex::homology_invariants(self, degree)
    }

    fn graded_integral_homology(
        &self,
    ) -> Result<noether::homalg::GradedHomology<IntegerRing>, TopologyError> {
        SimplicialComplex::graded_integral_homology(self)
    }
}

impl SimplicialChains for SimplicialSet {
    fn chain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError> {
        SimplicialSet::chain_complex(self, ring)
    }

    fn cochain_complex<R: Ring>(
        &self,
        ring: R,
    ) -> Result<noether::homalg::ChainComplex<R>, TopologyError> {
        SimplicialSet::cochain_complex(self, ring)
    }

    fn homology<R: Field>(
        &self,
        field: R,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<R>, TopologyError> {
        SimplicialSet::homology(self, field, degree)
    }

    fn betti_number<R: Field>(&self, field: R, degree: usize) -> Result<usize, TopologyError> {
        SimplicialSet::betti_number(self, field, degree)
    }

    fn graded_homology<R: Field>(
        &self,
        field: R,
    ) -> Result<noether::homalg::GradedHomology<R>, TopologyError> {
        SimplicialSet::graded_homology(self, field)
    }

    fn integral_homology(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::HomologyGroup<IntegerRing>, TopologyError> {
        SimplicialSet::integral_homology(self, degree)
    }

    fn homology_invariants(
        &self,
        degree: usize,
    ) -> Result<noether::homalg::FinitelyGeneratedAbelianGroup, TopologyError> {
        SimplicialSet::homology_invariants(self, degree)
    }

    fn graded_integral_homology(
        &self,
    ) -> Result<noether::homalg::GradedHomology<IntegerRing>, TopologyError> {
        SimplicialSet::graded_integral_homology(self)
    }
}

#[derive(Clone)]
enum CoefficientContext {
    Integer,
    Rational,
    FiniteField(FiniteField),
    NumberField(NumberField),
}

fn coefficient_context(ring: &Bound<'_, PyAny>) -> PyResult<CoefficientContext> {
    if ring.is_instance_of::<PyIntegerRing>() {
        Ok(CoefficientContext::Integer)
    } else if ring.is_instance_of::<PyRationalField>() {
        Ok(CoefficientContext::Rational)
    } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
        Ok(CoefficientContext::FiniteField(field.inner.clone()))
    } else if let Ok(field) = ring.extract::<PyRef<'_, PyNumberField>>() {
        Ok(CoefficientContext::NumberField(field.inner.clone()))
    } else {
        Err(unsupported_coefficients())
    }
}

fn optional_coefficient_context(
    ring: Option<&Bound<'_, PyAny>>,
    default: CoefficientContext,
) -> PyResult<CoefficientContext> {
    ring.map_or(Ok(default), coefficient_context)
}

fn chain_complex<T: SimplicialChains>(
    value: &T,
    ring: CoefficientContext,
) -> PyResult<PyChainComplex> {
    let inner = match ring {
        CoefficientContext::Integer => {
            ChainComplexInner::Integer(value.chain_complex(Z).map_err(topology_error)?)
        }
        CoefficientContext::Rational => {
            ChainComplexInner::Rational(value.chain_complex(Q).map_err(topology_error)?)
        }
        CoefficientContext::FiniteField(field) => {
            ChainComplexInner::FiniteField(value.chain_complex(field).map_err(topology_error)?)
        }
        CoefficientContext::NumberField(field) => {
            ChainComplexInner::NumberField(value.chain_complex(field).map_err(topology_error)?)
        }
    };
    Ok(PyChainComplex { inner })
}

fn cochain_complex<T: SimplicialChains>(
    value: &T,
    ring: CoefficientContext,
) -> PyResult<PyChainComplex> {
    let inner = match ring {
        CoefficientContext::Integer => {
            ChainComplexInner::Integer(value.cochain_complex(Z).map_err(topology_error)?)
        }
        CoefficientContext::Rational => {
            ChainComplexInner::Rational(value.cochain_complex(Q).map_err(topology_error)?)
        }
        CoefficientContext::FiniteField(field) => {
            ChainComplexInner::FiniteField(value.cochain_complex(field).map_err(topology_error)?)
        }
        CoefficientContext::NumberField(field) => {
            ChainComplexInner::NumberField(value.cochain_complex(field).map_err(topology_error)?)
        }
    };
    Ok(PyChainComplex { inner })
}

trait ModTwoCochainOperations {
    fn cup_i_operation(
        &self,
        cup_index: usize,
        left_degree: usize,
        left: &noether::linalg::Vector<FiniteField>,
        right_degree: usize,
        right: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError>;

    fn steenrod_square_operation(
        &self,
        square: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError>;
}

impl ModTwoCochainOperations for SimplicialComplex {
    fn cup_i_operation(
        &self,
        cup_index: usize,
        left_degree: usize,
        left: &noether::linalg::Vector<FiniteField>,
        right_degree: usize,
        right: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.cup_i(cup_index, left_degree, left, right_degree, right)
    }

    fn steenrod_square_operation(
        &self,
        square: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.steenrod_square(square, degree, cocycle)
    }
}

impl ModTwoCochainOperations for SimplicialSet {
    fn cup_i_operation(
        &self,
        cup_index: usize,
        left_degree: usize,
        left: &noether::linalg::Vector<FiniteField>,
        right_degree: usize,
        right: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.cup_i(cup_index, left_degree, left, right_degree, right)
    }

    fn steenrod_square_operation(
        &self,
        square: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.steenrod_square(square, degree, cocycle)
    }
}

fn cup_i_operation<T: ModTwoCochainOperations>(
    value: &T,
    cup_index: usize,
    left_degree: usize,
    left: &PyVector,
    right_degree: usize,
    right: &PyVector,
) -> PyResult<PyVector> {
    let (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) =
        (&left.inner, &right.inner)
    else {
        return Err(PyTypeError::new_err(
            "cup_i requires cochains over the same finite field of characteristic 2",
        ));
    };
    Ok(PyVector {
        inner: VectorInner::FiniteField(
            value
                .cup_i_operation(cup_index, left_degree, left, right_degree, right)
                .map_err(topology_error)?,
        ),
    })
}

fn steenrod_square_operation<T: ModTwoCochainOperations>(
    value: &T,
    square: usize,
    degree: usize,
    cocycle: &PyVector,
) -> PyResult<PyVector> {
    let VectorInner::FiniteField(cocycle) = &cocycle.inner else {
        return Err(PyTypeError::new_err(
            "steenrod_square requires a cocycle over a finite field of characteristic 2",
        ));
    };
    Ok(PyVector {
        inner: VectorInner::FiniteField(
            value
                .steenrod_square_operation(square, degree, cocycle)
                .map_err(topology_error)?,
        ),
    })
}

trait OddPrimaryCochainOperations {
    fn steenrod_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError>;

    fn bockstein_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError>;
}

impl OddPrimaryCochainOperations for SimplicialComplex {
    fn steenrod_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.steenrod_reduced_power(power, degree, cocycle)
    }

    fn bockstein_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.bockstein_reduced_power(power, degree, cocycle)
    }
}

impl OddPrimaryCochainOperations for SimplicialSet {
    fn steenrod_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.steenrod_reduced_power(power, degree, cocycle)
    }

    fn bockstein_reduced_power_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &noether::linalg::Vector<FiniteField>,
    ) -> Result<noether::linalg::Vector<FiniteField>, TopologyError> {
        self.bockstein_reduced_power(power, degree, cocycle)
    }
}

fn odd_primary_cochain_operation<T: OddPrimaryCochainOperations>(
    value: &T,
    power: usize,
    degree: usize,
    cocycle: &PyVector,
    bockstein: bool,
) -> PyResult<PyVector> {
    let VectorInner::FiniteField(cocycle) = &cocycle.inner else {
        return Err(PyTypeError::new_err(
            "odd-primary Steenrod operations require a cocycle over an odd prime field",
        ));
    };
    let inner = if bockstein {
        value.bockstein_reduced_power_operation(power, degree, cocycle)
    } else {
        value.steenrod_reduced_power_operation(power, degree, cocycle)
    };
    Ok(PyVector {
        inner: VectorInner::FiniteField(inner.map_err(topology_error)?),
    })
}

trait CupProductSource {
    fn cup_product_operation<R: Ring>(
        &self,
        left_degree: usize,
        left: &noether::linalg::Vector<R>,
        right_degree: usize,
        right: &noether::linalg::Vector<R>,
    ) -> Result<noether::linalg::Vector<R>, TopologyError>;
}

impl CupProductSource for SimplicialComplex {
    fn cup_product_operation<R: Ring>(
        &self,
        left_degree: usize,
        left: &noether::linalg::Vector<R>,
        right_degree: usize,
        right: &noether::linalg::Vector<R>,
    ) -> Result<noether::linalg::Vector<R>, TopologyError> {
        self.cup_product(left_degree, left, right_degree, right)
    }
}

impl CupProductSource for SimplicialSet {
    fn cup_product_operation<R: Ring>(
        &self,
        left_degree: usize,
        left: &noether::linalg::Vector<R>,
        right_degree: usize,
        right: &noether::linalg::Vector<R>,
    ) -> Result<noether::linalg::Vector<R>, TopologyError> {
        self.cup_product(left_degree, left, right_degree, right)
    }
}

fn cup_product<T: CupProductSource>(
    value: &T,
    left_degree: usize,
    left: &PyVector,
    right_degree: usize,
    right: &PyVector,
) -> PyResult<PyVector> {
    macro_rules! cup {
        ($variant:ident, $left:ident, $right:ident) => {
            value
                .cup_product_operation(left_degree, $left, right_degree, $right)
                .map(VectorInner::$variant)
                .map_err(topology_error)
        };
    }
    let inner = match (&left.inner, &right.inner) {
        (VectorInner::Integer(left), VectorInner::Integer(right)) => cup!(Integer, left, right),
        (VectorInner::Rational(left), VectorInner::Rational(right)) => cup!(Rational, left, right),
        (VectorInner::Residue(left), VectorInner::Residue(right)) => cup!(Residue, left, right),
        (VectorInner::FiniteField(left), VectorInner::FiniteField(right)) => {
            cup!(FiniteField, left, right)
        }
        (VectorInner::Padic(left), VectorInner::Padic(right)) => cup!(Padic, left, right),
        (VectorInner::PadicField(left), VectorInner::PadicField(right)) => {
            cup!(PadicField, left, right)
        }
        (VectorInner::Qadic(left), VectorInner::Qadic(right)) => cup!(Qadic, left, right),
        (VectorInner::QadicField(left), VectorInner::QadicField(right)) => {
            cup!(QadicField, left, right)
        }
        (VectorInner::NumberField(left), VectorInner::NumberField(right)) => {
            cup!(NumberField, left, right)
        }
        (VectorInner::NumberFieldOrder(left), VectorInner::NumberFieldOrder(right)) => {
            cup!(NumberFieldOrder, left, right)
        }
        (VectorInner::RealBall(left), VectorInner::RealBall(right)) => cup!(RealBall, left, right),
        (VectorInner::ComplexBall(left), VectorInner::ComplexBall(right)) => {
            cup!(ComplexBall, left, right)
        }
        (VectorInner::SeriesInteger(left), VectorInner::SeriesInteger(right)) => {
            cup!(SeriesInteger, left, right)
        }
        (VectorInner::SeriesRational(left), VectorInner::SeriesRational(right)) => {
            cup!(SeriesRational, left, right)
        }
        (VectorInner::SeriesResidue(left), VectorInner::SeriesResidue(right)) => {
            cup!(SeriesResidue, left, right)
        }
        (VectorInner::SeriesFiniteField(left), VectorInner::SeriesFiniteField(right)) => {
            cup!(SeriesFiniteField, left, right)
        }
        (VectorInner::SeriesPadic(left), VectorInner::SeriesPadic(right)) => {
            cup!(SeriesPadic, left, right)
        }
        (VectorInner::SeriesQadic(left), VectorInner::SeriesQadic(right)) => {
            cup!(SeriesQadic, left, right)
        }
        _ => {
            return Err(PyTypeError::new_err(
                "cup-product cochains must use the same coefficient context family",
            ));
        }
    }?;
    Ok(PyVector { inner })
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum SimplicialMorseReductionInner {
    Integer(SimplicialMorseReduction<IntegerRing>),
    Rational(SimplicialMorseReduction<RationalField>),
    FiniteField(SimplicialMorseReduction<FiniteField>),
    NumberField(SimplicialMorseReduction<NumberField>),
}

/// Verified critical-cell chain model of a simplicial complex.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialMorseReduction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialMorseReduction {
    inner: SimplicialMorseReductionInner,
}

#[pymethods]
impl PySimplicialMorseReduction {
    /// Source simplicial complex.
    #[getter]
    fn source(&self) -> PySimplicialComplex {
        let inner = match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => value.source().clone(),
            SimplicialMorseReductionInner::Rational(value) => value.source().clone(),
            SimplicialMorseReductionInner::FiniteField(value) => value.source().clone(),
            SimplicialMorseReductionInner::NumberField(value) => value.source().clone(),
        };
        PySimplicialComplex { inner }
    }

    /// Chain complex generated by critical cells.
    #[getter]
    fn reduced(&self) -> PyChainComplex {
        let inner = match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => {
                ChainComplexInner::Integer(value.reduced().clone())
            }
            SimplicialMorseReductionInner::Rational(value) => {
                ChainComplexInner::Rational(value.reduced().clone())
            }
            SimplicialMorseReductionInner::FiniteField(value) => {
                ChainComplexInner::FiniteField(value.reduced().clone())
            }
            SimplicialMorseReductionInner::NumberField(value) => {
                ChainComplexInner::NumberField(value.reduced().clone())
            }
        };
        PyChainComplex { inner }
    }

    /// Number of nonempty simplices before reduction.
    #[getter]
    fn original_cells(&self) -> usize {
        match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => value.original_cells(),
            SimplicialMorseReductionInner::Rational(value) => value.original_cells(),
            SimplicialMorseReductionInner::FiniteField(value) => value.original_cells(),
            SimplicialMorseReductionInner::NumberField(value) => value.original_cells(),
        }
    }

    /// Number of critical cells in each simplicial dimension.
    #[getter]
    fn critical_cells(&self) -> Vec<usize> {
        match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => value.critical_cells(),
            SimplicialMorseReductionInner::Rational(value) => value.critical_cells(),
            SimplicialMorseReductionInner::FiniteField(value) => value.critical_cells(),
            SimplicialMorseReductionInner::NumberField(value) => value.critical_cells(),
        }
    }

    /// Number of matched face/coface pairs.
    #[getter]
    fn matched_pairs(&self) -> usize {
        match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => value.matched_pairs(),
            SimplicialMorseReductionInner::Rational(value) => value.matched_pairs(),
            SimplicialMorseReductionInner::FiniteField(value) => value.matched_pairs(),
            SimplicialMorseReductionInner::NumberField(value) => value.matched_pairs(),
        }
    }

    /// Recheck the acyclic matching and Morse boundary.
    fn verify(&self) -> PyResult<()> {
        let result = match &self.inner {
            SimplicialMorseReductionInner::Integer(value) => value.verify(),
            SimplicialMorseReductionInner::Rational(value) => value.verify(),
            SimplicialMorseReductionInner::FiniteField(value) => value.verify(),
            SimplicialMorseReductionInner::NumberField(value) => value.verify(),
        };
        result.map_err(super::errors::verification)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SimplicialMorseReduction(original_cells={}, critical_cells={:?})",
            self.original_cells(),
            self.critical_cells()
        ))
    }
}

fn morse_reduction(
    value: &SimplicialComplex,
    ring: CoefficientContext,
) -> PyResult<PySimplicialMorseReduction> {
    let inner = match ring {
        CoefficientContext::Integer => SimplicialMorseReductionInner::Integer(
            value.morse_reduction(Z).map_err(topology_error)?,
        ),
        CoefficientContext::Rational => SimplicialMorseReductionInner::Rational(
            value.morse_reduction(Q).map_err(topology_error)?,
        ),
        CoefficientContext::FiniteField(field) => SimplicialMorseReductionInner::FiniteField(
            value.morse_reduction(field).map_err(topology_error)?,
        ),
        CoefficientContext::NumberField(field) => SimplicialMorseReductionInner::NumberField(
            value.morse_reduction(field).map_err(topology_error)?,
        ),
    };
    Ok(PySimplicialMorseReduction { inner })
}

fn homology<T: SimplicialChains>(
    value: &T,
    degree: usize,
    ring: CoefficientContext,
) -> PyResult<PyHomologyGroup> {
    let inner = match ring {
        CoefficientContext::Integer => {
            HomologyGroupInner::Integer(value.integral_homology(degree).map_err(topology_error)?)
        }
        CoefficientContext::Rational => {
            HomologyGroupInner::Rational(value.homology(Q, degree).map_err(topology_error)?)
        }
        CoefficientContext::FiniteField(field) => {
            HomologyGroupInner::FiniteField(value.homology(field, degree).map_err(topology_error)?)
        }
        CoefficientContext::NumberField(field) => {
            HomologyGroupInner::NumberField(value.homology(field, degree).map_err(topology_error)?)
        }
    };
    Ok(PyHomologyGroup { inner })
}

fn homology_invariants<T: SimplicialChains>(
    value: &T,
    degree: usize,
) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
    Ok(PyFinitelyGeneratedAbelianGroup {
        inner: value.homology_invariants(degree).map_err(topology_error)?,
    })
}

fn betti_number<T: SimplicialChains>(
    value: &T,
    degree: usize,
    ring: CoefficientContext,
) -> PyResult<usize> {
    match ring {
        CoefficientContext::Rational => value.betti_number(Q, degree).map_err(topology_error),
        CoefficientContext::Integer => Err(PyValueError::new_err(
            "betti_number requires field coefficients; use Q or another field",
        )),
        CoefficientContext::FiniteField(field) => {
            value.betti_number(field, degree).map_err(topology_error)
        }
        CoefficientContext::NumberField(field) => {
            value.betti_number(field, degree).map_err(topology_error)
        }
    }
}

fn graded_homology<T: SimplicialChains>(
    value: &T,
    ring: CoefficientContext,
) -> PyResult<PyGradedHomology> {
    let inner = match ring {
        CoefficientContext::Integer => {
            GradedHomologyInner::Integer(value.graded_integral_homology().map_err(topology_error)?)
        }
        CoefficientContext::Rational => {
            GradedHomologyInner::Rational(value.graded_homology(Q).map_err(topology_error)?)
        }
        CoefficientContext::FiniteField(field) => {
            GradedHomologyInner::FiniteField(value.graded_homology(field).map_err(topology_error)?)
        }
        CoefficientContext::NumberField(field) => {
            GradedHomologyInner::NumberField(value.graded_homology(field).map_err(topology_error)?)
        }
    };
    Ok(PyGradedHomology { inner })
}

/// A finite abstract simplicial complex given by its maximal simplices.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialComplex",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialComplex {
    pub(crate) inner: SimplicialComplex,
}

impl PySimplicialComplex {
    pub(crate) fn from_core(inner: SimplicialComplex) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySimplicialComplex {
    /// Construct the downward closure of `facets`.
    #[new]
    fn new(facets: Vec<Vec<u32>>) -> Self {
        Self {
            inner: SimplicialComplex::from_facets(&facets),
        }
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialComplex::TYPE_ID
    }

    /// Top dimension; the empty complex has dimension -1.
    #[getter]
    fn dimension(&self) -> i64 {
        self.inner.dimension()
    }

    /// Whether this complex contains no nonempty simplices.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of simplices in one dimension.
    fn count(&self, dimension: usize) -> usize {
        self.inner.count(dimension)
    }

    /// Total number of nonempty simplices.
    fn __len__(&self) -> usize {
        self.inner.total_cells()
    }

    /// Maximal simplices in canonical lexicographic order.
    #[getter]
    fn facets(&self) -> Vec<Vec<u32>> {
        self.inner
            .facets()
            .iter()
            .map(|simplex| simplex.vertices().to_vec())
            .collect()
    }

    /// Simplices in one dimension, in canonical boundary-matrix order.
    fn simplices(&self, dimension: usize) -> Vec<Vec<u32>> {
        self.inner
            .simplices(dimension)
            .iter()
            .map(|simplex| simplex.vertices().to_vec())
            .collect()
    }

    /// Euler characteristic computed independently from cell counts.
    fn euler_characteristic(&self) -> i64 {
        self.inner.euler_characteristic()
    }

    /// Construct the canonical checked chain complex over `ring`.
    fn chain_complex(&self, py: Python<'_>, ring: &Bound<'_, PyAny>) -> PyResult<PyChainComplex> {
        let ring = coefficient_context(ring)?;
        py.detach(|| chain_complex(&self.inner, ring))
    }

    /// Construct the canonical signed cochain complex over `ring`.
    fn cochain_complex(&self, py: Python<'_>, ring: &Bound<'_, PyAny>) -> PyResult<PyChainComplex> {
        let ring = coefficient_context(ring)?;
        py.detach(|| cochain_complex(&self.inner, ring))
    }

    /// Alexander--Whitney front/back faces for one simplex.
    fn alexander_whitney_diagonal(
        &self,
        dimension: usize,
        simplex: usize,
    ) -> PyResult<Vec<(Vec<u32>, Vec<u32>)>> {
        self.inner
            .alexander_whitney_diagonal(dimension, simplex)
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|(front, back)| (front.vertices().to_vec(), back.vertices().to_vec()))
                    .collect()
            })
            .map_err(topology_error)
    }

    /// Exact Alexander--Whitney cup product in the fixed simplex bases.
    fn cup_product(
        &self,
        py: Python<'_>,
        left_degree: usize,
        left: &PyVector,
        right_degree: usize,
        right: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| cup_product(&self.inner, left_degree, left, right_degree, right))
    }

    /// Medina-Mardones cup-i product on normalized characteristic-two
    /// cochains.
    fn cup_i(
        &self,
        py: Python<'_>,
        cup_index: usize,
        left_degree: usize,
        left: &PyVector,
        right_degree: usize,
        right: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| {
            cup_i_operation(
                &self.inner,
                cup_index,
                left_degree,
                left,
                right_degree,
                right,
            )
        })
    }

    /// Steenrod square of a checked characteristic-two cocycle.
    fn steenrod_square(
        &self,
        py: Python<'_>,
        square: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| steenrod_square_operation(&self.inner, square, degree, cocycle))
    }

    /// Odd-primary Steenrod reduced power of a checked cocycle.
    fn steenrod_reduced_power(
        &self,
        py: Python<'_>,
        power: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| odd_primary_cochain_operation(&self.inner, power, degree, cocycle, false))
    }

    /// Odd-primary beta-reduced-power operation on a checked cocycle.
    fn bockstein_reduced_power(
        &self,
        py: Python<'_>,
        power: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| odd_primary_cochain_operation(&self.inner, power, degree, cocycle, true))
    }

    /// Complete cohomology algebra over one exact finite field.
    fn cohomology_ring(
        &self,
        py: Python<'_>,
        field: &PyFiniteField,
    ) -> PyResult<PySimplicialCohomologyRing> {
        py.detach(|| {
            Ok(PySimplicialCohomologyRing {
                inner: self
                    .inner
                    .cohomology_ring(field.inner.clone())
                    .map_err(topology_error)?,
            })
        })
    }

    /// Construct a verified critical-cell chain model over `ring`.
    fn morse_reduction(
        &self,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
    ) -> PyResult<PySimplicialMorseReduction> {
        let ring = coefficient_context(ring)?;
        py.detach(|| morse_reduction(&self.inner, ring))
    }

    /// Construct only the critical-cell chain complex over `ring`.
    fn morse_chain_complex(
        &self,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
    ) -> PyResult<PyChainComplex> {
        let ring = coefficient_context(ring)?;
        py.detach(|| Ok(morse_reduction(&self.inner, ring)?.reduced()))
    }

    /// Return one Betti number over `Q` by default.
    #[pyo3(signature = (degree, *, ring = None))]
    fn betti_number(
        &self,
        py: Python<'_>,
        degree: usize,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Rational)?;
        py.detach(|| betti_number(&self.inner, degree, ring))
    }

    /// Compute one checked homology group, over `Z` by default.
    #[pyo3(signature = (degree, *, ring = None))]
    fn homology(
        &self,
        py: Python<'_>,
        degree: usize,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHomologyGroup> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Integer)?;
        py.detach(|| homology(&self.inner, degree, ring))
    }

    /// Compute lean integral homology invariants without retaining boundaries.
    fn homology_invariants(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        py.detach(|| homology_invariants(&self.inner, degree))
    }

    /// Compute lean integral homology invariants in every chain degree.
    fn graded_homology_invariants(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<PyFinitelyGeneratedAbelianGroup>> {
        py.detach(|| {
            self.inner
                .graded_homology_invariants()
                .map(|groups| {
                    groups
                        .into_iter()
                        .map(|inner| PyFinitelyGeneratedAbelianGroup { inner })
                        .collect()
                })
                .map_err(topology_error)
        })
    }

    /// Compute checked homology in every chain degree, over `Z` by default.
    #[pyo3(signature = (*, ring = None))]
    fn graded_homology(
        &self,
        py: Python<'_>,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyGradedHomology> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Integer)?;
        py.detach(|| graded_homology(&self.inner, ring))
    }

    /// Canonically relabel the used vertices as a contiguous initial segment.
    fn canonical_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.canonical_form().map_err(topology_error)?,
            })
        })
    }

    /// Dominated vertices with their least-label dominators.
    fn dominated_vertices(&self) -> Vec<PyDominatedVertex> {
        self.inner
            .dominated_vertices()
            .into_iter()
            .map(|inner| PyDominatedVertex { inner })
            .collect()
    }

    /// Replay one verified dominated-vertex deletion.
    fn strong_collapse(&self, step: &PyDominatedVertex) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .strong_collapse(step.inner)
                .map_err(topology_error)?,
        })
    }

    /// Construct a replayable reduction to the strong-homotopy core.
    fn strong_core_reduction(&self, py: Python<'_>) -> PyResult<PyStrongCoreReduction> {
        py.detach(|| {
            Ok(PyStrongCoreReduction {
                inner: self.inner.strong_core_reduction().map_err(topology_error)?,
            })
        })
    }

    /// Canonically relabeled strong-homotopy core.
    fn strong_core(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.strong_core().map_err(topology_error)?,
            })
        })
    }

    /// Every elementary free-face collapse available in this complex.
    fn elementary_collapses(&self) -> Vec<PyElementaryCollapse> {
        self.inner
            .elementary_collapses()
            .into_iter()
            .map(|inner| PyElementaryCollapse { inner })
            .collect()
    }

    /// Replay one verified elementary free-face collapse.
    fn elementary_collapse(&self, step: &PyElementaryCollapse) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .elementary_collapse(&step.inner)
                .map_err(topology_error)?,
        })
    }

    /// Replay and retain a proposed sequence of elementary collapses.
    fn collapse_sequence(
        &self,
        steps: Vec<PyElementaryCollapse>,
    ) -> PyResult<PySimplicialCollapseSequence> {
        let steps = steps.into_iter().map(|step| step.inner).collect();
        Ok(PySimplicialCollapseSequence {
            inner: self
                .inner
                .collapse_sequence(steps)
                .map_err(topology_error)?,
        })
    }

    /// Replay and retain a proposed simple-homotopy zigzag.
    fn simple_homotopy_zigzag(
        &self,
        moves: Vec<PySimplicialSimpleHomotopyMove>,
    ) -> PyResult<PySimplicialSimpleHomotopyZigzag> {
        Ok(PySimplicialSimpleHomotopyZigzag {
            inner: self
                .inner
                .simple_homotopy_zigzag(moves.into_iter().map(|movement| movement.inner).collect())
                .map_err(topology_error)?,
        })
    }

    /// Deterministically collapse until no free-face pair remains.
    fn greedy_collapse_reduction(&self, py: Python<'_>) -> PyResult<PySimplicialCollapseSequence> {
        py.detach(|| {
            Ok(PySimplicialCollapseSequence {
                inner: self
                    .inner
                    .greedy_collapse_reduction()
                    .map_err(topology_error)?,
            })
        })
    }

    /// Compute the exact automorphism group acting on vertex ids.
    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            self.inner
                .automorphism_group()
                .map(|inner| PyPermGroup { inner })
                .map_err(topology_error)
        })
    }

    /// Recheck signed boundary construction and `d^2 = 0` over the integers.
    fn verify(&self) -> PyResult<()> {
        self.inner
            .chain_complex(Z)
            .map_err(topology_error)?
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialComplex::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SimplicialComplex(dimension={}, cells={}, facets={})",
                self.inner.dimension(),
                self.inner.total_cells(),
                self.inner.facets().len()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let count = nonnegative_degree_count(self.inner.dimension());
        summary.set_item("type", SimplicialComplex::TYPE_ID)?;
        summary.set_item("dimension", self.inner.dimension())?;
        summary.set_item(
            "cell_counts",
            (0..count)
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .map(|dimension| self.inner.count(dimension))
                .collect::<Vec<_>>(),
        )?;
        if count > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("cell_counts_truncated", true)?;
        }
        summary.set_item("facets", self.inner.facets().len())?;
        Ok(summary.unbind())
    }
}

/// A finite simplicial set given by the faces of its non-degenerate simplices.
#[pyclass(
    module = "noether.topology",
    name = "SimplicialSet",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySimplicialSet {
    inner: SimplicialSet,
}

#[pymethods]
impl PySimplicialSet {
    /// Construct a checked non-degenerate presentation.
    ///
    /// A non-degenerate face is its integer index one dimension below. A
    /// degenerate face is a mapping with `simplex_dimension`, `simplex`, and
    /// the monotone `surjection` producing it.
    #[new]
    fn new(py: Python<'_>, faces: Vec<Vec<Vec<Py<PyAny>>>>) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSet::new(parse_faces(py, faces)?).map_err(topology_error)?,
        })
    }

    /// The minimal circle: one vertex and one loop.
    #[staticmethod]
    fn circle() -> Self {
        Self {
            inner: SimplicialSet::circle(),
        }
    }

    /// The minimal `dimension`-sphere.
    #[staticmethod]
    fn sphere(dimension: usize) -> Self {
        Self {
            inner: SimplicialSet::sphere(dimension),
        }
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        SimplicialSet::TYPE_ID
    }

    /// Top dimension carrying a non-degenerate simplex.
    #[getter]
    fn dimension(&self) -> i64 {
        self.inner.dimension()
    }

    /// Whether there are no non-degenerate simplices.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of non-degenerate simplices in one dimension.
    fn count(&self, dimension: usize) -> usize {
        self.inner.count(dimension)
    }

    /// Total number of non-degenerate simplices.
    fn __len__(&self) -> usize {
        self.inner.total_cells()
    }

    /// Non-degenerate simplex counts by dimension.
    #[getter]
    fn non_degenerate_counts(&self) -> Vec<usize> {
        (0..nonnegative_degree_count(self.inner.dimension()))
            .map(|dimension| self.inner.count(dimension))
            .collect()
    }

    /// Faces of one non-degenerate simplex.
    fn faces_of(
        &self,
        py: Python<'_>,
        dimension: usize,
        simplex: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let faces = self.inner.faces_of(dimension, simplex).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "there is no non-degenerate {dimension}-simplex {simplex}; {} exist",
                self.inner.count(dimension)
            ))
        })?;
        faces_to_python(py, faces)
    }

    /// Complete canonical non-degenerate face data.
    #[getter]
    fn faces(&self, py: Python<'_>) -> PyResult<Vec<Vec<Vec<Py<PyAny>>>>> {
        (0..nonnegative_degree_count(self.inner.dimension()))
            .map(|dimension| {
                (0..self.inner.count(dimension))
                    .map(|simplex| {
                        let faces = self.inner.faces_of(dimension, simplex).ok_or_else(|| {
                            PyValueError::new_err("validated simplicial-set face data is missing")
                        })?;
                        faces_to_python(py, faces)
                    })
                    .collect()
            })
            .collect()
    }

    /// Euler characteristic from non-degenerate cell counts.
    fn euler_characteristic(&self) -> i64 {
        self.inner.euler_characteristic()
    }

    /// Construct the canonical normalized chain complex over `ring`.
    fn chain_complex(&self, py: Python<'_>, ring: &Bound<'_, PyAny>) -> PyResult<PyChainComplex> {
        let ring = coefficient_context(ring)?;
        py.detach(|| chain_complex(&self.inner, ring))
    }

    /// Construct normalized simplicial cochains over `ring`.
    fn cochain_complex(&self, py: Python<'_>, ring: &Bound<'_, PyAny>) -> PyResult<PyChainComplex> {
        let ring = coefficient_context(ring)?;
        py.detach(|| cochain_complex(&self.inner, ring))
    }

    /// Alexander--Whitney cup product on normalized cochains.
    fn cup_product(
        &self,
        py: Python<'_>,
        left_degree: usize,
        left: &PyVector,
        right_degree: usize,
        right: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| cup_product(&self.inner, left_degree, left, right_degree, right))
    }

    /// Medina-Mardones cup-i product on normalized characteristic-two
    /// cochains.
    fn cup_i(
        &self,
        py: Python<'_>,
        cup_index: usize,
        left_degree: usize,
        left: &PyVector,
        right_degree: usize,
        right: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| {
            cup_i_operation(
                &self.inner,
                cup_index,
                left_degree,
                left,
                right_degree,
                right,
            )
        })
    }

    /// Steenrod square of a checked characteristic-two cocycle.
    fn steenrod_square(
        &self,
        py: Python<'_>,
        square: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| steenrod_square_operation(&self.inner, square, degree, cocycle))
    }

    /// Odd-primary Steenrod reduced power of a checked normalized cocycle.
    fn steenrod_reduced_power(
        &self,
        py: Python<'_>,
        power: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| odd_primary_cochain_operation(&self.inner, power, degree, cocycle, false))
    }

    /// Odd-primary beta-reduced-power operation on a checked normalized cocycle.
    fn bockstein_reduced_power(
        &self,
        py: Python<'_>,
        power: usize,
        degree: usize,
        cocycle: &PyVector,
    ) -> PyResult<PyVector> {
        py.detach(|| odd_primary_cochain_operation(&self.inner, power, degree, cocycle, true))
    }

    /// Complete normalized cohomology algebra over one exact finite field.
    fn cohomology_ring(
        &self,
        py: Python<'_>,
        field: &PyFiniteField,
    ) -> PyResult<PySimplicialSetCohomologyRing> {
        py.detach(|| {
            Ok(PySimplicialSetCohomologyRing {
                inner: self
                    .inner
                    .cohomology_ring(field.inner.clone())
                    .map_err(topology_error)?,
            })
        })
    }

    /// All available elementary collapses inverse to horn attachments.
    fn elementary_horn_collapses(&self) -> Vec<PySimplicialSetElementaryHornCollapse> {
        self.inner
            .elementary_horn_collapses()
            .into_iter()
            .map(|inner| PySimplicialSetElementaryHornCollapse { inner })
            .collect()
    }

    /// Replay one checked elementary horn collapse.
    fn elementary_horn_collapse(
        &self,
        step: &PySimplicialSetElementaryHornCollapse,
    ) -> PyResult<PySimplicialSet> {
        Ok(PySimplicialSet {
            inner: self
                .inner
                .elementary_horn_collapse(&step.inner)
                .map_err(topology_error)?,
        })
    }

    /// Replay and retain a checked sequence of horn collapses.
    fn horn_collapse_sequence(
        &self,
        steps: Vec<PySimplicialSetElementaryHornCollapse>,
    ) -> PyResult<PySimplicialSetHornCollapseSequence> {
        Ok(PySimplicialSetHornCollapseSequence {
            inner: self
                .inner
                .horn_collapse_sequence(steps.into_iter().map(|step| step.inner).collect())
                .map_err(topology_error)?,
        })
    }

    /// Replay and retain a checked zigzag of horn attachments and collapses.
    fn anodyne_zigzag(
        &self,
        moves: Vec<PySimplicialSetAnodyneMove>,
    ) -> PyResult<PySimplicialSetAnodyneZigzag> {
        Ok(PySimplicialSetAnodyneZigzag {
            inner: self
                .inner
                .anodyne_zigzag(moves.into_iter().map(|movement| movement.inner).collect())
                .map_err(topology_error)?,
        })
    }

    /// Deterministically collapse the least available horn pair to a terminal
    /// target.
    fn greedy_horn_collapse_reduction(
        &self,
        py: Python<'_>,
    ) -> PyResult<PySimplicialSetHornCollapseSequence> {
        Ok(PySimplicialSetHornCollapseSequence {
            inner: py
                .detach(|| self.inner.greedy_horn_collapse_reduction())
                .map_err(topology_error)?,
        })
    }

    /// Deterministically reduce normalized chains by replayable `1` and `-1`
    /// cancellations.
    fn normalized_chain_reduction(
        &self,
        py: Python<'_>,
        ring: &Bound<'_, PyAny>,
    ) -> PyResult<PyChainReduction> {
        let ring = coefficient_context(ring)?;
        let inner = py.detach(|| -> PyResult<_> {
            Ok(match ring {
                CoefficientContext::Integer => ChainReductionInner::Integer(
                    self.inner
                        .normalized_chain_reduction(Z)
                        .map_err(topology_error)?,
                ),
                CoefficientContext::Rational => ChainReductionInner::Rational(
                    self.inner
                        .normalized_chain_reduction(Q)
                        .map_err(topology_error)?,
                ),
                CoefficientContext::FiniteField(field) => ChainReductionInner::FiniteField(
                    self.inner
                        .normalized_chain_reduction(field)
                        .map_err(topology_error)?,
                ),
                CoefficientContext::NumberField(field) => ChainReductionInner::NumberField(
                    self.inner
                        .normalized_chain_reduction(field)
                        .map_err(topology_error)?,
                ),
            })
        })?;
        Ok(PyChainReduction { inner })
    }

    /// Return one Betti number over `Q` by default.
    #[pyo3(signature = (degree, *, ring = None))]
    fn betti_number(
        &self,
        py: Python<'_>,
        degree: usize,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Rational)?;
        py.detach(|| betti_number(&self.inner, degree, ring))
    }

    /// Compute one checked homology group, over `Z` by default.
    #[pyo3(signature = (degree, *, ring = None))]
    fn homology(
        &self,
        py: Python<'_>,
        degree: usize,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHomologyGroup> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Integer)?;
        py.detach(|| homology(&self.inner, degree, ring))
    }

    /// Compute lean integral homology invariants without retaining boundaries.
    fn homology_invariants(
        &self,
        py: Python<'_>,
        degree: usize,
    ) -> PyResult<PyFinitelyGeneratedAbelianGroup> {
        py.detach(|| homology_invariants(&self.inner, degree))
    }

    /// Compute checked homology in every degree, over `Z` by default.
    #[pyo3(signature = (*, ring = None))]
    fn graded_homology(
        &self,
        py: Python<'_>,
        ring: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyGradedHomology> {
        let ring = optional_coefficient_context(ring, CoefficientContext::Integer)?;
        py.detach(|| graded_homology(&self.inner, ring))
    }

    /// Canonically number the non-degenerate simplices in every dimension.
    fn canonical_form(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.canonical_form().map_err(topology_error)?,
            })
        })
    }

    /// Exact automorphism group on dimension-major non-degenerate cells.
    fn automorphism_group(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            self.inner
                .automorphism_group()
                .map(|inner| PyPermGroup { inner })
                .map_err(topology_error)
        })
    }

    /// Recheck the simplicial identities and normalized signed boundary.
    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .chain_complex(Z)
            .map_err(topology_error)?
            .verify()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SimplicialSet::from_json(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SimplicialSet(dimension={}, non_degenerate={:?})",
                self.inner.dimension(),
                self.non_degenerate_counts()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let counts = self.non_degenerate_counts();
        summary.set_item("type", SimplicialSet::TYPE_ID)?;
        summary.set_item("dimension", self.inner.dimension())?;
        summary.set_item(
            "non_degenerate_counts",
            counts
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        if counts.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("non_degenerate_counts_truncated", true)?;
        }
        summary.set_item("euler_characteristic", self.inner.euler_characteristic())?;
        Ok(summary.unbind())
    }
}

/// A checked finite real-valued filtration of an abstract simplicial complex.
#[pyclass(
    module = "noether.topology",
    name = "FilteredComplex",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyFilteredComplex {
    inner: FilteredComplex,
}

#[pymethods]
impl PyFilteredComplex {
    /// Construct from `(vertices, value)` entries in any order.
    #[new]
    fn new(entries: Vec<(Vec<u32>, f64)>) -> PyResult<Self> {
        let entries = entries
            .into_iter()
            .map(|(vertices, value)| (Simplex::new(vertices), value))
            .collect();
        Ok(Self {
            inner: FilteredComplex::new(entries).map_err(topology_error)?,
        })
    }

    /// Give every simplex of `complex` the same filtration value.
    #[staticmethod]
    #[pyo3(signature = (complex, value = 0.0))]
    fn constant(complex: &PySimplicialComplex, value: f64) -> PyResult<Self> {
        Ok(Self {
            inner: FilteredComplex::constant(&complex.inner, value).map_err(topology_error)?,
        })
    }

    /// Construct a Vietoris–Rips filtration from any checked finite metric.
    #[staticmethod]
    #[pyo3(signature = (metric, *, max_scale, max_dimension))]
    fn vietoris_rips(
        py: Python<'_>,
        metric: &Bound<'_, PyAny>,
        max_scale: f64,
        max_dimension: usize,
    ) -> PyResult<Self> {
        if let Ok(points) = metric.extract::<PyRef<'_, PyPointCloud>>() {
            let points = points.inner.clone();
            return py.detach(move || {
                Ok(Self {
                    inner: FilteredComplex::vietoris_rips(&points, max_scale, max_dimension)
                        .map_err(topology_error)?,
                })
            });
        }

        let matrix = checked_python_metric(metric)?;
        py.detach(move || {
            Ok(Self {
                inner: FilteredComplex::vietoris_rips(&matrix, max_scale, max_dimension)
                    .map_err(topology_error)?,
            })
        })
    }

    /// Construct a Čech filtration from a checked Euclidean point cloud.
    #[staticmethod]
    #[pyo3(signature = (points, *, max_scale, max_dimension))]
    fn cech(
        py: Python<'_>,
        points: &PyPointCloud,
        max_scale: f64,
        max_dimension: usize,
    ) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: FilteredComplex::cech(&points.inner, max_scale, max_dimension)
                    .map_err(topology_error)?,
            })
        })
    }

    /// Construct the alpha filtration carried by an existing triangulation.
    #[staticmethod]
    #[pyo3(signature = (triangulation, *, max_scale, max_dimension, max_face_slots = None))]
    fn alpha(
        py: Python<'_>,
        triangulation: &PyTriangulation,
        max_scale: f64,
        max_dimension: usize,
        max_face_slots: Option<usize>,
    ) -> PyResult<Self> {
        py.detach(|| {
            let inner = match max_face_slots {
                Some(limit) => FilteredComplex::alpha_with_max_face_slots(
                    &triangulation.inner,
                    max_scale,
                    max_dimension,
                    limit,
                ),
                None => FilteredComplex::alpha(&triangulation.inner, max_scale, max_dimension),
            };
            Ok(Self {
                inner: inner.map_err(topology_error)?,
            })
        })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        FilteredComplex::TYPE_ID
    }

    /// Largest simplex dimension; the empty filtered complex has dimension -1.
    #[getter]
    fn dimension(&self) -> i64 {
        self.inner.dimension()
    }

    /// Whether no nonempty simplex is present.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of stored simplices.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Simplices in canonical filtration order.
    #[getter]
    fn simplices(&self) -> Vec<Vec<u32>> {
        self.inner
            .simplices()
            .iter()
            .map(|simplex| simplex.vertices().to_vec())
            .collect()
    }

    /// Finite filtration values aligned with `simplices`.
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Canonical `(simplex, value)` entries.
    #[getter]
    fn entries(&self) -> Vec<(Vec<u32>, f64)> {
        self.inner
            .simplices()
            .iter()
            .zip(self.inner.values())
            .map(|(simplex, value)| (simplex.vertices().to_vec(), *value))
            .collect()
    }

    /// The sublevel simplicial complex with value at most `scale`.
    fn slice_at(&self, scale: f64) -> PyResult<PySimplicialComplex> {
        Ok(PySimplicialComplex {
            inner: self.inner.slice_at(scale).map_err(topology_error)?,
        })
    }

    /// Persistent homology over `Q` or a finite field.
    ///
    /// Prime fields use the specialized sparse engine reduction. Extension
    /// fields use the generic facade reduction as the exact correctness floor.
    #[pyo3(signature = (*, ring))]
    fn persistent_homology(&self, py: Python<'_>, ring: &Bound<'_, PyAny>) -> PyResult<PyBarcode> {
        if ring.is_instance_of::<PyRationalField>() {
            return py.detach(|| {
                Ok(PyBarcode {
                    inner: BarcodeInner::Rational(
                        self.inner.persistent_homology(Q).map_err(topology_error)?,
                    ),
                })
            });
        }

        let field = ring.extract::<PyRef<'_, PyFiniteField>>().map_err(|_| {
            PyTypeError::new_err(
                "persistent homology requires Q or a FiniteField; integral persistence has no barcode",
            )
        })?;
        let context = field.inner.clone();
        py.detach(|| {
            let barcode = self
                .inner
                .persistent_homology(context)
                .map_err(topology_error)?;
            Ok(PyBarcode {
                inner: BarcodeInner::FiniteField(barcode),
            })
        })
    }

    /// Recheck finiteness, uniqueness, face closure, and monotonicity.
    fn verify(&self) -> PyResult<()> {
        let entries = self
            .inner
            .simplices()
            .iter()
            .cloned()
            .zip(self.inner.values().iter().copied())
            .collect();
        let rebuilt = FilteredComplex::new(entries).map_err(topology_error)?;
        if rebuilt == self.inner {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "filtered-complex canonical reconstruction changed its value",
            ))
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "FilteredComplex(dimension={}, cells={})",
                self.inner.dimension(),
                self.inner.len()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", FilteredComplex::TYPE_ID)?;
        summary.set_item("dimension", self.inner.dimension())?;
        summary.set_item("cells", self.inner.len())?;
        summary.set_item("value_min", self.inner.values().first().copied())?;
        summary.set_item("value_max", self.inner.values().last().copied())?;
        Ok(summary.unbind())
    }
}

fn checked_python_metric(metric: &Bound<'_, PyAny>) -> PyResult<noether::geometry::DistanceMatrix> {
    const MAX_MATERIALIZED_ENTRIES: usize = 16_777_216;

    let len = metric
        .call_method0("__len__")
        .and_then(|value| value.extract::<usize>())
        .map_err(|_| {
            PyTypeError::new_err(
                "a Metric must define __len__() -> int and distance(i, j) -> float",
            )
        })?;
    if u32::try_from(len).is_err() {
        return Err(PyValueError::new_err(format!(
            "{len} metric points cannot be represented by u32 simplex labels"
        )));
    }
    let entries = len.checked_mul(len).ok_or_else(|| {
        PyValueError::new_err(format!("a {len} by {len} metric matrix is too large"))
    })?;
    if entries > MAX_MATERIALIZED_ENTRIES {
        return Err(PyValueError::new_err(format!(
            "a structural Python Metric on {len} points would materialize {entries} distances; use a Rust-backed PointCloud or metric for this size"
        )));
    }
    let mut rows = vec![vec![0.0; len]; len];
    for (i, row) in rows.iter_mut().enumerate() {
        for (j, distance) in row.iter_mut().enumerate() {
            *distance = metric
                .call_method1("distance", (i, j))
                .and_then(|value| value.extract::<f64>())
                .map_err(|_| {
                    PyTypeError::new_err(format!("Metric.distance({i}, {j}) must return a float"))
                })?;
        }
    }
    noether::geometry::DistanceMatrix::from_full(&rows).map_err(PyValueError::new_err)
}

#[derive(Clone)]
enum BarcodeInner {
    Rational(Barcode<RationalField>),
    FiniteField(Barcode<FiniteField>),
}

impl BarcodeInner {
    fn bars(&self) -> &[(usize, f64, Option<f64>)] {
        match self {
            Self::Rational(barcode) => barcode.bars(),
            Self::FiniteField(barcode) => barcode.bars(),
        }
    }

    fn dimensions(&self) -> Vec<usize> {
        match self {
            Self::Rational(barcode) => barcode.dimensions(),
            Self::FiniteField(barcode) => barcode.dimensions(),
        }
    }

    fn betti_at(&self, value: f64) -> Result<Vec<usize>, TopologyError> {
        match self {
            Self::Rational(barcode) => barcode.betti_at(value),
            Self::FiniteField(barcode) => barcode.betti_at(value),
        }
    }

    fn in_dimension(&self, dimension: usize) -> Self {
        match self {
            Self::Rational(barcode) => Self::Rational(barcode.in_dimension(dimension)),
            Self::FiniteField(barcode) => Self::FiniteField(barcode.in_dimension(dimension)),
        }
    }

    fn field_label(&self) -> String {
        match self {
            Self::Rational(_) => "Q".to_owned(),
            Self::FiniteField(barcode) if barcode.field().degree() == 1 => {
                format!("F_{}", barcode.field().characteristic())
            }
            Self::FiniteField(barcode) => format!(
                "F_{}^{}",
                barcode.field().characteristic(),
                barcode.field().degree()
            ),
        }
    }
}

/// Persistent-homology intervals carrying their exact coefficient field.
#[pyclass(module = "noether.topology", name = "Barcode", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyBarcode {
    inner: BarcodeInner,
}

#[pymethods]
impl PyBarcode {
    /// Construct checked `(dimension, birth, death)` intervals over `ring`.
    #[new]
    fn new(bars: Vec<(usize, f64, Option<f64>)>, ring: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = if ring.is_instance_of::<PyRationalField>() {
            BarcodeInner::Rational(Barcode::new(Q, bars).map_err(topology_error)?)
        } else if let Ok(field) = ring.extract::<PyRef<'_, PyFiniteField>>() {
            BarcodeInner::FiniteField(
                Barcode::new(field.inner.clone(), bars).map_err(topology_error)?,
            )
        } else {
            return Err(PyTypeError::new_err(
                "a Barcode coefficient context must be Q or a FiniteField",
            ));
        };
        Ok(Self { inner })
    }

    /// Versioned language-neutral semantic identity.
    #[classattr]
    fn type_id() -> &'static str {
        Barcode::<RationalField>::TYPE_ID
    }

    /// Coefficient field used for this computation.
    #[getter]
    fn field(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            BarcodeInner::Rational(_) => Ok(Py::new(py, PyRationalField)?.into_any()),
            BarcodeInner::FiniteField(barcode) => Ok(Py::new(
                py,
                PyFiniteField {
                    inner: barcode.field().clone(),
                },
            )?
            .into_any()),
        }
    }

    /// Canonically ordered `(dimension, birth, death)` intervals.
    #[getter]
    fn bars(&self) -> Vec<(usize, f64, Option<f64>)> {
        self.inner.bars().to_vec()
    }

    /// Homological dimensions carrying at least one interval.
    #[getter]
    fn dimensions(&self) -> Vec<usize> {
        self.inner.dimensions()
    }

    /// Essential intervals, whose death is `None`.
    #[getter]
    fn essential(&self) -> Vec<(usize, f64, Option<f64>)> {
        self.inner
            .bars()
            .iter()
            .filter(|(_, _, death)| death.is_none())
            .copied()
            .collect()
    }

    /// Interval lengths, with `None` for essential classes.
    #[getter]
    fn lifetimes(&self) -> Vec<Option<f64>> {
        self.inner
            .bars()
            .iter()
            .map(|(_, birth, death)| death.map(|death| death - birth))
            .collect()
    }

    /// Restrict to one homological dimension.
    fn in_dimension(&self, dimension: usize) -> Self {
        Self {
            inner: self.inner.in_dimension(dimension),
        }
    }

    /// Betti numbers at one finite filtration value.
    fn betti_at(&self, value: f64) -> PyResult<Vec<usize>> {
        self.inner.betti_at(value).map_err(topology_error)
    }

    /// Bottleneck distance, with dimensions and coefficient context checked.
    fn bottleneck_distance(&self, other: &Self) -> PyResult<f64> {
        match (&self.inner, &other.inner) {
            (BarcodeInner::Rational(left), BarcodeInner::Rational(right)) => {
                left.bottleneck_distance(right).map_err(topology_error)
            }
            (BarcodeInner::FiniteField(left), BarcodeInner::FiniteField(right)) => {
                left.bottleneck_distance(right).map_err(topology_error)
            }
            _ => Err(PyValueError::new_err(
                "cannot compare barcodes over different coefficient fields",
            )),
        }
    }

    /// Wasserstein distance, with dimensions and coefficient context checked.
    #[pyo3(signature = (other, *, power = 2.0))]
    fn wasserstein_distance(&self, other: &Self, power: f64) -> PyResult<f64> {
        match (&self.inner, &other.inner) {
            (BarcodeInner::Rational(left), BarcodeInner::Rational(right)) => left
                .wasserstein_distance(right, power)
                .map_err(topology_error),
            (BarcodeInner::FiniteField(left), BarcodeInner::FiniteField(right)) => left
                .wasserstein_distance(right, power)
                .map_err(topology_error),
            _ => Err(PyValueError::new_err(
                "cannot compare barcodes over different coefficient fields",
            )),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.bars().len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<(usize, f64, Option<f64>)> {
        let len = self.inner.bars().len() as isize;
        let index = if index < 0 { index + len } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("Barcode index out of range"));
        }
        Ok(self.inner.bars()[index as usize])
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyList::new(py, self.inner.bars())?.try_iter()?.into())
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Barcode(field={}, bars={}, dimensions={:?})",
                self.inner.field_label(),
                self.inner.bars().len(),
                self.inner.dimensions()
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (BarcodeInner::Rational(left), BarcodeInner::Rational(right)) => left == right,
            (BarcodeInner::FiniteField(left), BarcodeInner::FiniteField(right)) => left == right,
            _ => false,
        }
    }

    /// Bounded semantic summary for interactive and LLM-facing clients.
    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        let dimensions = self.inner.dimensions();
        summary.set_item("type", Barcode::<RationalField>::TYPE_ID)?;
        summary.set_item("field", self.inner.field_label())?;
        summary.set_item("bars", self.inner.bars().len())?;
        summary.set_item(
            "dimensions",
            dimensions
                .iter()
                .take(crate::py::repr::SUMMARY_ITEM_LIMIT)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        if dimensions.len() > crate::py::repr::SUMMARY_ITEM_LIMIT {
            summary.set_item("dimensions_truncated", true)?;
        }
        summary.set_item("essential", self.essential().len())?;
        Ok(summary.unbind())
    }
}

fn unsupported_coefficients() -> PyErr {
    PyTypeError::new_err(
        "topological chains currently support Z, Q, FiniteField, and NumberField coefficients",
    )
}
