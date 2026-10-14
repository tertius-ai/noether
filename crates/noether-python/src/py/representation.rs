// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of weight characters and type-A representations.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use std::hash::{DefaultHasher, Hash, Hasher};

use noether::groups::{
    PolynomialGlRepresentation, RepresentationError, WeightBasisConvention, WeightCharacter,
    WeightLattice, WeightLatticeMap,
};
use noether::{Integer, SemanticType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::facade_scalar::{extract_integer, integer_to_python};
use super::partition::PyPartition;
use super::polynomial_facade::PyPolynomial;
use super::series::PyPowerSeries;

fn representation_error(error: RepresentationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn serialization_error(error: impl ToString) -> PyErr {
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

fn extract_weight(value: &Bound<'_, PyAny>) -> PyResult<Vec<Integer>> {
    value
        .try_iter()?
        .map(|entry| extract_integer(&entry?))
        .collect()
}

fn weight_to_python(py: Python<'_>, weight: &[Integer]) -> PyResult<Py<PyTuple>> {
    let entries = weight
        .iter()
        .map(|entry| integer_to_python(py, entry))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, entries)?.unbind())
}

fn extract_terms(value: &Bound<'_, PyAny>) -> PyResult<Vec<(Vec<Integer>, Integer)>> {
    value
        .try_iter()?
        .map(|term| {
            let term = term?;
            let tuple = term.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err("character terms must be (weight, multiplicity) pairs")
            })?;
            if tuple.len() != 2 {
                return Err(PyTypeError::new_err(
                    "character terms must be (weight, multiplicity) pairs",
                ));
            }
            Ok((
                extract_weight(&tuple.get_item(0)?)?,
                extract_integer(&tuple.get_item(1)?)?,
            ))
        })
        .collect()
}

#[pyclass(
    module = "noether.groups",
    name = "WeightLattice",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// A free weight lattice with a named basis and an explicit basis convention.
pub struct PyWeightLattice {
    inner: WeightLattice,
}

impl Serialize for PyWeightLattice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyWeightLattice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WeightLattice::deserialize(deserializer).map(Self::from_core)
    }
}

impl PyWeightLattice {
    pub(crate) fn from_core(inner: WeightLattice) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWeightLattice {
    #[new]
    fn new(basis_names: Vec<String>) -> PyResult<Self> {
        WeightLattice::new(basis_names)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    #[staticmethod]
    fn standard(rank: usize) -> PyResult<Self> {
        WeightLattice::standard(rank)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    #[staticmethod]
    fn type_a(n: usize) -> PyResult<Self> {
        WeightLattice::type_a(n)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        WeightLattice::TYPE_ID
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn basis_names(&self) -> Vec<String> {
        self.inner.basis_names().to_vec()
    }

    #[getter]
    fn basis_convention(&self) -> &'static str {
        match self.inner.convention() {
            WeightBasisConvention::Standard => "standard",
            WeightBasisConvention::TypeAFundamental { .. } => "type_a_fundamental",
        }
    }

    #[getter]
    fn type_a_degree(&self) -> Option<usize> {
        match self.inner.convention() {
            WeightBasisConvention::Standard => None,
            WeightBasisConvention::TypeAFundamental { degree } => Some(*degree),
        }
    }

    fn homomorphism(
        &self,
        target: &PyWeightLattice,
        basis_images: &Bound<'_, PyAny>,
    ) -> PyResult<PyWeightLatticeMap> {
        let images = basis_images
            .try_iter()?
            .map(|image| extract_weight(&image?))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .homomorphism(target.inner.clone(), images)
            .map(PyWeightLatticeMap::from_core)
            .map_err(representation_error)
    }

    fn identity_map(&self) -> PyWeightLatticeMap {
        PyWeightLatticeMap::from_core(self.inner.identity_map())
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(serialization_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        WeightLattice::from_json(value)
            .map(Self::from_core)
            .map_err(serialization_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }
}

#[pyclass(
    module = "noether.groups",
    name = "WeightLatticeMap",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// A homomorphism between two explicit weight lattices.
pub struct PyWeightLatticeMap {
    inner: WeightLatticeMap,
}

impl Serialize for PyWeightLatticeMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyWeightLatticeMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WeightLatticeMap::deserialize(deserializer).map(Self::from_core)
    }
}

impl PyWeightLatticeMap {
    pub(crate) fn from_core(inner: WeightLatticeMap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWeightLatticeMap {
    #[classattr]
    fn type_id() -> &'static str {
        WeightLatticeMap::TYPE_ID
    }

    #[getter]
    fn source(&self) -> PyWeightLattice {
        PyWeightLattice::from_core(self.inner.source().clone())
    }

    #[getter]
    fn target(&self) -> PyWeightLattice {
        PyWeightLattice::from_core(self.inner.target().clone())
    }

    #[getter]
    fn basis_images(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let images = self
            .inner
            .basis_images()
            .iter()
            .map(|image| weight_to_python(py, image))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, images)?.unbind())
    }

    fn apply(&self, weight: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let image = self
            .inner
            .apply(&extract_weight(weight)?)
            .map_err(representation_error)?;
        weight_to_python(py, &image)
    }

    fn then(&self, next: &Self) -> PyResult<Self> {
        self.inner
            .then(&next.inner)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(serialization_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        WeightLatticeMap::from_json(value)
            .map(Self::from_core)
            .map_err(serialization_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "WeightLatticeMap(source_rank={}, target_rank={})",
            self.inner.source().rank(),
            self.inner.target().rank()
        ))
    }
}

#[pyclass(
    module = "noether.groups",
    name = "WeightCharacter",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// A finite formal character on an explicit weight lattice.
pub struct PyWeightCharacter {
    inner: WeightCharacter,
}

impl Serialize for PyWeightCharacter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyWeightCharacter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WeightCharacter::deserialize(deserializer).map(Self::from_core)
    }
}

impl PyWeightCharacter {
    pub(crate) fn from_core(inner: WeightCharacter) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWeightCharacter {
    #[new]
    fn new(lattice: &PyWeightLattice, terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        WeightCharacter::new(lattice.inner.clone(), extract_terms(terms)?)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        WeightCharacter::TYPE_ID
    }

    #[getter]
    fn lattice(&self) -> PyWeightLattice {
        PyWeightLattice::from_core(self.inner.lattice().clone())
    }

    #[getter]
    fn terms(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let terms = self
            .inner
            .terms()
            .map(|(weight, multiplicity)| {
                let weight = weight_to_python(py, weight)?.into_any();
                let multiplicity = integer_to_python(py, multiplicity)?;
                Ok(PyTuple::new(py, [weight, multiplicity])?.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, terms)?.unbind())
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.dimension())
    }

    fn multiplicity(&self, weight: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let multiplicity = self
            .inner
            .multiplicity(&extract_weight(weight)?)
            .map_err(representation_error)?;
        integer_to_python(py, &multiplicity)
    }

    fn checked_add(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .checked_add(&other.inner)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    fn tensor(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .tensor(&other.inner)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    fn dual(&self) -> Self {
        Self::from_core(self.inner.dual())
    }

    fn restrict(&self, map: &PyWeightLatticeMap) -> PyResult<Self> {
        self.inner
            .restrict(&map.inner)
            .map(Self::from_core)
            .map_err(representation_error)
    }

    fn chern_class(&self, degree: u64) -> PyResult<PyPolynomial> {
        self.inner
            .chern_class(degree)
            .map(PyPolynomial::from_integer)
            .map_err(representation_error)
    }

    fn total_chern_class(&self, maximum_degree: u64) -> PyResult<PyPowerSeries> {
        self.inner
            .total_chern_class(maximum_degree)
            .map(PyPowerSeries::from_integer)
            .map_err(representation_error)
    }

    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(representation_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(serialization_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        WeightCharacter::from_json(value)
            .map(Self::from_core)
            .map_err(serialization_error)
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.checked_add(other)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.tensor(other)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("lattice_rank", self.inner.lattice().rank())?;
        summary.set_item("weight_count", self.inner.terms().count())?;
        summary.set_item("dimension", self.inner.dimension().to_string())?;
        Ok(summary.unbind())
    }
}

#[pyclass(
    module = "noether.groups",
    name = "PolynomialGlRepresentation",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// The irreducible polynomial representation of GL indexed by a partition.
pub struct PyPolynomialGlRepresentation {
    inner: PolynomialGlRepresentation,
}

impl Serialize for PyPolynomialGlRepresentation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyPolynomialGlRepresentation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        PolynomialGlRepresentation::deserialize(deserializer).map(Self::from_core)
    }
}

impl PyPolynomialGlRepresentation {
    pub(crate) fn from_core(inner: PolynomialGlRepresentation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPolynomialGlRepresentation {
    #[new]
    fn new(rank: usize, partition: &PyPartition) -> PyResult<Self> {
        PolynomialGlRepresentation::new(rank, partition.core().clone())
            .map(Self::from_core)
            .map_err(representation_error)
    }

    #[classattr]
    fn type_id() -> &'static str {
        PolynomialGlRepresentation::TYPE_ID
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn partition(&self) -> PyPartition {
        PyPartition::from_core(self.inner.partition().clone())
    }

    #[getter]
    fn highest_weight(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        weight_to_python(py, &self.inner.highest_weight())
    }

    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.degree())
    }

    #[getter]
    fn weyl_dimension(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dimension = self.inner.weyl_dimension().map_err(representation_error)?;
        integer_to_python(py, &dimension)
    }

    fn gl_character(&self) -> PyResult<PyWeightCharacter> {
        self.inner
            .gl_character()
            .map(PyWeightCharacter::from_core)
            .map_err(representation_error)
    }

    fn sl_character(&self) -> PyResult<PyWeightCharacter> {
        self.inner
            .sl_character()
            .map(PyWeightCharacter::from_core)
            .map_err(representation_error)
    }

    #[getter]
    fn scalar_central_character(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        integer_to_python(py, &self.inner.scalar_central_character())
    }

    fn descends_to_sl_quotient(&self, order: usize) -> PyResult<bool> {
        self.inner
            .descends_to_sl_quotient(order)
            .map_err(representation_error)
    }

    /// Exact multiple of the standard `SL_n` second Chern class.
    fn second_chern_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let index = self
            .inner
            .second_chern_index()
            .map_err(representation_error)?;
        integer_to_python(py, &index)
    }

    fn second_chern_index_from_weights(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let index = self
            .inner
            .second_chern_index_from_weights()
            .map_err(representation_error)?;
        integer_to_python(py, &index)
    }

    fn verify_second_chern_index(&self) -> PyResult<()> {
        self.inner
            .verify_second_chern_index()
            .map_err(representation_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(serialization_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        PolynomialGlRepresentation::from_json(value)
            .map(Self::from_core)
            .map_err(serialization_error)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> isize {
        canonical_hash(&self.inner)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("rank", self.inner.rank())?;
        summary.set_item("partition", self.inner.partition().parts())?;
        summary.set_item("degree", self.inner.degree().to_string())?;
        summary.set_item(
            "weyl_dimension",
            self.inner
                .weyl_dimension()
                .map_err(representation_error)?
                .to_string(),
        )?;
        Ok(summary.unbind())
    }
}
