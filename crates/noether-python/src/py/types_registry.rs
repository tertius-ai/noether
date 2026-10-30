// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Python adapters for the Rust facade's accepted semantic wire types.
//!
//! [`noether::accepted_type_schemas`] is the authority for identities,
//! versions, and schemas. This module contains only language conversion:
//! registering the matching pyclass and translating canonical JSON to or from
//! that wrapper.
#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::{PyClass, PyClassInitializer, PyTypeInfo};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::py::hyperelliptic::{PyHyperellipticCurve, PyHyperellipticFrobeniusEvidence};
use crate::py::surface::{PyFrobeniusPolynomial, PyZetaFunction};
use noether::category::SteinerCategory;
use noether::combin::{
    Composition, Graph, KashiwaraCrystal, Matroid, Partition, Poset, SymmetricFunction, Tableau,
    Tree,
};
use noether::commalg::{
    FrobeniusPolynomial, HyperellipticCurve, HyperellipticFrobeniusEvidence, ZetaFunction,
};
use noether::commalg::{PolynomialHomomorphism, PolynomialRing, RationalFunctionFieldMap};
use noether::groups::{
    PolynomialGlRepresentation, WeightCharacter, WeightLattice, WeightLatticeMap,
};
use noether::homalg::{
    ExteriorAlgebra, ExteriorElement, FinitelyGeneratedAbelianGroup, ImageSubmodule,
    KernelSubmodule, ModulePowerHomomorphism, PresentedModulePowerHomomorphism,
};
use noether::numthy::{FiniteField, FiniteFieldEmbedding};
use noether::topology::{
    EilenbergMacLaneIntegralCohomology, EilenbergMacLaneIntegralCohomologyPresentation,
    EilenbergMacLaneIntegralHomology, EilenbergMacLaneModularCohomology,
    EilenbergMacLaneModularCohomologyTruncation, EilenbergMacLaneSpace,
};
use noether::{
    IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction, IntegerRing,
    PrimeSubfieldEmbedding, RationalField, SemanticType,
};

use crate::py::category::PySteinerCategory;
use crate::py::coefficient_maps::{
    PyIntegerFiniteFieldMap, PyIntegerRationalEmbedding, PyIntegerReduction,
    PyPrimeSubfieldEmbedding,
};
use crate::py::crystals::PyKashiwaraCrystal;
use crate::py::graph::PyGraph;
use crate::py::homalg::PyFinitelyGeneratedAbelianGroup;
use crate::py::matroid::PyMatroid;
use crate::py::module_facade::{PyImageSubmodule, PyKernelSubmodule};
use crate::py::module_power::{
    PyExteriorAlgebra, PyExteriorElement, PyModulePowerHomomorphism,
    PyPresentedModulePowerHomomorphism,
};
use crate::py::numthy::PyFiniteFieldEmbedding;
use crate::py::partition::{PyComposition, PyPartition};
use crate::py::polynomial_facade::PyPolynomialHomomorphism;
use crate::py::poset::PyPoset;
use crate::py::rational_function::PyRationalFunctionFieldMap;
use crate::py::representation::{
    PyPolynomialGlRepresentation, PyWeightCharacter, PyWeightLattice, PyWeightLatticeMap,
};
use crate::py::symmetric_function::PySymmetricFunction;
use crate::py::tableau::PyTableau;
use crate::py::topology_facade::{
    PyEilenbergMacLaneIntegralCohomology, PyEilenbergMacLaneIntegralCohomologyPresentation,
    PyEilenbergMacLaneIntegralHomology, PyEilenbergMacLaneModularCohomology,
    PyEilenbergMacLaneModularCohomologyTruncation, PyEilenbergMacLaneSpace,
};
use crate::py::tree::PyTree;

type RegisteredPolynomialHomomorphism =
    PolynomialHomomorphism<IntegerRing, RationalField, IntegerRationalEmbedding>;
type RegisteredRationalFunctionFieldMap =
    RationalFunctionFieldMap<FiniteField, FiniteField, PrimeSubfieldEmbedding>;
type RegisteredExteriorAlgebra = ExteriorAlgebra<PolynomialRing<RationalField>>;
type RegisteredExteriorElement = ExteriorElement<PolynomialRing<RationalField>>;
type RegisteredModulePowerHomomorphism = ModulePowerHomomorphism<PolynomialRing<RationalField>>;
type RegisteredPresentedModulePowerHomomorphism =
    PresentedModulePowerHomomorphism<PolynomialRing<RationalField>>;

struct TypeAdapter {
    type_id: &'static str,
    py_name: fn(Python<'_>) -> PyResult<String>,
    hydrate: fn(Python<'_>, &str, &str) -> PyResult<Py<PyAny>>,
    dump: fn(&Bound<'_, PyAny>) -> Option<PyResult<String>>,
    matches: fn(&Bound<'_, PyAny>) -> bool,
    add_class: fn(&Bound<'_, PyModule>) -> PyResult<()>,
}

fn py_name<T: PyTypeInfo>(py: Python<'_>) -> PyResult<String> {
    let type_object = T::type_object(py);
    let name = pyo3::types::PyTypeMethods::name(&type_object)?;
    Ok(pyo3::types::PyStringMethods::to_string_lossy(&name).into_owned())
}

fn hydrate_adapter<T>(py: Python<'_>, type_id: &str, json: &str) -> PyResult<Py<PyAny>>
where
    T: PyClass + DeserializeOwned + Send + Into<PyClassInitializer<T>>,
{
    let value: T = py.detach(|| serde_json::from_str(json)).map_err(|error| {
        PyValueError::new_err(format!(
            "hydrate({type_id}): invalid canonical JSON: {error}"
        ))
    })?;
    Ok(Py::new(py, value)?.into_any())
}

fn dump_adapter<T>(obj: &Bound<'_, PyAny>) -> Option<PyResult<String>>
where
    T: PyClass + Serialize + Sync,
{
    let cell = obj.cast::<T>().ok()?;
    let guard = cell.borrow();
    let value: &T = &guard;
    Some(
        obj.py()
            .detach(|| serde_json::to_string(value))
            .map_err(|error| {
                PyValueError::new_err(format!("canonical JSON serialization failed: {error}"))
            }),
    )
}

fn matches<T: PyClass>(obj: &Bound<'_, PyAny>) -> bool {
    obj.cast::<T>().is_ok()
}

fn add_class<T: PyClass>(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<T>()
}

macro_rules! adapter {
    ($python:ty, $semantic:ty) => {
        TypeAdapter {
            type_id: <$semantic as SemanticType>::TYPE_ID,
            py_name: py_name::<$python>,
            hydrate: hydrate_adapter::<$python>,
            dump: dump_adapter::<$python>,
            matches: matches::<$python>,
            add_class: add_class::<$python>,
        }
    };
}

static TYPE_ADAPTERS: &[TypeAdapter] = &[
    adapter!(PyHyperellipticCurve, HyperellipticCurve),
    adapter!(
        PyHyperellipticFrobeniusEvidence,
        HyperellipticFrobeniusEvidence
    ),
    adapter!(PyFrobeniusPolynomial, FrobeniusPolynomial),
    adapter!(PyZetaFunction, ZetaFunction),
    adapter!(PyIntegerRationalEmbedding, IntegerRationalEmbedding),
    adapter!(PyIntegerReduction, IntegerReduction),
    adapter!(PyIntegerFiniteFieldMap, IntegerFiniteFieldMap),
    adapter!(PyPrimeSubfieldEmbedding, PrimeSubfieldEmbedding),
    adapter!(PyFiniteFieldEmbedding, FiniteFieldEmbedding),
    adapter!(PyPolynomialHomomorphism, RegisteredPolynomialHomomorphism),
    adapter!(
        PyRationalFunctionFieldMap,
        RegisteredRationalFunctionFieldMap
    ),
    adapter!(PySteinerCategory, SteinerCategory),
    adapter!(PyKashiwaraCrystal, KashiwaraCrystal),
    adapter!(PyGraph, Graph),
    adapter!(PyMatroid, Matroid),
    adapter!(PyPartition, Partition),
    adapter!(PyComposition, Composition),
    adapter!(PyPoset, Poset),
    adapter!(PySymmetricFunction, SymmetricFunction),
    adapter!(PyTableau, Tableau),
    adapter!(PyWeightLattice, WeightLattice),
    adapter!(PyWeightLatticeMap, WeightLatticeMap),
    adapter!(PyWeightCharacter, WeightCharacter),
    adapter!(PyPolynomialGlRepresentation, PolynomialGlRepresentation),
    adapter!(PyTree, Tree),
    adapter!(PyImageSubmodule, ImageSubmodule<IntegerRing>),
    adapter!(PyKernelSubmodule, KernelSubmodule<IntegerRing>),
    adapter!(PyExteriorAlgebra, RegisteredExteriorAlgebra),
    adapter!(PyExteriorElement, RegisteredExteriorElement),
    adapter!(
        PyFinitelyGeneratedAbelianGroup,
        FinitelyGeneratedAbelianGroup
    ),
    adapter!(PyModulePowerHomomorphism, RegisteredModulePowerHomomorphism),
    adapter!(
        PyPresentedModulePowerHomomorphism,
        RegisteredPresentedModulePowerHomomorphism
    ),
    adapter!(PyEilenbergMacLaneSpace, EilenbergMacLaneSpace),
    adapter!(
        PyEilenbergMacLaneIntegralHomology,
        EilenbergMacLaneIntegralHomology
    ),
    adapter!(
        PyEilenbergMacLaneIntegralCohomology,
        EilenbergMacLaneIntegralCohomology
    ),
    adapter!(
        PyEilenbergMacLaneIntegralCohomologyPresentation,
        EilenbergMacLaneIntegralCohomologyPresentation
    ),
    adapter!(
        PyEilenbergMacLaneModularCohomology,
        EilenbergMacLaneModularCohomology
    ),
    adapter!(
        PyEilenbergMacLaneModularCohomologyTruncation,
        EilenbergMacLaneModularCohomologyTruncation
    ),
];

fn lookup(type_id: &str) -> Option<&'static TypeAdapter> {
    TYPE_ADAPTERS
        .iter()
        .find(|adapter| adapter.type_id == type_id)
}

/// Translate canonical JSON for an accepted Rust semantic type into Python.
pub fn hydrate(py: Python<'_>, type_id: &str, json: &str) -> PyResult<Py<PyAny>> {
    let adapter = lookup(type_id).ok_or_else(|| {
        PyValueError::new_err(format!(
            "hydrate: unknown type_id {type_id:?} (not in the Rust facade registry)"
        ))
    })?;
    (adapter.hydrate)(py, type_id, json)
}

/// Translate a registered Python wrapper to its Rust-owned wire identity and JSON.
pub fn dump(obj: &Bound<'_, PyAny>) -> Option<PyResult<(String, String)>> {
    for adapter in TYPE_ADAPTERS {
        if let Some(result) = (adapter.dump)(obj) {
            return Some(result.map(|json| (adapter.type_id.to_owned(), json)));
        }
    }
    None
}

#[pyfunction]
#[pyo3(name = "_dump")]
pub fn py_dump(obj: &Bound<'_, PyAny>) -> PyResult<Option<(String, String)>> {
    dump(obj).transpose()
}

#[pyfunction]
#[pyo3(name = "_type_id_of")]
pub fn py_type_id_of(obj: &Bound<'_, PyAny>) -> Option<String> {
    if let Ok(type_id) = obj.getattr("type") {
        if let Ok(type_id) = type_id.extract::<String>() {
            if lookup(&type_id).is_some() {
                return Some(type_id);
            }
        }
    }
    TYPE_ADAPTERS
        .iter()
        .find(|adapter| (adapter.matches)(obj))
        .map(|adapter| adapter.type_id.to_owned())
}

fn validate_adapter_projection(py: Python<'_>) -> PyResult<()> {
    let accepted = noether::accepted_type_schemas();
    if accepted.len() != TYPE_ADAPTERS.len() {
        return Err(PyValueError::new_err(format!(
            "Python transport has {} type adapters, but the Rust facade accepts {} wire types",
            TYPE_ADAPTERS.len(),
            accepted.len()
        )));
    }

    let mut seen_names = std::collections::HashSet::new();
    let mut seen_ids = std::collections::HashSet::new();
    for adapter in TYPE_ADAPTERS {
        let registration = noether::type_schema(adapter.type_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Python transport adapter {:?} is absent from the Rust facade registry",
                adapter.type_id
            ))
        })?;
        if !seen_ids.insert(adapter.type_id) {
            return Err(PyValueError::new_err(format!(
                "Python transport repeats semantic type {:?}",
                adapter.type_id
            )));
        }
        let name = (adapter.py_name)(py)?;
        if !seen_names.insert(name.clone()) {
            return Err(PyValueError::new_err(format!(
                "Python transport repeats pyclass name {name:?}"
            )));
        }
        let expected_name = registration
            .description()
            .python_path
            .and_then(|path| path.rsplit_once('.').map(|(_, name)| name));
        if expected_name != Some(name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Python class {name:?} does not match Rust path {:?}",
                registration.description().python_path
            )));
        }
    }
    Ok(())
}

/// Register every Python adapter after proving it exactly projects Rust truth.
pub fn register_all_types(module: &Bound<'_, PyModule>) -> PyResult<()> {
    validate_adapter_projection(module.py())?;
    for adapter in TYPE_ADAPTERS {
        (adapter.add_class)(module)?;
    }
    Ok(())
}

/// Rust-owned `(type_id, version)` entries exposed to schema tooling.
pub fn registered_types() -> Vec<(&'static str, u32)> {
    noether::accepted_type_schemas()
        .iter()
        .map(|registration| (registration.description().semantic_id, registration.version))
        .collect()
}

#[pyfunction]
pub fn schema_for(type_id: &str) -> PyResult<String> {
    let registration = noether::type_schema(type_id).ok_or_else(|| {
        PyValueError::new_err(format!(
            "schema_for: unknown type_id {type_id:?} (not in the Rust facade registry)"
        ))
    })?;
    serde_json::to_string_pretty(&registration.schema())
        .map_err(|error| PyValueError::new_err(format!("schema_for({type_id:?}) failed: {error}")))
}

#[pyfunction]
#[pyo3(name = "_hydrate")]
pub fn py_hydrate(py: Python<'_>, type_id: &str, json: &str) -> PyResult<Py<PyAny>> {
    hydrate(py, type_id, json)
}

#[pyfunction]
#[pyo3(name = "_registered_type_ids")]
pub fn py_registered_type_ids() -> Vec<(String, u32)> {
    registered_types()
        .into_iter()
        .map(|(type_id, version)| (type_id.to_owned(), version))
        .collect()
}

/// Register private transport functions used by package tooling.
pub fn register_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(schema_for, module)?)?;
    module.add_function(wrap_pyfunction!(py_hydrate, module)?)?;
    module.add_function(wrap_pyfunction!(py_dump, module)?)?;
    module.add_function(wrap_pyfunction!(py_type_id_of, module)?)?;
    module.add_function(wrap_pyfunction!(py_registered_type_ids, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_exactly_cover_the_rust_registry() {
        let adapter_ids = TYPE_ADAPTERS
            .iter()
            .map(|adapter| adapter.type_id)
            .collect::<std::collections::HashSet<_>>();
        let rust_ids = noether::accepted_type_schemas()
            .iter()
            .map(|registration| registration.description().semantic_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(adapter_ids, rust_ids);
    }
}
